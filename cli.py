#!/usr/bin/env python3
"""workflow-engine CLI — load a workflow definition, run one step.

A workflow is supplied as a directory containing an `__init__.py` (or a
single `config.py`) that exports a `WORKFLOW` (or legacy `SKILL_CONFIG`)
constant of type `core.WorkflowConfig`.

Usage:

  python3 cli.py --workflow PATH/TO/workflow_dir \\
      --session sess-001 --user "..."

  python3 cli.py --workflow PATH --session sess-001 --status
  python3 cli.py --workflow PATH --session sess-001 --reset

Output: JSON with workflow, phase, iteration, mini_prompt,
instructions_for_bot, state_summary, state_file.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path

# Make core/ importable when running directly from a checkout.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def load_workflow_config(workflow_path: Path):
    """Load WORKFLOW (or legacy SKILL_CONFIG) constant from a workflow path.

    Two layouts accepted:
      • directory: must contain `__init__.py` exporting WORKFLOW.
        We add the parent dir to sys.path and import by directory name.
        Relative imports inside the package (e.g. `from . import prompts`)
        work naturally.
      • single file: a config.py with `WORKFLOW` at module level.
    """
    if not workflow_path.exists():
        raise SystemExit(f"workflow path does not exist: {workflow_path}")

    if workflow_path.is_dir():
        init_py = workflow_path / "__init__.py"
        if not init_py.exists():
            raise SystemExit(
                f"workflow dir missing __init__.py: {workflow_path}"
            )
        parent = workflow_path.parent
        pkg_name = workflow_path.name
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        try:
            mod = importlib.import_module(pkg_name)
        except ImportError as e:
            raise SystemExit(f"failed to import workflow {pkg_name!r}: {e}")
    else:
        # Single-file: load by absolute path, no package context.
        spec = importlib.util.spec_from_file_location(
            workflow_path.stem, workflow_path
        )
        if spec is None or spec.loader is None:
            raise SystemExit(f"could not load workflow file: {workflow_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    cfg = getattr(mod, "WORKFLOW", None) or getattr(mod, "SKILL_CONFIG", None)
    if cfg is None:
        raise SystemExit(
            f"workflow module {workflow_path} does not export WORKFLOW "
            f"(or legacy SKILL_CONFIG)"
        )
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--workflow",
        required=False,
        default=None,
        help="Path to workflow directory (or single .py file) exporting WORKFLOW.",
    )

    # ─── identity (registry-backed lifecycle) ──────────────────────
    # The plugin (or any caller) resolves identity BEFORE invoking CLI:
    #   1. Compute agent_id + conversation_id from the request context.
    #   2. Look up active invocation via Registry.find_active(...).
    #   3. If none, Registry.start(...) → new invocation_id.
    #   4. Pass invocation_id to this CLI.
    # CLI is a dumb worker — does NOT resolve identity itself.
    #
    # ``--session`` is kept as a back-compat alias for ``--invocation-id``
    # (and as the file-name key when the file-based legacy backend is
    # used). New callers should prefer ``--invocation-id``.
    ap.add_argument(
        "--invocation-id", default=None,
        help="Workflow invocation id (registry primary key). New callers "
             "should use this; legacy callers may use --session as an alias.",
    )
    ap.add_argument(
        "--session", default=None,
        help="DEPRECATED alias for --invocation-id (file-based back-compat).",
    )
    ap.add_argument(
        "--agent-id", default=None,
        help="Required for --start. Identifies who runs the workflow "
             "(main, chief-XYZ, worker-N).",
    )
    ap.add_argument(
        "--conversation-id", default=None,
        help="Required for --start. Stable per chat (tg-dm-<chat_id>, etc).",
    )
    ap.add_argument(
        "--is-test", action="store_true",
        help="Mark new invocation as test — janitor uses tighter timeout.",
    )

    # ─── workflow step inputs ──────────────────────────────────────
    ap.add_argument("--user", default=None, help="Latest user message")
    ap.add_argument("--prev-bot", default=None, help="Previous bot reply for slot extraction")

    # ─── backend selection ─────────────────────────────────────────
    ap.add_argument(
        "--state-dir", default=None,
        help="Override on-disk state dir (artifacts, history). Registry "
             "backend ignores this for live state.",
    )
    ap.add_argument(
        "--no-registry", action="store_true",
        help="Disable registry backend, use file-only (legacy). Default: "
             "registry on (sqlite at WORKFLOW_REGISTRY_DB, default "
             "/opt/data/state.db).",
    )

    # ─── lifecycle commands ────────────────────────────────────────
    ap.add_argument("--status", action="store_true", help="Report state without advancing")
    ap.add_argument("--reset", action="store_true", help="Cancel + clear state for this invocation")
    ap.add_argument(
        "--start", action="store_true",
        help="Create a fresh invocation (Registry.start) and print its id. "
             "Requires --agent-id and --conversation-id. Returns "
             "{invocation_id, workflow, agent_id, conversation_id}.",
    )
    ap.add_argument(
        "--find-active", action="store_true",
        help="Look up active invocation by --agent-id + --conversation-id. "
             "Returns the id or null.",
    )
    ap.add_argument(
        "--cancel", action="store_true",
        help="Mark --invocation-id as CANCELLED (same as --reset alias).",
    )

    args = ap.parse_args()

    if not args.workflow:
        print(json.dumps({"error": "--workflow PATH required"}), file=sys.stderr)
        return 2

    config = load_workflow_config(Path(args.workflow).expanduser().resolve())

    from core import run, run_status, run_reset, Registry

    state_dir = Path(args.state_dir) if args.state_dir else None
    registry = None if args.no_registry else Registry()
    invocation_id = args.invocation_id or args.session

    # ── lifecycle-only commands (no run) ────────────────────────────
    if args.start:
        if registry is None:
            print(json.dumps({"error": "--start requires registry backend"}),
                  file=sys.stderr)
            return 2
        if not args.agent_id or not args.conversation_id:
            print(json.dumps({
                "error": "--start requires --agent-id and --conversation-id",
            }), file=sys.stderr)
            return 2
        rec = registry.start(
            workflow_name=config.name,
            agent_id=args.agent_id,
            conversation_id=args.conversation_id,
            is_test=args.is_test,
        )
        print(json.dumps({
            "invocation_id": rec.invocation_id,
            "workflow": rec.workflow_name,
            "agent_id": rec.agent_id,
            "conversation_id": rec.conversation_id,
            "started_ts": rec.started_ts,
            "is_test": rec.is_test,
        }, ensure_ascii=False, indent=2))
        return 0

    if args.find_active:
        if registry is None:
            print(json.dumps({"error": "--find-active requires registry"}),
                  file=sys.stderr)
            return 2
        if not args.agent_id or not args.conversation_id:
            print(json.dumps({
                "error": "--find-active requires --agent-id and --conversation-id",
            }), file=sys.stderr)
            return 2
        rec = registry.find_active(
            workflow_name=config.name,
            agent_id=args.agent_id,
            conversation_id=args.conversation_id,
        )
        print(json.dumps(
            {"invocation_id": rec.invocation_id, "phase": rec.phase,
             "iteration": rec.iteration} if rec else {"invocation_id": None},
            ensure_ascii=False, indent=2,
        ))
        return 0

    if args.cancel:
        if not invocation_id:
            print(json.dumps({"error": "--cancel requires --invocation-id"}),
                  file=sys.stderr)
            return 2
        out = run_reset(config, invocation_id, state_dir, registry)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    # ── existing run/status/reset paths (need invocation_id) ────────
    if not invocation_id:
        print(json.dumps({
            "error": "--invocation-id (or legacy --session) required",
        }), file=sys.stderr)
        return 2

    if args.reset:
        out = run_reset(config, invocation_id, state_dir, registry)
    elif args.status:
        out = run_status(config, invocation_id, state_dir, registry)
    else:
        if not args.user:
            print(json.dumps({"error": "--user required for advance"}), file=sys.stderr)
            return 2
        out = run(config, invocation_id, args.user, args.prev_bot,
                  state_dir, registry)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
