"""Generic runner — load state, run one decision step, build prompt, save.

Thin glue layer that:
1. Loads WorkflowState from JSON (or creates fresh)
2. Appends new user msg + previous bot reply to history
3. Constructs WorkflowMachine + calls decide() → target phase
4. Resolves target phase's prompt_builder + builds mini_prompt
5. Saves state
6. Returns a structured result dict for the CLI to JSON-serialize

Workflow-specific code lives outside this package (caller passes a
WorkflowConfig); the runner knows nothing workflow-specific.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .machine import WorkflowMachine
from .registry import Registry, InvocationRecord
from .state import WorkflowState, WorkflowConfig

STATE_DIR_DEFAULT = Path(
    os.environ.get(
        "WORKFLOW_STATE_DIR",
        str(Path.home() / ".hermes" / "workflow_state"),
    )
)


def _state_path(state_dir: Path, workflow_name: str, session: str) -> Path:
    """Legacy per-workflow subdir + session.json layout.

    Retained for the file-based fallback path in ``run()`` (when callers
    pass no registry / no invocation_id). New code MUST go through the
    Registry-backed lifecycle.
    """
    return state_dir / workflow_name / f"{session}.json"


def _artifact_path(state_dir: Path, workflow_name: str, invocation: str) -> Path:
    """Per-invocation final YAML artifact (P5).

    Lives in ``<state_dir>/<workflow>/artifacts/<invocation_id>.yaml`` —
    the invocation_id is the new identity (was ``session`` previously).
    Path is on disk because downstream skill consumers read it as a file.
    """
    return state_dir / workflow_name / "artifacts" / f"{invocation}.yaml"


def _compute_clarity_score(wstate, schema) -> float:
    """Programmatic clarity_score from schema weights + veto penalties.

    Components (each 0..1):
      completeness   filled_required / required_total
      confidence     mean confidence over required slots
      stability      1 - contradictions/iteration (rough)
      grounding      1 - (training_grounded violations / required_total)
      contradictions 1 - min(1, len(contradictions)/iteration)

    Then subtract per-veto severity penalty from any violations
    recorded into ``wstate.extras["last_violations"]`` by P4 detectors.
    """
    from .schema import SchemaSlots
    from .anti_patterns import apply_penalty, Violation
    slots = wstate.slots
    weights = schema.clarity_score.weights or {}

    completeness = slots.completeness() if hasattr(slots, "completeness") else 0.0

    # Mean confidence over required slots only
    if isinstance(slots, SchemaSlots):
        req = slots.required_keys()
        conf = slots.confidence_dict()
        confidence = (sum(conf.get(k, 0.0) for k in req) / len(req)) if req else 0.0
        req_total = max(1, len(req))
    else:
        confidence = 0.0
        req_total = 1

    contradictions = 1.0 - min(1.0, len(wstate.contradictions) / max(1, wstate.iteration))
    stability = contradictions

    # Grounding now derived from training_grounded violations
    violations = wstate.extras.get("last_violations") or []
    training_v = sum(1 for v in violations if v.get("name") == "training_grounded")
    grounding = max(0.0, 1.0 - training_v / req_total)

    components = {
        "completeness": completeness,
        "stability": stability,
        "confidence": confidence,
        "grounding": grounding,
        "contradictions": contradictions,
    }
    weighted_sum = 0.0
    weight_total = 0.0
    for key, w in weights.items():
        weighted_sum += components.get(key, 0.0) * float(w)
        weight_total += float(w)
    base = (weighted_sum / weight_total) if weight_total > 0 else 0.0

    # Apply veto penalties
    veto_list = [
        Violation(name=v["name"], severity=v["severity"], detail=v["detail"])
        for v in violations
    ]
    return apply_penalty(base, veto_list)


def run(
    config: WorkflowConfig,
    session: str,
    user_msg: str,
    prev_bot_msg: str | None = None,
    state_dir: Path | None = None,
    registry: Registry | None = None,
) -> dict[str, Any]:
    """Run one workflow step and return result dict.

    Args:
        config: workflow's WORKFLOW config
        session: invocation_id (preferred) OR legacy session-as-filename.
                 When ``registry`` is provided this is treated as the
                 ``invocation_id`` and state lives in sqlite. Otherwise
                 it's used as a filename under ``state_dir`` (legacy).
        user_msg: latest user message
        prev_bot_msg: bot's previous reply (for slot extraction)
        state_dir: override file-fallback location (default
                   ``~/.hermes/workflow_state/``). Ignored when
                   ``registry`` is supplied for the active-state path.
                   Still used for artifact + per-turn history files.
        registry: Registry instance. When set, state load+save go through
                  sqlite (preferred path — multi-agent safe, atomic).

    Returns:
        {
            "workflow": <name>,
            "phase": <target phase name>,
            "iteration": <int>,
            "state_summary": {...},
            "mini_prompt": "<markdown>",
            "instructions_for_bot": "<one-liner>",
            "state_file": "<path-or-registry-url>",
            "invocation_id": <str>,
        }
    """
    sdir = Path(state_dir or STATE_DIR_DEFAULT)
    state_path = _state_path(sdir, config.name, session)

    # Backend resolution. Registry path: load from DB; file path: legacy.
    if registry is not None:
        rec = registry.get(session)
        if rec is None:
            # Caller forgot to register — auto-bootstrap a row so the engine
            # can proceed. Plugin-level resolution is preferred; this is a
            # safety net for tests / direct cli usage.
            rec = registry.start(
                workflow_name=config.name,
                agent_id="unknown",
                conversation_id=f"adhoc-{session}",
                invocation_id=session,
            )
        wstate = WorkflowState.from_blob(
            rec.state_blob, config.slots_cls, config.name,
            session_id=rec.invocation_id,
        )
        # The phase from the DB row tells us where we are; honor it
        # unless this is the very first turn (iteration == 0 + INIT).
        if rec.phase and wstate.phase == "INIT" and rec.phase != "INIT":
            wstate.phase = rec.phase
    else:
        wstate = WorkflowState.load(state_path, config.slots_cls, config.name)

    # ── P6: cancellation pre-check ──────────────────────────────────
    # If the user explicitly asks the workflow to reset BEFORE anything
    # else (extractor / decide_fn) runs, wipe state and start fresh on
    # this same call. The cancellation message itself becomes turn 1 of
    # the new session.
    #
    # Lifecycle: when running registry-backed we mark the CURRENT
    # invocation as CANCELLED (audit-trail-preserving), then ask the
    # plugin/CLI to start a fresh invocation on the next turn. Plugin
    # detects cancel-intent itself (faster than waiting for the engine
    # to come back with phase=CANCELLED), but we still defend in depth.
    from . import detectors as _det
    if _det.detect_cancellation(user_msg):
        if registry is not None:
            registry.finish(session, reason="CANCELLED")
            now = time.time() if False else __import__("time").time()
            wstate = WorkflowState(
                session_id=session,
                workflow_name=config.name,
                created_ts=now,
                updated_ts=now,
                slots=config.slots_cls(),
            )
        else:
            if state_path.exists():
                state_path.unlink()
            wstate = WorkflowState.load(state_path, config.slots_cls, config.name)
        wstate.action_log.append(
            "P6: cancellation detected → state reset"
        )

    if prev_bot_msg:
        wstate.add_bot_msg(prev_bot_msg)
    if user_msg:
        wstate.add_user_msg(user_msg)

    wstate.iteration += 1

    # ── P6: unfill signal ────────────────────────────────────────────
    # User said "не X, а Y" or similar — flag a contradiction so the
    # extractor's re-read on this turn gets priority over the prior
    # state value (extractor's calibration rules already prefer recent
    # messages, but the flag surfaces the user intent to humans).
    if _det.detect_unfill_signal(user_msg):
        wstate.extras["unfill_signaled"] = True
        wstate.action_log.append("P6: unfill signal detected")

    # ── P2: LLM extractor ────────────────────────────────────────────
    # Before routing, ask a small fast LLM to (re)read the conversation
    # against the schema and fill / revise slots with per-slot
    # confidence. Skipped silently when:
    #   - workflow uses the legacy dataclass SlotsBase (no schema)
    #   - WORKFLOW_EXTRACTOR_ENABLED=0
    #   - extractor call fails (network, anthropic SDK missing, etc.)
    # The legacy regex-based detector in decide_fn keeps running either
    # way — extractor is additive, not a replacement.
    try:
        from .schema import SchemaSlots
        if isinstance(wstate.slots, SchemaSlots) and getattr(config, "schema", None) is not None:
            from .extractor import extract_slots as _extract, apply_results
            results = _extract(
                config.schema,
                wstate.slots,
                wstate.user_history,
                wstate.bot_history,
            )
            if results:
                changes = apply_results(
                    wstate.slots, results, config.schema,
                    log=wstate.action_log,
                )
                if changes == 0:
                    wstate.action_log.append("extractor: no changes")
            else:
                wstate.action_log.append("extractor: skipped / no results")

            # ── P4: programmatic anti-pattern detection ──────────────
            from .anti_patterns import check_extraction, check_bot_reply
            extr_violations = check_extraction(
                config.schema, wstate.slots, wstate.user_history,
            )
            reply_violations: list = []
            if prev_bot_msg:
                reply_violations = check_bot_reply(
                    config.schema, prev_bot_msg, wstate.user_history,
                    iteration=wstate.iteration, phase=wstate.phase,
                )
            all_v = extr_violations + reply_violations
            for v in all_v:
                # Record into contradictions so clarity_score sees it
                # and into action_log for human inspection.
                wstate.contradictions.append(str(v))
                wstate.action_log.append(f"anti-pattern: {v}")
            # Stash full violations into extras for the artifact builder.
            if all_v:
                wstate.extras["last_violations"] = [
                    {"name": v.name, "severity": v.severity, "detail": v.detail}
                    for v in all_v
                ]
    except Exception as e:
        # Never let extractor failure stop the engine — log and continue.
        wstate.action_log.append(f"extractor: error ({type(e).__name__}: {e})")

    machine = WorkflowMachine(config, wstate)
    target_phase = machine.decide(user_msg, prev_bot_msg)
    wstate.phase = target_phase

    phase_spec = config.get_phase(target_phase)
    if phase_spec is None:
        mini_prompt = f"<error: unknown phase {target_phase}>"
        instructions = "Engine error — fall back to manual flow"
    else:
        mini_prompt = phase_spec.prompt_builder(wstate, user_msg)
        if (
            target_phase == "LOCK"
            and config.mandatory_lock_phrase
            and config.mandatory_lock_phrase not in mini_prompt
        ):
            mini_prompt = (
                f"<engine-warning: LOCK template missing mandatory phrase "
                f"{config.mandatory_lock_phrase!r}>\n\n{mini_prompt}"
            )
        instructions = _instructions_for_phase(target_phase)

    wstate.action_log.append(
        f"iter={wstate.iteration} phase={target_phase} "
        f"filled={wstate.slots.filled_required()}/{len(wstate.slots.required_keys())}"
    )
    # Persist back to whichever backend we loaded from.
    if registry is not None:
        registry.save_state(
            invocation_id=session,
            state_blob=wstate.to_blob(),
            phase=target_phase,
            iteration=wstate.iteration,
        )
    else:
        wstate.save(state_path)

    # ── P7: per-turn snapshot for audit trail ────────────────────────
    # Mirror the just-saved state into
    # <history>/<invocation_id>/v{NNNN}.json so we can replay slot
    # evolution, confidence deltas, and contradictions arrival across
    # turns. History stays on disk (audit-only, no transactional needs).
    try:
        from .history import write_snapshot
        import json as _json
        if registry is not None:
            snapshot_dict = _json.loads(wstate.to_blob())
        else:
            snapshot_dict = _json.loads(state_path.read_text(encoding="utf-8"))
        write_snapshot(
            sdir, config.name, session,
            wstate.iteration, snapshot_dict,
        )
    except Exception as e:
        wstate.action_log.append(f"history: snapshot failed ({type(e).__name__}: {e})")

    # ── P5: final artifact ───────────────────────────────────────────
    # When the workflow reaches its terminal phase (typically DONE),
    # emit a YAML artifact next to the state file. The artifact is what
    # downstream skills consume; `желание[]` (raw user-wishes audit) is
    # included for traceability but `export.fields` defines the subset
    # that crosses the skill boundary (only `goal` etc.).
    artifact_path = _artifact_path(sdir, config.name, session)
    artifact_data: dict[str, Any] | None = None
    schema = getattr(config, "schema", None)
    is_terminal = bool(
        schema and target_phase == schema.phases.terminal
    )
    if is_terminal and schema is not None:
        from .schema import SchemaSlots, build_final_artifact
        if isinstance(wstate.slots, SchemaSlots):
            clarity = _compute_clarity_score(wstate, schema)
            artifact_data = build_final_artifact(
                schema,
                wstate.slots,
                session_id=session,
                turns_used=wstate.iteration,
                clarity_score=clarity,
                желание=wstate.user_history,
                bot_history=wstate.bot_history,
            )
            try:
                import yaml as _yaml
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                artifact_path.write_text(
                    _yaml.safe_dump(
                        artifact_data,
                        allow_unicode=True,
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                wstate.action_log.append(
                    f"artifact: wrote {artifact_path.name} "
                    f"clarity={clarity:.3f}"
                )
                # Re-save so the action_log line lands in storage too.
                if registry is not None:
                    registry.save_state(
                        invocation_id=session,
                        state_blob=wstate.to_blob(),
                        phase=target_phase,
                        iteration=wstate.iteration,
                    )
                else:
                    wstate.save(state_path)
            except Exception as e:
                wstate.action_log.append(
                    f"artifact: write failed ({type(e).__name__}: {e})"
                )

    # ── lifecycle: mark invocation finished on terminal phase ───────
    # Registry-backed path only — file-based legacy callers continue to
    # keep DONE files around until janitor / manual cleanup.
    if is_terminal and registry is not None:
        try:
            registry.finish(session, reason="DONE")
        except Exception as e:
            wstate.action_log.append(
                f"lifecycle: finish() failed ({type(e).__name__}: {e})"
            )

    result = {
        "workflow": config.name,
        "phase": target_phase,
        "iteration": wstate.iteration,
        "state_summary": wstate.summary(),
        "mini_prompt": mini_prompt,
        "instructions_for_bot": instructions,
        "state_file": (
            f"registry:workflow_invocations/{session}"
            if registry is not None
            else str(state_path)
        ),
        "invocation_id": session,
    }
    if artifact_data is not None:
        result["artifact_file"] = str(artifact_path)
        result["artifact"] = artifact_data
    return result


async def run_async(
    config: WorkflowConfig,
    session: str,
    user_msg: str,
    prev_bot_msg: str | None = None,
    state_dir: Path | None = None,
    registry: Registry | None = None,
) -> dict[str, Any]:
    """Async wrapper around ``run()`` — see run() docstring for details."""
    import asyncio
    return await asyncio.to_thread(
        run, config, session, user_msg, prev_bot_msg, state_dir, registry,
    )


def run_status(
    config: WorkflowConfig,
    session: str,
    state_dir: Path | None = None,
    registry: Registry | None = None,
) -> dict[str, Any]:
    """Return state summary without advancing the workflow.

    Registry-backed path: query DB by invocation_id.
    File-backed path: read state file (legacy).
    """
    if registry is not None:
        rec = registry.get(session)
        if rec is None:
            return {
                "error": "no state",
                "invocation_id": session,
                "workflow": config.name,
            }
        wstate = WorkflowState.from_blob(
            rec.state_blob, config.slots_cls, config.name,
            session_id=rec.invocation_id,
        )
        summary = wstate.summary()
        summary["action_log_tail"] = wstate.action_log[-10:]
        summary["finished_ts"] = rec.finished_ts
        summary["finished_reason"] = rec.finished_reason
        summary["is_test"] = rec.is_test
        return summary

    sdir = Path(state_dir or STATE_DIR_DEFAULT)
    state_path = _state_path(sdir, config.name, session)
    if not state_path.exists():
        return {
            "error": "no state",
            "session": session,
            "workflow": config.name,
            "state_file": str(state_path),
        }
    wstate = WorkflowState.load(state_path, config.slots_cls, config.name)
    summary = wstate.summary()
    summary["action_log_tail"] = wstate.action_log[-10:]
    return summary


def run_reset(
    config: WorkflowConfig,
    session: str,
    state_dir: Path | None = None,
    registry: Registry | None = None,
) -> dict[str, Any]:
    """Cancel + clear persisted state for this invocation.

    Registry-backed: mark invocation finished with reason CANCELLED.
    File-backed: delete the state file (legacy).
    """
    if registry is not None:
        rec = registry.get(session)
        if rec is None:
            return {
                "reset": False,
                "invocation_id": session,
                "workflow": config.name,
                "note": "no state existed",
            }
        registry.cancel(session)
        return {
            "reset": True,
            "invocation_id": session,
            "workflow": config.name,
            "note": "marked CANCELLED in registry",
        }

    sdir = Path(state_dir or STATE_DIR_DEFAULT)
    state_path = _state_path(sdir, config.name, session)
    if state_path.exists():
        state_path.unlink()
        return {"reset": True, "session": session, "workflow": config.name}
    return {
        "reset": False,
        "session": session,
        "workflow": config.name,
        "note": "no state existed",
    }


def _instructions_for_phase(phase: str) -> str:
    """One-liner imperatives. Generic phase-name conventions.

    Workflows can override via PhaseSpec metadata in future if needed.
    """
    return {
        "INIT": "Engine is initializing; no reply yet.",
        "DECOMPOSE": "Use the template AS YOUR REPLY. Do not add tool calls; do not start work.",
        "ASK": "Use the template AS YOUR REPLY. Ask ONE focused question.",
        "REFLECT": "Use the template AS YOUR REPLY. Wait for user confirmation.",
        "LOCK": "Use the template AS YOUR REPLY. Lock phrase is MANDATORY and verbatim.",
        "DONE": "Workflow complete. Execute the action from the lock block.",
    }.get(phase, f"Phase {phase} — use mini_prompt verbatim.")
