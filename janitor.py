#!/usr/bin/env python3
"""workflow-engine janitor — sweep abandoned + prune retention.

Run periodically (cron, default every 10 min). Does two things:

  1. ``janitor_sweep`` — mark active invocations whose ``last_active_ts``
     is older than the configured timeout as ``ABANDONED_TIMEOUT``.
     Tests (``is_test=1``) get a tight 5-min timeout; regular runs get
     3h by default. Both overridable via env.
  2. ``purge_finished_older_than`` — retention. Hard-delete finished
     rows older than ``WORKFLOW_RETENTION_DAYS`` (default 30).

Env knobs:
  WORKFLOW_REGISTRY_DB          override sqlite path
  WORKFLOW_ABANDON_TIMEOUT      seconds, default 10800 (3h)
  WORKFLOW_TEST_ABANDON_TIMEOUT seconds, default 300 (5min)
  WORKFLOW_RETENTION_DAYS       days, default 30
  WORKFLOW_JANITOR_DRY_RUN      "1" → only report, don't mutate

Output (always JSON to stdout, one line):
  {
    "swept": N,
    "swept_ids": [...],
    "purged_retention": N,
    "stats_after": {...},
  }
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make sibling `core/` importable when invoked directly.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def main() -> int:
    from core.registry import Registry

    db_path = os.environ.get("WORKFLOW_REGISTRY_DB", "/opt/data/state.db")
    default_timeout = _env_float("WORKFLOW_ABANDON_TIMEOUT", 10800.0)
    test_timeout = _env_float("WORKFLOW_TEST_ABANDON_TIMEOUT", 300.0)
    retention_days = _env_int("WORKFLOW_RETENTION_DAYS", 30)
    dry_run = os.environ.get("WORKFLOW_JANITOR_DRY_RUN", "").lower() in {"1", "true", "yes"}

    r = Registry(db_path)

    if dry_run:
        # List what would be swept, but don't update.
        active = r.list_active()
        import time as _time
        now = _time.time()
        would_sweep = []
        for inv in active:
            timeout = test_timeout if inv.is_test else default_timeout
            if now - inv.last_active_ts > timeout:
                would_sweep.append(inv.invocation_id)
        print(json.dumps({
            "dry_run": True,
            "would_sweep": would_sweep,
            "would_sweep_count": len(would_sweep),
            "stats": r.stats(),
        }, ensure_ascii=False))
        return 0

    swept = r.janitor_sweep(
        default_timeout_sec=default_timeout,
        test_timeout_sec=test_timeout,
    )
    purged = r.purge_finished_older_than(days=retention_days)

    print(json.dumps({
        "swept": len(swept),
        "swept_ids": [inv.invocation_id for inv in swept],
        "purged_retention": purged,
        "stats_after": r.stats(),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
