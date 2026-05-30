"""Per-turn state snapshots + diff helpers.

Each call to ``runner.run()`` writes a numbered snapshot under
``<state_dir>/<workflow>/history/<session>/v{NNNN}.json`` so that
clarity_score evolution, slot stabilization, and contradictions
arrival can all be reconstructed after the fact.

Why
    The single ``<session>.json`` only holds the latest state — fine
    for runtime, useless for "show me how confidence on истинная_цель
    moved across the 5 turns we had". F1 metrics (stability, mean
    confidence delta) need a history.

What
    write_snapshot() — append-only per-turn JSON. Filename uses 4-digit
    zero-padding so directory listing sorts chronologically.
    diff_states() — structured diff between two state dicts; surfaces
    slot value changes, confidence deltas, contradictions added.
    iter_history() — iterator over (iteration, state_dict).

Test
    Round-trip a 3-turn sequence in tests/test_history.py (planned).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def history_dir(state_dir: Path, workflow_name: str, session: str) -> Path:
    return state_dir / workflow_name / "history" / session


def _snapshot_filename(iteration: int) -> str:
    return f"v{iteration:04d}.json"


def write_snapshot(
    state_dir: Path,
    workflow_name: str,
    session: str,
    iteration: int,
    state_dict: dict[str, Any],
) -> Path:
    """Persist one turn's state for the audit trail. Idempotent on path."""
    hd = history_dir(state_dir, workflow_name, session)
    hd.mkdir(parents=True, exist_ok=True)
    path = hd / _snapshot_filename(iteration)
    path.write_text(
        json.dumps(state_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def iter_history(
    state_dir: Path,
    workflow_name: str,
    session: str,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield (iteration, state_dict) in chronological order."""
    hd = history_dir(state_dir, workflow_name, session)
    if not hd.exists():
        return
    files = sorted(hd.glob("v[0-9][0-9][0-9][0-9].json"))
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        # Filename "v0042.json" → 42
        try:
            iteration = int(f.stem.lstrip("v"))
        except ValueError:
            iteration = data.get("iteration", 0)
        yield iteration, data


def diff_states(
    prev: dict[str, Any] | None,
    curr: dict[str, Any],
) -> dict[str, Any]:
    """Structured diff between two state snapshots.

    Returns:
        {
            "phase": (old, new) or None when unchanged,
            "slot_changes": {slot: (old_value, new_value), ...},
            "confidence_deltas": {slot: (old_conf, new_conf), ...},
            "added_contradictions": [str, ...],
            "added_action_log": [str, ...],
        }
    """
    out: dict[str, Any] = {
        "phase": None,
        "slot_changes": {},
        "confidence_deltas": {},
        "added_contradictions": [],
        "added_action_log": [],
    }
    if prev is None:
        out["phase"] = (None, curr.get("phase"))
        slots_curr = (curr.get("slots") or {}).get("_values") or {}
        confs_curr = (curr.get("slots") or {}).get("_confidence") or {}
        for k, v in slots_curr.items():
            if v is not None:
                out["slot_changes"][k] = (None, v)
        for k, c in confs_curr.items():
            if c:
                out["confidence_deltas"][k] = (0.0, c)
        out["added_contradictions"] = list(curr.get("contradictions") or [])
        out["added_action_log"] = list(curr.get("action_log") or [])
        return out

    if prev.get("phase") != curr.get("phase"):
        out["phase"] = (prev.get("phase"), curr.get("phase"))

    p_slots = (prev.get("slots") or {}).get("_values") or {}
    c_slots = (curr.get("slots") or {}).get("_values") or {}
    for k in set(p_slots) | set(c_slots):
        a, b = p_slots.get(k), c_slots.get(k)
        if a != b:
            out["slot_changes"][k] = (a, b)

    p_confs = (prev.get("slots") or {}).get("_confidence") or {}
    c_confs = (curr.get("slots") or {}).get("_confidence") or {}
    for k in set(p_confs) | set(c_confs):
        a, b = float(p_confs.get(k, 0.0)), float(c_confs.get(k, 0.0))
        if abs(a - b) > 1e-6:
            out["confidence_deltas"][k] = (a, b)

    p_contr = set(prev.get("contradictions") or [])
    c_contr = list(curr.get("contradictions") or [])
    out["added_contradictions"] = [c for c in c_contr if c not in p_contr]

    p_log = set(prev.get("action_log") or [])
    c_log = list(curr.get("action_log") or [])
    out["added_action_log"] = [l for l in c_log if l not in p_log]

    return out
