"""DecomposeReflectLock — 6-phase clarification cycle.

Schema-driven state-machine pattern: INIT → DECOMPOSE → ASK/REFLECT →
LOCK → DONE. Used by any skill that needs to fill a slot schema
through dialog, confirm with the user, and lock a final commitment.

The decide_fn is universal: it gates phase advancement on
schema-declared confidence thresholds and the same anti-spinning
safety valves used by desire-to-goal (LOCK→DONE first; push-for-action
shortcut; max_iterations escape).

Required prompts module — the consuming workflow's `prompts.py` must
expose these names as callables ``(wstate, user_msg) → str``:
  decompose, ask, reflect, lock, done
INIT has no prompt (engine doesn't reply on the very first call).
"""
from __future__ import annotations

from types import ModuleType
from typing import Any, Callable

from core import detectors
from core.state import PhaseSpec, TransitionSpec


TRANSITIONS: list[TransitionSpec] = [
    TransitionSpec(trigger="begin", source="INIT", dest="DECOMPOSE"),
    TransitionSpec(trigger="advance_to_ask", source=["DECOMPOSE", "REFLECT"], dest="ASK"),
    TransitionSpec(trigger="advance_to_reflect", source=["ASK", "DECOMPOSE"], dest="REFLECT"),
    TransitionSpec(trigger="advance_to_lock", source=["ASK", "REFLECT", "DECOMPOSE"], dest="LOCK"),
    TransitionSpec(trigger="finish", source="LOCK", dest="DONE"),
]


def build_phases(prompts: ModuleType) -> list[PhaseSpec]:
    """Wire prompts.<name> callables to PhaseSpec entries."""
    return [
        PhaseSpec(name="INIT", prompt_builder=lambda s, u: "<init phase, no reply yet>"),
        PhaseSpec(name="DECOMPOSE", prompt_builder=prompts.decompose),
        PhaseSpec(name="ASK", prompt_builder=prompts.ask),
        PhaseSpec(name="REFLECT", prompt_builder=prompts.reflect),
        PhaseSpec(name="LOCK", prompt_builder=prompts.lock),
        PhaseSpec(name="DONE", prompt_builder=prompts.done),
    ]


def build_decide_fn(
    schema: Any,
    *,
    label_map: dict[str, list[str]] | None = None,
) -> Callable[..., str]:
    """Return a decide_fn closure bound to the given schema.

    Rule order (top wins):
      1. INIT or iteration==1 → DECOMPOSE
      2. current_phase == LOCK → DONE   (delivered lock, user just acked)
      3. iteration >= max_iterations → LOCK   (safety valve)
      4. user_pushed_for_action + any required filled → LOCK
      5. all required filled (above threshold_high) and motivation OK:
           from ASK/DECOMPOSE → REFLECT
           from REFLECT → LOCK
      6. otherwise → ASK
    """
    label_map = label_map or {}

    def _decide(
        wstate: Any,
        user_msg: str,
        prev_bot_msg: str,
        current_phase: str,
    ) -> str:
        # Legacy detector still runs as a belt-and-suspenders backup —
        # if extractor failed for any reason the regex sweep may still
        # pull at least the headline slot from a structured bot reply.
        if prev_bot_msg and label_map:
            extracted = detectors.extract_slots(prev_bot_msg, label_map)
            for k, v in extracted.items():
                if v and not wstate.slots.get(k):
                    wstate.slots.set(k, v)

        if detectors.detect_user_pushed_for_action(user_msg):
            wstate.extras["user_pushed_for_action"] = True

        # Motivation tracking. Two independent triggers (either flips
        # motivation_answered=True):
        #   1. Extractor confidence on `мотивация` slot ≥ threshold_high
        #      — the LLM already decided the user answered the "why".
        #      This is the PRIMARY signal; the keyword heuristic below
        #      was the only signal until 2026-05-27 and produced
        #      false-negatives for terse answers like "Финансовый
        #      результат." / "cashflow каждый месяц." which lack the
        #      qualifier connectives.
        #   2. Legacy keyword sweep — user said «потому что / чтобы /
        #      зачем тебе» on a non-refusal turn. Kept as belt-and-
        #      suspenders for cases where extractor missed.
        if wstate.extras.get("motivation_asked"):
            try:
                mot_conf = wstate.slots._confidence.get("мотивация", 0.0)
                threshold_high = schema.confidence.threshold_high
            except AttributeError:
                mot_conf, threshold_high = 0.0, 1.0
            if mot_conf >= threshold_high:
                wstate.extras["motivation_answered"] = True
            elif user_msg and not detectors.is_non_answer(user_msg, "open"):
                if any(
                    q in user_msg.lower()
                    for q in detectors.DEFAULT_QUALIFIER_HINTS
                ):
                    wstate.extras["motivation_answered"] = True

        if wstate.iteration <= 1 or current_phase == "INIT":
            wstate.extras["motivation_asked"] = True
            return "DECOMPOSE"

        if current_phase == "LOCK":
            return "DONE"

        if wstate.iteration >= schema.phases.max_iterations:
            return "LOCK"

        user_pushed = wstate.extras.get("user_pushed_for_action", False)
        motivation_ok = wstate.extras.get("motivation_answered", False)
        filled = wstate.slots.filled_required()
        all_filled = (filled == len(wstate.slots.required_keys()))

        # ── P6: validate_lock gate ────────────────────────────────────
        # Before advancing into LOCK, ensure required slots actually
        # meet the schema's confidence threshold. validate_lock returns
        # (ok, reasons); when not ok we stay at ASK and surface the
        # reasons in extras so the next turn's bot reply can target the
        # weakest slot.
        def _lock_ok() -> bool:
            ok, reasons = detectors.validate_lock(wstate.slots, schema)
            if not ok:
                wstate.extras["lock_blocked_reasons"] = reasons
            else:
                wstate.extras.pop("lock_blocked_reasons", None)
            return ok

        if user_pushed and filled >= 1:
            if _lock_ok():
                return "LOCK"
            return "ASK"

        if all_filled and motivation_ok:
            if current_phase in ("ASK", "DECOMPOSE"):
                return "REFLECT"
            if current_phase == "REFLECT" and _lock_ok():
                return "LOCK"

        return "ASK"

    return _decide
