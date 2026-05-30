"""Programmatic anti-pattern detectors.

Anti-patterns previously lived as plain-text instructions in
``prompts.py`` ("don't list brands without user mention", "don't fill
slots from training data", "don't interpret a non-answer as an
answer"). The main bot was supposed to honor them, but adherence was
prompt-conditional and drifted.

These checks run programmatically against extractor results and bot
replies. Violations:
  - dock the clarity_score via the veto rules in schema.clarity_score.veto
  - get logged into wstate.contradictions (visible in state.json) and
    wstate.action_log
  - can be inspected by tests / dashboards without re-parsing prompts

Each detector is a small, pure function: takes ``(schema, slots,
user_history, bot_history)`` (or a subset) and returns a list of
``Violation`` records — never raises.

When to call:
  - ``check_extraction(...)`` runs in runner.run() right after the
    extractor applies results, so means_as_goal / training_grounded
    are caught at the source.
  - ``check_bot_reply(...)`` runs against the most-recent bot reply
    (engine receives it next turn as ``prev_bot_msg``); rushed_to_action
    detection looks at action_log + iteration.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional


VETO_PENALTY = {
    "means_as_goal": 0.5,         # slot истинная_цель == slot средство
    "training_grounded": 0.3,     # value not derivable from any user message
    "rushed_to_action": 0.4,      # exited workflow before required completion
    "brand_listing": 0.2,         # bot enumerated brand names user never said
}


@dataclass
class Violation:
    name: str
    severity: float       # 0..1 — drives clarity_score penalty
    detail: str           # human-readable

    def __str__(self) -> str:
        return f"anti-pattern[{self.name}] sev={self.severity:.2f}: {self.detail}"


# ─── Helper: text grounding ────────────────────────────────────────────


_WORD_SPLIT = re.compile(r"[\W_]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _WORD_SPLIT.split(text or "") if len(t) >= 3}


def _is_value_grounded(value: str, user_corpus: str) -> bool:
    """Heuristic: any 3+-char content token of `value` appears in
    `user_corpus`.

    Cheap; misses paraphrases. The extractor's confidence score is the
    primary trust signal — this just catches the worst hallucinations
    (full slot value with zero overlap to anything user said).
    """
    if not value or not value.strip():
        return True
    user_tokens = _tokens(user_corpus)
    value_tokens = _tokens(value)
    return bool(user_tokens & value_tokens)


# ─── Detectors ─────────────────────────────────────────────────────────


def detect_means_as_goal(slots: Any, label_aliases: dict[str, str] | None = None) -> Optional[Violation]:
    """slot 'истинная_цель' = slot 'средство' → user's true goal is the tool.

    The most common drift: bot interprets "I want a trading bot" as
    ``goal=trading bot``, which is the means, not the end. Engine
    flags this so reviewers see it in state.json and clarity_score
    deducts the configured veto.
    """
    # Generic naming — workflows that use different slot names can pass
    # label_aliases like {"goal": "истинная_цель", "means": "средство"}.
    label_aliases = label_aliases or {"goal": "истинная_цель", "means": "средство"}
    goal_key = label_aliases.get("goal", "истинная_цель")
    means_key = label_aliases.get("means", "средство")
    goal_v = slots.get(goal_key) if hasattr(slots, "get") else None
    means_v = slots.get(means_key) if hasattr(slots, "get") else None
    if not goal_v or not means_v:
        return None
    if str(goal_v).strip().lower() == str(means_v).strip().lower():
        return Violation(
            name="means_as_goal",
            severity=VETO_PENALTY["means_as_goal"],
            detail=f"{goal_key}={goal_v!r} equals {means_key}={means_v!r}",
        )
    return None


def detect_training_grounded(
    slots: Any,
    user_history: list[str],
) -> list[Violation]:
    """Any filled slot whose tokens don't appear anywhere in user msgs.

    Catches the bot confabulating "ANT 2.0" or "Bybit Futures" when
    the user never said those words. Heuristic — paraphrase / abstraction
    can false-positive; severity is low (0.3) so it nudges rather than
    nukes the score.
    """
    user_corpus = " ".join(user_history or [])
    violations: list[Violation] = []
    keys = slots.all_keys() if hasattr(slots, "all_keys") else list(slots.keys())
    for k in keys:
        v = slots.get(k) if hasattr(slots, "get") else slots.get(k, None)
        if not v:
            continue
        if not _is_value_grounded(str(v), user_corpus):
            violations.append(Violation(
                name="training_grounded",
                severity=VETO_PENALTY["training_grounded"],
                detail=f"slot {k}={v!r} not derivable from any user message",
            ))
    return violations


def detect_rushed_to_action(
    iteration: int,
    phase: str,
    schema_terminal: str,
    schema_lock: str,
    *,
    min_turns_before_lock: int = 2,
) -> Optional[Violation]:
    """Hit terminal phase too fast.

    If we LOCK on turn ≤ ``min_turns_before_lock`` we likely skipped
    the clarification the skill is supposed to do.
    """
    if phase not in (schema_lock, schema_terminal):
        return None
    if iteration <= min_turns_before_lock:
        return Violation(
            name="rushed_to_action",
            severity=VETO_PENALTY["rushed_to_action"],
            detail=f"reached {phase} on iteration {iteration} (min {min_turns_before_lock + 1} expected)",
        )
    return None


_BRAND_PROBE_RE = re.compile(
    r"\b(bybit|binance|amocrm|homeassistant|home\s*assistant|google\s*home|"
    r"openai|claude|chatgpt|telegram|whatsapp|nikon|sony|airbnb|tesla)\b",
    re.IGNORECASE,
)


def detect_brand_listing(
    bot_reply: str,
    user_history: list[str],
) -> Optional[Violation]:
    """Bot enumerated brand names the user never mentioned.

    Common DECOMPOSE / ASK drift — bot tries to "be helpful" by listing
    five trading platforms or smart-home stacks even when the user said
    nothing about brands. Cheap to detect: substring match the brand
    probe regex against the bot's reply and check that the user did NOT
    mention them.
    """
    if not bot_reply:
        return None
    bot_brands = {m.group(0).lower() for m in _BRAND_PROBE_RE.finditer(bot_reply)}
    if not bot_brands:
        return None
    user_corpus = " ".join(user_history or []).lower()
    new_brands = {b for b in bot_brands if b not in user_corpus}
    if not new_brands:
        return None
    return Violation(
        name="brand_listing",
        severity=VETO_PENALTY["brand_listing"],
        detail=f"bot mentioned brands user didn't: {sorted(new_brands)}",
    )


# ─── Aggregation ───────────────────────────────────────────────────────


def check_extraction(
    schema: Any,
    slots: Any,
    user_history: list[str],
) -> list[Violation]:
    """Run all extraction-time detectors. Returns a list (possibly empty)."""
    veto = schema.clarity_score.veto if hasattr(schema, "clarity_score") else {}
    out: list[Violation] = []
    if veto.get("means_as_goal", True):
        v = detect_means_as_goal(slots)
        if v:
            out.append(v)
    if veto.get("training_grounded", True):
        out.extend(detect_training_grounded(slots, user_history))
    return out


def check_bot_reply(
    schema: Any,
    bot_reply: str,
    user_history: list[str],
    *,
    iteration: int,
    phase: str,
) -> list[Violation]:
    """Run all reply-time detectors (called next turn with prev_bot_msg)."""
    veto = schema.clarity_score.veto if hasattr(schema, "clarity_score") else {}
    out: list[Violation] = []
    if veto.get("rushed_to_action", True):
        v = detect_rushed_to_action(
            iteration, phase,
            schema.phases.terminal, schema.phases.required_completion,
        )
        if v:
            out.append(v)
    if veto.get("brand_listing", True):
        v = detect_brand_listing(bot_reply, user_history)
        if v:
            out.append(v)
    return out


def apply_penalty(base_score: float, violations: list[Violation]) -> float:
    """Subtract veto penalties from a base clarity_score, clamped 0..1."""
    penalty = sum(v.severity for v in violations)
    return max(0.0, min(1.0, base_score - penalty))
