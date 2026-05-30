"""Generic detectors — pure functions reusable across all cyclic skills.

Each detector takes raw text + optional context, returns a bool / score /
extracted-value. No state mutation, no side effects. Easy to unit-test.

Skills may add their own domain-specific detectors in
`skills/<name>/detectors.py` — those should import from here when
overlap exists (e.g. for tokenization).
"""
from __future__ import annotations

import re
from typing import Any, Iterable


# ─── Tokenization ──────────────────────────────────────────────────────


_WORD_RE = re.compile(r"[a-zа-яё]+", re.IGNORECASE | re.UNICODE)


def tokens(text: str | None, min_len: int = 3) -> set[str]:
    """Lowercase alphabetic word tokens, deduped, min-length filter.

    min_len=3 strips Russian particles («и», «а», «то») and English
    stopwords («a», «to», «is») while keeping content words.
    """
    if not text:
        return set()
    return {t.lower() for t in _WORD_RE.findall(text) if len(t) >= min_len}


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity. Both empty → 1.0 (vacuously similar)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def has_stem_match(token: str, token_set: set[str], min_stem: int = 4) -> bool:
    """True iff token shares N-char stem with any in token_set.

    Defends against Russian declension (крипторынок ↔ крипторынке share
    «крип» 4-char stem). For tokens shorter than min_stem, falls back
    to exact membership.
    """
    if not token or len(token) < min_stem:
        return token in token_set
    stem = token[:min_stem]
    for t in token_set:
        if len(t) < min_stem:
            if t == token:
                return True
            continue
        if t.startswith(stem) or token.startswith(t[:min_stem]):
            return True
    return False


# ─── Non-answer detection ──────────────────────────────────────────────


_NON_ANSWER_PATTERNS = re.compile(
    r"начинай|начни|\bделай\b|уже\s+(всё\s+|все\s+)?(ответил|сказал)|"
    r"defaults?\b|как\s+(считаешь|хочешь)|на\s+твой\s+вкус|продолжай|"
    r"go\s*ahead|пох(уй|ер)|не\s+хочу\s+(говорить|называть)",
    re.IGNORECASE,
)

_SURFACE_AFFIRM_PATTERNS = re.compile(
    r"^\s*(ок|угу|да|\+|yes|ага)\s*[.,!?]?\s*$",
    re.IGNORECASE,
)


def is_non_answer(user_msg: str | None, prev_question_type: str = "open") -> bool:
    """True if user's reply dodges the question.

    Args:
        user_msg: latest user text
        prev_question_type: "open" (motivation, why), "yesno" (confirm),
            "choice" (pick from list). Surface affirmations are non-answers
            for OPEN questions but valid for YESNO confirmation.

    Returns:
        True if reply is a non-answer (engine should re-ask).

    Examples:
        is_non_answer("начинай с defaults", "open") → True
        is_non_answer("да", "yesno") → False (valid confirm)
        is_non_answer("да", "open") → True (surface affirm to open Q)
        is_non_answer("хочу финансовой независимости", "open") → False
    """
    if not user_msg:
        return True
    text = user_msg.strip()
    if _NON_ANSWER_PATTERNS.search(text):
        return True
    if prev_question_type == "open" and _SURFACE_AFFIRM_PATTERNS.match(text):
        return True
    return False


# ─── User-pushed-for-action detection ──────────────────────────────────


_USER_PUSH_PATTERNS = re.compile(
    r"начинай|начни|defaults?\b|уже\s+(всё\s+)?(ответил|сказал)|"
    r"если\s+(можешь|можно)\s+начать|спрашивай\s+один|"
    r"go\s*ahead|поехали",
    re.IGNORECASE,
)


def detect_user_pushed_for_action(user_msg: str | None) -> bool:
    """True if user explicitly told the bot to start work.

    Signal to fast-track from ASK/REFLECT to LOCK without waiting for
    full slot completion.
    """
    if not user_msg:
        return False
    return bool(_USER_PUSH_PATTERNS.search(user_msg))


# ─── Means-as-goal detection ───────────────────────────────────────────


# Tokens that, alone or with other means tokens, signal that the user's
# stated "goal" is just a means/proxy without underlying motivation.
DEFAULT_MEANS_TOKENS = {
    "бот", "сайт", "канал", "приложение", "сервис", "магазин",
    "чат-бот", "чатбот", "торговля", "трейдинг",
    "прибыль", "доход", "деньги",
}

# Tokens that, if present alongside means tokens, qualify the
# formulation enough to NOT trigger the means-as-goal flag.
# («ради финансовой независимости» — means «прибыль» + qualifier «ради»).
DEFAULT_QUALIFIER_HINTS = {
    "для", "чтобы", "ради", "потому",
    "независимость", "свобода", "развитие", "рост",
    "обучение", "практика", "помочь", "помощь", "решить",
    "здоровье", "комфорт", "удобство",
}


def is_means_as_goal(
    goal_text: str | None,
    means_tokens: set[str] = DEFAULT_MEANS_TOKENS,
    qualifier_hints: set[str] = DEFAULT_QUALIFIER_HINTS,
) -> bool:
    """True if the goal is just a means/proxy without qualifying context.

    Skills can override means_tokens / qualifier_hints for their domain
    (e.g. risk-assessment might add «снизить риск» as a quasi-means).
    """
    if not goal_text:
        return False
    norm = goal_text.lower()
    toks = tokens(norm)
    has_means = any(t in means_tokens for t in toks)
    if not has_means:
        return False
    has_qualifier = any(t in qualifier_hints for t in toks) or any(
        q in norm for q in qualifier_hints
    )
    return not has_qualifier


# ─── Slot extraction from markdown ─────────────────────────────────────


_DEFAULT_PLACEHOLDER_RE = re.compile(
    r"^\s*(не\s+указан[оаы]?|не\s+определен[оаы]?|неизвестно|нет\s+данных)",
    re.IGNORECASE,
)


def extract_slots(
    bot_text: str,
    label_map: dict[str, list[str]],
    placeholder_re: re.Pattern = _DEFAULT_PLACEHOLDER_RE,
) -> dict[str, str | None]:
    """Parse REFLECT-style markdown into a dict of slot_key → value.

    Args:
        bot_text: bot's previous reply containing markdown slot lines
        label_map: {slot_key: [display_label1, display_label2, ...]}
                   Each label is a Russian/English heading the bot
                   might write. First match wins.
        placeholder_re: regex matching «not specified» phrases

    Returns:
        {slot_key: extracted_value_or_None}. None if not found OR
        if extraction matched a placeholder phrase.

    Example label_map for desire-to-goal:
        {
            "истинная_цель": ["Истинная цель", "True goal", "Цель"],
            "средство": ["Средство", "Means"],
            "место": ["Место/контекст", "Место", "Place"],
        }

    Patterns matched (markdown):
        — **Истинная цель**: <value-up-to-next-bullet>
        - **True goal**: <value>
        **Цель**: <value>
    """
    pattern_tmpl = (
        r"\*\*{label}\*\*\s*[:.]?\s*(.+?)"
        r"(?=\n[ \t]*[—\-\*]\s\*\*|\n\n|\Z)"
    )
    out: dict[str, str | None] = {k: None for k in label_map}

    for key, label_list in label_map.items():
        for label in label_list:
            pat = pattern_tmpl.format(label=re.escape(label))
            m = re.search(pat, bot_text, flags=re.DOTALL | re.IGNORECASE)
            if not m:
                continue
            val = m.group(1).strip()
            # Strip parenthetical examples, bold/dash decorations
            val_clean = re.sub(r"\([^)]*\)", "", val).strip()
            val_clean = re.sub(
                r"^[*_\-—•:.,\s]+|[*_\-—•:.,\s]+$", "", val_clean
            )
            if placeholder_re.match(val_clean) or not val_clean:
                # Found the label but value is placeholder — leave None
                break
            out[key] = val
            break

    return out


# ─── Lock-signal detection ─────────────────────────────────────────────


DEFAULT_LOCK_PATTERNS = [
    r"истинн\w*\s+цель\s+определен",
    r"цель\s+зафиксирован",
    r"фиксирую\s+цель",
    r"цель\s+ясн[ао]\s*,?\s*начина[юе]",
    r"понял\s+окончательно",
    r"цель\s+понятн[ао]\s*,?\s*перехожу",
]
_DEFAULT_LOCK_RE = re.compile("|".join(DEFAULT_LOCK_PATTERNS), re.IGNORECASE)


def has_lock_signal(bot_text: str | None, patterns: list[str] = None) -> bool:
    """True if bot's text contains a recognized lock-phrase.

    Skills can supply alternative patterns for their domain.
    """
    if not bot_text:
        return False
    if patterns is None:
        return bool(_DEFAULT_LOCK_RE.search(bot_text))
    rx = re.compile("|".join(patterns), re.IGNORECASE)
    return bool(rx.search(bot_text))


# ─── Grounding (slot value vs user vocabulary) ────────────────────────


def grounding_score(
    slot_values: Iterable[str | None],
    user_texts: Iterable[str],
    overlap_threshold: float = 0.2,
) -> float:
    """Fraction of slot values whose tokens overlap user vocabulary.

    For each non-empty slot value, computes asymmetric overlap (how much
    of the value's tokens appear in the user-text union, with stem
    matching for declension tolerance). Slot is "grounded" if overlap
    ≥ threshold.

    Returns:
        Average grounded-fraction across non-empty slots, 0..1.
        Empty input → 0.
        No user vocab → 0.5 (neutral, can't ground against nothing).
    """
    filled = [v for v in slot_values if v]
    if not filled:
        return 0.0

    user_vocab: set[str] = set()
    for t in user_texts:
        user_vocab |= tokens(t or "")
    if not user_vocab:
        return 0.5

    grounded = []
    for v in filled:
        v_toks = tokens(v)
        if not v_toks:
            continue
        covered = sum(1 for t in v_toks if has_stem_match(t, user_vocab))
        grounded.append(1.0 if (covered / len(v_toks)) >= overlap_threshold else 0.0)

    return sum(grounded) / len(grounded) if grounded else 0.0


# ─── P6: Universal cancellation / unfill / lock-validator ─────────────


_CANCELLATION_PATTERNS = re.compile(
    r"\b(забудь(те)?|сброс(ь|ить)?|reset|cancel|отмен(и|ите|яй)|"
    r"начни\s+сначала|stop(\s+it)?|не\s+туда|это\s+не\s+то|"
    r"перестань|"
    r"забей|"
    r"скинь\s+(всё|все|контекст))\b",
    re.IGNORECASE,
)


def detect_cancellation(user_msg: str | None) -> bool:
    """User asked the workflow to reset / start over / abandon.

    Programmatic — does NOT call any LLM. Triggers an unconditional
    state reset in runner.run() before extractor / decide_fn run.
    """
    if not user_msg:
        return False
    return bool(_CANCELLATION_PATTERNS.search(user_msg))


# Matches "не Х, а Y" / "X — это не то / неправильно" / "не Х, лучше Y"
_UNFILL_PATTERNS = re.compile(
    r"\b(не\s+\S+,\s+а\s+|"
    r"не\s+правильно|"
    r"это\s+не\s+(то|так)|"
    r"(а\s+)?(лучше|вернее)\s+|"
    r"забудь\s+про\s+|"
    r"перепиши)\b",
    re.IGNORECASE,
)


def detect_unfill_signal(user_msg: str | None) -> bool:
    """User explicitly contradicted a prior slot fill.

    Engine consumers can use this to bump the contradictions counter
    and let the extractor re-read the most recent message with priority.
    Doesn't pinpoint WHICH slot — extractor does that on the rewrite.
    """
    if not user_msg:
        return False
    return bool(_UNFILL_PATTERNS.search(user_msg))


def validate_lock(
    slots: Any,
    schema: Any,
    *,
    require_high_confidence: bool = True,
) -> tuple[bool, list[str]]:
    """Check whether the workflow may legally enter the LOCK phase.

    Returns (ok, reasons). When ``require_high_confidence`` (default
    True), each required slot must have confidence above the schema's
    ``threshold_high``. Without confidence (legacy SlotsBase) the check
    is value-presence only.
    """
    reasons: list[str] = []
    req = slots.required_keys() if hasattr(slots, "required_keys") else []
    for key in req:
        val = slots.get(key) if hasattr(slots, "get") else None
        if not val:
            reasons.append(f"required slot empty: {key}")
            continue
        if require_high_confidence and hasattr(slots, "get_confidence"):
            conf = slots.get_confidence(key)
            high = (
                schema.confidence.threshold_high
                if hasattr(schema, "confidence")
                else 0.75
            )
            if conf < high:
                reasons.append(
                    f"required slot {key} below confidence threshold "
                    f"({conf:.2f} < {high:.2f})"
                )
    return (len(reasons) == 0, reasons)
