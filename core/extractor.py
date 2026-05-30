"""LLM-backed slot extractor — schema-driven, with per-slot confidence.

Replaces "main bot does slot inference inline via prompt" with a
dedicated fast LLM (haiku-4-5) that reads:
  * the SkillSchema (slot definitions, descriptions, examples)
  * the current SchemaSlots state (so it doesn't re-extract what's done)
  * a short conversation window (last N user + bot turns)
and returns JSON ``{slot_name: {value, confidence, reasoning}}``.

Why
    The previous design relied on the main agent embedding regex-able
    markers in its replies and the engine scraping them with
    ``detectors.extract_slots(prev_bot_msg, LABEL_MAP)``. Brittle:
    drifts the moment the main bot reformats. A dedicated extractor
    with schema-constrained JSON output is deterministic on a per-call
    basis and decouples slot reasoning from main-bot prose.

What
    ``extract_slots()`` returns ``dict[str, ExtractionResult]``. Caller
    (runner.py) writes ``value`` into ``wstate.slots`` and ``confidence``
    via ``slots.set_confidence()``. Routing logic in decide_fn can then
    check ``slots.is_high_confidence(key)`` to gate phase advancement.

Test
    Anthropic SDK call is mocked in tests; the prompt + JSON parsing
    paths run unconditionally so prompt drift surfaces immediately.

Endpoint
    Default: ``http://127.0.0.1:3456`` (host Meridian — same proxy used
    by ``claude-agent-sdk`` provider, which the hermes container reaches
    because hermes-core runs in ``network_mode: host``).
    Override via ``WORKFLOW_EXTRACTOR_BASE_URL`` env var.
    Model override: ``WORKFLOW_EXTRACTOR_MODEL`` (default ``claude-haiku-4-5``).
    Disable extractor entirely: ``WORKFLOW_EXTRACTOR_ENABLED=0`` — runner
    falls back to the legacy regex-only path.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from core.schema import SchemaSlots, SkillSchema

logger = logging.getLogger(__name__)


# ─── Result types ──────────────────────────────────────────────────────


@dataclass
class ExtractionResult:
    value: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class ExtractorConfig:
    base_url: str = "http://127.0.0.1:3456"
    api_key: str = "not-needed"
    model: str = "claude-haiku-4-5"
    max_tokens: int = 1024
    timeout_seconds: float = 60.0
    conversation_window: int = 6   # last N turns (user + bot interleaved)
    enabled: bool = True

    @classmethod
    def from_env(cls) -> "ExtractorConfig":
        """Build extractor config from the engine-wide LLM switch.

        Resolution:
          1. Per-stage env: ``WORKFLOW_EXTRACTOR_MODEL`` (overrides engine)
          2. Engine-wide env: ``WORKFLOW_LLM_MODEL`` (single switch)
          3. **If neither sets MODEL → stage disabled** (NO silent LLM call).

        ``enabled=False`` means runner skips slot extraction entirely;
        engine continues with deterministic phase routing + programmatic
        detectors. The operator must opt-in to LLM by setting
        ``WORKFLOW_LLM_MODEL`` — there is no implicit default.
        """
        from .llm_config import get_stage_llm_config
        stage = get_stage_llm_config("EXTRACTOR")
        if stage is None:
            # No LLM model configured at any layer → stage off.
            return cls(enabled=False, model="", api_key="", base_url="")
        return cls(
            base_url=stage.api_base,
            api_key=stage.api_key,
            model=stage.model,
            max_tokens=int(os.getenv("WORKFLOW_EXTRACTOR_MAX_TOKENS", "1024")),
            timeout_seconds=float(os.getenv("WORKFLOW_EXTRACTOR_TIMEOUT", "60")),
            conversation_window=int(os.getenv("WORKFLOW_EXTRACTOR_WINDOW", "6")),
            enabled=True,
        )


# ─── Prompt builder ────────────────────────────────────────────────────


_SYSTEM_PROMPT = """\
You are a slot extractor for a clarification workflow. Read a schema, the conversation so far, and the slots already filled. Extract slot values that the user has EXPLICITLY stated.

Strict rules:
- Only fill a slot if the user mentioned the value directly or in close paraphrase.
- Never invent brand names, product names, numbers, or technical stacks that the user did not say. Training-data fills are forbidden.
- If a slot is already filled and the new message does not contradict it, leave value unchanged (return the existing value with the same confidence).
- If the user changed their mind, prefer the most recent statement.
- Confidence calibration:
    0.95 — direct quote, no interpretation
    0.75 — close paraphrase / unambiguous
    0.50 — reasonable inference from context
    0.25 — guess
    0.00 — no signal
- For optional slots not addressed: value=null, confidence=0.
- For required slots not yet addressed: value=null, confidence=0 (do NOT guess to please).

Output JSON ONLY. No prose, no markdown fences. Schema:
{
  "<slot_name>": {"value": "<string or null>", "confidence": <0..1>, "reasoning": "<one short sentence>"}
}
Return exactly one object covering every slot in the schema.
"""


def _render_schema_section(schema: SkillSchema) -> str:
    lines = ["SCHEMA:"]
    for name, spec in schema.slots.items():
        required = "REQUIRED" if spec.required else "optional"
        lines.append(f"- {name} ({required})")
        if spec.description:
            for d in spec.description.strip().splitlines():
                lines.append(f"    {d.strip()}")
        if spec.examples:
            ex = "; ".join(spec.examples[:3])
            lines.append(f"    examples: {ex}")
    return "\n".join(lines)


def _render_state_section(slots: SchemaSlots) -> str:
    values = slots.as_dict()
    conf = slots.confidence_dict()
    payload = {
        name: {"value": values.get(name), "confidence": conf.get(name, 0.0)}
        for name in values
    }
    return "CURRENT_STATE:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _render_conversation_section(
    user_history: list[str],
    bot_history: list[str],
    window: int,
) -> str:
    # Interleave user/bot turns chronologically. user_history[i] usually
    # corresponds to the response BEFORE bot_history[i], but engine
    # doesn't strictly guarantee 1:1 alignment, so we pair by index and
    # tolerate length mismatch.
    pairs: list[str] = []
    max_len = max(len(user_history), len(bot_history))
    for i in range(max_len):
        if i < len(user_history):
            pairs.append(f"USER: {user_history[i]}")
        if i < len(bot_history):
            pairs.append(f"BOT: {bot_history[i]}")
    tail = pairs[-window * 2:]    # 2 lines per turn (user + bot)
    return "CONVERSATION (most recent first-to-last):\n" + "\n".join(tail or ["<empty>"])


def build_prompt(
    schema: SkillSchema,
    slots: SchemaSlots,
    user_history: list[str],
    bot_history: list[str],
    *,
    window: int = 6,
) -> str:
    return "\n\n".join([
        _render_schema_section(schema),
        _render_state_section(slots),
        _render_conversation_section(user_history, bot_history, window),
        "Return the JSON object now.",
    ])


# ─── Anthropic SDK call ────────────────────────────────────────────────


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_result_dict(raw: str) -> dict[str, Any]:
    """Extract a JSON object from the model's text response.

    Tolerates models that wrap output in markdown fences or prepend
    ``Here's the JSON:`` despite the JSON-only instruction.
    """
    raw = raw.strip()
    # Strip optional fences
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fall back to greedy { ... } match
        m = _JSON_OBJECT_RE.search(raw)
        if not m:
            raise
        return json.loads(m.group(0))


def _call_anthropic(
    prompt: str,
    cfg: ExtractorConfig,
) -> str:
    """Single Anthropic Messages call. Returns the raw text response."""
    try:
        import anthropic  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "anthropic SDK not installed in this venv — required for the "
            "workflow-engine extractor. Install via "
            "`pip install 'hermes-agent[anthropic]'` or set "
            "WORKFLOW_EXTRACTOR_ENABLED=0 to fall back to the legacy "
            "regex-only path."
        ) from e

    client = anthropic.Anthropic(
        base_url=cfg.base_url.rstrip("/"),
        api_key=cfg.api_key,
        timeout=cfg.timeout_seconds,
    )
    resp = client.messages.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = []
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


# ─── Public API ────────────────────────────────────────────────────────


def extract_slots(
    schema: SkillSchema,
    slots: SchemaSlots,
    user_history: list[str],
    bot_history: list[str],
    *,
    cfg: Optional[ExtractorConfig] = None,
) -> dict[str, ExtractionResult]:
    """Call the extractor LLM and return per-slot results.

    On failure (network, JSON parse, SDK import, etc.) returns an empty
    dict so the runner can degrade gracefully into the legacy
    regex-based detector. The runner logs the failure into
    ``wstate.action_log`` for visibility in /workflow_state/*.json.
    """
    cfg = cfg or ExtractorConfig.from_env()
    if not cfg.enabled:
        return {}

    if not user_history and not bot_history:
        return {}

    prompt = build_prompt(
        schema, slots, user_history, bot_history,
        window=cfg.conversation_window,
    )

    try:
        raw = _call_anthropic(prompt, cfg)
        data = _coerce_result_dict(raw)
    except Exception as e:
        logger.warning("extractor call failed: %s", e)
        return {}

    results: dict[str, ExtractionResult] = {}
    for slot_name in schema.all_slot_names():
        spec = data.get(slot_name)
        if not isinstance(spec, dict):
            continue
        value = spec.get("value")
        if value == "":
            value = None
        # Coerce non-strings (numbers) to string, keep null as None
        if value is not None and not isinstance(value, str):
            value = str(value)
        try:
            conf = float(spec.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        results[slot_name] = ExtractionResult(
            value=value,
            confidence=max(0.0, min(1.0, conf)),
            reasoning=str(spec.get("reasoning", "")).strip(),
        )

    return results


def apply_results(
    slots: SchemaSlots,
    results: dict[str, ExtractionResult],
    schema: SkillSchema,
    *,
    log: Optional[list[str]] = None,
) -> int:
    """Write extractor results into the SchemaSlots container.

    Returns the number of slots whose value or confidence changed.
    Updates only when confidence >= threshold_low and value is non-null.
    Lower-confidence results still update the confidence score so the
    routing logic can see "we asked, got weak answer" vs "we never asked".
    """
    threshold_low = schema.confidence.threshold_low
    changed = 0
    for name, result in results.items():
        prev_value = slots.get(name)
        prev_conf = slots.get_confidence(name)
        # Always write the latest confidence — gives visibility into
        # what the extractor saw on this turn.
        if abs(result.confidence - prev_conf) > 1e-6:
            slots.set_confidence(name, result.confidence)
            changed += 1
        # Only overwrite the slot value when:
        #   - extractor returned a non-null value AND
        #   - confidence cleared the low-confidence floor AND
        #   - either the slot was empty, or the new value is different
        #     AND its confidence beats what we had before
        if result.value is not None and result.confidence >= threshold_low:
            if prev_value is None:
                slots.set(name, result.value)
                changed += 1
                if log is not None:
                    log.append(
                        f"extractor: filled {name}={result.value!r} "
                        f"conf={result.confidence:.2f}"
                    )
            elif result.value != prev_value and result.confidence > prev_conf:
                slots.set(name, result.value)
                changed += 1
                if log is not None:
                    log.append(
                        f"extractor: revised {name}: {prev_value!r} → "
                        f"{result.value!r} conf={result.confidence:.2f}"
                    )
    return changed
