"""Engine-wide LLM configuration — single source of truth.

Centralises the "which LLM should the workflow-engine call" question.
Stages that consume LLM responses (extractor, future LLM-anti-pattern
detectors, etc.) all read from here so the operator can toggle every
LLM-dependent stage with one env knob.

Env contract
------------
WORKFLOW_LLM_MODEL       Model name. **Empty / unset → ALL LLM stages
                         skip and the engine runs in deterministic mode**
                         (programmatic detectors + regex matching only).
                         Example: ``claude-haiku-4-5``, ``gpt-4o-mini``,
                         ``gemini-2.0-flash``.

WORKFLOW_LLM_API_BASE    Endpoint (OpenAI-compatible). Defaults to the
                         host Meridian proxy ``http://127.0.0.1:3456``
                         which routes Claude through the operator's Max
                         subscription. Override for direct API or other
                         providers.

WORKFLOW_LLM_API_KEY     Auth header value. Defaults to ``not-needed``
                         (Meridian's no-auth pseudo-key). Set to real
                         API key when calling provider directly.

Per-stage overrides (optional, take precedence over the engine defaults):

WORKFLOW_EXTRACTOR_ENABLED   "0" → skip extractor even if WORKFLOW_LLM_MODEL set
WORKFLOW_EXTRACTOR_MODEL     extractor-specific model
WORKFLOW_EXTRACTOR_BASE_URL  extractor-specific endpoint
WORKFLOW_EXTRACTOR_API_KEY   extractor-specific key

When per-stage env is unset, the stage uses the engine-wide value. When
``WORKFLOW_LLM_MODEL`` is empty AND the stage-specific MODEL is also
empty, the stage SKIPS — exposing this to callers via ``llm_enabled()``.

Why this matters
----------------
- Tests: run engine in deterministic mode (no LLM cost / no flakiness)
  by leaving ``WORKFLOW_LLM_MODEL`` unset.
- Prod: set ``WORKFLOW_LLM_MODEL=claude-haiku-4-5`` once at compose
  level — all stages light up.
- Future stages don't need their own env scaffold — they ask
  ``get_llm_config()`` and either use it or no-op.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LLMConfig:
    model: str
    api_base: str
    api_key: str

    @property
    def enabled(self) -> bool:
        return bool(self.model)


def get_engine_llm_config() -> Optional[LLMConfig]:
    """Return engine-wide LLM config, or ``None`` if disabled.

    Disabled means ``WORKFLOW_LLM_MODEL`` is empty (or unset). All LLM
    stages must treat ``None`` as "skip me".
    """
    model = os.environ.get("WORKFLOW_LLM_MODEL", "").strip()
    if not model:
        return None
    return LLMConfig(
        model=model,
        api_base=os.environ.get(
            "WORKFLOW_LLM_API_BASE", "http://127.0.0.1:3456",
        ).rstrip("/"),
        api_key=os.environ.get("WORKFLOW_LLM_API_KEY", "not-needed"),
    )


def get_stage_llm_config(stage_prefix: str) -> Optional[LLMConfig]:
    """Return per-stage LLM config falling back to engine-wide.

    Args:
        stage_prefix: e.g. ``"EXTRACTOR"`` for the slot extractor.
                     Looks up ``WORKFLOW_<stage>_MODEL`` /
                     ``..._BASE_URL`` / ``..._API_KEY``. Missing values
                     fall through to the engine-wide config.

    Returns:
        ``LLMConfig`` if any model name resolves, else ``None``.
        Also honours ``WORKFLOW_<stage>_ENABLED=0`` to force-disable a
        stage even when a model is otherwise available.
    """
    enabled_env = os.environ.get(f"WORKFLOW_{stage_prefix}_ENABLED")
    if enabled_env is not None and enabled_env.strip() in {"0", "false", "False"}:
        return None
    engine_cfg = get_engine_llm_config()
    stage_model = os.environ.get(f"WORKFLOW_{stage_prefix}_MODEL", "").strip()
    stage_base = os.environ.get(f"WORKFLOW_{stage_prefix}_BASE_URL", "").strip()
    stage_key = os.environ.get(f"WORKFLOW_{stage_prefix}_API_KEY", "").strip()

    if not stage_model and engine_cfg is None:
        # Nothing configured at either layer → stage disabled.
        return None

    return LLMConfig(
        model=stage_model or (engine_cfg.model if engine_cfg else ""),
        api_base=stage_base or (engine_cfg.api_base if engine_cfg else "http://127.0.0.1:3456"),
        api_key=stage_key or (engine_cfg.api_key if engine_cfg else "not-needed"),
    )


def llm_enabled() -> bool:
    """Quick check used by callers that just want to know yes/no."""
    return get_engine_llm_config() is not None
