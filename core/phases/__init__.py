"""Reusable phase patterns for workflow-engine skills.

Each module exposes ``build_*_phases``, ``build_*_decide_fn`` and a
``TRANSITIONS`` list so a skill's config.py can wire a complete state
machine in ~5 lines instead of re-declaring the pattern.

Currently shipped:
  - decompose_reflect_lock: 6-phase clarification cycle used by
    desire-to-goal and any future skill that needs schema-driven
    "fill required slots, confirm, lock" flow.

Adding a new reusable pattern: create a module here that returns
phases/transitions/decide_fn tuples; document the canonical signature
and any required prompt_builder names in its docstring.
"""
from .decompose_reflect_lock import (
    TRANSITIONS as DRL_TRANSITIONS,
    build_decide_fn as build_DRL_decide_fn,
    build_phases as build_DRL_phases,
)

__all__ = [
    "DRL_TRANSITIONS",
    "build_DRL_decide_fn",
    "build_DRL_phases",
]
