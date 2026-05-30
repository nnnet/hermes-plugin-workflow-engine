"""WorkflowState — slot container + per-session JSON persistence.

Each workflow defines its own SlotsBase subclass with named fields. The
state container wraps any SlotsBase instance + tracks iteration, phase,
history, contradictions, and workflow-specific flags via `extras` dict.

Persistence: state serializes to JSON. Workflow-specific extras go through
the `extras` dict (not class fields) so the same persistence layer
works across all workflows without per-workflow schema versioning.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Callable

STATE_VERSION = 1


# ─── Slots base ────────────────────────────────────────────────────────


@dataclass
class SlotsBase:
    """Base class for workflow-specific slot containers.

    Subclasses declare named fields (one per slot) typed as `str | None`.
    Override `required_keys()` to list the subset that gates completion.

    Example for desire-to-goal:

        @dataclass
        class DesireToGoalSlots(SlotsBase):
            истинная_цель: str | None = None
            средство: str | None = None
            место: str | None = None
            команда: str | None = None

            def required_keys(self) -> list[str]:
                return ["истинная_цель", "средство", "место"]
    """

    def required_keys(self) -> list[str]:
        """Override in subclass to list required slot keys."""
        return [f.name for f in fields(self)]

    def all_keys(self) -> list[str]:
        return [f.name for f in fields(self)]

    def get(self, key: str) -> Any:
        return getattr(self, key, None)

    def set(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)

    def filled_required(self) -> int:
        return sum(1 for k in self.required_keys() if self.get(k))

    def completeness(self) -> float:
        req = self.required_keys()
        return self.filled_required() / len(req) if req else 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SlotsBase":
        """Reconstruct from a dict; ignore unknown keys gracefully."""
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in (d or {}).items() if k in valid_keys}
        return cls(**filtered)


# ─── Workflow config ───────────────────────────────────────────────────


@dataclass
class PhaseSpec:
    """One phase in the workflow.

    name: phase name (state machine state)
    prompt_builder: callable(state, user_msg, **kwargs) → str
                    returns the markdown the bot should use as its reply
    on_enter: optional callable(state) → None for side effects
              (e.g. set a flag when entering a phase)
    """
    name: str
    prompt_builder: Callable[..., str]
    on_enter: Callable[[Any], None] | None = None


@dataclass
class TransitionSpec:
    """One transition between phases.

    trigger: method name on the machine (e.g. "advance_to_ask")
    source: state name(s) we can transition FROM
    dest: state name we transition TO
    condition: optional callable(state, ctx) → bool; if False, transition
               is skipped and the machine considers the next applicable.
    """
    trigger: str
    source: str | list[str]
    dest: str
    condition: Callable[[Any, dict], bool] | None = None


@dataclass
class WorkflowConfig:
    """Static config describing a workflow.

    Provided by each workflow's `__init__.py` as `WORKFLOW`.

    A workflow is a state graph + per-state prompt builders + a routing
    function. The graph can be a cycle, a linear pipeline, a branching
    tree, or a DAG — the engine doesn't care.

    Fields:
      name: unique workflow identifier (used as state-dir subfolder)
      slots_cls: subclass of SlotsBase
      phases: list of PhaseSpec (state names + per-phase prompt builders)
      transitions: list of TransitionSpec
      initial_phase: state name for fresh sessions (typically "INIT")
      decide_fn: callable(state, user_msg, prev_bot_msg, current_phase) → str
                 — workflow routing logic, returns target phase name
      mandatory_lock_phrase: optional substring that MUST appear in the
                             LOCK phase's mini_prompt template (engine
                             checks at build time; failsafe)
    """
    name: str
    slots_cls: type
    phases: list[PhaseSpec]
    transitions: list[TransitionSpec]
    initial_phase: str
    decide_fn: Callable[..., str]
    mandatory_lock_phrase: str | None = None
    # When the workflow uses a SchemaSlots subclass, this carries the
    # loaded SkillSchema so the runner can drive the LLM extractor.
    # None for legacy workflows that hand-write a SlotsBase subclass.
    schema: Any = None

    def phase_names(self) -> list[str]:
        return [p.name for p in self.phases]

    def get_phase(self, name: str) -> PhaseSpec | None:
        for p in self.phases:
            if p.name == name:
                return p
        return None

    def transitions_dicts(self) -> list[dict[str, Any]]:
        """Convert TransitionSpec → pytransitions transition dict format."""
        out = []
        for t in self.transitions:
            d = {"trigger": t.trigger, "source": t.source, "dest": t.dest}
            if t.condition is not None:
                d["conditions"] = t.condition  # type: ignore[assignment]
            out.append(d)
        return out


# ─── Workflow state container ──────────────────────────────────────────


@dataclass
class WorkflowState:
    """All persistent state for one workflow session.

    Persists as JSON. Workflow-specific flags (motivation_answered,
    user_pushed_for_action, etc.) go into `extras` so the schema is
    stable across workflows.
    """
    session_id: str
    workflow_name: str
    version: int = STATE_VERSION
    created_ts: float = 0.0
    updated_ts: float = 0.0
    iteration: int = 0
    phase: str = "INIT"
    slots: Any = None  # SlotsBase instance — concrete type set by load()
    user_history: list[str] = field(default_factory=list)
    bot_history: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    action_log: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    # ─── persistence ──────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path, slots_cls: type, workflow_name: str) -> "WorkflowState":
        """Load state from JSON or create a fresh one.

        Args:
            path: file path keyed by session_id
            slots_cls: workflow's SlotsBase subclass for slot reconstruction
            workflow_name: persisted into state for cross-check
        """
        if not path.exists():
            now = time.time()
            return cls(
                session_id=path.stem,
                workflow_name=workflow_name,
                created_ts=now,
                updated_ts=now,
                slots=slots_cls(),
            )

        raw = json.loads(path.read_text(encoding="utf-8"))
        slots_dict = raw.pop("slots", None) or {}

        # Mismatched workflow_name = data from a different workflow in same
        # session id space. Don't blend — start fresh and warn.
        # Back-compat: accept legacy `skill_name` field from cyclic-engine v0.
        saved_name = raw.pop("workflow_name", None) or raw.pop("skill_name", "")
        if saved_name and saved_name != workflow_name:
            now = time.time()
            return cls(
                session_id=path.stem,
                workflow_name=workflow_name,
                created_ts=now,
                updated_ts=now,
                slots=slots_cls(),
                action_log=[f"⚠ replaced stale state from workflow={saved_name}"],
            )

        st = cls(workflow_name=workflow_name, **raw)
        st.slots = slots_cls.from_dict(slots_dict)
        return st

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_blob(indent=2), encoding="utf-8")

    # ─── blob serialisation (registry / sqlite back-end) ────────────
    #
    # ``to_blob`` / ``from_blob`` mirror the file-based ``save``/``load``
    # methods but produce / consume a JSON STRING instead of a file path.
    # Used by ``core.registry.Registry`` which stores state in a sqlite
    # ``state_blob`` column. Keeps the file-based methods working for any
    # caller (tests, legacy tooling) — they delegate to the blob layer.

    def to_blob(self, *, indent: int | None = None) -> str:
        """Serialize the live state to a JSON string."""
        self.updated_ts = time.time()
        slots_data: dict[str, Any] = {}
        if self.slots is not None:
            if hasattr(self.slots, "serialize"):
                slots_data = self.slots.serialize()
            elif hasattr(self.slots, "as_dict"):
                slots_data = self.slots.as_dict()
        data = {
            "session_id": self.session_id,
            "workflow_name": self.workflow_name,
            "version": self.version,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "iteration": self.iteration,
            "phase": self.phase,
            "slots": slots_data,
            "user_history": list(self.user_history),
            "bot_history": list(self.bot_history),
            "contradictions": list(self.contradictions),
            "action_log": list(self.action_log),
            "extras": dict(self.extras),
        }
        return json.dumps(data, ensure_ascii=False, indent=indent)

    @classmethod
    def from_blob(
        cls, blob: str, slots_cls: type, workflow_name: str,
        session_id: str,
    ) -> "WorkflowState":
        """Inverse of ``to_blob``.

        Empty / missing blob → fresh state (with the requested
        ``session_id`` — used as identity in the file-less back-end).
        Workflow-name mismatch → fresh state with a warning in the log
        (same policy as ``load``).
        """
        if not blob or blob.strip() in ("", "{}"):
            now = time.time()
            return cls(
                session_id=session_id,
                workflow_name=workflow_name,
                created_ts=now,
                updated_ts=now,
                slots=slots_cls(),
            )
        raw = json.loads(blob)
        slots_dict = raw.pop("slots", None) or {}
        saved_name = raw.pop("workflow_name", None) or raw.pop("skill_name", "")
        if saved_name and saved_name != workflow_name:
            now = time.time()
            return cls(
                session_id=session_id,
                workflow_name=workflow_name,
                created_ts=now,
                updated_ts=now,
                slots=slots_cls(),
                action_log=[f"⚠ replaced stale state from workflow={saved_name}"],
            )
        # Force session_id to the registry-supplied invocation_id even if
        # the blob contains a legacy value — single source of truth.
        raw.pop("session_id", None)
        st = cls(
            session_id=session_id,
            workflow_name=workflow_name,
            **raw,
        )
        st.slots = slots_cls.from_dict(slots_dict)
        return st

    # ─── convenience ──────────────────────────────────────────────

    def add_user_msg(self, msg: str | None) -> None:
        if msg:
            self.user_history.append(msg)

    def add_bot_msg(self, msg: str | None) -> None:
        if msg:
            self.bot_history.append(msg)

    def union_user_text(self) -> str:
        return " ".join(self.user_history)

    def union_bot_text(self) -> str:
        return " ".join(self.bot_history)

    def summary(self) -> dict[str, Any]:
        """Compact JSON-friendly snapshot for logs/tool returns."""
        return {
            "session_id": self.session_id,
            "workflow_name": self.workflow_name,
            "phase": self.phase,
            "iteration": self.iteration,
            "slots": self.slots.as_dict() if self.slots else {},
            "completeness": self.slots.completeness() if self.slots else 0.0,
            "user_msg_count": len(self.user_history),
            "bot_msg_count": len(self.bot_history),
            "contradictions_count": len(self.contradictions),
            "extras": dict(self.extras),
        }
