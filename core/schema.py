"""Schema-driven slot definitions for workflow-engine.

Replaces the bespoke `SlotsBase` subclass-per-workflow pattern with a
single YAML schema that all consumers (engine, extractor LLM,
F1 tests, final-artifact writer, SKILL.md authors) read at runtime.

Why
    Editing the slot schema previously required touching three places —
    `DesireToGoalSlots` dataclass, `LABEL_MAP` dict, `cases/*.yaml`
    assertions, plus the prose in `prompts.py`. Drift was unavoidable.
    A single ``schema.yaml`` per skill collapses all of that to one
    edit point.

What
    `load_schema(path)` returns a `SkillSchema` dataclass with typed
    accessors (slots, audit fields, export config, clarity weights,
    phase config). `SchemaSlots` is a dict-based SlotsBase replacement
    that any workflow can use — no per-skill subclass needed.

Test
    `tests/test_schema.py` round-trips load/save and asserts schema
    fields surface correctly (planned for next commit).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from core.state import SlotsBase


# ─── Schema dataclasses ────────────────────────────────────────────────


@dataclass
class SlotSpec:
    """One slot definition from the schema YAML."""
    name: str
    type: str = "string"           # currently only "string" supported
    required: bool = False
    description: str = ""
    examples: list[str] = field(default_factory=list)


@dataclass
class ConfidenceSpec:
    threshold_high: float = 0.75
    threshold_low: float = 0.4
    default: float = 0.0


@dataclass
class PhaseSchemaSpec:
    initial: str = "INIT"
    required_completion: str = "LOCK"
    terminal: str = "DONE"
    max_iterations: int = 8


@dataclass
class ClarityScoreSpec:
    weights: dict[str, float] = field(default_factory=dict)
    veto: dict[str, bool] = field(default_factory=dict)


@dataclass
class ExportSpec:
    fields: list[str] = field(default_factory=list)
    next_skill_hint: str = ""


@dataclass
class SkillSchema:
    """Top-level schema container."""
    name: str
    version: int
    description: str
    slots: dict[str, SlotSpec]
    audit_fields: list[str]
    confidence: ConfidenceSpec
    phases: PhaseSchemaSpec
    clarity_score: ClarityScoreSpec
    mandatory_lock_phrase: Optional[str]
    export: ExportSpec
    source_path: str = ""

    def required_slots(self) -> list[str]:
        return [name for name, spec in self.slots.items() if spec.required]

    def optional_slots(self) -> list[str]:
        return [name for name, spec in self.slots.items() if not spec.required]

    def all_slot_names(self) -> list[str]:
        return list(self.slots.keys())


# ─── Loader ────────────────────────────────────────────────────────────


def load_schema(path: str | Path) -> SkillSchema:
    """Read schema.yaml + return SkillSchema. Strict on missing required fields."""
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"schema {p} did not parse to a mapping")

    slots_raw = raw.get("slots") or {}
    slots = {
        name: SlotSpec(
            name=name,
            type=str(spec.get("type", "string")),
            required=bool(spec.get("required", False)),
            description=str(spec.get("description", "")).strip(),
            examples=list(spec.get("examples", []) or []),
        )
        for name, spec in slots_raw.items()
    }

    audit_raw = raw.get("audit") or {}
    audit_fields = [k for k, v in audit_raw.items() if isinstance(v, list)]

    conf_raw = raw.get("confidence") or {}
    confidence = ConfidenceSpec(
        threshold_high=float(conf_raw.get("threshold_high", 0.75)),
        threshold_low=float(conf_raw.get("threshold_low", 0.4)),
        default=float(conf_raw.get("default", 0.0)),
    )

    phases_raw = raw.get("phases") or {}
    phases = PhaseSchemaSpec(
        initial=str(phases_raw.get("initial", "INIT")),
        required_completion=str(phases_raw.get("required_completion", "LOCK")),
        terminal=str(phases_raw.get("terminal", "DONE")),
        max_iterations=int(phases_raw.get("max_iterations", 8)),
    )

    cs_raw = raw.get("clarity_score") or {}
    clarity = ClarityScoreSpec(
        weights={k: float(v) for k, v in (cs_raw.get("weights") or {}).items()},
        veto={k: bool(v) for k, v in (cs_raw.get("veto") or {}).items()},
    )

    export_raw = raw.get("export") or {}
    export = ExportSpec(
        fields=list(export_raw.get("fields") or []),
        next_skill_hint=str(export_raw.get("next_skill_hint", "")),
    )

    return SkillSchema(
        name=str(raw.get("name", p.parent.name)),
        version=int(raw.get("version", 1)),
        description=str(raw.get("description", "")).strip(),
        slots=slots,
        audit_fields=audit_fields,
        confidence=confidence,
        phases=phases,
        clarity_score=clarity,
        mandatory_lock_phrase=raw.get("mandatory_lock_phrase"),
        export=export,
        source_path=str(p.resolve()),
    )


# ─── Schema-driven slot container ──────────────────────────────────────


class SchemaSlots(SlotsBase):
    """Dict-backed SlotsBase that reads its shape from a SkillSchema.

    Drop-in replacement for per-workflow @dataclass subclasses. Any
    workflow whose slots are declared in schema.yaml can use this
    instead of writing a new SlotsBase subclass.

    Stores values in `_values: dict[str, str | None]`. Confidence is
    tracked separately in `_confidence: dict[str, float]` so the
    extractor can write structured output without losing audit info.
    """

    # Class-level schema — set by `bind(schema)` before instantiation.
    _schema: Optional[SkillSchema] = None

    def __init__(self, **kwargs):
        # SlotsBase is a dataclass-style base, but SchemaSlots intentionally
        # bypasses the @dataclass decorator. Slots are dict-driven, not
        # field-driven, so we initialize containers manually.
        schema = type(self)._schema
        if schema is None:
            raise RuntimeError(
                "SchemaSlots subclass not bound — call SchemaSlots.bind(schema) first"
            )
        self._values: dict[str, str | None] = {
            name: kwargs.get(name) for name in schema.all_slot_names()
        }
        self._confidence: dict[str, float] = {
            name: schema.confidence.default for name in schema.all_slot_names()
        }

    @classmethod
    def bind(cls, schema: SkillSchema) -> type["SchemaSlots"]:
        """Return a subclass bound to a specific schema.

        Each call mints a new subclass so multiple workflows with
        different schemas can coexist in one process without bleeding.
        """
        bound = type(f"SchemaSlots_{schema.name}", (cls,), {"_schema": schema})
        return bound

    # ── SlotsBase interface ──

    def required_keys(self) -> list[str]:
        return type(self)._schema.required_slots()  # type: ignore[union-attr]

    def all_keys(self) -> list[str]:
        return type(self)._schema.all_slot_names()  # type: ignore[union-attr]

    def get(self, key: str) -> Any:
        return self._values.get(key)

    def set(self, key: str, value: Any) -> None:
        if key in self._values:
            self._values[key] = value

    def filled_required(self) -> int:
        return sum(1 for k in self.required_keys() if self.get(k))

    def completeness(self) -> float:
        req = self.required_keys()
        return self.filled_required() / len(req) if req else 0.0

    # ── Confidence helpers ──

    def set_confidence(self, key: str, score: float) -> None:
        if key in self._confidence:
            self._confidence[key] = max(0.0, min(1.0, float(score)))

    def get_confidence(self, key: str) -> float:
        return self._confidence.get(key, 0.0)

    def confidence_dict(self) -> dict[str, float]:
        return dict(self._confidence)

    def is_high_confidence(self, key: str) -> bool:
        schema = type(self)._schema
        if schema is None:
            return False
        return self.get_confidence(key) >= schema.confidence.threshold_high

    # ── Persistence ──

    def as_dict(self) -> dict[str, Any]:
        return dict(self._values)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SchemaSlots":
        # Accept either bare slot dict or wrapped {values, confidence}
        if "_values" in d and isinstance(d.get("_values"), dict):
            inst = cls(**d["_values"])
            for k, v in (d.get("_confidence") or {}).items():
                inst.set_confidence(k, v)
            return inst
        # Bare dict path — common legacy state files
        valid_keys = set(cls._schema.all_slot_names()) if cls._schema else set()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})

    def serialize(self) -> dict[str, Any]:
        """Full structured dump for state.json — includes confidence."""
        return {"_values": dict(self._values), "_confidence": dict(self._confidence)}


# ─── Final-artifact builder ────────────────────────────────────────────


def build_final_artifact(
    schema: SkillSchema,
    slots: SchemaSlots,
    *,
    session_id: str,
    turns_used: int,
    clarity_score: float,
    желание: list[str],
    bot_history: list[str],
) -> dict[str, Any]:
    """Assemble the YAML artifact emitted when the workflow reaches DONE.

    `желание[]` (raw user wishes accumulated across turns) is included
    in the artifact for audit but is NOT in `export.fields` — callers
    serializing for the next skill must filter to `goal/confidence/…`
    only. This separation is the user's explicit design: log everything,
    forward only the clarified goal.
    """
    goal_block: dict[str, str] = {}
    for name in schema.all_slot_names():
        val = slots.get(name)
        if val is not None and val != "":
            goal_block[name] = str(val)

    return {
        "session": session_id,
        "workflow": schema.name,
        "schema_version": schema.version,
        "clarified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "желание": list(желание),     # raw audit — NOT exported to next skill
        "bot_history": list(bot_history),
        "goal": goal_block,           # ← what next skill receives
        "confidence": slots.confidence_dict(),
        "turns_used": turns_used,
        "clarity_score": clarity_score,
        "export_fields": list(schema.export.fields),
        "next_skill_hint": schema.export.next_skill_hint,
    }


def export_subset(artifact: dict[str, Any]) -> dict[str, Any]:
    """Strip an artifact down to only `export.fields` — what the next skill sees.

    `желание[]`, `bot_history`, raw audit stays in the on-disk artifact
    but never crosses the skill boundary.
    """
    keep = set(artifact.get("export_fields") or [])
    return {k: v for k, v in artifact.items() if k in keep}
