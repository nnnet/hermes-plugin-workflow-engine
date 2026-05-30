# Adding a new workflow

A workflow is a state graph + per-state prompt builders + a routing
function. Lives anywhere on disk; the engine loads it via `--workflow PATH`.

Typical layout (recommended): inside the skill that uses it, in a
`workflow/` subdirectory, so the workflow definition travels with the
skill that invokes it.

## Walkthrough — adding a `risk-assessment` workflow

### 1. Directory + module init

```bash
mkdir -p ~/.hermes/skills/<skill-path>/workflow
touch ~/.hermes/skills/<skill-path>/workflow/__init__.py
```

`workflow/__init__.py`:

```python
from .config import WORKFLOW

__all__ = ["WORKFLOW"]
```

### 2. Slot schema + decide_fn (`config.py`)

```python
from dataclasses import dataclass
from core.state import SlotsBase, WorkflowConfig, PhaseSpec, TransitionSpec
from core import detectors
from . import prompts


@dataclass
class RiskAssessmentSlots(SlotsBase):
    risk_type: str | None = None
    blast_radius: str | None = None
    likelihood: str | None = None
    severity: str | None = None

    def required_keys(self):
        return ["risk_type", "blast_radius", "severity"]


LABEL_MAP = {
    "risk_type": ["Тип риска", "Risk type"],
    "blast_radius": ["Радиус", "Blast radius"],
    "likelihood": ["Вероятность", "Likelihood"],
    "severity": ["Тяжесть", "Severity"],
}


def _decide(wstate, user_msg, prev_bot_msg, current_phase):
    if prev_bot_msg:
        extracted = detectors.extract_slots(prev_bot_msg, LABEL_MAP)
        for k, v in extracted.items():
            if v and not wstate.slots.get(k):
                wstate.slots.set(k, v)

    if wstate.iteration <= 1:
        return "DECOMPOSE"

    if wstate.slots.filled_required() == len(wstate.slots.required_keys()):
        if current_phase != "ASSESS":
            return "ASSESS"
        return "REPORT"

    return "ASK"


PHASES = [
    PhaseSpec(name="INIT", prompt_builder=lambda s, u: ""),
    PhaseSpec(name="DECOMPOSE", prompt_builder=prompts.decompose),
    PhaseSpec(name="ASK", prompt_builder=prompts.ask),
    PhaseSpec(name="ASSESS", prompt_builder=prompts.assess),
    PhaseSpec(name="REPORT", prompt_builder=prompts.report),
]

TRANSITIONS = [
    TransitionSpec(trigger="begin", source="INIT", dest="DECOMPOSE"),
    TransitionSpec(trigger="advance_to_ask",
                   source=["DECOMPOSE", "ASSESS"], dest="ASK"),
    TransitionSpec(trigger="advance_to_assess",
                   source=["ASK", "DECOMPOSE"], dest="ASSESS"),
    TransitionSpec(trigger="advance_to_report",
                   source="ASSESS", dest="REPORT"),
]

WORKFLOW = WorkflowConfig(
    name="risk-assessment",
    slots_cls=RiskAssessmentSlots,
    phases=PHASES,
    transitions=TRANSITIONS,
    initial_phase="INIT",
    decide_fn=_decide,
    mandatory_lock_phrase=None,
)
```

### 3. Phase prompts (`prompts.py`)

```python
def decompose(state, user_msg):
    return f"""You received a risk topic:

> {user_msg}

Decompose into 4 slots..."""
```

### 4. Invoke from your SKILL.md

```bash
python3 /opt/workflow-engine/cli.py \
    --workflow ~/.hermes/skills/<skill-path>/workflow \
    --session <chat_id> \
    --user "<latest user msg>" \
    --prev-bot "<previous bot reply or empty>"
```

State persists in `~/.hermes/workflow_state/risk-assessment/<session>.json`.

## What you reuse from core (no need to rewrite)

| Need | Use |
|---|---|
| Tokenization | `core.detectors.tokens()` |
| Russian stem match | `core.detectors.has_stem_match()` |
| Jaccard similarity | `core.detectors.jaccard()` |
| Non-answer detection | `core.detectors.is_non_answer()` |
| User push detection | `core.detectors.detect_user_pushed_for_action()` |
| Means/proxy goal | `core.detectors.is_means_as_goal()` |
| Slot extraction | `core.detectors.extract_slots()` |
| Lock-phrase check | `core.detectors.has_lock_signal()` |
| Grounding score | `core.detectors.grounding_score()` |
| State machine | `core.machine.WorkflowMachine` |
| Persistence | `core.state.WorkflowState` |

## What you supply per-workflow

- Slot schema (3-10 named fields)
- 3-7 phases with prompt builders
- Transitions between phases
- decide_fn (routing logic — usually ~30-50 lines)
- LABEL_MAP for slot extraction
- Optional: domain-specific detectors (extend `core.detectors`)
- Optional: `mandatory_lock_phrase` if there is a hard-veto phrase

## Common pitfalls

1. **Don't name the state container `self.state`** — pytransitions
   binds that attribute. Use `self.wstate` (the engine handles this).
2. **Transitions must be reachable** — if decide_fn returns target X
   from current Y but no Y→X transition exists, the machine stays put.
3. **State JSON is per-workflow** — sessions are isolated by workflow
   name in `~/.hermes/workflow_state/<workflow>/<session>.json`.
4. **mandatory_lock_phrase only fires at phase named LOCK** — rename
   your exit phase or set the field to `None` if not applicable.
