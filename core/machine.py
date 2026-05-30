"""WorkflowMachine — pytransitions wrapper over WorkflowConfig.

Each workflow provides a WorkflowConfig with phases + transitions + decide_fn.
This wrapper instantiates a pytransitions Machine, exposes phase trigger
methods on `self`, and delegates routing to workflow.decide_fn.

The workflow's decide_fn returns the target phase name; the wrapper finds
the appropriate trigger and fires it.

Important: pytransitions binds `self.state` on the model to track the
current state. We use `self.wstate` for our WorkflowState container to
avoid the name collision.
"""
from __future__ import annotations

try:
    from transitions import Machine
except ImportError:  # pragma: no cover — engine entry handles this
    Machine = None  # type: ignore[misc,assignment]

from .state import WorkflowState, WorkflowConfig


class WorkflowMachine:
    """Wraps a WorkflowConfig + WorkflowState in a pytransitions Machine."""

    def __init__(self, config: WorkflowConfig, wstate: WorkflowState):
        if Machine is None:
            raise ImportError(
                "pytransitions not installed. Run: pip install transitions"
            )

        self.config = config
        self.wstate = wstate  # NOT self.state — collides with pytransitions

        states = config.phase_names()
        initial = (
            wstate.phase
            if wstate.phase in states
            else config.initial_phase
        )

        self._machine = Machine(
            model=self,
            states=states,
            transitions=config.transitions_dicts(),
            initial=initial,
            auto_transitions=False,
        )

        for phase in config.phases:
            if phase.on_enter:
                getattr(self._machine, f"on_enter_{phase.name}")(phase.on_enter)

    def decide(
        self,
        user_msg: str,
        prev_bot_msg: str | None = None,
    ) -> str:
        """Delegate routing to workflow, fire matching trigger, return target.

        decide_fn signature:
            decide_fn(wstate, user_msg, prev_bot_msg, current_phase) → str
              returns target phase name

        After deciding, find a transition whose source includes current
        and dest equals target, fire it. If multiple match, the first
        applicable wins. If none match (e.g. target == current), no-op.
        """
        current = self.state  # pytransitions current state name
        target = self.config.decide_fn(
            self.wstate,
            user_msg or "",
            prev_bot_msg or "",
            current,
        )

        if target == current:
            return target

        for t in self.config.transitions:
            sources = t.source if isinstance(t.source, list) else [t.source]
            if current in sources and t.dest == target:
                trigger_fn = getattr(self, t.trigger, None)
                if trigger_fn:
                    try:
                        trigger_fn()
                    except Exception:
                        continue
                    return target

        return current

    def current_phase(self) -> str:
        return self.state  # pytransitions attr
