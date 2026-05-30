"""workflow-engine.core — generic primitives for stateful workflows.

Public surface: WorkflowConfig + PhaseSpec + TransitionSpec + SlotsBase
let a workflow declare its state graph. The engine handles persistence,
state-machine wiring, and prompt building.

State backends:
- Registry (sqlite) — preferred; multi-agent safe, atomic, transactional.
  Identity = (workflow_name, agent_id, conversation_id, invocation_id).
- File (legacy) — retained for tooling that still passes raw paths.
"""
from .state import SlotsBase, WorkflowState, WorkflowConfig, PhaseSpec, TransitionSpec
from .machine import WorkflowMachine
from .runner import run, run_status, run_reset
from .registry import Registry, InvocationRecord
from .llm_config import (
    LLMConfig, get_engine_llm_config, get_stage_llm_config, llm_enabled,
)
from . import detectors

__all__ = [
    "SlotsBase",
    "WorkflowState",
    "WorkflowConfig",
    "PhaseSpec",
    "TransitionSpec",
    "WorkflowMachine",
    "run",
    "run_status",
    "run_reset",
    "Registry",
    "InvocationRecord",
    "LLMConfig",
    "get_engine_llm_config",
    "get_stage_llm_config",
    "llm_enabled",
    "detectors",
]
