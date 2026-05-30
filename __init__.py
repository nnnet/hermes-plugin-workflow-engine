"""workflow-engine — generic state-machine orchestrator (engine package).

This module is the Python package entry point. The actual engine
runs as a CLI tool (``cli.py``), invoked via subprocess by consumer
plugins like ``desire-to-goal-driver``. We expose only ``register()``
here so the Hermes plugin scanner can discover the package, label it
in the dashboard UI, and avoid the "Failed to load" warning that
would otherwise appear for any plugin.yaml without a register entry
point.

No hooks are registered — by design. The engine is invoked
out-of-process; see ``cli.py`` for the actual API surface.
"""

from __future__ import annotations


def register(ctx) -> None:
    """Plugin entry point — no hooks needed.

    The workflow-engine is an out-of-process tool invoked via
    ``/opt/workflow-engine/cli.py`` by consumer plugins. This stub
    exists purely so the Hermes plugin scanner recognises the
    package and records it as a loaded plugin in the dashboard.
    """
    return None
