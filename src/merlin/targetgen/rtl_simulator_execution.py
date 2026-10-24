"""Run one synchronous oracle call under hardware-byte and simulator-slot checks."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from .rtl_simulator_pin import verify_selected
from .rtl_simulator_slot import verilator_slot


@dataclass(frozen=True)
class SimulatorExecution:
    """The selected backend and execution policy for one oracle call."""

    backend: Any
    engine: str
    target: str
    timeout: float
    unavailable: type[Exception] = RuntimeError


def run_verified(execution: SimulatorExecution, operation):
    """Reopen an optional hardware pin around an operation using the remaining budget."""
    started = time.monotonic()
    backend, engine, target = execution.backend, execution.engine, execution.target
    timeout, unavailable = execution.timeout, execution.unavailable
    slot = verilator_slot(timeout=timeout) if engine == "verilator" else nullcontext(timeout)
    with slot as remaining:
        admitted = time.monotonic()
        binding = verify_selected(backend, engine, target) if engine != "spike" else None
        remaining -= time.monotonic() - admitted
        if remaining <= 0:
            raise unavailable("RTL simulator verification exhausted the capsule deadline")
        result = operation(remaining)
        if binding is not None:
            after = verify_selected(backend, engine, target)
            if binding["execution_identity"] != after["execution_identity"]:
                raise unavailable("RTL simulator provenance changed during execution")
            if isinstance(result.get("oracle"), dict):
                result["oracle"]["provenance"] = binding
        if engine == "verilator":
            timing = result.setdefault("timing", {})
            timing["oracle_wait_s"] = float(timing.get("oracle_wait_s") or 0) + admitted - started
        return result
