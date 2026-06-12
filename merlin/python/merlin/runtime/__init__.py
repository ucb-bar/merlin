"""Merlin-owned runtime substrate (real, dependency-free).

Merlin owns the runtime abstraction: the command-buffer format, the execution model, and the
metrics/trace schemas. Targets provide adapters. This package is the reference simulator
backend that executes a command buffer with real integer arithmetic and produces real metrics,
a trace, committed outputs, and an independent reference recomputation for correctness.
"""
from __future__ import annotations

from .tensor import Tensor
from .metrics import Metrics, COMMON_METRIC_NAMES
from .commandbuffer import load_command_buffer, validate_command_buffer, materialize_inputs
from .simulator import simulate, SimulationError
from .reference import reference_outputs, outputs_match

__all__ = [
    "Tensor", "Metrics", "COMMON_METRIC_NAMES",
    "load_command_buffer", "validate_command_buffer", "materialize_inputs",
    "simulate", "SimulationError", "reference_outputs", "outputs_match",
]
