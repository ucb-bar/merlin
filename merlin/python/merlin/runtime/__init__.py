"""Merlin-owned runtime substrate (real, dependency-free).

Merlin owns the runtime abstraction: the command-buffer format, the execution model, and the
metrics/trace schemas. Targets provide adapters. This package is the reference simulator
backend that executes a command buffer with real integer arithmetic and produces real metrics,
a trace, committed outputs, and an independent reference recomputation for correctness.
"""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Tensor": "tensor", "Metrics": "metrics", "COMMON_METRIC_NAMES": "metrics",
    "load_command_buffer": "commandbuffer", "validate_command_buffer": "commandbuffer",
    "materialize_inputs": "commandbuffer", "simulate": "simulator",
    "SimulationError": "simulator", "reference_outputs": "reference", "outputs_match": "reference",
}

__all__ = [
    "Tensor", "Metrics", "COMMON_METRIC_NAMES",
    "load_command_buffer", "validate_command_buffer", "materialize_inputs",
    "simulate", "SimulationError", "reference_outputs", "outputs_match",
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
