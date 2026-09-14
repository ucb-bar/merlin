"""Emit an interface_candidate justified by design pressure -- a thin front for the one emitter.

``merlin.kernels.emit.interface_candidate`` builds the full L5 candidate (what the compiler must prove,
the hardware and runtime must provide, and the four lowering variants every candidate is evaluated
under). This entry point keeps the design-pressure calling convention and supplies ``justified_by``
from it, so both producers emit the same shape instead of two drifting copies of it.
"""
from __future__ import annotations

from typing import Iterable

from merlin.kernels.emit.interface_candidate import emit_interface_candidate as _emit


def emit_interface_candidate(
    name: str,
    interface_ops: Iterable[str],
    interface_types: Iterable[str],
    design_pressure_name: str,
    policies: Iterable[str],
    validate: bool = True,
) -> dict:
    """Build a schema-shaped interface candidate justified by design pressure + policies."""
    return _emit(name, interface_ops, interface_types,
                 justified_by={"design_pressure": design_pressure_name, "policies": list(policies)},
                 validate=validate)
