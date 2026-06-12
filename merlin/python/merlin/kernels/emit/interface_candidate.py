"""Emit an interface_candidate dict (conforming to ``interface_candidate.schema.yaml``).

The interface candidate is the L5 deliverable: it states what the *compiler must prove*, what
the *hardware must provide*, and what the *runtime must provide* for an abstraction to be
exposed — plus the four lowering variants (baseline / software_visible / hardware_managed /
oracle) that the DSE workstream evaluates to answer "expose to SW, hide in HW, or drop?".
"""
from __future__ import annotations

from typing import Iterable

from merlin.common import schemas

# The standard four lowering variants every interface candidate must be evaluated under.
LOWERING_VARIANTS = ("baseline", "software_visible", "hardware_managed", "oracle")


def emit_interface_candidate(
    name: str,
    interface_ops: Iterable[str],
    interface_types: Iterable[str],
    justified_by: dict,
    compiler_must_prove: Iterable[str] = (),
    hardware_must_provide: Iterable[str] = (),
    runtime_must_provide: Iterable[str] = (),
    lowering_variants: Iterable[str] = LOWERING_VARIANTS,
    extra: dict | None = None,
    validate: bool = True,
) -> dict:
    """Build a schema-shaped interface candidate with the L5 contract + lowering variants."""
    cand = {
        "name": name,
        "interface_ops": list(interface_ops),
        "interface_types": list(interface_types),
        "justified_by": dict(justified_by),
        "compiler_must_prove": list(compiler_must_prove),
        "hardware_must_provide": list(hardware_must_provide),
        "runtime_must_provide": list(runtime_must_provide),
        "lowering_variants": list(lowering_variants),
    }
    if extra:
        cand.update(extra)
    if validate:
        schemas.validate_or_raise(cand, "interface_candidate")
    return cand
