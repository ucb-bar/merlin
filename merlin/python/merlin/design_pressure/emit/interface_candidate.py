"""Emit an interface_candidate dict (conforming to ``interface_candidate.schema.yaml``)."""
from __future__ import annotations

from typing import Iterable

from merlin.common import schemas


def emit_interface_candidate(
    name: str,
    interface_ops: Iterable[str],
    interface_types: Iterable[str],
    design_pressure_name: str,
    policies: Iterable[str],
    validate: bool = True,
) -> dict:
    """Build a schema-shaped interface candidate justified by design pressure + policies."""
    cand = {
        "name": name,
        "interface_ops": list(interface_ops),
        "interface_types": list(interface_types),
        "justified_by": {
            "design_pressure": design_pressure_name,
            "policies": list(policies),
        },
    }
    if validate:
        schemas.validate_or_raise(cand, "interface_candidate")
    return cand
