"""Closed declaration vocabulary for existing integer-reduction source proofs.

This module selects a source recognizer; it does not admit host placement,
prove a linked build, or establish numerical equivalence.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from merlin.compiler.frontend.linalg_patterns import InvalidLinalgPattern

STATIC_INTEGER_REDUCTION_SOURCE_BODY_SCHEMA = "merlin.static_integer_reduction_source_body.v1"

# The one frontend/root-Linalg/kind mapping shared by host admission and the
# independently linked private source witness. Scalar children are not roots.
INTEGER_REDUCTION_TARGETS = {
    "aten.sum.dim_IntList": ("linalg.reduce", "sum"),
    "aten.cumsum.default": ("linalg.generic", "cumsum"),
    "aten.min.dim": ("linalg.generic", "i64_min_first_index"),
}


def serialized_reduction_pattern(pattern: object) -> dict:
    """Use the same strict JSON shape before and after an admission receipt is saved."""
    return json.loads(json.dumps(asdict(pattern), allow_nan=False))


def validate_static_integer_reduction_source_body(declaration: object) -> dict[str, str]:
    """Validate a closed source-only selector, granting no operation support."""
    if (
        not isinstance(declaration, dict)
        or set(declaration) != {"schema", "operation"}
        or declaration.get("schema") != STATIC_INTEGER_REDUCTION_SOURCE_BODY_SCHEMA
        or not isinstance(declaration.get("operation"), str)
        or declaration.get("operation") not in {kind for _, kind in INTEGER_REDUCTION_TARGETS.values()}
    ):
        raise InvalidLinalgPattern("source_body requires a closed integer-reduction v1 declaration")
    return dict(declaration)
