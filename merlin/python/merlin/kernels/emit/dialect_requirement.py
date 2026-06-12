"""Emit a dialect_requirement dict (conforming to ``dialect_requirement.schema.yaml``).

The L6 deliverable: what a target dialect must provide (ops/types/verifiers + lowering
path) to implement a promoted interface candidate. This is *input to* TargetGen's
``dialect_plan`` — kernel mining proposes requirements, it never creates dialect ops.
A requirement exists only because an interface candidate cleared the promotion gate;
``status`` stays ``proposed`` until Stage-F target-lowering validation.
"""
from __future__ import annotations

from typing import Iterable

from merlin.common import schemas

# Verifier conditions each known interface needs a target dialect to enforce.
VERIFIERS: dict[str, list[str]] = {
    "resident_packed_tensor": ["capacity_constraint", "lifetime_constraint",
                               "layout_constraint"],
    "accumulator_commit": ["no_intervening_materialization", "output_dtype_known",
                           "epilogue_adjacency"],
    "async_pipeline": ["double_buffer_capacity", "completion_before_use"],
}


def emit_dialect_requirement(
    source_abstraction: str,
    required_ops: Iterable[str],
    required_types: Iterable[str],
    target: str = "toy_npu",
    required_verifiers: Iterable[str] | None = None,
    lowering_target: Iterable[str] = ("command_buffer", "simulator"),
    extra: dict | None = None,
    validate: bool = True,
) -> dict:
    """Build a schema-shaped L6 dialect requirement for one interface candidate."""
    verifiers = (list(required_verifiers) if required_verifiers is not None
                 else VERIFIERS.get(source_abstraction, ["op_specific_review_needed"]))
    req = {
        "source_abstraction": source_abstraction,
        "target": target,
        "required_ops": list(required_ops),
        "required_types": list(required_types),
        "required_verifiers": verifiers,
        "lowering_target": list(lowering_target),
        "status": "proposed",
    }
    if extra:
        req.update(extra)
    if validate:
        schemas.validate_or_raise(req, "dialect_requirement")
    return req
