"""Structural integer-contraction observations for capture export checks."""

from merlin.common import mlir_query as query

from .capture_refusals import M2MUnavailable


def receipt_export_complete(receipt: dict | None, partition: dict | None) -> bool:
    """Check worker export evidence, retaining v1's absent-batch default."""
    if receipt is None:
        return True
    from merlin.capture.integerization import non_contraction_program

    if non_contraction_program(receipt):
        # No contraction to export: the program's integer work is its annotated non-contraction ops.
        return bool(partition is not None and receipt["golden_agreement"]["status"] == "passed")
    return bool(
        partition is not None
        and receipt.get("accumulator_bound_checked") is True
        and receipt["exported_integer_mm_count"] + receipt["exported_integer_bmm_count"] > 0
        and receipt["integer_mm_emitted"] == receipt["exported_integer_mm_count"]
        and receipt.get("integer_bmm_emitted", 0) == receipt["exported_integer_bmm_count"]
        and receipt["golden_agreement"]["status"] == "passed"
    )


def _forward(module):
    for op in query.walk(module, "func.func"):
        if getattr(getattr(op, "sym_name", None), "data", None) == "forward":
            return op
    raise ValueError("no @forward function")


def _integer_contraction_rank(op) -> int | None:
    if query.op_name(op) == "linalg.generic" and query.attr_str(op, "prov.op") != "int_matmul":
        return None
    if len(op.operands) < 2 or not op.results:
        return None
    lhs, rhs = (query.type_shape_dtype(value.type)[1] for value in op.operands[:2])
    shape, result = query.type_shape_dtype(op.results[0].type)
    return len(shape) if (lhs, rhs, result) == ("i8", "i8", "i32") else None


def exported_integer_counts(program: str) -> tuple[int, int]:
    """Count matrix and batched matrix contractions in the exported entry."""
    try:
        forward = _forward(query.parse(program))
        structural = structural_bmm = 0
        for op in query.walk(forward, "linalg.matmul", "linalg.generic"):
            rank = _integer_contraction_rank(op)
            if rank is not None:
                structural += rank == 2
                structural_bmm += rank > 2
        return structural, structural_bmm
    except Exception as exc:
        raise M2MUnavailable(f"cannot verify integerized model export: {exc}") from exc
