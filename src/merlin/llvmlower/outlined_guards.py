"""Closed returned-flag guard transport; no controller-side inference math."""

from __future__ import annotations


def guard_tensor(op):
    """Match the exact standard rank-zero I1 extract/assert source boundary.

    The original IR is retained. The command buffer can read the completed
    tensor's canonical byte instead of materializing a scalar ABI buffer.
    Unknown source effects, extra indices and other conditions remain refused.
    """
    from xdsl.dialects.builtin import StringAttr, TensorType, i1
    from xdsl.dialects.cf import AssertOp
    from xdsl.dialects.tensor import ExtractOp

    if not isinstance(op, AssertOp):
        raise ValueError("only an actual standard assertion can become a returned-flag guard")
    extract = op.arg.owner
    if (
        not isinstance(extract, ExtractOp)
        or extract.indices
        or extract.result.type != i1
        or not isinstance(extract.tensor.type, TensorType)
        or extract.tensor.type.get_shape() != ()
        or extract.tensor.type.element_type != i1
        or not isinstance(op.msg, StringAttr)
        or set(op.properties) != {"msg"}
        or any(not key.startswith("prov.") for key in op.attributes)
        or any(not key.startswith("prov.") for key in extract.attributes)
    ):
        raise ValueError("assertion needs an exact canonical rank-zero I1 returned flag")
    return extract.tensor


def discardable_driver_op(op):
    """Only source-proved pure glue may disappear during DAG liveness pruning."""
    from xdsl.dialects.builtin import TensorType
    from xdsl.dialects.linalg.ops import FillOp, YieldOp
    from xdsl.traits import is_side_effect_free

    if is_side_effect_free(op):
        return True
    # xDSL does not attach a pure-tensor effect implementation to FillOp. Its
    # standard immutable-tensor form, with the exact yield-only body, is pure;
    # a buffer/memref form or an unknown body does not inherit that statement.
    return (
        isinstance(op, FillOp)
        and all(isinstance(value.type, TensorType) for value in (*op.outputs, *op.results))
        and bool(op.results)
        and len(op.body.blocks) == 1
        and len(tuple(op.body.block.ops)) == 1
        and isinstance(op.body.block.last_op, YieldOp)
    )
