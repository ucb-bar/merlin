"""Shared structural recognition of admitted Linalg scalar regions.

The reader and independent source verifier use this one checked pattern so
provenance labels or operation names cannot silently stand in for the body.
"""

from __future__ import annotations


class InvalidLinalgPattern(ValueError):
    pass


def recognize_signed_i8_i32_matmul(op) -> tuple:
    """Recognize a rank-two signed i8 matmul with ordered i32 wrap accumulation."""
    from xdsl.dialects.builtin import TensorType
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    if len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
        raise InvalidLinalgPattern("linalg.generic is not a two-input, one-init, one-result contraction")
    lhs, rhs = op.inputs
    init = op.outputs[0]
    values = (lhs, rhs, init, op.results[0])
    if not all(isinstance(value.type, TensorType) for value in values) or [
        str(value.type.element_type) for value in values
    ] != ["i8", "i8", "i32", "i32"]:
        raise InvalidLinalgPattern("linalg.generic is not signed i8 x i8 -> i32 matmul")
    if op.results[0].type != init.type:
        raise InvalidLinalgPattern("linalg.generic result type differs from its initialized output")

    maps = tuple(attribute.data for attribute in op.indexing_maps)
    d0, d1, d2 = (AffineDimExpr(i) for i in range(3))
    expected = (
        AffineMap(3, 0, (d0, d2)),
        AffineMap(3, 0, (d2, d1)),
        AffineMap(3, 0, (d0, d1)),
    )
    if maps != expected:
        raise InvalidLinalgPattern("linalg.generic indexing maps are not rank-2 matmul maps")
    if tuple(attribute.data for attribute in op.iterator_types) != (
        IteratorType.PARALLEL, IteratorType.PARALLEL, IteratorType.REDUCTION,
    ):
        raise InvalidLinalgPattern("linalg.generic iterators are not two parallel and one reduction")
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise InvalidLinalgPattern("linalg.generic requires a single scalar body block")
    block = op.regions[0].block
    args = list(block.args)
    if len(args) != 3 or [str(arg.type) for arg in args] != ["i8", "i8", "i32"]:
        raise InvalidLinalgPattern("linalg.generic body argument types do not match signed matmul")
    body_ops = list(block.ops)
    if [inner.name for inner in body_ops] != [
        "arith.extsi", "arith.extsi", "arith.muli", "arith.addi", "linalg.yield",
    ]:
        raise InvalidLinalgPattern("linalg.generic body is not signed widen, multiply, add, yield")
    ex_lhs, ex_rhs, mul, add, yld = body_ops
    if (list(ex_lhs.operands), list(ex_rhs.operands)) != ([args[0]], [args[1]]):
        raise InvalidLinalgPattern("linalg.generic body does not widen both input elements")
    if [str(ex_lhs.results[0].type), str(ex_rhs.results[0].type)] != ["i32", "i32"]:
        raise InvalidLinalgPattern("linalg.generic body widens to a different integer width")
    if set(mul.operands) != {ex_lhs.results[0], ex_rhs.results[0]} or len(mul.operands) != 2:
        raise InvalidLinalgPattern("linalg.generic body product does not use both widened inputs")
    if set(add.operands) != {mul.results[0], args[2]} or len(add.operands) != 2:
        raise InvalidLinalgPattern("linalg.generic body sum does not use product and accumulator")
    if list(yld.operands) != [add.results[0]]:
        raise InvalidLinalgPattern("linalg.generic body yields a different value")
    for arithmetic in (mul, add):
        if str(arithmetic.results[0].type) != "i32" or arithmetic.overflow_flags.data:
            raise InvalidLinalgPattern("linalg.generic body has an unsupported arithmetic width or overflow flag")
    return lhs, rhs, init
