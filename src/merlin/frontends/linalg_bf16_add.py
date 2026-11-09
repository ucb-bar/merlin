"""Recognize explicit FP32 add/subtract with a BF16 bit chop."""

from __future__ import annotations

from .linalg_patterns import InvalidLinalgPattern


def _recognize_bf16_f32_chop_binary(op, *, subtract: bool) -> tuple[object, object, object, str, str]:
    """Check the exact scalar program; leave its pairwise domain as a premise."""
    from xdsl.dialects.arith import (
        AddfOp,
        BitcastOp,
        ConstantOp,
        ExtFOp,
        ShRUIOp,
        SubfOp,
        TruncIOp,
    )
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        BFloat16Type,
        DenseArrayBase,
        Float32Type,
        IntegerAttr,
        IntegerType,
        NoneAttr,
        StringAttr,
        TensorType,
    )
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.dialects.linalg.ops import GenericOp, YieldOp
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    mode = "subtract" if subtract else "add"
    arithmetic_op = SubfOp if subtract else AddfOp
    admitted_domain = ("finite-normal-exact-fp32-difference-or-zero-v1" if subtract
                       else "finite-normal-exact-fp32-sum-v1")
    if type(op) is not GenericOp or len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
        raise InvalidLinalgPattern(f"BF16 {mode} needs two inputs and one tensor result")
    if op.successors or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
        raise InvalidLinalgPattern(f"BF16 {mode} has an unknown Linalg property or successor")
    if any(
        not key.startswith("prov.") and key not in {"merlin.numeric_semantics", "merlin.input_domain"}
        for key in op.attributes
    ):
        raise InvalidLinalgPattern(f"BF16 {mode} has an unknown Linalg attribute")
    policy = op.attributes.get("merlin.numeric_semantics")
    domain = op.attributes.get("merlin.input_domain")
    if (
        not isinstance(policy, StringAttr)
        or policy.data != "selected-f32-rne-bf16-chop-v1"
        or not isinstance(domain, StringAttr)
        or domain.data != admitted_domain
    ):
        raise InvalidLinalgPattern(f"BF16 {mode} needs its exact numerical policy and pairwise domain")
    segments = op.properties["operandSegmentSizes"]
    if (
        not isinstance(segments, DenseArrayBase)
        or str(segments.elt_type) != "i32"
        or tuple(segments.iter_values()) != (2, 1)
    ):
        raise InvalidLinalgPattern(f"BF16 {mode} has inconsistent operand segments")
    values = (*op.inputs, *op.outputs, *op.results)
    if any(
        not isinstance(value.type, TensorType)
        or not isinstance(value.type.element_type, BFloat16Type)
        or not isinstance(value.type.encoding, NoneAttr)
        for value in values
    ):
        raise InvalidLinalgPattern(f"BF16 {mode} needs default-encoded BF16 tensors")
    shapes = tuple(tuple(value.type.get_shape()) for value in values)
    if (
        len(shapes[0]) != 2
        or any(shape != shapes[0] for shape in shapes)
        or any(type(dimension) is not int or dimension <= 0 for dimension in shapes[0])
    ):
        raise InvalidLinalgPattern(f"BF16 {mode} needs one positive static rank-two shape")
    d0, d1 = AffineDimExpr(0), AffineDimExpr(1)
    identity = AffineMap(2, 0, (d0, d1))
    if (
        len(op.get_indexing_maps()) != 3
        or any(
            not isinstance(index_map, AffineMapAttr) or index_map.data != identity
            for index_map in op.get_indexing_maps()
        )
        or not isinstance(op.iterator_types, ArrayAttr)
        or any(not isinstance(iterator, IteratorTypeAttr) for iterator in op.iterator_types)
        or tuple(iterator.data for iterator in op.iterator_types) != (IteratorType.PARALLEL,) * 2
    ):
        raise InvalidLinalgPattern(f"BF16 {mode} needs parallel identity index maps")
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise InvalidLinalgPattern(f"BF16 {mode} needs one scalar body")
    block = op.regions[0].block
    if len(block.args) != 3 or any(not isinstance(arg.type, BFloat16Type) for arg in block.args):
        raise InvalidLinalgPattern(f"BF16 {mode} scalar arguments differ from its tensor elements")
    body = tuple(block.ops)
    if tuple(type(inner) for inner in body) != (
        ExtFOp,
        ExtFOp,
        arithmetic_op,
        BitcastOp,
        ConstantOp,
        ShRUIOp,
        TruncIOp,
        BitcastOp,
        YieldOp,
    ):
        raise InvalidLinalgPattern(f"BF16 {mode} body must widen, {mode} in FP32, and chop bits")
    lhs, rhs, add, bits, shift, high, narrow, chopped, yield_op = body
    literal = shift.properties.get("value")
    if (
        any(inner.attributes for inner in body)
        or any(inner.properties for inner in (lhs, rhs, bits, high, narrow, chopped, yield_op))
        or set(add.properties) != {"fastmath"}
        or add.fastmath.data
        or set(shift.properties) != {"value"}
        or not isinstance(literal, IntegerAttr)
        or str(literal.type) != "i32"
        or literal.value.data != 16
        or tuple(lhs.operands) != (block.args[0],)
        or tuple(rhs.operands) != (block.args[1],)
        or tuple(add.operands) != (lhs.results[0], rhs.results[0])
        or tuple(bits.operands) != tuple(add.results)
        or tuple(high.operands) != (bits.results[0], shift.results[0])
        or tuple(narrow.operands) != tuple(high.results)
        or tuple(chopped.operands) != tuple(narrow.results)
        or tuple(yield_op.operands) != tuple(chopped.results)
        or not isinstance(lhs.results[0].type, Float32Type)
        or not isinstance(rhs.results[0].type, Float32Type)
        or not isinstance(add.results[0].type, Float32Type)
        or not isinstance(bits.results[0].type, IntegerType)
        or str(bits.results[0].type) != "i32"
        or not isinstance(shift.results[0].type, IntegerType)
        or str(shift.results[0].type) != "i32"
        or not isinstance(high.results[0].type, IntegerType)
        or str(high.results[0].type) != "i32"
        or not isinstance(narrow.results[0].type, IntegerType)
        or str(narrow.results[0].type) != "i16"
        or not isinstance(chopped.results[0].type, BFloat16Type)
    ):
        raise InvalidLinalgPattern(f"BF16 {mode} does not yield an unflagged FP32 result with BF16 bit chop")
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - parsed untrusted IR must refuse
        raise InvalidLinalgPattern(f"BF16 {mode} Linalg operation does not verify") from exc
    return op.inputs[0], op.inputs[1], op.outputs[0], policy.data, domain.data


def recognize_bf16_f32_chop_add(op) -> tuple[object, object, object, str, str]:
    return _recognize_bf16_f32_chop_binary(op, subtract=False)


def recognize_bf16_f32_chop_sub(op) -> tuple[object, object, object, str, str]:
    return _recognize_bf16_f32_chop_binary(op, subtract=True)
