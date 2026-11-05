"""Recognize a declared BF16-to-E4M3 truncation in parsed Linalg IR."""

from __future__ import annotations

from .linalg_patterns import InvalidLinalgPattern


def recognize_bf16_fp8_truncation(op) -> tuple[object, object, str, str, int]:
    """Check the scalar cast and maps; return its declared policy and domain.

    The source producer owns the input-domain assertion. A target rule must
    separately establish that its physical conversion implements the declared
    policy on that domain; this check grants no target or runtime admission.
    """
    from xdsl.dialects.arith import ConstantOp, MulfOp, TruncFOp
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        BFloat16Type,
        DenseArrayBase,
        FloatAttr,
        IntegerAttr,
        NoneAttr,
        StringAttr,
        TensorType,
    )
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.dialects.linalg.ops import GenericOp, YieldOp
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    from merlin.xdsl_dialects.fp8 import Float8E4M3FNType

    if type(op) is not GenericOp or len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1:
        raise InvalidLinalgPattern("BF16-to-FP8 truncation needs one input and one result")
    if op.successors or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
        raise InvalidLinalgPattern("BF16-to-FP8 truncation has an unknown property or successor")
    if any(not key.startswith("prov.") and key not in {
        "merlin.numeric_semantics", "merlin.input_domain", "merlin.scale_code"
    } for key in op.attributes):
        raise InvalidLinalgPattern("BF16-to-FP8 truncation has an unknown attribute")
    policy = op.attributes.get("merlin.numeric_semantics")
    domain = op.attributes.get("merlin.input_domain")
    scale = op.attributes.get("merlin.scale_code")
    if (not isinstance(policy, StringAttr) or not policy.data
        or not isinstance(domain, StringAttr) or not domain.data
        or not isinstance(scale, IntegerAttr) or str(scale.type) != "i32"
        or not 0 <= scale.value.data <= 255):
        raise InvalidLinalgPattern("BF16-to-FP8 truncation needs policy, domain, and scale code")
    segments = op.properties["operandSegmentSizes"]
    if (not isinstance(segments, DenseArrayBase) or str(segments.elt_type) != "i32"
        or tuple(segments.iter_values()) != (1, 1)):
        raise InvalidLinalgPattern("BF16-to-FP8 truncation has inconsistent operand segments")
    source, destination, result = (*op.inputs, *op.outputs, *op.results)
    if any(not isinstance(value.type, TensorType) or not isinstance(value.type.encoding, NoneAttr)
           for value in (source, destination, result)):
        raise InvalidLinalgPattern("BF16-to-FP8 truncation needs default-encoded tensors")
    if (not isinstance(source.type.element_type, BFloat16Type)
        or any(not isinstance(value.type.element_type, Float8E4M3FNType)
               for value in (destination, result))):
        raise InvalidLinalgPattern("BF16-to-FP8 truncation has wrong element types")
    shapes = tuple(tuple(value.type.get_shape()) for value in (source, destination, result))
    if len(shapes[0]) != 2 or any(shape != shapes[0] for shape in shapes) or (
        any(type(dimension) is not int or dimension <= 0 for dimension in shapes[0])
    ):
        raise InvalidLinalgPattern("BF16-to-FP8 truncation needs one positive static rank-two shape")
    d0, d1 = AffineDimExpr(0), AffineDimExpr(1)
    identity = AffineMap(2, 0, (d0, d1))
    if (len(op.get_indexing_maps()) != 2 or any(
        not isinstance(index_map, AffineMapAttr) or index_map.data != identity
        for index_map in op.get_indexing_maps()
    ) or not isinstance(op.iterator_types, ArrayAttr) or any(
        not isinstance(iterator, IteratorTypeAttr) for iterator in op.iterator_types
    ) or tuple(iterator.data for iterator in op.iterator_types) != (IteratorType.PARALLEL,) * 2):
        raise InvalidLinalgPattern("BF16-to-FP8 truncation needs parallel identity maps")
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise InvalidLinalgPattern("BF16-to-FP8 truncation needs one scalar body")
    block = op.regions[0].block
    if (len(block.args) != 2 or not isinstance(block.args[0].type, BFloat16Type)
        or not isinstance(block.args[1].type, Float8E4M3FNType)):
        raise InvalidLinalgPattern("BF16-to-FP8 scalar block arguments have wrong types")
    body = tuple(block.ops)
    if scale.value.data == 127:
        if len(body) != 2 or tuple(type(inner) for inner in body) != (TruncFOp, YieldOp):
            raise InvalidLinalgPattern("unit BF16-to-FP8 body must be truncf then yield")
        cast, yield_op = body
        cast_input = block.args[0]
    else:
        if len(body) != 4 or tuple(type(inner) for inner in body) != (ConstantOp, MulfOp, TruncFOp, YieldOp):
            raise InvalidLinalgPattern("scaled BF16-to-FP8 body needs a constant inverse scale and mulf")
        constant, multiply, cast, yield_op = body
        factor = constant.properties.get("value")
        expected = 2.0 ** (127 - scale.value.data)
        if (
            constant.attributes or set(constant.properties) != {"value"}
            or not isinstance(factor, FloatAttr) or not isinstance(factor.type, BFloat16Type)
            or factor.value.data != expected
            or multiply.attributes or set(multiply.properties) != {"fastmath"}
            or multiply.fastmath.data
            or tuple(multiply.operands) != (block.args[0], constant.results[0])
            or not isinstance(multiply.results[0].type, BFloat16Type)
        ):
            raise InvalidLinalgPattern("scaled BF16-to-FP8 body has a wrong inverse scale or multiply")
        cast_input = multiply.results[0]
    if (cast.attributes or cast.properties or tuple(cast.operands) != (cast_input,)
        or not isinstance(cast.results[0].type, Float8E4M3FNType)
        or yield_op.attributes or yield_op.properties
        or tuple(yield_op.operands) != tuple(cast.results)):
        raise InvalidLinalgPattern("BF16-to-FP8 body does not yield the input cast")
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - parsed untrusted IR must refuse
        raise InvalidLinalgPattern("BF16-to-FP8 Linalg operation does not verify") from exc
    return source, destination, policy.data, domain.data, scale.value.data
