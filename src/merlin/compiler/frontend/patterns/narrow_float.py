"""Recognize explicit FP32 add/subtract with a BF16 bit chop."""

from __future__ import annotations

from math import copysign

from merlin.compiler.frontend.patterns.linalg_patterns import InvalidLinalgPattern

# ---- linalg_bf16_add ---------------------------------------------------------------------------


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
    admitted_domain = (
        "finite-normal-exact-fp32-difference-or-zero-v1" if subtract else "finite-normal-exact-fp32-sum-v1"
    )
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


# ---- linalg_bf16_mul ---------------------------------------------------------------------------
# Recognize an explicit nearest-even BF16 product in a Linalg scalar body.


def recognize_bf16_product_rne(op) -> tuple[object, object, object, str, str]:
    """Check the exact scalar program; normal inputs/results remain premises."""
    from xdsl.dialects.arith import (
        AddiOp,
        AndIOp,
        BitcastOp,
        ConstantOp,
        ExtFOp,
        MulfOp,
        ShRUIOp,
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

    if type(op) is not GenericOp or len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
        raise InvalidLinalgPattern("BF16 multiply needs two inputs and one tensor result")
    if op.successors or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
        raise InvalidLinalgPattern("BF16 multiply has an unknown Linalg property or successor")
    if any(
        not key.startswith("prov.") and key not in {"merlin.numeric_semantics", "merlin.input_domain"}
        for key in op.attributes
    ):
        raise InvalidLinalgPattern("BF16 multiply has an unknown Linalg attribute")
    policy = op.attributes.get("merlin.numeric_semantics")
    domain = op.attributes.get("merlin.input_domain")
    if (
        not isinstance(policy, StringAttr)
        or policy.data != "bf16-product-rne-v1"
        or not isinstance(domain, StringAttr)
        or domain.data != "finite-normal-bf16-product-normal-v1"
    ):
        raise InvalidLinalgPattern("BF16 multiply needs its numerical policy and bounded domain")
    segments = op.properties["operandSegmentSizes"]
    if (
        not isinstance(segments, DenseArrayBase)
        or str(segments.elt_type) != "i32"
        or tuple(segments.iter_values()) != (2, 1)
    ):
        raise InvalidLinalgPattern("BF16 multiply has inconsistent operand segments")
    values = (*op.inputs, *op.outputs, *op.results)
    if any(
        not isinstance(value.type, TensorType)
        or not isinstance(value.type.element_type, BFloat16Type)
        or not isinstance(value.type.encoding, NoneAttr)
        for value in values
    ):
        raise InvalidLinalgPattern("BF16 multiply needs default-encoded BF16 tensors")
    shapes = tuple(tuple(value.type.get_shape()) for value in values)
    if (
        len(shapes[0]) != 2
        or any(shape != shapes[0] for shape in shapes)
        or any(type(dimension) is not int or dimension <= 0 for dimension in shapes[0])
    ):
        raise InvalidLinalgPattern("BF16 multiply needs one positive static rank-two shape")
    d0, d1 = AffineDimExpr(0), AffineDimExpr(1)
    identity = AffineMap(2, 0, (d0, d1))
    if (
        len(op.get_indexing_maps()) != 3
        or any(not isinstance(item, AffineMapAttr) or item.data != identity for item in op.get_indexing_maps())
        or not isinstance(op.iterator_types, ArrayAttr)
        or any(not isinstance(item, IteratorTypeAttr) for item in op.iterator_types)
        or tuple(item.data for item in op.iterator_types) != (IteratorType.PARALLEL,) * 2
    ):
        raise InvalidLinalgPattern("BF16 multiply needs parallel identity index maps")
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise InvalidLinalgPattern("BF16 multiply needs one scalar body")
    block = op.regions[0].block
    if len(block.args) != 3 or any(not isinstance(arg.type, BFloat16Type) for arg in block.args):
        raise InvalidLinalgPattern("BF16 multiply scalar arguments differ from its tensor elements")
    body = tuple(block.ops)
    if tuple(type(inner) for inner in body) != (
        ExtFOp,
        ExtFOp,
        MulfOp,
        BitcastOp,
        ConstantOp,
        ConstantOp,
        ConstantOp,
        ShRUIOp,
        AndIOp,
        AddiOp,
        AddiOp,
        ShRUIOp,
        TruncIOp,
        BitcastOp,
        YieldOp,
    ):
        raise InvalidLinalgPattern("BF16 multiply body must use the checked nearest-even bit program")
    (lhs, rhs, product, bits, shift, half, one, high, lsb, bias, biased, rounded, narrow, result, yield_op) = body
    literals = (shift.properties.get("value"), half.properties.get("value"), one.properties.get("value"))
    if (
        any(inner.attributes for inner in body)
        or any(inner.properties for inner in (lhs, rhs, bits, high, lsb, rounded, narrow, result, yield_op))
        or set(product.properties) != {"fastmath"}
        or product.fastmath.data
        or any(
            set(inner.properties) != {"overflowFlags"} or inner.properties["overflowFlags"].data
            for inner in (bias, biased)
        )
        or any(set(inner.properties) != {"value"} for inner in (shift, half, one))
        or any(not isinstance(literal, IntegerAttr) or str(literal.type) != "i32" for literal in literals)
        or tuple(literal.value.data for literal in literals) != (16, 32767, 1)
        or tuple(lhs.operands) != (block.args[0],)
        or tuple(rhs.operands) != (block.args[1],)
        or tuple(product.operands) != (lhs.results[0], rhs.results[0])
        or tuple(bits.operands) != tuple(product.results)
        or tuple(high.operands) != (bits.results[0], shift.results[0])
        or tuple(lsb.operands) != (high.results[0], one.results[0])
        or tuple(bias.operands) != (half.results[0], lsb.results[0])
        or tuple(biased.operands) != (bits.results[0], bias.results[0])
        or tuple(rounded.operands) != (biased.results[0], shift.results[0])
        or tuple(narrow.operands) != tuple(rounded.results)
        or tuple(result.operands) != tuple(narrow.results)
        or tuple(yield_op.operands) != tuple(result.results)
        or any(not isinstance(inner.results[0].type, Float32Type) for inner in (lhs, rhs, product))
        or not isinstance(bits.results[0].type, IntegerType)
        or str(bits.results[0].type) != "i32"
        or any(str(inner.results[0].type) != "i32" for inner in (shift, half, one, high, lsb, bias, biased, rounded))
        or str(narrow.results[0].type) != "i16"
        or not isinstance(result.results[0].type, BFloat16Type)
    ):
        raise InvalidLinalgPattern("BF16 multiply does not yield unflagged FP32 product with nearest-even BF16 bits")
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - parsed untrusted IR must refuse
        raise InvalidLinalgPattern("BF16 multiply Linalg operation does not verify") from exc
    return op.inputs[0], op.inputs[1], op.outputs[0], policy.data, domain.data


# ---- linalg_bf16_relu --------------------------------------------------------------------------
# Recognize one exact BF16 Linalg ReLU source form with an explicit domain.


def recognize_bf16_sign_clamp_relu(op) -> tuple[object, object]:
    """Return input and destination for a parsed elementwise maximum with +0.

    The source declaration supplies a finite-normal-or-zero premise. This
    function proves the Linalg body and indexing geometry, not that runtime
    inputs satisfy the premise or that an accelerator implements the operation.
    """
    from xdsl.dialects.arith import ConstantOp, MaximumfOp
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        BFloat16Type,
        DenseArrayBase,
        FloatAttr,
        NoneAttr,
        StringAttr,
        TensorType,
    )
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.dialects.linalg.ops import GenericOp, YieldOp
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    if type(op) is not GenericOp or len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1:
        raise InvalidLinalgPattern("BF16 ReLU requires one input and one tensor result")
    if op.successors or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
        raise InvalidLinalgPattern("BF16 ReLU has an unknown Linalg property or successor")
    if any(
        not key.startswith("prov.") and key not in {"merlin.numeric_semantics", "merlin.input_domain"}
        for key in op.attributes
    ):
        raise InvalidLinalgPattern("BF16 ReLU has an unknown Linalg attribute")
    policy = op.attributes.get("merlin.numeric_semantics")
    domain = op.attributes.get("merlin.input_domain")
    if (
        not isinstance(policy, StringAttr)
        or policy.data != "bf16-sign-clamp-v1"
        or not isinstance(domain, StringAttr)
        or domain.data != "finite-normal-or-zero-bf16"
    ):
        raise InvalidLinalgPattern("BF16 ReLU requires its exact numerical policy and input domain")
    segments = op.properties["operandSegmentSizes"]
    if (
        not isinstance(segments, DenseArrayBase)
        or str(segments.elt_type) != "i32"
        or (tuple(segments.iter_values()) != (1, 1))
    ):
        raise InvalidLinalgPattern("BF16 ReLU has an inconsistent operand segment")
    values = (*op.inputs, *op.outputs, *op.results)
    if any(
        not isinstance(value.type, TensorType)
        or not isinstance(value.type.element_type, BFloat16Type)
        or not isinstance(value.type.encoding, NoneAttr)
        for value in values
    ):
        raise InvalidLinalgPattern("BF16 ReLU requires default-encoded BF16 tensors")
    shapes = tuple(tuple(value.type.get_shape()) for value in values)
    if (
        len(shapes[0]) != 2
        or any(shape != shapes[0] for shape in shapes)
        or (any(type(dim) is not int or dim <= 0 for dim in shapes[0]))
    ):
        raise InvalidLinalgPattern("BF16 ReLU requires one positive static rank-two shape")
    maps = op.get_indexing_maps()
    d0, d1 = AffineDimExpr(0), AffineDimExpr(1)
    identity = AffineMap(2, 0, (d0, d1))
    if (
        len(maps) != 2
        or any(not isinstance(item, AffineMapAttr) or item.data != identity for item in maps)
        or not isinstance(op.iterator_types, ArrayAttr)
        or any(not isinstance(item, IteratorTypeAttr) for item in op.iterator_types)
        or tuple(item.data for item in op.iterator_types) != (IteratorType.PARALLEL,) * 2
    ):
        raise InvalidLinalgPattern("BF16 ReLU requires parallel identity index maps")
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise InvalidLinalgPattern("BF16 ReLU requires one scalar body")
    block = op.regions[0].block
    if len(block.args) != 2 or any(not isinstance(arg.type, BFloat16Type) for arg in block.args):
        raise InvalidLinalgPattern("BF16 ReLU scalar arguments differ from BF16 tensor elements")
    body = tuple(block.ops)
    if len(body) != 3 or tuple(type(inner) for inner in body) != (ConstantOp, MaximumfOp, YieldOp):
        raise InvalidLinalgPattern("BF16 ReLU body must be constant, maximumf, yield")
    zero, maximum, yield_op = body
    literal = zero.properties.get("value")
    if (
        zero.attributes
        or set(zero.properties) != {"value"}
        or not isinstance(literal, FloatAttr)
        or not isinstance(literal.type, BFloat16Type)
        or literal.value.data != 0.0
        or copysign(1.0, literal.value.data) != 1.0
        or tuple(zero.results) != (maximum.operands[1],)
        or maximum.operands[0] is not block.args[0]
        or maximum.attributes
        or set(maximum.properties) != {"fastmath"}
        or maximum.fastmath.data
        or yield_op.attributes
        or yield_op.properties
        or tuple(yield_op.operands) != tuple(maximum.results)
    ):
        raise InvalidLinalgPattern("BF16 ReLU does not yield exact maximumf(input, +0)")
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - parsed untrusted IR must refuse
        raise InvalidLinalgPattern("BF16 ReLU Linalg operation does not verify") from exc
    return op.inputs[0], op.outputs[0]


# ---- linalg_fp8_cast ---------------------------------------------------------------------------
# Recognize a declared BF16-to-E4M3 truncation in parsed Linalg IR.


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

    from merlin.compiler.dialects.xdsl_context import Float8E4M3FNType

    if type(op) is not GenericOp or len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1:
        raise InvalidLinalgPattern("BF16-to-FP8 truncation needs one input and one result")
    if op.successors or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
        raise InvalidLinalgPattern("BF16-to-FP8 truncation has an unknown property or successor")
    if any(
        not key.startswith("prov.")
        and key not in {"merlin.numeric_semantics", "merlin.input_domain", "merlin.scale_code"}
        for key in op.attributes
    ):
        raise InvalidLinalgPattern("BF16-to-FP8 truncation has an unknown attribute")
    policy = op.attributes.get("merlin.numeric_semantics")
    domain = op.attributes.get("merlin.input_domain")
    scale = op.attributes.get("merlin.scale_code")
    if (
        not isinstance(policy, StringAttr)
        or not policy.data
        or not isinstance(domain, StringAttr)
        or not domain.data
        or not isinstance(scale, IntegerAttr)
        or str(scale.type) != "i32"
        or not 0 <= scale.value.data <= 255
    ):
        raise InvalidLinalgPattern("BF16-to-FP8 truncation needs policy, domain, and scale code")
    segments = op.properties["operandSegmentSizes"]
    if (
        not isinstance(segments, DenseArrayBase)
        or str(segments.elt_type) != "i32"
        or tuple(segments.iter_values()) != (1, 1)
    ):
        raise InvalidLinalgPattern("BF16-to-FP8 truncation has inconsistent operand segments")
    source, destination, result = (*op.inputs, *op.outputs, *op.results)
    if any(
        not isinstance(value.type, TensorType) or not isinstance(value.type.encoding, NoneAttr)
        for value in (source, destination, result)
    ):
        raise InvalidLinalgPattern("BF16-to-FP8 truncation needs default-encoded tensors")
    if not isinstance(source.type.element_type, BFloat16Type) or any(
        not isinstance(value.type.element_type, Float8E4M3FNType) for value in (destination, result)
    ):
        raise InvalidLinalgPattern("BF16-to-FP8 truncation has wrong element types")
    shapes = tuple(tuple(value.type.get_shape()) for value in (source, destination, result))
    if (
        len(shapes[0]) != 2
        or any(shape != shapes[0] for shape in shapes)
        or (any(type(dimension) is not int or dimension <= 0 for dimension in shapes[0]))
    ):
        raise InvalidLinalgPattern("BF16-to-FP8 truncation needs one positive static rank-two shape")
    d0, d1 = AffineDimExpr(0), AffineDimExpr(1)
    identity = AffineMap(2, 0, (d0, d1))
    if (
        len(op.get_indexing_maps()) != 2
        or any(
            not isinstance(index_map, AffineMapAttr) or index_map.data != identity
            for index_map in op.get_indexing_maps()
        )
        or not isinstance(op.iterator_types, ArrayAttr)
        or any(not isinstance(iterator, IteratorTypeAttr) for iterator in op.iterator_types)
        or tuple(iterator.data for iterator in op.iterator_types) != (IteratorType.PARALLEL,) * 2
    ):
        raise InvalidLinalgPattern("BF16-to-FP8 truncation needs parallel identity maps")
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise InvalidLinalgPattern("BF16-to-FP8 truncation needs one scalar body")
    block = op.regions[0].block
    if (
        len(block.args) != 2
        or not isinstance(block.args[0].type, BFloat16Type)
        or not isinstance(block.args[1].type, Float8E4M3FNType)
    ):
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
            constant.attributes
            or set(constant.properties) != {"value"}
            or not isinstance(factor, FloatAttr)
            or not isinstance(factor.type, BFloat16Type)
            or factor.value.data != expected
            or multiply.attributes
            or set(multiply.properties) != {"fastmath"}
            or multiply.fastmath.data
            or tuple(multiply.operands) != (block.args[0], constant.results[0])
            or not isinstance(multiply.results[0].type, BFloat16Type)
        ):
            raise InvalidLinalgPattern("scaled BF16-to-FP8 body has a wrong inverse scale or multiply")
        cast_input = multiply.results[0]
    if (
        cast.attributes
        or cast.properties
        or tuple(cast.operands) != (cast_input,)
        or not isinstance(cast.results[0].type, Float8E4M3FNType)
        or yield_op.attributes
        or yield_op.properties
        or tuple(yield_op.operands) != tuple(cast.results)
    ):
        raise InvalidLinalgPattern("BF16-to-FP8 body does not yield the input cast")
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - parsed untrusted IR must refuse
        raise InvalidLinalgPattern("BF16-to-FP8 Linalg operation does not verify") from exc
    return source, destination, policy.data, domain.data, scale.value.data


# ---- linalg_ordered_fp8 ------------------------------------------------------------------------
# Recognize an explicitly annotated FP8/BF16 ordered reduction region.


def recognize_ordered_fp8_bf16_reset_matmul(op) -> tuple[object, object, object, str, str]:
    """Check a closed FMA body and return A, W, init, policy and domain.

    The K order and input domain are declarations made by the source producer.
    This recognizer checks that the body and maps have the declared form; it
    cannot prove runtime input values satisfy the domain or that target
    arithmetic implements the named policy. The caller validates that init is
    either a positive-zero fill or an ordered accumulator with matching policy.
    """
    from xdsl.dialects.arith import ExtFOp
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        BFloat16Type,
        DenseArrayBase,
        NoneAttr,
        StringAttr,
        TensorType,
    )
    from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
    from xdsl.dialects.linalg.ops import GenericOp, YieldOp
    from xdsl.dialects.math import FmaOp
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    from merlin.compiler.dialects.xdsl_context import Float8E4M3FNType

    if type(op) is not GenericOp or len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
        raise InvalidLinalgPattern("ordered FP8 matmul needs two inputs and one result")
    if op.successors or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
        raise InvalidLinalgPattern("ordered FP8 matmul has unknown Linalg properties or successors")
    if any(
        not key.startswith("prov.")
        and key
        not in {
            "merlin.numeric_semantics",
            "merlin.input_domain",
            "merlin.reduction_order",
        }
        for key in op.attributes
    ):
        raise InvalidLinalgPattern("ordered FP8 matmul has unknown Linalg attributes")
    semantics = op.attributes.get("merlin.numeric_semantics")
    domain = op.attributes.get("merlin.input_domain")
    order = op.attributes.get("merlin.reduction_order")
    if (
        not isinstance(semantics, StringAttr)
        or not semantics.data
        or not isinstance(domain, StringAttr)
        or not domain.data
        or not isinstance(order, StringAttr)
        or order.data != "k-ascending"
    ):
        raise InvalidLinalgPattern("ordered FP8 matmul requires semantics, domain, and K order")
    segments = op.properties["operandSegmentSizes"]
    if (
        not isinstance(segments, DenseArrayBase)
        or str(segments.elt_type) != "i32"
        or (tuple(segments.iter_values()) != (2, 1))
    ):
        raise InvalidLinalgPattern("ordered FP8 matmul has inconsistent operand segments")
    a, w, init = (*op.inputs, *op.outputs)
    result = op.results[0]
    if any(
        not isinstance(value.type, TensorType) or not isinstance(value.type.encoding, NoneAttr)
        for value in (a, w, init, result)
    ) or (
        not isinstance(a.type.element_type, Float8E4M3FNType)
        or not isinstance(w.type.element_type, Float8E4M3FNType)
        or not isinstance(init.type.element_type, BFloat16Type)
        or not isinstance(result.type.element_type, BFloat16Type)
    ):
        raise InvalidLinalgPattern("ordered FP8 matmul needs default-encoded FP8 inputs and BF16 result")
    a_shape, w_shape, init_shape, result_shape = (tuple(value.type.get_shape()) for value in (a, w, init, result))
    if (
        any(
            len(shape) != 2 or any(type(dim) is not int or dim <= 0 for dim in shape)
            for shape in (a_shape, w_shape, init_shape, result_shape)
        )
        or a_shape[1] != w_shape[1]
        or init_shape != result_shape
        or result_shape != (a_shape[0], w_shape[0])
    ):
        raise InvalidLinalgPattern("ordered FP8 matmul has incompatible static A[M,K], W[N,K], C[M,N]")
    d0, d1, d2 = (AffineDimExpr(axis) for axis in range(3))
    expected_maps = (
        AffineMap(3, 0, (d0, d2)),
        AffineMap(3, 0, (d1, d2)),
        AffineMap(3, 0, (d0, d1)),
    )
    maps = op.get_indexing_maps()
    if (
        len(maps) != 3
        or any(
            not isinstance(item, AffineMapAttr) or item.data != expected for item, expected in zip(maps, expected_maps)
        )
        or not isinstance(op.iterator_types, ArrayAttr)
        or any(not isinstance(item, IteratorTypeAttr) for item in op.iterator_types)
        or tuple(item.data for item in op.iterator_types)
        != (IteratorType.PARALLEL, IteratorType.PARALLEL, IteratorType.REDUCTION)
    ):
        raise InvalidLinalgPattern("ordered FP8 matmul has wrong maps or iterator order")
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise InvalidLinalgPattern("ordered FP8 matmul needs one scalar body")
    block = op.regions[0].block
    if len(block.args) != 3 or tuple(type(arg.type) for arg in block.args) != (
        Float8E4M3FNType,
        Float8E4M3FNType,
        BFloat16Type,
    ):
        raise InvalidLinalgPattern("ordered FP8 matmul scalar arguments have wrong types")
    body = tuple(block.ops)
    if len(body) != 4 or tuple(type(inner) for inner in body) != (ExtFOp, ExtFOp, FmaOp, YieldOp):
        raise InvalidLinalgPattern("ordered FP8 matmul needs extf, extf, fma, yield")
    ext_a, ext_w, fma, yield_op = body
    if (
        any(inner.attributes or inner.properties for inner in (ext_a, ext_w, yield_op))
        or tuple(ext_a.operands) != (block.args[0],)
        or tuple(ext_w.operands) != (block.args[1],)
        or not isinstance(ext_a.results[0].type, BFloat16Type)
        or not isinstance(ext_w.results[0].type, BFloat16Type)
        or fma.attributes
        or set(fma.properties) != {"fastmath"}
        or fma.fastmath.data
        or tuple(fma.operands) != (ext_a.results[0], ext_w.results[0], block.args[2])
        or tuple(yield_op.operands) != tuple(fma.results)
    ):
        raise InvalidLinalgPattern("ordered FP8 matmul body changes the BF16 FMA or accumulator")
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - parsed untrusted IR must refuse
        raise InvalidLinalgPattern("ordered FP8 matmul Linalg operation does not verify") from exc
    return a, w, init, semantics.data, domain.data


def is_positive_bf16_zero(op) -> bool:
    """Recognize a scalar +0 BF16 constant without treating tensor.empty as zero."""
    from xdsl.dialects.arith import ConstantOp
    from xdsl.dialects.builtin import BFloat16Type, FloatAttr

    if type(op) is not ConstantOp or len(op.results) != 1 or op.attributes or set(op.properties) != {"value"}:
        return False
    value = op.properties["value"]
    return (
        isinstance(value, FloatAttr)
        and isinstance(value.type, BFloat16Type)
        and value.value.data == 0.0
        and copysign(1.0, value.value.data) == 1.0
    )
