"""Recognize an explicit nearest-even BF16 product in a Linalg scalar body."""

from __future__ import annotations

from .linalg_patterns import InvalidLinalgPattern


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
        ExtFOp, ExtFOp, MulfOp, BitcastOp, ConstantOp, ConstantOp, ConstantOp,
        ShRUIOp, AndIOp, AddiOp, AddiOp, ShRUIOp, TruncIOp, BitcastOp, YieldOp,
    ):
        raise InvalidLinalgPattern("BF16 multiply body must use the checked nearest-even bit program")
    (lhs, rhs, product, bits, shift, half, one, high, lsb,
     bias, biased, rounded, narrow, result, yield_op) = body
    literals = (shift.properties.get("value"), half.properties.get("value"), one.properties.get("value"))
    if (
        any(inner.attributes for inner in body)
        or any(inner.properties for inner in (lhs, rhs, bits, high, lsb, rounded, narrow, result, yield_op))
        or set(product.properties) != {"fastmath"} or product.fastmath.data
        or any(set(inner.properties) != {"overflowFlags"} or inner.properties["overflowFlags"].data
               for inner in (bias, biased))
        or any(set(inner.properties) != {"value"} for inner in (shift, half, one))
        or any(not isinstance(literal, IntegerAttr) or str(literal.type) != "i32"
               for literal in literals)
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
        or any(str(inner.results[0].type) != "i32" for inner in (shift, half, one, high, lsb,
                                                                  bias, biased, rounded))
        or str(narrow.results[0].type) != "i16"
        or not isinstance(result.results[0].type, BFloat16Type)
    ):
        raise InvalidLinalgPattern("BF16 multiply does not yield unflagged FP32 product with nearest-even BF16 bits")
    try:
        op.verify()
    except Exception as exc:  # noqa: BLE001 - parsed untrusted IR must refuse
        raise InvalidLinalgPattern("BF16 multiply Linalg operation does not verify") from exc
    return op.inputs[0], op.inputs[1], op.outputs[0], policy.data, domain.data
