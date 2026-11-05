"""Recognize an explicitly annotated FP8/BF16 ordered reduction region."""

from __future__ import annotations

from math import copysign

from .linalg_patterns import InvalidLinalgPattern


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

    from merlin.xdsl_dialects.fp8 import Float8E4M3FNType

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
