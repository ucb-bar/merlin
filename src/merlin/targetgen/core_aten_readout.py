"""Static layout proof for a flat host readout of a batched integer MAC."""

from __future__ import annotations

import math


def planned_readout_geometry(contraction):
    """Derive a row-major [B*M,N] view without changing any MAC axis."""
    from xdsl.ir.affine import AffineExpr, AffineMap

    if len(contraction.operands) != 3 or len(contraction.results) != 1:
        return None
    shapes = [tuple(value.type.get_shape()) for value in (*contraction.operands, contraction.results[0])]
    lhs, rhs, initial, result = shapes
    if any(len(shape) != 3 or any(extent <= 0 for extent in shape) for shape in shapes):
        return None
    batch, m, k = lhs
    if rhs != (batch, k, result[-1]) or initial != result or result != (batch, m, rhs[-1]):
        return None
    d = AffineExpr.dimension
    expected = [
        AffineMap(4, 0, (d(0), d(1), d(3))),
        AffineMap(4, 0, (d(0), d(3), d(2))),
        AffineMap(4, 0, (d(0), d(1), d(2))),
    ]
    if [amap.data for amap in contraction.get_indexing_maps()] != expected:
        return None
    return {
        "source_shape": list(result),
        "flat_shape": [batch * m, result[-1]],
        "element_order": "row_major",
        "mac_batch_axes_preserved": True,
    }


def _unique_user(value):
    uses = list(value.uses)
    if len(uses) != 1:
        raise ValueError("batched readout requires a unique direct view/arithmetic chain")
    return uses[0].operation


def _reshape(value, shape, views):
    """Require the existing canonical full-collapse/static-expand representation."""
    from xdsl.dialects.builtin import TensorType
    from xdsl.dialects.tensor import CollapseShapeOp, ExpandShapeOp
    from xdsl.utils.exceptions import VerifyException

    source = tuple(value.type.get_shape())
    total = math.prod(source)
    if math.prod(shape) != total or any(extent <= 0 for extent in (*source, *shape)):
        raise ValueError("batched readout changes static element count")
    for cls, expected, rank in ((CollapseShapeOp, (total,), len(source)), (ExpandShapeOp, tuple(shape), len(shape))):
        op = _unique_user(value)
        if (
            not isinstance(op, cls)
            or tuple(op.operands) != (value,)
            or len(op.results) != 1
            or op.results[0].type != TensorType(value.type.element_type, expected)
        ):
            raise ValueError("batched readout needs an exact static collapse/expand view")
        groups = [[item.value.data for item in group] for group in op.properties["reassociation"]]
        if groups != [list(range(rank))]:
            raise ValueError("batched readout reassociation is not row-major flattening")
        try:
            op.verify()
        except VerifyException as exc:
            raise ValueError("batched readout contains an invalid static view") from exc
        views.append(op)
        value = op.results[0]
    return value


def _reader(value, operation_type, arity, *, strict=False):
    from xdsl.dialects import arith
    from xdsl.dialects.linalg.ops import GenericOp, YieldOp
    from xdsl.ir.affine import AffineExpr, AffineMap

    candidates = []
    for use in value.uses:
        op = use.operation
        if (
            not isinstance(op, GenericOp)
            or len(op.results) != 1
            or len(op.inputs) != arity
            or len(op.outputs) != 1
            or op.inputs[0] is not value
        ):
            continue
        block = op.body.block
        if len(block.args) != arity + 1 or len(block.ops) != 2:
            continue
        arithmetic, yielded = list(block.ops)
        if (
            not isinstance(arithmetic, operation_type)
            or tuple(arithmetic.operands) != tuple(block.args[:arity])
            or not isinstance(yielded, YieldOp)
            or tuple(yielded.operands) != (arithmetic.results[0],)
        ):
            continue
        if isinstance(arithmetic, arith.MulfOp) and arithmetic.fastmath.data:
            continue
        if strict:
            shape = value.type.get_shape()
            identity = AffineMap.identity(len(shape))
            maps = [amap.data for amap in op.get_indexing_maps()]
            if (
                op.results[0].type.get_shape() != shape
                or maps[0] != identity
                or maps[-1] != identity
                or any(iterator.data.value != "parallel" for iterator in op.iterator_types)
            ):
                continue
            if arity == 2:
                scale_shape = op.inputs[1].type.get_shape()
                expected = (
                    identity
                    if scale_shape == shape
                    else AffineMap(len(shape), 0, ())
                    if not scale_shape
                    else AffineMap(len(shape), 0, (AffineExpr.dimension(len(shape) - 1),))
                )
                if (
                    maps[1] != expected
                    or (scale_shape and scale_shape not in (shape, (shape[-1],)))
                    or any(str(v.type.element_type) != "f32" for v in op.operands)
                ):
                    continue
        candidates.append(op)
    if len(candidates) != 1 or (strict and _unique_user(value) is not candidates[0]):
        raise ValueError("integer readout requires a unique closed direct cast and two f32 multiplies")
    return candidates[0]


def host_readout_bridge(contraction):
    """Return arithmetic, restored output and a proof of the actual captured views."""
    from xdsl.dialects import arith

    geometry = planned_readout_geometry(contraction)
    rank = len(contraction.results[0].type.get_shape())
    views = []
    value = contraction.results[0]
    if rank == 3:
        if geometry is None:
            raise ValueError("batched readout has unsupported static MAC geometry")
        value = _reshape(value, geometry["flat_shape"], views)
    cast = _reader(value, arith.SIToFPOp, 1, strict=rank == 3)
    if str(value.type.element_type) != "i32" or str(cast.results[0].type.element_type) != "f32":
        raise ValueError("integer readout requires signed i32 to f32 conversion")
    first = _reader(cast.results[0], arith.MulfOp, 2, strict=rank == 3)
    second = _reader(first.results[0], arith.MulfOp, 2, strict=rank == 3)
    restored = second.results[0]
    if rank == 3:
        restored = _reshape(restored, geometry["source_shape"], views)
    proof = {
        "status": "verified_captured_ssa",
        "cast_rank": len(cast.results[0].type.get_shape()),
        "views": [op.name for op in views],
        **(geometry or {}),
    }
    return (cast, first, second), restored, proof
