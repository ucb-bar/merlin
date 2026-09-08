"""Structural reader for canonical direct 2-D integer convolution.

The frontend keeps convolution as a compound-affine ``linalg.generic``.  This module recognizes
that semantic form without using model names or fixed ResNet shapes.  It also folds the canonical
zero-padding ``tensor.insert_slice`` view into geometry, allowing the target scheduler to generate
padding while packing a streamed im2col row instead of materialising a padded activation.
"""
from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import IntegerType, TensorType
from xdsl.ir import Operation, SSAValue
from xdsl.ir.affine import (AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr,
                            AffineDimExpr)


@dataclass(frozen=True)
class DirectConv:
    activation: SSAValue
    weight: SSAValue
    output: SSAValue
    batch: int
    ci: int
    hi: int
    wi: int
    co: int
    kh: int
    kw: int
    ho: int
    wo: int
    stride_h: int
    stride_w: int
    dilation_h: int
    dilation_w: int
    pad_top: int
    pad_left: int
    pad_bottom: int
    pad_right: int

    @property
    def k(self) -> int:
        return self.ci * self.kh * self.kw

    @property
    def direct_dma(self) -> bool:
        """The activation already is the KxW row matrix; no packing is required."""
        return (self.kh, self.kw, self.stride_h, self.stride_w,
                self.dilation_h, self.dilation_w,
                self.pad_top, self.pad_left, self.pad_bottom, self.pad_right) == (
                    1, 1, 1, 1, 1, 1, 0, 0, 0, 0)


def _shape(value: SSAValue) -> tuple[int, ...] | None:
    ty = value.type
    if not isinstance(ty, TensorType):
        return None
    return tuple(int(d) for d in ty.get_shape())


def _integer_width(value: SSAValue) -> int | None:
    ty = value.type
    if not isinstance(ty, TensorType) or not isinstance(ty.element_type, IntegerType):
        return None
    return int(ty.element_type.width.data)


def _linear(expr) -> tuple[dict[int, int], int] | None:
    """Return integer coefficients + constant for a linear affine expression."""
    if isinstance(expr, AffineDimExpr):
        return {expr.position: 1}, 0
    if isinstance(expr, AffineConstantExpr):
        return {}, int(expr.value)
    if not isinstance(expr, AffineBinaryOpExpr):
        return None
    lhs, rhs = _linear(expr.lhs), _linear(expr.rhs)
    if lhs is None or rhs is None:
        return None
    if expr.kind == AffineBinaryOpKind.Add:
        out = dict(lhs[0])
        for dim, coefficient in rhs[0].items():
            out[dim] = out.get(dim, 0) + coefficient
        return {d: c for d, c in out.items() if c}, lhs[1] + rhs[1]
    if expr.kind == AffineBinaryOpKind.Mul:
        # Affine multiplication is legal only when one side is constant.
        if lhs[0] and rhs[0]:
            return None
        if lhs[0]:
            scale, variable = rhs[1], lhs
        elif rhs[0]:
            scale, variable = lhs[1], rhs
        else:
            return {}, lhs[1] * rhs[1]
        return ({d: c * scale for d, c in variable[0].items()}, variable[1] * scale)
    return None


def _static(op: Operation, key: str) -> list[int] | None:
    attr = op.properties.get(key)
    if attr is None:
        attr = op.attributes.get(key)
    return None if attr is None else [int(v) for v in attr.get_values()]


def _constant_zero(value: SSAValue) -> bool:
    owner = value.owner if isinstance(value.owner, Operation) else None
    if owner is None or owner.name != "arith.constant":
        return False
    attr = owner.properties.get("value")
    if attr is None:
        attr = owner.attributes.get("value")
    raw = getattr(getattr(attr, "value", None), "data", None)
    return raw is not None and float(raw) == 0.0


def _unwrap_zero_padding(value: SSAValue) -> tuple[SSAValue, tuple[int, int, int, int]] | None:
    owner = value.owner if isinstance(value.owner, Operation) else None
    if owner is None or owner.name != "tensor.insert_slice":
        return value, (0, 0, 0, 0)
    offsets = _static(owner, "static_offsets")
    sizes = _static(owner, "static_sizes")
    strides = _static(owner, "static_strides")
    source, dest = owner.operands[:2]
    src_shape, dst_shape = _shape(source), _shape(dest)
    dest_owner = dest.owner if isinstance(dest.owner, Operation) else None
    if (src_shape is None or dst_shape is None or len(src_shape) != 4 or len(dst_shape) != 4
            or offsets is None or sizes != list(src_shape) or strides != [1, 1, 1, 1]
            or offsets[:2] != [0, 0] or src_shape[:2] != dst_shape[:2]
            or dest_owner is None or dest_owner.name != "tensor.splat"
            or not _constant_zero(dest_owner.operands[0])):
        return None
    pt, pl = offsets[2], offsets[3]
    pb, pr = dst_shape[2] - src_shape[2] - pt, dst_shape[3] - src_shape[3] - pl
    if min(pt, pl, pb, pr) < 0:
        return None
    return source, (pt, pl, pb, pr)


def recognize(op: Operation) -> DirectConv | None:
    """Return canonical direct-convolution geometry, or ``None`` without guessing."""
    if op.name != "linalg.generic" or len(op.operands) != 3 or len(op.results) != 1:
        return None
    if [_integer_width(v) for v in op.operands[:2]] != [8, 8] or _integer_width(op.results[0]) != 32:
        return None
    maps_attr = op.properties.get("indexing_maps")
    iters_attr = op.properties.get("iterator_types")
    if maps_attr is None or iters_attr is None:
        return None
    maps = [item.data for item in maps_attr.data]
    iters = [str(item) for item in iters_attr.data]
    if (len(maps) != 3 or any(m.num_dims != 7 or m.num_symbols for m in maps)
            or [len(m.results) for m in maps] != [4, 4, 4]
            or len(iters) != 7
            or any("parallel" not in item for item in iters[:4])
            or any("reduction" not in item for item in iters[4:])):
        return None
    linear = [[_linear(expr) for expr in amap.results] for amap in maps]
    if any(item is None for row in linear for item in row):
        return None
    if any(item[1] != 0 for row in linear for item in row):
        return None
    # n/co/oh/ow, co/ci/kh/kw and the two simple activation axes are canonical.
    dims = lambda row: [item[0] for item in row]
    if dims(linear[2]) != [{0: 1}, {1: 1}, {2: 1}, {3: 1}]:
        return None
    if dims(linear[1]) != [{1: 1}, {4: 1}, {5: 1}, {6: 1}]:
        return None
    if dims(linear[0][:2]) != [{0: 1}, {4: 1}]:
        return None
    y, x = linear[0][2], linear[0][3]
    if y[1] != 0 or x[1] != 0 or set(y[0]) != {2, 5} or set(x[0]) != {3, 6}:
        return None
    sh, dh, sw, dw = y[0][2], y[0][5], x[0][3], x[0][6]
    if min(sh, sw, dh, dw) <= 0:
        return None
    padded = _unwrap_zero_padding(op.operands[0])
    if padded is None:
        return None
    activation, padding = padded
    act_shape, weight_shape, out_shape = _shape(activation), _shape(op.operands[1]), _shape(op.results[0])
    padded_shape = _shape(op.operands[0])
    if any(shape is None or len(shape) != 4 or any(n <= 0 for n in shape)
           for shape in (act_shape, weight_shape, out_shape, padded_shape)):
        return None
    if _shape(op.operands[2]) != out_shape or _integer_width(op.operands[2]) != 32:
        return None
    n, ci, hi, wi = act_shape
    co, wci, kh, kw = weight_shape
    on, oco, ho, wo = out_shape
    if (n, ci, co) != (on, wci, oco):
        return None
    expect_h = (padded_shape[2] - dh * (kh - 1) - 1) // sh + 1
    expect_w = (padded_shape[3] - dw * (kw - 1) - 1) // sw + 1
    if (ho, wo) != (expect_h, expect_w):
        return None
    body_names = [item.name for item in op.regions[0].blocks[0].ops]
    if body_names != ["arith.extsi", "arith.extsi", "arith.muli", "arith.addi", "linalg.yield"]:
        return None
    body = op.regions[0].blocks[0]
    if len(body.args) != 3 or [str(v.type) for v in body.args] != ["i8", "i8", "i32"]:
        return None
    a, b, product, total, yielded = list(body.ops)
    if (tuple(a.operands) != (body.args[0],) or tuple(b.operands) != (body.args[1],)
            or tuple(product.operands) != (a.results[0], b.results[0])
            or tuple(total.operands) != (product.results[0], body.args[2])
            or tuple(yielded.operands) != (total.results[0],)
            or any(len(node.results) != 1 or str(node.results[0].type) != "i32"
                   for node in (a, b, product, total))):
        return None
    return DirectConv(activation, op.operands[1], op.results[0], n, ci, hi, wi, co, kh, kw,
                      ho, wo, sh, sw, dh, dw, *padding)
