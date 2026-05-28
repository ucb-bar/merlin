"""Bindings recognizer for NHWC int8 quantized Conv2D — Transpose-free.

Mirrors `nchw_int8_conv.py`'s DAG walk but anchored on the NHWC variant
(`linalg.conv_2d_nhwc_hwcf_q`) so the lowered QNN graph is pure NHWC:
no `Transpose` adapter, weight HWIO is already in QNN's native layout,
no byte permutation needed.

This is the **HTA-compatible** path: HTA on QAIRT 2.45 doesn't have
Transpose at all; Adreno GPU rejects our specific Transpose
declaration. By aligning the input IR to NHWC upstream (re-export the
ONNX in NHWC, run an IREE preprocessing pass, or layout-convert the
graph offline), the partitioner emits NHWC slices that lower cleanly
on both backends.
"""

from __future__ import annotations

import math
from typing import Any

from .base import (
    dense_to_bytes,
    elem_dtype_of,
    find_func,
    find_named_op,
    find_named_ops,
    parse_dense_2d_attr,
    shape_of,
    walk_inner_ops,
)

NAME = "nhwc_int8_conv_dequant"


def _scalar_float_constant(op: Any) -> float | None:
    if op.operation.name != "arith.constant":
        return None
    if "value" not in op.attributes:
        return None
    from iree.compiler import ir

    try:
        return ir.FloatAttr(op.attributes["value"]).value
    except (ValueError, TypeError):
        return None


def _splat_int_constant(op: Any) -> int | None:
    if op.operation.name != "arith.constant":
        return None
    if "value" not in op.attributes:
        return None
    from iree.compiler import ir

    try:
        return ir.IntegerAttr(op.attributes["value"]).value
    except (ValueError, TypeError):
        return None


def _defining_op(value: Any) -> Any | None:
    owner = value.owner
    return owner if hasattr(owner, "operation") else None


def _bias_quantize_scale(generic_op: Any) -> float | None:
    for region in generic_op.operation.regions:
        for block in region.blocks:
            for inner in block.operations:
                if inner.name != "arith.divf":
                    continue
                if len(inner.operands) < 2:
                    continue
                src = _defining_op(inner.operands[1])
                if src is None:
                    continue
                v = _scalar_float_constant(src)
                if v is not None:
                    return v
    return None


def _dequant_scale(generic_op: Any) -> float | None:
    for region in generic_op.operation.regions:
        for block in region.blocks:
            for inner in block.operations:
                if inner.name != "arith.mulf":
                    continue
                if len(inner.operands) < 2:
                    continue
                src = _defining_op(inner.operands[1])
                if src is None:
                    continue
                v = _scalar_float_constant(src)
                if v is not None:
                    return v
    return None


def _pad_amounts_nhwc(pad_op: Any) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """tensor.pad static_low / static_high in NHWC layout (dims 1, 2 are H, W)."""
    if pad_op.operation.name != "tensor.pad":
        return None
    if "static_low" not in pad_op.attributes or "static_high" not in pad_op.attributes:
        return None
    lo = pad_op.attributes["static_low"]
    hi = pad_op.attributes["static_high"]
    try:
        lo_list = [int(lo[i]) for i in range(len(lo))]
        hi_list = [int(hi[i]) for i in range(len(hi))]
    except (TypeError, ValueError):
        return None
    if len(lo_list) != 4 or len(hi_list) != 4:
        return None
    # NHWC: dim 0 = N, dim 1 = H, dim 2 = W, dim 3 = C
    return ((lo_list[1], hi_list[1]), (lo_list[2], hi_list[2]))


def try_recognize(module: Any, *, fp_dtype: str = "float32", **_: object) -> Any | None:
    func = find_func(module)
    if func is None:
        return None
    conv_ops = find_named_ops(func, "linalg.conv_2d_nhwc_hwcf_q")
    if not conv_ops:
        return None
    conv = conv_ops[0]

    # ins(padded, weight, in_zp, w_zp); outs(broadcasted_bias)
    if len(conv.operands) < 5:
        return None
    padded_v = conv.operands[0]
    weight_v = conv.operands[1]
    in_zp_v = conv.operands[2]
    w_zp_v = conv.operands[3]
    bcast_v = conv.operands[4]

    strides = parse_dense_2d_attr(conv, "strides")
    dilation = parse_dense_2d_attr(conv, "dilations")
    if strides is None or dilation is None:
        return None

    pad_op = _defining_op(padded_v)
    if pad_op is None or pad_op.operation.name != "tensor.pad":
        return None
    pad_amounts = _pad_amounts_nhwc(pad_op)
    if pad_amounts is None:
        return None
    pad_low_hw = (pad_amounts[0][0], pad_amounts[1][0])
    pad_high_hw = (pad_amounts[0][1], pad_amounts[1][1])

    pre_pad_input_v = pad_op.operands[0]
    in_shape = shape_of(pre_pad_input_v)
    if len(in_shape) != 4 or in_shape[0] != 1:
        return None
    if elem_dtype_of(pre_pad_input_v) != "i8":
        return None

    # Weight: i8 HWIO 4D
    weight_op = _defining_op(weight_v)
    if weight_op is None or weight_op.operation.name != "arith.constant":
        return None
    if elem_dtype_of(weight_op.results[0]) != "i8":
        return None
    weight_shape = shape_of(weight_op.results[0])  # (Kh, Kw, Ic, Oc)
    if len(weight_shape) != 4:
        return None
    weight_bytes_i8 = dense_to_bytes(weight_op, "i8")
    if weight_bytes_i8 is None:
        return None
    # int8 → uint8 storage remap (zp +128). HTA on QAIRT 2.45 and Adreno
    # GPU only accept QNN_DATATYPE_UFIXED_POINT_8 for Conv2d; SFIXED_POINT_8
    # is HTP-only. Mathematically identical: q_u = q_i + 128, zp_u = zp_i + 128.
    weight_bytes_hwio = bytes((b + 128) & 0xFF for b in weight_bytes_i8)

    in_zp_op = _defining_op(in_zp_v)
    w_zp_op = _defining_op(w_zp_v)
    in_zp = _splat_int_constant(in_zp_op) if in_zp_op else None
    w_zp = _splat_int_constant(w_zp_op) if w_zp_op else None
    if in_zp is None or w_zp is None:
        return None

    bcast_op = _defining_op(bcast_v)
    if bcast_op is None or bcast_op.operation.name != "linalg.broadcast":
        return None
    bias_i32_v = bcast_op.operands[0]
    bias_q_op = _defining_op(bias_i32_v)
    if bias_q_op is None or bias_q_op.operation.name != "linalg.generic":
        return None
    bias_scale = _bias_quantize_scale(bias_q_op)
    if bias_scale is None:
        return None
    bias_const_op = _defining_op(bias_q_op.operands[0])
    if bias_const_op is None or bias_const_op.operation.name != "arith.constant":
        return None
    if elem_dtype_of(bias_const_op.results[0]) != "f32":
        return None
    bias_bytes_f32 = dense_to_bytes(bias_const_op, "f32")
    if bias_bytes_f32 is None:
        return None

    # Dequant generic (i32 → f32)
    conv_result = conv.results[0]
    dequant_op: Any | None = None
    for op in walk_inner_ops(func):
        if op.operation.name != "linalg.generic":
            continue
        if any(o.get_name() == conv_result.get_name() for o in op.operands):
            dequant_op = op
            break
    if dequant_op is None:
        return None
    output_scale = _dequant_scale(dequant_op)
    if output_scale is None:
        return None

    return_op = find_named_op(func, "func.return")
    if return_op is None or len(return_op.operands) != 1:
        return None
    out_val = return_op.operands[0]
    # The fixture's func.return is f32 (post-dequant), but HTA on
    # QAIRT 2.45 doesn't have a Dequantize op — HTA output is always
    # int8 with the conv's output q-params. We slice the dequant off
    # and let the IREE side run it as a separate dispatch.
    if elem_dtype_of(out_val) != "f32":
        return None
    f32_output_shape = shape_of(out_val)
    # The QNN graph's output is the i8 conv result; same NHWC shape
    # as the f32 dequant output, just int8.
    output_shape = f32_output_shape
    output_dtype = "int8"

    # Lower to NHWC QNN graph (no Transpose).
    from iree.compiler import ir as _ir
    from qnn_ir import (
        QnnGraphDesc,
        QuantParams,
        TensorDesc,
        conv2d_node,
    )

    sym_name = _ir.StringAttr(func.attributes["sym_name"]).value

    n, h_in, w_in, ic = in_shape
    kh, kw, ic_w, oc = weight_shape
    if ic != ic_w:
        raise ValueError(f"input/weight channel mismatch: {ic} vs {ic_w}")

    # Split bias_scale = input_scale × weight_scale via sqrt (any
    # consistent factorization is QNN-equivalent at runtime).
    # uint8 storage: zp_u = zp_i + 128 so q_u·scale represents the
    # same float value as the original int8.
    s = math.sqrt(bias_scale) if bias_scale > 0 else 1.0
    input_qp = QuantParams(scale=s, offset=int(in_zp) + 128)
    weight_qp = QuantParams(scale=s, offset=int(w_zp) + 128)
    bias_qp = QuantParams(scale=float(bias_scale), offset=0)
    output_qp = QuantParams(scale=float(output_scale), offset=128)

    # Bias bytes (sfixed_point_32) — quantize per-channel.
    import struct

    n_bias = oc
    f32_vals = struct.unpack(f"<{n_bias}f", bias_bytes_f32)
    INT32_MIN = -(1 << 31)
    INT32_MAX = (1 << 31) - 1
    if bias_scale > 0:
        bias_q = [max(INT32_MIN, min(INT32_MAX, int(round(v / bias_scale)))) for v in f32_vals]
    else:
        bias_q = [0] * n_bias
    bias_bytes_i32 = struct.pack(f"<{n_bias}i", *bias_q)

    out_n, out_h, out_w, out_c = output_shape

    # The QNN graph is conv-only: input → Conv2d → output (i8). The
    # f32 dequant in the source IR moves to a separate IREE dispatch
    # downstream (HTA has no Dequantize op on QAIRT 2.45).
    tensors = (
        TensorDesc(
            name="input",
            shape=tuple(in_shape),
            dtype="uint8",
            role="input",
            quant=input_qp,
        ),
        TensorDesc(
            name="weight",
            shape=tuple(weight_shape),  # HWIO
            dtype="uint8",
            role="static",
            static_data=weight_bytes_hwio,
            quant=weight_qp,
        ),
        TensorDesc(
            name="bias",
            shape=(oc,),
            dtype="sfixed_point_32",
            role="static",
            static_data=bias_bytes_i32,
            quant=bias_qp,
        ),
        TensorDesc(
            name="output",
            shape=tuple(output_shape),
            dtype="uint8",  # uint8 — HTA/GPU only accept UFIXED_POINT_8
            role="output",
            quant=output_qp,
        ),
    )

    nodes = (
        conv2d_node(
            name="conv_op",
            input_tensor="input",
            weight_tensor="weight",
            bias_tensor="bias",
            output_tensor="output",
            strides=strides,
            pad_before_after_hw=(
                (pad_low_hw[0], pad_high_hw[0]),
                (pad_low_hw[1], pad_high_hw[1]),
            ),
            dilation=dilation,
            group=1,
        ),
    )
    return QnnGraphDesc(name=sym_name, tensors=tensors, nodes=nodes)
