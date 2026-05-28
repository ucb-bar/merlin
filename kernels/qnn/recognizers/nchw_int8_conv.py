"""Bindings recognizer for the NCHW int8 quantized Conv2D DAG that real
IREE-emitted yolov8 IR produces (closes #102 — Step 4b).

The DAG, by SSA position relative to a `linalg.conv_2d_nchw_fchw_q` op:

    bias_f32  -- (linalg.generic: divf scale, roundeven, addf zp,
                  maximumf MIN, minimumf MAX, fptosi)
                  --> bias_i32  (1D)

    bias_i32  -- (linalg.broadcast dimensions=[0, 2, 3])
                  --> broadcasted_bias_i32  (NCHW)

    input_i8 (func arg or upstream)  -- (tensor.pad low/high)  -->  padded_i8

    conv_i32 = linalg.conv_2d_nchw_fchw_q {strides, dilations}
        ins(padded_i8, weight_i8, input_zp_i32, weight_zp_i32)
        outs(broadcasted_bias_i32)

    conv_i32  -- (linalg.generic: sitofp, mulf output_scale)
                  --> output_f32  (NCHW)

This module exposes:
  - `ParsedNchwInt8Conv`     : extracted parameters dataclass
  - `parse_yolov8_conv`      : module → ParsedNchwInt8Conv | None
  - `lower_nchw_int8_conv`   : ParsedNchwInt8Conv → QnnGraphDesc
  - `try_recognize`          : module → QnnGraphDesc | None (in REGISTRY)

Lowering produces a 4-node graph (no activation), 5-node (Relu /
Sigmoid / Tanh inserted as `ElementWiseNeuron`), or 6-node (SiLU =
`ElementWiseNeuron(Sigmoid)` + `ElementWiseMultiply`). The constant
spine is Transpose NCHW→NHWC, Conv2d (NHWC × HWIO + sfixed32 bias),
... activation ..., Transpose NHWC→NCHW, Dequantize.

The OIhw→HWIO weight byte permutation runs at emission time via numpy.
Per-element weight/bias payloads are extracted via `dense_to_bytes` and
survive both splat and non-splat dense constants.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from .base import (
    dense_to_bytes,
    elem_dtype_of,
    find_func,
    find_named_op,
    find_named_ops,
    integer_attr_value,  # noqa: F401  - reserved for follow-up
    is_ranked_tensor,
    parse_dense_2d_attr,
    shape_of,
    walk_inner_ops,
)

NAME = "nchw_int8_conv_dequant"


@dataclasses.dataclass(frozen=True)
class ParsedNchwInt8Conv:
    """Extraction result for one yolov8-style NCHW int8 conv DAG."""

    func_name: str
    # NCHW shapes
    input_shape: tuple[int, ...]  # (N, IC, H_in, W_in) — pre-pad
    padded_input_shape: tuple[int, ...]  # post-pad
    weight_shape: tuple[int, ...]  # (OC, IC, Kh, Kw) — linalg OIhw form
    output_shape: tuple[int, ...]  # (N, OC, H_out, W_out)
    # Conv attributes
    strides: tuple[int, int]
    dilation: tuple[int, int]
    pad_low_hw: tuple[int, int]  # NCHW dims 2, 3
    pad_high_hw: tuple[int, int]
    # Quantization
    input_zero_point: int
    weight_zero_point: int
    bias_scale: float  # divisor in the bias-quantize generic
    output_scale: float  # multiplier in the dequant generic
    # Static payloads (full per-element bytes, both splat and non-splat
    # constants supported via `dense_to_bytes`).
    weight_bytes_oihw: bytes  # i8 OIhw, len = OC*IC*Kh*Kw
    bias_bytes_f32: bytes  # source f32 bias, len = OC*4 (little-endian)
    # Optional fused activation that follows the dequant. Recognized
    # values (each maps to a specific QNN op shape in the lowering):
    #   None      : no activation; just Conv2d → Dequantize
    #   "Relu"    : ElementWiseNeuron(Relu)        — HTA fold-friendly
    #   "Sigmoid" : ElementWiseNeuron(Sigmoid)
    #   "Tanh"    : ElementWiseNeuron(Tanh)
    #   "SiLU"    : ElementWiseNeuron(Sigmoid) + ElementWiseMultiply
    #               (yolov8's actual activation; multi-op fusion)
    fused_activation: str | None = None


def _splat_int_constant(op: Any) -> int | None:
    """If `op` is a scalar `arith.constant <int> : i*`, return the int."""
    if op.operation.name != "arith.constant":
        return None
    if "value" not in op.attributes:
        return None
    val = op.attributes["value"]
    # Integer scalar attr (not a dense one).
    from iree.compiler import ir

    try:
        return ir.IntegerAttr(val).value
    except (ValueError, TypeError):
        return None


def _scalar_float_constant(op: Any) -> float | None:
    """If `op` is `arith.constant <float> : f32`, return the float."""
    if op.operation.name != "arith.constant":
        return None
    if "value" not in op.attributes:
        return None
    from iree.compiler import ir

    try:
        return ir.FloatAttr(op.attributes["value"]).value
    except (ValueError, TypeError):
        return None


def _defining_op(value: Any) -> Any | None:
    """Return the op that defines this SSA value, or None for block args."""
    owner = value.owner
    # In the bindings, `Value.owner` returns either an `Operation` or a
    # `Block` (for block arguments). We only want operation owners.
    return owner if hasattr(owner, "operation") else None


def _bias_quantize_scale(generic_op: Any) -> float | None:
    """Walk the bias-quantize `linalg.generic` body to find the `divf`
    operand that supplies the scale (the second operand of divf, which
    is an SSA def from an outer `arith.constant f32`)."""
    for region in generic_op.operation.regions:
        for block in region.blocks:
            for inner in block.operations:
                if inner.name != "arith.divf":
                    continue
                if len(inner.operands) < 2:
                    continue
                rhs = inner.operands[1]
                src = _defining_op(rhs)
                if src is None:
                    continue
                v = _scalar_float_constant(src)
                if v is not None:
                    return v
    return None


def _dequant_scale(generic_op: Any) -> float | None:
    """Walk the dequant `linalg.generic` body to find the `mulf` operand
    that supplies the output scale."""
    for region in generic_op.operation.regions:
        for block in region.blocks:
            for inner in block.operations:
                if inner.name != "arith.mulf":
                    continue
                if len(inner.operands) < 2:
                    continue
                rhs = inner.operands[1]
                src = _defining_op(rhs)
                if src is None:
                    continue
                v = _scalar_float_constant(src)
                if v is not None:
                    return v
    return None


def _maximumf_with_zero(op: Any) -> bool:
    """Return True iff `op` is `arith.maximumf %x, %0.0` (commutative-
    tolerant): one of the operands traces to a scalar f32 constant 0.0.
    """
    if op.name != "arith.maximumf" or len(op.operands) != 2:
        return False
    for operand in op.operands:
        src = _defining_op(operand)
        if src is None:
            continue
        v = _scalar_float_constant(src)
        if v is not None and v == 0.0:
            return True
    return False


def _is_quantize_to_i8_generic(generic_op: Any) -> bool:
    """Return True iff `generic_op` is a `linalg.generic` that quantizes
    f32 → i8 via `divf scale → roundeven → addf zp → maximumf MIN →
    minimumf MAX → fptosi`. This is the per-element quantize body that
    IREE emits after a dequant when the next op needs an i8 input."""
    if generic_op.operation.name != "linalg.generic":
        return False
    body_op_names: list[str] = []
    for region in generic_op.operation.regions:
        for block in region.blocks:
            for inner in block.operations:
                if inner.name == "linalg.yield":
                    continue
                body_op_names.append(inner.name)
    expected = [
        "arith.divf",
        "math.roundeven",
        "arith.addf",
        "arith.maximumf",
        "arith.minimumf",
        "arith.fptosi",
    ]
    return body_op_names == expected


def _is_dequant_from_i8_generic(generic_op: Any) -> bool:
    """Return True iff `generic_op` is `linalg.generic` doing the i8 → f32
    dequant: `sitofp` → `mulf scale`."""
    if generic_op.operation.name != "linalg.generic":
        return False
    body_op_names: list[str] = []
    for region in generic_op.operation.regions:
        for block in region.blocks:
            for inner in block.operations:
                if inner.name == "linalg.yield":
                    continue
                body_op_names.append(inner.name)
    return body_op_names == ["arith.sitofp", "arith.mulf"]


def _strip_requantize_roundtrip(f32_value: Any, func: Any) -> Any:
    """Walk past a `quantize → dequantize` round-trip starting from an
    f32 SSA value, returning the f32 value emitted at the other end (or
    the original value if no round-trip is present).

    Real IREE-emitted yolov8 IR inserts these round-trips around every
    activation: `f32 → quantize-i8 → dequant-f32 → activation`. The
    recognizer walks past them so the activation classifier sees the
    "post-roundtrip" f32 value rather than the bare dequant output.
    """
    src_name = f32_value.get_name()
    quant_op: Any | None = None
    for op in walk_inner_ops(func):
        if op.operation.name != "linalg.generic":
            continue
        if not _is_quantize_to_i8_generic(op):
            continue
        if any(o.get_name() == src_name for o in op.operands):
            quant_op = op
            break
    if quant_op is None:
        return f32_value
    quant_result = quant_op.results[0]
    dequant_op: Any | None = None
    for op in walk_inner_ops(func):
        if op.operation.name != "linalg.generic":
            continue
        if not _is_dequant_from_i8_generic(op):
            continue
        if any(o.get_name() == quant_result.get_name() for o in op.operands):
            dequant_op = op
            break
    if dequant_op is None:
        return f32_value
    return dequant_op.results[0]


def _is_two_input_mulf_generic(generic_op: Any) -> bool:
    """Return True iff `generic_op` is a `linalg.generic` with exactly two
    tensor inputs and a body that is a single `arith.mulf` (used to detect
    `x * sigmoid(x)` for SiLU fusion)."""
    if generic_op.operation.name != "linalg.generic":
        return False
    # Two ins(...) tensor operands. linalg.generic operand layout is
    # [ins..., outs...] in MLIR; we count tensor-typed operands and
    # require exactly 3 (2 inputs + 1 init), which is the SiLU multiply
    # shape.
    if len(generic_op.operands) != 3:
        return False
    body_ops: list[Any] = []
    for region in generic_op.operation.regions:
        for block in region.blocks:
            for inner in block.operations:
                if inner.name == "linalg.yield":
                    continue
                body_ops.append(inner)
    return len(body_ops) == 1 and body_ops[0].name == "arith.mulf"


def _classify_activation_generic(generic_op: Any) -> str | None:
    """Classify a `linalg.generic` body as one of the recognized
    fixed-shape activations and return the QNN ElementWiseNeuron
    operation name (`"Relu"`, `"Sigmoid"`, `"Tanh"`) or None.

    Patterns (single-input single-output `linalg.generic` bodies):
      Relu     : exactly `arith.maximumf %x, %0.0` then yield
      Sigmoid  : `arith.negf` → `math.exp` → `arith.addf %, %1.0`
                 → `arith.divf %1.0, %` then yield (1/(1+exp(-x)))
      Tanh     : exactly `math.tanh %x` then yield
    """
    if generic_op.operation.name != "linalg.generic":
        return None

    body_ops: list[Any] = []
    for region in generic_op.operation.regions:
        for block in region.blocks:
            for inner in block.operations:
                if inner.name == "linalg.yield":
                    continue
                body_ops.append(inner)

    if len(body_ops) == 1:
        op = body_ops[0]
        if _maximumf_with_zero(op):
            return "Relu"
        if op.name == "math.tanh":
            return "Tanh"
        return None

    # Sigmoid: 4 ops in order — negf, exp, addf, divf — with constants
    # 1.0 on the addf RHS and divf LHS (matching the closed-form
    # `1 / (1 + exp(-x))`).
    op_names = [op.name for op in body_ops]
    if op_names == ["arith.negf", "math.exp", "arith.addf", "arith.divf"]:
        addf, divf = body_ops[2], body_ops[3]
        # addf has 2 operands (exp_result, const 1.0); divf has 2
        # operands (const 1.0, addf_result). The constant 1.0 in either
        # position is the giveaway.
        for op in (addf, divf):
            if len(op.operands) != 2:
                return None
            saw_one = False
            for operand in op.operands:
                src = _defining_op(operand)
                if src is None:
                    continue
                v = _scalar_float_constant(src)
                if v is not None and v == 1.0:
                    saw_one = True
                    break
            if not saw_one:
                return None
        return "Sigmoid"

    return None


def _pad_amounts(pad_op: Any) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Read `low` / `high` static padding from a `tensor.pad` op as
    NCHW (low_h, low_w, high_h, high_w). Returns ((low_h, high_h), (low_w, high_w))."""
    if pad_op.operation.name != "tensor.pad":
        return None
    # `tensor.pad` carries two DenseI64ArrayAttr-style attributes:
    # `static_low` and `static_high`. The layout matches operand rank.
    lo = pad_op.attributes.get("static_low") if hasattr(pad_op.attributes, "get") else None
    hi = pad_op.attributes.get("static_high") if hasattr(pad_op.attributes, "get") else None
    if lo is None and "static_low" in pad_op.attributes:
        lo = pad_op.attributes["static_low"]
    if hi is None and "static_high" in pad_op.attributes:
        hi = pad_op.attributes["static_high"]
    if lo is None or hi is None:
        return None
    try:
        lo_list = [int(lo[i]) for i in range(len(lo))]
        hi_list = [int(hi[i]) for i in range(len(hi))]
    except (TypeError, ValueError):
        return None
    if len(lo_list) != 4 or len(hi_list) != 4:
        return None
    # NCHW: dims 0,1 are N,C (no pad); dims 2,3 are H,W.
    return ((lo_list[2], hi_list[2]), (lo_list[3], hi_list[3]))


def parse_yolov8_conv(module: Any) -> ParsedNchwInt8Conv | None:
    """Walk the parsed module looking for the bias-quant + pad + broadcast
    + conv_2d_nchw_fchw_q + dequant DAG. Returns `None` on no-match.
    """
    func = find_func(module)
    if func is None:
        return None

    conv_ops = find_named_ops(func, "linalg.conv_2d_nchw_fchw_q")
    if not conv_ops:
        return None
    conv = conv_ops[0]

    # Conv operands: ins(padded, weight, in_zp, w_zp) outs(broadcasted_bias)
    if len(conv.operands) < 5:
        return None
    padded_v = conv.operands[0]
    weight_v = conv.operands[1]
    in_zp_v = conv.operands[2]
    w_zp_v = conv.operands[3]
    bcast_v = conv.operands[4]

    # Strides + dilations
    strides = parse_dense_2d_attr(conv, "strides")
    dilation = parse_dense_2d_attr(conv, "dilations")
    if strides is None or dilation is None:
        return None

    # Padded input — defining op is tensor.pad
    pad_op = _defining_op(padded_v)
    if pad_op is None or pad_op.operation.name != "tensor.pad":
        return None
    pad_amounts = _pad_amounts(pad_op)
    if pad_amounts is None:
        return None
    pad_low_hw = (pad_amounts[0][0], pad_amounts[1][0])
    pad_high_hw = (pad_amounts[0][1], pad_amounts[1][1])

    # Pre-pad input
    pre_pad_input_v = pad_op.operands[0]
    if not is_ranked_tensor(pre_pad_input_v):
        return None
    input_shape = shape_of(pre_pad_input_v)
    if len(input_shape) != 4 or input_shape[0] != 1:
        return None
    if elem_dtype_of(pre_pad_input_v) != "i8":
        return None

    # Padded shape
    padded_shape = shape_of(padded_v)
    if len(padded_shape) != 4:
        return None

    # Weight: arith.constant i8 NCHW (linalg OIhw)
    weight_op = _defining_op(weight_v)
    if weight_op is None or weight_op.operation.name != "arith.constant":
        return None
    if elem_dtype_of(weight_op.results[0]) != "i8":
        return None
    weight_shape = shape_of(weight_op.results[0])
    if len(weight_shape) != 4:
        return None
    weight_bytes_oihw = dense_to_bytes(weight_op, "i8")
    if weight_bytes_oihw is None:
        return None

    # Zero-points (scalar i32 constants)
    in_zp_op = _defining_op(in_zp_v)
    w_zp_op = _defining_op(w_zp_v)
    if in_zp_op is None or w_zp_op is None:
        return None
    in_zp = _splat_int_constant(in_zp_op)
    w_zp = _splat_int_constant(w_zp_op)
    if in_zp is None or w_zp is None:
        return None

    # Broadcast op — defining op of the conv's outs operand
    bcast_op = _defining_op(bcast_v)
    if bcast_op is None or bcast_op.operation.name != "linalg.broadcast":
        return None
    if len(bcast_op.operands) < 1:
        return None
    bias_i32_v = bcast_op.operands[0]

    # Bias-quantize generic — defines bias_i32_v
    bias_q_op = _defining_op(bias_i32_v)
    if bias_q_op is None or bias_q_op.operation.name != "linalg.generic":
        return None
    bias_scale = _bias_quantize_scale(bias_q_op)
    if bias_scale is None:
        return None
    if len(bias_q_op.operands) < 1:
        return None
    bias_f32_v = bias_q_op.operands[0]
    bias_const_op = _defining_op(bias_f32_v)
    if bias_const_op is None or bias_const_op.operation.name != "arith.constant":
        return None
    if elem_dtype_of(bias_const_op.results[0]) != "f32":
        return None
    bias_bytes_f32 = dense_to_bytes(bias_const_op, "f32")
    if bias_bytes_f32 is None:
        return None

    # Dequant generic — consumer of conv result
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

    # Optional fused activation: a `linalg.generic` whose body matches a
    # recognized fixed-shape activation (Relu, Sigmoid, Tanh), consuming
    # the dequant's f32 output. yolov8's actual activation is SiLU
    # (sigmoid-then-multiply, multi-op) which lands as a follow-up
    # specialised lowering; the simple single-op activations recognized
    # here cover Relu / Sigmoid / Tanh chains used by other models and
    # by hand-curated fixtures.
    fused_activation: str | None = None
    dequant_result = dequant_op.results[0]
    activation_op: Any | None = None
    # Real IREE-emitted yolov8 IR inserts a `quantize → dequantize` round-
    # trip between the conv's f32 dequant and the activation. Walk past
    # it so the activation classifier sees the post-roundtrip f32 value.
    pre_activation_value = _strip_requantize_roundtrip(dequant_result, func)

    for op in walk_inner_ops(func):
        if op.operation.name != "linalg.generic":
            continue
        if not any(o.get_name() == pre_activation_value.get_name() for o in op.operands):
            continue
        kind = _classify_activation_generic(op)
        if kind is not None:
            fused_activation = kind
            activation_op = op
            break

    # Escalate `Sigmoid` → `SiLU` when a multiply generic combines the
    # sigmoid output with the (post-roundtrip) dequant output, possibly
    # with another `quantize → dequantize` round-trip between sigmoid
    # and multiply (which is what real yolov8 emits).
    if fused_activation == "Sigmoid" and activation_op is not None:
        post_sigmoid_value = _strip_requantize_roundtrip(activation_op.results[0], func)
        for m in walk_inner_ops(func):
            if m.operation.name != "linalg.generic":
                continue
            operand_ssa = {opnd.get_name() for opnd in m.operands}
            if (
                post_sigmoid_value.get_name() in operand_ssa
                and pre_activation_value.get_name() in operand_ssa
                and _is_two_input_mulf_generic(m)
            ):
                fused_activation = "SiLU"
                activation_op = m  # multiply is the new terminal op
                break

    # Final output shape from func.return
    return_op = find_named_op(func, "func.return")
    if return_op is None or len(return_op.operands) != 1:
        return None
    output_v = return_op.operands[0]
    if elem_dtype_of(output_v) != "f32":
        return None
    output_shape = shape_of(output_v)
    # The dequant output is the func return when no activation; with an
    # activation the activation's output is the func return.
    expected_terminal = activation_op if activation_op is not None else dequant_op
    if output_shape != tuple(shape_of(expected_terminal.results[0])):
        return None

    from iree.compiler import ir  # noqa: PLC0415  - lazy

    sym_name = ir.StringAttr(func.attributes["sym_name"]).value

    return ParsedNchwInt8Conv(
        func_name=sym_name,
        input_shape=tuple(input_shape),
        padded_input_shape=tuple(padded_shape),
        weight_shape=tuple(weight_shape),
        output_shape=tuple(output_shape),
        strides=strides,
        dilation=dilation,
        pad_low_hw=pad_low_hw,
        pad_high_hw=pad_high_hw,
        input_zero_point=int(in_zp),
        weight_zero_point=int(w_zp),
        bias_scale=float(bias_scale),
        output_scale=float(output_scale),
        weight_bytes_oihw=weight_bytes_oihw,
        bias_bytes_f32=bias_bytes_f32,
        fused_activation=fused_activation,
    )


def _split_bias_scale(bias_scale: float) -> tuple[float, float]:
    """Split `bias_scale = input_scale * weight_scale` into individual
    scales. The QNN backend's runtime math depends only on the product, so
    any consistent split is mathematically equivalent. We pick
    `input_scale = weight_scale = sqrt(bias_scale)`.

    Front-ends that *do* know the original input/weight scales should
    encode them on `func.func` attributes (`merlin.qnn.input_scale`,
    `merlin.qnn.weight_scale`); the parser will pick those up in a
    follow-up. The math is correct regardless because QNN computes
    `output = (sum(input_q * weight_q) + bias_q) * (input_scale *
    weight_scale / output_scale)`.
    """
    s = math.sqrt(bias_scale) if bias_scale > 0 else 1.0
    return s, s


def _permute_oihw_to_hwio(oihw_bytes: bytes, weight_shape: tuple[int, ...]) -> bytes:
    """Permute a weight constant's bytes from linalg's OIhw layout to
    QNN's HWIO layout (Conv2d's filter form).

    Uses numpy for the permutation: O(N) vs O(OC*IC*Kh*Kw) Python loop.
    Splat constants (all bytes equal) are layout-invariant; we shortcut
    them so unit tests don't need numpy in the trivial path.
    """
    if len(weight_shape) != 4:
        raise ValueError(f"weight must be 4D, got shape {weight_shape}")
    expected = 1
    for d in weight_shape:
        expected *= int(d)
    if len(oihw_bytes) != expected:
        raise ValueError(
            f"weight bytes length {len(oihw_bytes)} disagrees with " f"shape {weight_shape} (expected {expected})"
        )
    if not oihw_bytes:
        return oihw_bytes
    if oihw_bytes == bytes([oihw_bytes[0]]) * len(oihw_bytes):
        return oihw_bytes

    import numpy as np  # noqa: PLC0415  - lazy

    oc, ic, kh, kw = (int(d) for d in weight_shape)
    arr = np.frombuffer(oihw_bytes, dtype=np.int8).reshape(oc, ic, kh, kw)
    # OIhw → HWIO: axes (0=O, 1=I, 2=H, 3=W) → (H, W, I, O) = (2, 3, 1, 0)
    permuted = np.ascontiguousarray(arr.transpose(2, 3, 1, 0))
    return permuted.tobytes()


def _quantize_bias_f32_to_i32(bias_bytes_f32: bytes, bias_scale: float, bias_zero_point: int = 0) -> bytes:
    """Quantize a per-channel f32 bias to per-channel i32 using the
    bias-quantize formula from the source IR:
        q = clamp(round(real / scale) + zp, INT32_MIN, INT32_MAX)
    Output is little-endian packed i32 bytes, length = 4 × len(input)/4.
    """
    import struct  # noqa: PLC0415  - lazy

    if len(bias_bytes_f32) % 4 != 0:
        raise ValueError(f"bias_bytes_f32 length {len(bias_bytes_f32)} not a multiple of 4")
    n = len(bias_bytes_f32) // 4
    f32_vals = struct.unpack(f"<{n}f", bias_bytes_f32)
    if bias_scale == 0.0:
        return struct.pack(f"<{n}i", *([0] * n))

    INT32_MIN = -(1 << 31)
    INT32_MAX = (1 << 31) - 1
    q_vals: list[int] = []
    for v in f32_vals:
        q = int(round(v / bias_scale)) + int(bias_zero_point)
        q_vals.append(max(INT32_MIN, min(INT32_MAX, q)))
    return struct.pack(f"<{n}i", *q_vals)


def lower_nchw_int8_conv(parsed: ParsedNchwInt8Conv) -> Any:
    """Build a runtime-shaped `QnnGraphDesc` from the parser's output.

    Graph structure (5 nodes, 7 tensors). NCHW input is transposed to
    NHWC for QNN's Conv2d (which is mathematically NHWC), and the result
    is transposed back to NCHW before a Dequantize node converts the
    quantized i8 output to f32.

        input_i8        APP_WRITE  NCHW i8
            └─ Transpose perm=(0,2,3,1)
        nhwc_in         NATIVE     NHWC i8
            └─ Conv2d (NHWC × HWIO + bias_sfixed32)
        nhwc_conv       NATIVE     NHWC i8
            └─ Transpose perm=(0,3,1,2)
        nchw_quant      NATIVE     NCHW i8
            └─ Dequantize
        output_f32      APP_READ   NCHW f32

    Q-param assignment splits `bias_scale` evenly across input and weight
    via `_split_bias_scale` (same product → same QNN runtime math).
    Output q-params come from the dequant multiplier observed in source.
    """
    from qnn_ir import (  # noqa: PLC0415  - lazy
        QnnGraphDesc,
        QuantParams,
        TensorDesc,
        binary_op_node,
        conv2d_node,
        dequantize_node,
        element_wise_neuron_node,
        transpose_node,
    )

    n, ic, h_in, w_in = parsed.input_shape
    oc, ic_w, kh, kw = parsed.weight_shape
    if ic != ic_w:
        raise ValueError(f"input channels ({ic}) and weight in-channels ({ic_w}) disagree")
    n_out, oc_out, h_out, w_out = parsed.output_shape
    if oc != oc_out:
        raise ValueError(f"weight out-channels ({oc}) and output channels ({oc_out}) disagree")

    input_scale, weight_scale = _split_bias_scale(parsed.bias_scale)
    input_qp = QuantParams(scale=input_scale, offset=parsed.input_zero_point)
    weight_qp = QuantParams(scale=weight_scale, offset=parsed.weight_zero_point)
    bias_qp = QuantParams(scale=parsed.bias_scale, offset=0)
    output_qp = QuantParams(scale=parsed.output_scale, offset=0)

    # Static weight payload: permute OIhw → HWIO at emit time.
    expected_w = oc * ic * kh * kw
    if len(parsed.weight_bytes_oihw) != expected_w:
        raise ValueError(
            f"weight bytes length {len(parsed.weight_bytes_oihw)} disagrees "
            f"with shape {parsed.weight_shape} (expected {expected_w})"
        )
    hwio_bytes = _permute_oihw_to_hwio(parsed.weight_bytes_oihw, parsed.weight_shape)

    # Static bias payload (sfixed_point_32) — quantize source f32 bias
    # element-wise via the IR's quantize formula.
    expected_b = oc * 4  # f32 = 4 bytes per channel
    if len(parsed.bias_bytes_f32) != expected_b:
        raise ValueError(
            f"bias_bytes_f32 length {len(parsed.bias_bytes_f32)} disagrees " f"with OC={oc} (expected {expected_b})"
        )
    bias_bytes = _quantize_bias_f32_to_i32(parsed.bias_bytes_f32, parsed.bias_scale)

    # Tensor inventory.
    nhwc_in_shape = (n, h_in, w_in, ic)
    nhwc_conv_shape = (n, h_out, w_out, oc)
    hwio_weight_shape = (kh, kw, ic, oc)
    nchw_quant_shape = (n, oc, h_out, w_out)

    tensors_list: list[Any] = [
        TensorDesc(
            name="input",
            shape=parsed.input_shape,
            dtype="int8",
            role="input",
            quant=input_qp,
        ),
        TensorDesc(
            name="nhwc_in",
            shape=nhwc_in_shape,
            dtype="int8",
            role="native",
            quant=input_qp,
        ),
        TensorDesc(
            name="weight",
            shape=hwio_weight_shape,
            dtype="int8",
            role="static",
            static_data=hwio_bytes,
            quant=weight_qp,
        ),
        TensorDesc(
            name="bias",
            shape=(oc,),
            dtype="sfixed_point_32",
            role="static",
            static_data=bias_bytes,
            quant=bias_qp,
        ),
        TensorDesc(
            name="nhwc_conv",
            shape=nhwc_conv_shape,
            dtype="int8",
            role="native",
            quant=output_qp,
        ),
    ]
    nodes_list: list[Any] = [
        transpose_node(
            name="nchw_to_nhwc_in",
            input_tensor="input",
            output_tensor="nhwc_in",
            perm=(0, 2, 3, 1),
        ),
        conv2d_node(
            name="conv_op",
            input_tensor="nhwc_in",
            weight_tensor="weight",
            bias_tensor="bias",
            output_tensor="nhwc_conv",
            strides=parsed.strides,
            pad_before_after_hw=(
                (parsed.pad_low_hw[0], parsed.pad_high_hw[0]),
                (parsed.pad_low_hw[1], parsed.pad_high_hw[1]),
            ),
            dilation=parsed.dilation,
            group=1,
        ),
    ]

    # Optional fused activation. Three shapes:
    #   - None              : no activation; Transpose consumes nhwc_conv
    #   - single-op (Relu /
    #     Sigmoid / Tanh)   : insert ElementWiseNeuron(...) → nhwc_act,
    #                          shared output q-params with nhwc_conv so
    #                          HTA's fold_relu_activation_into_conv can
    #                          collapse the pair on supported activations
    #   - "SiLU"            : insert ElementWiseNeuron(Sigmoid) → nhwc_sig,
    #                          then ElementWiseMultiply(nhwc_conv,
    #                          nhwc_sig) → nhwc_act. Output q-params on
    #                          nhwc_act follow `output_scale^2` (the
    #                          quantized product of two i8 tensors with
    #                          scale=output_scale; offset=0).
    transpose_input = "nhwc_conv"
    # Q-params at the input of the post-Conv Transpose (== input of the
    # final Dequantize). Defaults to `output_qp` for plain conv and
    # single-op activations whose output dtype/scale matches Conv2d's.
    # SiLU's elementwise multiply produces a different scale so we
    # override below.
    terminal_qp = output_qp
    if parsed.fused_activation is None:
        pass
    elif parsed.fused_activation == "SiLU":
        tensors_list.append(
            TensorDesc(
                name="nhwc_sig",
                shape=nhwc_conv_shape,
                dtype="int8",
                role="native",
                quant=output_qp,
            )
        )
        # Quantized product q-params: scale = output_scale * output_scale.
        silu_qp = QuantParams(scale=parsed.output_scale * parsed.output_scale, offset=0)
        tensors_list.append(
            TensorDesc(
                name="nhwc_act",
                shape=nhwc_conv_shape,
                dtype="int8",
                role="native",
                quant=silu_qp,
            )
        )
        nodes_list.append(
            element_wise_neuron_node(
                name="sigmoid_op",
                input_tensor="nhwc_conv",
                output_tensor="nhwc_sig",
                operation="Sigmoid",
            )
        )
        nodes_list.append(
            binary_op_node(
                name="silu_mul_op",
                op_type="ElementWiseMultiply",
                lhs="nhwc_conv",
                rhs="nhwc_sig",
                output_tensor="nhwc_act",
            )
        )
        transpose_input = "nhwc_act"
        terminal_qp = silu_qp
    else:
        tensors_list.append(
            TensorDesc(
                name="nhwc_act",
                shape=nhwc_conv_shape,
                dtype="int8",
                role="native",
                quant=output_qp,
            )
        )
        nodes_list.append(
            element_wise_neuron_node(
                name="act_op",
                input_tensor="nhwc_conv",
                output_tensor="nhwc_act",
                operation=parsed.fused_activation,
            )
        )
        transpose_input = "nhwc_act"

    tensors_list.extend(
        [
            TensorDesc(
                name="nchw_quant",
                shape=nchw_quant_shape,
                dtype="int8",
                role="native",
                quant=terminal_qp,
            ),
            TensorDesc(
                name="output",
                shape=parsed.output_shape,
                dtype="float32",
                role="output",
            ),
        ]
    )
    nodes_list.extend(
        [
            transpose_node(
                name="nhwc_to_nchw_out",
                input_tensor=transpose_input,
                output_tensor="nchw_quant",
                perm=(0, 3, 1, 2),
            ),
            dequantize_node(
                name="dequant_op",
                input_tensor="nchw_quant",
                output_tensor="output",
            ),
        ]
    )

    return QnnGraphDesc(
        name=parsed.func_name,
        tensors=tuple(tensors_list),
        nodes=tuple(nodes_list),
    )


def try_recognize(module: Any, *, fp_dtype: str = "float32", **_: object) -> Any | None:
    """Match the yolov8 NCHW int8 conv DAG and lower to a `QnnGraphDesc`.

    Returns `None` if the parser doesn't fire so the dispatcher can try
    the next recognizer.
    """
    parsed = parse_yolov8_conv(module)
    if parsed is None:
        return None
    return lower_nchw_int8_conv(parsed)
