"""MLIR → QNN graph emitter (PR-C0 first compiler milestone).

Recognises a fixed input pattern — `linalg.conv_2d_nhwc_hwcf` followed by
a `linalg.generic` bias-add followed by a `linalg.generic` ReLU — and
emits an equivalent `.qnn.cpp` source file via the `qnn_ir.QnnGraphDesc`
intermediate representation. The emitted C++ goes through the same
downstream pipeline as our hand-authored kernels:

    .qnn.cpp  →  qnn_build.py --on-board  →  libQnnModel.so  →
        qnn-context-binary-generator --backend libQnnGpu.so  →  .qnn-ctx

Scope (deliberately minimal, per the PR-C0 plan):
- Conv2D f32 + ReLU f32, optional bias
- Static shapes only (NHWC input, HWCF weight)
- Stride 1 first, explicit padding
- N=1, no quantization
- One input, one weight constant, one bias constant, one output
- GPU backend only

Why a Python regex parser instead of the full MLIR API: keeps PR-C0 small
and self-contained. The contract (MLIR text in, .qnn.cpp out) is what
matters; later we can swap the parser for `iree.compiler.ir.Module.parse`
without touching the emitter or the QnnGraphDesc layer.

CLI:
    python kernels/qnn/emit.py \\
        --mlir benchmarks/QRB5165/mlir/conv2d_relu_smoke.mlir \\
        --name conv2d_relu_emitted \\
        --output build/qnn_emit/conv2d_relu_emitted.qnn.cpp
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from .ir import (  # noqa: E402
    QnnGraphDesc,
    QuantParams,
    TensorDesc,
    binary_op_node,
    concat_node,
    conv2d_node,
    depthwise_conv2d_node,
    f16_to_bytes,
    f32_to_bytes,
    pool_max_2d_node,
    relu_node,
    reshape_node,
    unary_op_node,
)


def _i32_to_bytes(values: list[int] | tuple[int, ...]) -> bytes:
    import struct

    return struct.pack(f"<{len(values)}i", *values)


def _u8_to_bytes(values: list[int] | tuple[int, ...]) -> bytes:
    return bytes(values)


_LOG = logging.getLogger("qnn_emit")


# ---------------------------------------------------------------------------
# MLIR text parsing
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ParsedConv2dRelu:
    """Result of recognizing the Conv2D + bias-add + ReLU pattern."""

    func_name: str
    input_shape: tuple[int, ...]  # NHWC, e.g. (1, 8, 8, 3)
    weight_shape: tuple[int, ...]  # HWCF, e.g. (3, 3, 3, 4) for kh,kw,in,out
    weight_constant_value: float
    bias_shape: tuple[int, ...]  # (out_channels,)
    bias_constant_value: float
    strides: tuple[int, int]  # (h, w)
    dilation: tuple[int, int]  # (h, w)
    output_shape: tuple[int, ...]  # NHWC


_TENSOR_TYPE_RE = re.compile(r"tensor<([\dx]+)x(?P<dtype>[a-zA-Z0-9]+)>")


def _parse_tensor_shape(s: str) -> tuple[tuple[int, ...], str]:
    m = _TENSOR_TYPE_RE.match(s.strip())
    if not m:
        raise ValueError(f"unrecognised tensor type literal: '{s}'")
    parts = m.group(0)[len("tensor<") : -1]
    dim_str, _, dtype = parts.rpartition("x")
    dims = tuple(int(d) for d in dim_str.split("x"))
    return dims, dtype


def parse_conv2d_relu_mlir(text: str) -> ParsedConv2dRelu:
    """Parse a fixed-pattern Conv2D + bias + ReLU MLIR module. Raises a
    descriptive ValueError on any mismatch with the expected shape — this
    is intentional for PR-C0; later expansions add more patterns.
    """
    func_match = re.search(
        r"func\.func\s+@(?P<name>\w+)\s*\(\s*%[\w\d_]+\s*:\s*"
        r"(?P<in_ty>tensor<[^>]+>)\s*\)\s*->\s*(?P<out_ty>tensor<[^>]+>)",
        text,
    )
    if not func_match:
        raise ValueError("no `func.func @<name>(<input>: tensor<...>) -> tensor<...>` " "found in MLIR")
    func_name = func_match.group("name")
    input_shape, in_dtype = _parse_tensor_shape(func_match.group("in_ty"))
    output_shape, out_dtype = _parse_tensor_shape(func_match.group("out_ty"))
    if in_dtype != "f32" or out_dtype != "f32":
        raise ValueError(f"emitter currently supports f32 only, got input={in_dtype} " f"output={out_dtype}")
    if len(input_shape) != 4 or input_shape[0] != 1:
        raise ValueError(f"emitter currently requires NHWC input with N=1, got {input_shape}")

    # Recognise the weight constant: `arith.constant dense<<value>> : tensor<...>`.
    weight_match = re.search(
        r"%\w+\s*=\s*arith\.constant\s+dense<(?P<val>[-\d\.eE+]+)>\s*:\s*" r"(?P<ty>tensor<\d+x\d+x\d+x\d+xf32>)",
        text,
    )
    if not weight_match:
        raise ValueError("no 4D fp32 weight constant (arith.constant dense<...>) found")
    weight_shape, _ = _parse_tensor_shape(weight_match.group("ty"))
    weight_value = float(weight_match.group("val"))

    # Bias constant: 1D fp32 dense.
    bias_match = re.search(
        r"%\w+\s*=\s*arith\.constant\s+dense<(?P<val>[-\d\.eE+]+)>\s*:\s*" r"(?P<ty>tensor<\d+xf32>)",
        text,
    )
    if not bias_match:
        raise ValueError("no 1D fp32 bias constant found")
    bias_shape, _ = _parse_tensor_shape(bias_match.group("ty"))
    bias_value = float(bias_match.group("val"))
    if bias_shape[0] != weight_shape[3]:
        raise ValueError(
            f"bias shape {bias_shape} doesn't match weight out channels "
            f"{weight_shape[3]} (HWCF: kh, kw, in_ch, out_ch)"
        )

    # Conv op: extract strides + dilations attributes.
    conv_match = re.search(
        r"linalg\.conv_2d_nhwc_hwcf\s*\{(?P<attrs>[^}]+)\}",
        text,
    )
    if not conv_match:
        raise ValueError(
            "no `linalg.conv_2d_nhwc_hwcf` op found — emitter currently " "supports only the NHWC × HWCF conv form"
        )
    attrs = conv_match.group("attrs")
    strides_m = re.search(r"strides\s*=\s*dense<(?P<vals>[\d,\s\[\]]+)>", attrs)
    dil_m = re.search(r"dilations\s*=\s*dense<(?P<vals>[\d,\s\[\]]+)>", attrs)
    if not strides_m or not dil_m:
        raise ValueError("conv op must declare both `strides` and `dilations` " "attributes as `dense<...>` literals")

    def _parse_dense_2d(v: str) -> tuple[int, int]:
        v = v.strip()
        # Either `1` (scalar broadcast) or `[h, w]`.
        if v.startswith("["):
            inner = v.strip("[] ")
            parts = [p.strip() for p in inner.split(",") if p.strip()]
            if len(parts) != 2:
                raise ValueError(f"expected 2-element dense<>, got '{v}'")
            return (int(parts[0]), int(parts[1]))
        return (int(v), int(v))

    strides = _parse_dense_2d(strides_m.group("vals"))
    dilation = _parse_dense_2d(dil_m.group("vals"))

    # Verify the bias-add and ReLU patterns are present (defensive — if
    # absent, the emitter would silently produce a Conv-only graph).
    if "arith.addf" not in text:
        raise ValueError("no `arith.addf` found — emitter expects a bias-add stage " "between conv and relu")
    if "arith.maximumf" not in text:
        raise ValueError("no `arith.maximumf` found — emitter expects a ReLU stage " "after bias-add")

    return ParsedConv2dRelu(
        func_name=func_name,
        input_shape=input_shape,
        weight_shape=weight_shape,
        weight_constant_value=weight_value,
        bias_shape=bias_shape,
        bias_constant_value=bias_value,
        strides=strides,
        dilation=dilation,
        output_shape=output_shape,
    )


# ---------------------------------------------------------------------------
# Pattern → QnnGraphDesc lowering
# ---------------------------------------------------------------------------


def _expected_conv_output_shape(
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    strides: tuple[int, int],
    dilation: tuple[int, int],
    pad_amount_hw: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[int, ...]:
    """NHWC × HWCF Conv2D output shape with explicit padding."""
    n, h_in, w_in, _ = input_shape
    kh, kw, _, c_out = weight_shape
    pad_h = pad_amount_hw[0][0] + pad_amount_hw[0][1]
    pad_w = pad_amount_hw[1][0] + pad_amount_hw[1][1]
    h_out = (h_in + pad_h - dilation[0] * (kh - 1) - 1) // strides[0] + 1
    w_out = (w_in + pad_w - dilation[1] * (kw - 1) - 1) // strides[1] + 1
    return (n, h_out, w_out, c_out)


def lower_conv2d_relu(
    parsed: ParsedConv2dRelu,
    *,
    compute_dtype: str = "float32",
) -> QnnGraphDesc:
    """Build a QnnGraphDesc from the recognized pattern. The emitter
    produces a 2-node graph (Conv2d → Relu) — bias-add is folded into
    Conv2d's optional bias input, matching QNN's native fused conv.

    `compute_dtype` selects the on-device tensor dtype for input / weight /
    bias / intermediate / output. QNN GPU (Adreno) supports `"float32"` and
    `"float16"` for Conv2D; `"float16"` halves the on-device bandwidth and
    is the recommended setting for the GPU path on QRB5165 (the v66 Hexagon
    has no BFLOAT_16 support, and HTA is uint8-only).
    """
    pad_amount: tuple[tuple[int, int], tuple[int, int]] = ((0, 0), (0, 0))
    expected_out = _expected_conv_output_shape(
        parsed.input_shape,
        parsed.weight_shape,
        parsed.strides,
        parsed.dilation,
        pad_amount,
    )
    if expected_out != parsed.output_shape:
        raise ValueError(
            f"output shape mismatch: parsed func returns {parsed.output_shape},"
            f" but Conv2D NHWC×HWCF with strides={parsed.strides} "
            f"dilation={parsed.dilation} pad={pad_amount} produces {expected_out}. "
            f"Check the MLIR fixture or emitter padding assumption."
        )

    if compute_dtype not in ("float32", "float16"):
        raise ValueError(f"lower_conv2d_relu compute_dtype must be 'float32' or 'float16';" f" got '{compute_dtype}'")
    pack = f16_to_bytes if compute_dtype == "float16" else f32_to_bytes

    n_weight = 1
    for d in parsed.weight_shape:
        n_weight *= d
    weight_bytes = pack([parsed.weight_constant_value] * n_weight)
    n_bias = parsed.bias_shape[0]
    bias_bytes = pack([parsed.bias_constant_value] * n_bias)

    tensors = (
        TensorDesc(name="input", shape=parsed.input_shape, dtype=compute_dtype, role="input"),
        TensorDesc(
            name="weight", shape=parsed.weight_shape, dtype=compute_dtype, role="static", static_data=weight_bytes
        ),
        TensorDesc(name="bias", shape=parsed.bias_shape, dtype=compute_dtype, role="static", static_data=bias_bytes),
        TensorDesc(name="conv_out", shape=parsed.output_shape, dtype=compute_dtype, role="native"),
        TensorDesc(name="output", shape=parsed.output_shape, dtype=compute_dtype, role="output"),
    )
    nodes = (
        conv2d_node(
            name="conv_op",
            input_tensor="input",
            weight_tensor="weight",
            bias_tensor="bias",
            output_tensor="conv_out",
            strides=parsed.strides,
            pad_before_after_hw=pad_amount,
            dilation=parsed.dilation,
            group=1,
        ),
        relu_node(name="relu_op", input_tensor="conv_out", output_tensor="output"),
    )
    return QnnGraphDesc(name=parsed.func_name, tensors=tensors, nodes=nodes)


# ---------------------------------------------------------------------------
# uint8 Conv2D recogniser — `merlin.qnn.conv2d_uint8` op with per-tensor
# scale-offset attrs encoded on the func.func.
#
# This is the bridge between the emitter and any front-end (IREE, hand-
# authored) that produces a quantized Conv2D in NHWC layout. The MLIR
# fixture format is intentionally close to what a `merlin-qnn-legalize`
# pass would produce: a custom op carrying the conv's strides/dilations
# plus per-tensor q-params on the func.func attributes. Real IREE-emitted
# IR (`linalg.conv_2d_nchw_fchw_q` + dequant chain) is recognised via a
# different recogniser tracked separately.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ParsedUint8Conv:
    func_name: str
    input_shape: tuple[int, ...]  # NHWC u8
    weight_shape: tuple[int, ...]  # HWCF u8 (kh, kw, in_ch, out_ch)
    weight_constant_value: int  # raw uint8 value
    bias_shape: tuple[int, ...]  # (out_ch,) i32
    bias_constant_value: int
    strides: tuple[int, int]
    dilation: tuple[int, int]
    output_shape: tuple[int, ...]
    input_qp: QuantParams
    weight_qp: QuantParams
    bias_qp: QuantParams
    output_qp: QuantParams


def _parse_qparams_attr(text: str, attr_name: str) -> QuantParams | None:
    """Pull `merlin.qnn.<attr_name> = {scale = X.X : f32, offset = N : i32}`
    out of the function's attribute dictionary."""
    pat = (
        r"merlin\.qnn\."
        + re.escape(attr_name)
        + r"\s*=\s*\{\s*scale\s*=\s*(?P<scale>[-\d\.eE+]+)\s*:\s*f32\s*,"
        + r"\s*offset\s*=\s*(?P<offset>-?\d+)\s*:\s*i32\s*\}"
    )
    m = re.search(pat, text)
    if not m:
        return None
    return QuantParams(scale=float(m.group("scale")), offset=int(m.group("offset")))


def parse_uint8_conv_mlir(text: str) -> ParsedUint8Conv | None:
    """Recognise a fixture-form uint8 Conv2D wrapped as `merlin.qnn.conv2d_uint8`.

    The fixture encodes per-tensor q-params via func attrs to keep the
    parser simple. Returns None when the pattern isn't present.
    """
    if "merlin.qnn.conv2d_uint8" not in text:
        return None

    func_match = re.search(
        r"func\.func\s+@(?P<name>\w+)\s*\(\s*%[\w\d_]+\s*:\s*"
        r"(?P<in_ty>tensor<[^>]+>)\s*\)\s*->\s*"
        r"(?P<out_ty>tensor<[^>]+>)",
        text,
    )
    if not func_match:
        return None
    in_shape, in_dt = _parse_tensor_shape(func_match.group("in_ty"))
    out_shape, out_dt = _parse_tensor_shape(func_match.group("out_ty"))
    if in_dt not in ("ui8", "i8") or out_dt not in ("ui8", "i8"):
        return None
    if len(in_shape) != 4 or in_shape[0] != 1:
        return None

    in_qp = _parse_qparams_attr(text, "input_qparams")
    w_qp = _parse_qparams_attr(text, "weight_qparams")
    b_qp = _parse_qparams_attr(text, "bias_qparams")
    out_qp = _parse_qparams_attr(text, "output_qparams")
    if not (in_qp and w_qp and b_qp and out_qp):
        return None

    # Weight constant: 4D uint8/int8 dense<...> tensor.
    w_match = re.search(
        r"%\w+\s*=\s*arith\.constant\s+dense<(?P<val>-?\d+)>\s*:\s*" r"(?P<ty>tensor<\d+x\d+x\d+x\d+x(?:ui8|i8)>)",
        text,
    )
    if not w_match:
        return None
    weight_shape, _ = _parse_tensor_shape(w_match.group("ty"))
    weight_value = int(w_match.group("val"))

    # Bias constant: 1D i32 dense<...>.
    b_match = re.search(
        r"%\w+\s*=\s*arith\.constant\s+dense<(?P<val>-?\d+)>\s*:\s*" r"(?P<ty>tensor<\d+xi32>)",
        text,
    )
    if not b_match:
        return None
    bias_shape, _ = _parse_tensor_shape(b_match.group("ty"))
    bias_value = int(b_match.group("val"))

    # Strides + dilations attrs on the merlin.qnn.conv2d_uint8 op.
    op_match = re.search(
        r'"merlin\.qnn\.conv2d_uint8"[^{]*\{(?P<attrs>[^}]+)\}',
        text,
    )
    if not op_match:
        return None
    attrs = op_match.group("attrs")
    strides_m = re.search(r"strides\s*=\s*dense<(?P<vals>[\d,\s\[\]]+)>", attrs)
    dil_m = re.search(r"dilations\s*=\s*dense<(?P<vals>[\d,\s\[\]]+)>", attrs)
    if not strides_m or not dil_m:
        return None

    def _parse_dense_2d(v: str) -> tuple[int, int]:
        v = v.strip()
        if v.startswith("["):
            parts = [p.strip() for p in v.strip("[] ").split(",") if p.strip()]
            return (int(parts[0]), int(parts[1]))
        return (int(v), int(v))

    return ParsedUint8Conv(
        func_name=func_match.group("name"),
        input_shape=in_shape,
        weight_shape=weight_shape,
        weight_constant_value=weight_value,
        bias_shape=bias_shape,
        bias_constant_value=bias_value,
        strides=_parse_dense_2d(strides_m.group("vals")),
        dilation=_parse_dense_2d(dil_m.group("vals")),
        output_shape=out_shape,
        input_qp=in_qp,
        weight_qp=w_qp,
        bias_qp=b_qp,
        output_qp=out_qp,
    )


def lower_uint8_conv(parsed: ParsedUint8Conv) -> QnnGraphDesc:
    """Lower the parsed uint8 Conv2D pattern to a QNN graph with proper
    q-params on each tensor descriptor.

    Output dtype follows the recogniser: `ui8` MLIR types map to
    `QNN_DATATYPE_UFIXED_POINT_8` (HTA's required dtype); `i8` maps to
    `SFIXED_POINT_8` (DSP's accepted form).
    """
    expected_out = _expected_conv_output_shape(
        parsed.input_shape,
        parsed.weight_shape,
        parsed.strides,
        parsed.dilation,
        ((0, 0), (0, 0)),
    )
    if expected_out != parsed.output_shape:
        raise ValueError(
            f"output shape mismatch: parsed {parsed.output_shape}, "
            f"expected {expected_out} from Conv2D NHWC×HWCF math"
        )

    n_w = 1
    for d in parsed.weight_shape:
        n_w *= d
    weight_bytes = _u8_to_bytes([parsed.weight_constant_value] * n_w)
    n_b = parsed.bias_shape[0]
    bias_bytes = _i32_to_bytes([parsed.bias_constant_value] * n_b)

    tensors = (
        TensorDesc(name="input", shape=parsed.input_shape, dtype="uint8", role="input", quant=parsed.input_qp),
        TensorDesc(
            name="weight",
            shape=parsed.weight_shape,
            dtype="uint8",
            role="static",
            static_data=weight_bytes,
            quant=parsed.weight_qp,
        ),
        TensorDesc(
            name="bias",
            shape=parsed.bias_shape,
            dtype="sfixed_point_32",
            role="static",
            static_data=bias_bytes,
            quant=parsed.bias_qp,
        ),
        TensorDesc(name="output", shape=parsed.output_shape, dtype="uint8", role="output", quant=parsed.output_qp),
    )
    nodes = (
        conv2d_node(
            name="conv_op",
            input_tensor="input",
            weight_tensor="weight",
            bias_tensor="bias",
            output_tensor="output",
            strides=parsed.strides,
            dilation=parsed.dilation,
            pad_before_after_hw=((0, 0), (0, 0)),
            group=1,
        ),
    )
    return QnnGraphDesc(name=parsed.func_name, tensors=tensors, nodes=nodes)


# ---------------------------------------------------------------------------
# Elementwise binary recogniser (Add / Mul / Sub / Div)
# ---------------------------------------------------------------------------


# linalg.generic body op → QNN op type. Extend as new arith ops are added.
_BINARY_BODY_TO_QNN_OP = {
    "arith.addf": "ElementWiseAdd",
    "arith.subf": "ElementWiseSubtract",
    "arith.mulf": "ElementWiseMultiply",
    "arith.divf": "ElementWiseDivide",
}


@dataclasses.dataclass(frozen=True)
class ParsedElementwiseBinary:
    func_name: str
    op_type: str  # QNN op_type string ("ElementWiseAdd", etc.)
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


def parse_elementwise_binary_mlir(
    text: str,
) -> ParsedElementwiseBinary | None:
    """Recognise `func.func @name(%a: T, %b: T) -> T` whose body is a
    single linalg.generic with one of the supported binary arith ops. Returns
    None when the pattern doesn't match (caller dispatches to the next
    recogniser).
    """
    func_match = re.search(
        r"func\.func\s+@(?P<name>\w+)\s*\(\s*%[\w\d_]+\s*:\s*"
        r"(?P<a_ty>tensor<[^>]+>)\s*,\s*%[\w\d_]+\s*:\s*"
        r"(?P<b_ty>tensor<[^>]+>)\s*\)\s*->\s*"
        r"(?P<out_ty>tensor<[^>]+>)",
        text,
    )
    if not func_match:
        return None
    a_shape, a_dt = _parse_tensor_shape(func_match.group("a_ty"))
    b_shape, b_dt = _parse_tensor_shape(func_match.group("b_ty"))
    out_shape, out_dt = _parse_tensor_shape(func_match.group("out_ty"))
    if a_dt != "f32" or b_dt != "f32" or out_dt != "f32":
        return None
    if a_shape != b_shape or a_shape != out_shape:
        return None  # broadcasting not yet supported

    # Body must contain one of the supported arith ops and no others.
    found = [op for op in _BINARY_BODY_TO_QNN_OP if op in text]
    if len(found) != 1:
        return None
    # Reject if the file also contains a Conv2D / Pool / etc. — that would
    # be a different pattern.
    if "linalg.conv_2d" in text or "linalg.pooling" in text:
        return None

    return ParsedElementwiseBinary(
        func_name=func_match.group("name"),
        op_type=_BINARY_BODY_TO_QNN_OP[found[0]],
        input_shape=a_shape,
        output_shape=out_shape,
    )


def lower_elementwise_binary(parsed: ParsedElementwiseBinary) -> QnnGraphDesc:
    tensors = (
        TensorDesc(name="a", shape=parsed.input_shape, dtype="float32", role="input"),
        TensorDesc(name="b", shape=parsed.input_shape, dtype="float32", role="input"),
        TensorDesc(name="output", shape=parsed.output_shape, dtype="float32", role="output"),
    )
    nodes = (binary_op_node(name="op", op_type=parsed.op_type, lhs="a", rhs="b", output_tensor="output"),)
    return QnnGraphDesc(name=parsed.func_name, tensors=tensors, nodes=nodes)


# ---------------------------------------------------------------------------
# Elementwise unary recogniser (Relu / Sigmoid / Tanh / Relu6 / HardSwish)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ParsedElementwiseUnary:
    func_name: str
    op_type: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


def parse_elementwise_unary_mlir(
    text: str,
) -> ParsedElementwiseUnary | None:
    """Recognise a single-input single-output elementwise unary op. Pattern
    classification is by linalg.generic body shape:

    Relu     :  arith.maximumf %a, %zero
    Tanh     :  math.tanh %a
    Sigmoid  :  arith.negf + math.exp + arith.addf 1 + arith.divf 1 (chain)
    """
    func_match = re.search(
        r"func\.func\s+@(?P<name>\w+)\s*\(\s*%[\w\d_]+\s*:\s*"
        r"(?P<in_ty>tensor<[^>]+>)\s*\)\s*->\s*"
        r"(?P<out_ty>tensor<[^>]+>)",
        text,
    )
    if not func_match:
        return None
    in_shape, in_dt = _parse_tensor_shape(func_match.group("in_ty"))
    out_shape, out_dt = _parse_tensor_shape(func_match.group("out_ty"))
    if in_dt != "f32" or out_dt != "f32" or in_shape != out_shape:
        return None
    if "linalg.conv_2d" in text or "linalg.pooling" in text:
        return None
    if "arith.addf" in text and "arith.divf" not in text:
        return None  # bias-add with no Sigmoid chain — different pattern

    body = text  # crude: recogniser walks the whole module text
    if "math.tanh" in body and "math.exp" not in body:
        op_type = "Tanh"
    elif "math.exp" in body and "arith.divf" in body and "arith.addf" in body:
        op_type = "Sigmoid"
    elif "arith.maximumf" in body and "arith.minimumf" in body:
        op_type = "Relu6"
    elif "arith.maximumf" in body and "arith.minimumf" not in body:
        op_type = "Relu"
    else:
        return None

    return ParsedElementwiseUnary(
        func_name=func_match.group("name"),
        op_type=op_type,
        input_shape=in_shape,
        output_shape=out_shape,
    )


def lower_elementwise_unary(parsed: ParsedElementwiseUnary) -> QnnGraphDesc:
    tensors = (
        TensorDesc(name="input", shape=parsed.input_shape, dtype="float32", role="input"),
        TensorDesc(name="output", shape=parsed.output_shape, dtype="float32", role="output"),
    )
    nodes = (unary_op_node(name="op", op_type=parsed.op_type, input_tensor="input", output_tensor="output"),)
    return QnnGraphDesc(name=parsed.func_name, tensors=tensors, nodes=nodes)


# ---------------------------------------------------------------------------
# DepthwiseConv2D recogniser (linalg.depthwise_conv_2d_nhwc_hwc)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ParsedDepthwiseConv:
    func_name: str
    input_shape: tuple[int, ...]  # NHWC
    weight_shape: tuple[int, ...]  # HWC (kh, kw, channels)
    weight_constant_value: float
    output_shape: tuple[int, ...]  # NHWC
    strides: tuple[int, int]
    dilation: tuple[int, int]


def parse_depthwise_conv_mlir(
    text: str,
) -> ParsedDepthwiseConv | None:
    """Recognise `linalg.depthwise_conv_2d_nhwc_hwc` with a single static
    weight constant. No bias / no fused activation in this first pass —
    the recogniser falls through if the file contains downstream ops the
    pattern can't yet handle.
    """
    if "linalg.depthwise_conv_2d_nhwc_hwc" not in text:
        return None
    func_match = re.search(
        r"func\.func\s+@(?P<name>\w+)\s*\(\s*%[\w\d_]+\s*:\s*"
        r"(?P<in_ty>tensor<[^>]+>)\s*\)\s*->\s*(?P<out_ty>tensor<[^>]+>)",
        text,
    )
    if not func_match:
        return None
    in_shape, in_dt = _parse_tensor_shape(func_match.group("in_ty"))
    out_shape, out_dt = _parse_tensor_shape(func_match.group("out_ty"))
    if in_dt != "f32" or out_dt != "f32":
        return None
    if len(in_shape) != 4 or in_shape[0] != 1:
        return None

    # Weight constant: 3D fp32 dense<...> : tensor<KhxKwxCxf32>
    w_match = re.search(
        r"%\w+\s*=\s*arith\.constant\s+dense<(?P<val>[-\d\.eE+]+)>\s*:\s*" r"(?P<ty>tensor<\d+x\d+x\d+xf32>)",
        text,
    )
    if not w_match:
        return None
    weight_shape, _ = _parse_tensor_shape(w_match.group("ty"))
    weight_value = float(w_match.group("val"))

    # Strides / dilation attrs
    op_match = re.search(
        r"linalg\.depthwise_conv_2d_nhwc_hwc\s*\{(?P<attrs>[^}]+)\}",
        text,
    )
    if not op_match:
        return None
    attrs = op_match.group("attrs")
    strides_m = re.search(r"strides\s*=\s*dense<(?P<vals>[\d,\s\[\]]+)>", attrs)
    dil_m = re.search(r"dilations\s*=\s*dense<(?P<vals>[\d,\s\[\]]+)>", attrs)
    if not strides_m or not dil_m:
        return None

    def _parse_dense_2d(v: str) -> tuple[int, int]:
        v = v.strip()
        if v.startswith("["):
            parts = [p.strip() for p in v.strip("[] ").split(",") if p.strip()]
            return (int(parts[0]), int(parts[1]))
        return (int(v), int(v))

    strides = _parse_dense_2d(strides_m.group("vals"))
    dilation = _parse_dense_2d(dil_m.group("vals"))

    return ParsedDepthwiseConv(
        func_name=func_match.group("name"),
        input_shape=in_shape,
        weight_shape=weight_shape,
        weight_constant_value=weight_value,
        output_shape=out_shape,
        strides=strides,
        dilation=dilation,
    )


def lower_depthwise_conv(parsed: ParsedDepthwiseConv) -> QnnGraphDesc:
    """Lower depthwise conv to QNN. Linalg's HWC weight is reshaped to
    QNN's HWCM (kh, kw, channels, channel_multiplier=1) — same byte layout,
    just a different rank declaration on the QNN tensor.

    QNN's DepthWiseConv2d validation requires a bias input (the spec
    accepts an optional bias but the GPU backend rejects graphs without
    one); we emit a zero bias automatically when the MLIR source doesn't
    declare one. Subsequent passes can fold an explicit bias-add into
    this stage when present.
    """
    kh, kw, ch = parsed.weight_shape
    n_weight = kh * kw * ch
    weight_bytes = f32_to_bytes([parsed.weight_constant_value] * n_weight)
    qnn_weight_shape = (kh, kw, ch, 1)  # HWCM with multiplier = 1
    out_channels = parsed.output_shape[-1]
    bias_bytes = f32_to_bytes([0.0] * out_channels)
    tensors = (
        TensorDesc(name="input", shape=parsed.input_shape, dtype="float32", role="input"),
        TensorDesc(name="weight", shape=qnn_weight_shape, dtype="float32", role="static", static_data=weight_bytes),
        TensorDesc(name="bias", shape=(out_channels,), dtype="float32", role="static", static_data=bias_bytes),
        TensorDesc(name="output", shape=parsed.output_shape, dtype="float32", role="output"),
    )
    nodes = (
        depthwise_conv2d_node(
            name="dwconv_op",
            input_tensor="input",
            weight_tensor="weight",
            bias_tensor="bias",
            output_tensor="output",
            strides=parsed.strides,
            pad_before_after_hw=((0, 0), (0, 0)),
            dilation=parsed.dilation,
        ),
    )
    return QnnGraphDesc(name=parsed.func_name, tensors=tensors, nodes=nodes)


# ---------------------------------------------------------------------------
# MaxPool recogniser (linalg.pooling_nhwc_max)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ParsedMaxPool:
    func_name: str
    input_shape: tuple[int, ...]  # NHWC
    output_shape: tuple[int, ...]  # NHWC
    filter_size: tuple[int, int]
    strides: tuple[int, int]
    dilation: tuple[int, int]


def parse_maxpool_mlir(text: str) -> ParsedMaxPool | None:
    """Recognise `linalg.pooling_nhwc_max` with a single fp32 input."""
    if "linalg.pooling_nhwc_max" not in text:
        return None
    func_match = re.search(
        r"func\.func\s+@(?P<name>\w+)\s*\(\s*%[\w\d_]+\s*:\s*"
        r"(?P<in_ty>tensor<[^>]+>)\s*\)\s*->\s*(?P<out_ty>tensor<[^>]+>)",
        text,
    )
    if not func_match:
        return None
    in_shape, in_dt = _parse_tensor_shape(func_match.group("in_ty"))
    out_shape, out_dt = _parse_tensor_shape(func_match.group("out_ty"))
    if in_dt != "f32" or out_dt != "f32":
        return None
    if len(in_shape) != 4:
        return None

    # The pooling op's filter buffer is the second `ins` operand and has
    # 2D shape `tensor<KhxKwxf32>`. The MLIR form is
    #   ins(%input, %window : tensor<NxHxWxCxf32>, tensor<KhxKwxf32>)
    # so the filter tensor type appears after a comma, before the closing
    # paren of `ins(...)`.
    f_match = re.search(
        r",\s*tensor<(?P<dims>\d+x\d+)xf32>\s*\)",
        text,
    )
    if not f_match:
        return None
    fdims = tuple(int(d) for d in f_match.group("dims").split("x"))
    if len(fdims) != 2:
        return None

    op_match = re.search(
        r"linalg\.pooling_nhwc_max\s*\{(?P<attrs>[^}]+)\}",
        text,
    )
    if not op_match:
        return None
    attrs = op_match.group("attrs")
    strides_m = re.search(r"strides\s*=\s*dense<(?P<vals>[\d,\s\[\]]+)>", attrs)
    dil_m = re.search(r"dilations\s*=\s*dense<(?P<vals>[\d,\s\[\]]+)>", attrs)
    if not strides_m or not dil_m:
        return None

    def _parse_dense_2d(v: str) -> tuple[int, int]:
        v = v.strip()
        if v.startswith("["):
            parts = [p.strip() for p in v.strip("[] ").split(",") if p.strip()]
            return (int(parts[0]), int(parts[1]))
        return (int(v), int(v))

    return ParsedMaxPool(
        func_name=func_match.group("name"),
        input_shape=in_shape,
        output_shape=out_shape,
        filter_size=fdims,
        strides=_parse_dense_2d(strides_m.group("vals")),
        dilation=_parse_dense_2d(dil_m.group("vals")),
    )


def lower_maxpool(parsed: ParsedMaxPool) -> QnnGraphDesc:
    tensors = (
        TensorDesc(name="input", shape=parsed.input_shape, dtype="float32", role="input"),
        TensorDesc(name="output", shape=parsed.output_shape, dtype="float32", role="output"),
    )
    nodes = (
        pool_max_2d_node(
            name="pool_op",
            input_tensor="input",
            output_tensor="output",
            filter_size=parsed.filter_size,
            strides=parsed.strides,
            pad_before_after_hw=((0, 0), (0, 0)),
            rounding_mode=0,  # floor
        ),
    )
    return QnnGraphDesc(name=parsed.func_name, tensors=tensors, nodes=nodes)


# ---------------------------------------------------------------------------
# Concat recogniser (tensor.concat)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ParsedConcat:
    func_name: str
    input_arg_names: tuple[str, ...]  # MLIR-level SSA names from func args
    input_shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]
    axis: int


def parse_concat_mlir(text: str) -> ParsedConcat | None:
    """Recognise a `tensor.concat dim(N) %a, %b, ... : (T1, T2, ...) -> Tout`
    pattern. The MLIR func signature must list all the inputs that get
    concat'd; each must be fp32 with all dims matching except the axis dim.
    """
    if "tensor.concat" not in text:
        return None

    # Func signature: 2+ tensor args → 1 tensor return.
    sig = re.search(
        r"func\.func\s+@(?P<name>\w+)\s*\((?P<args>[^)]*)\)\s*->\s*" r"(?P<out_ty>tensor<[^>]+>)",
        text,
    )
    if not sig:
        return None
    args_str = sig.group("args")
    arg_pairs = re.findall(r"(%[\w\d_]+)\s*:\s*(tensor<[^>]+>)", args_str)
    if len(arg_pairs) < 2:
        return None
    arg_names = tuple(a[0].lstrip("%") for a in arg_pairs)
    arg_types = [a[1] for a in arg_pairs]
    arg_shapes = []
    for ty in arg_types:
        sh, dt = _parse_tensor_shape(ty)
        if dt != "f32":
            return None
        arg_shapes.append(sh)

    out_shape, out_dt = _parse_tensor_shape(sig.group("out_ty"))
    if out_dt != "f32":
        return None

    # Concat op: extract dim(N) and verify the inputs match the func args.
    concat_match = re.search(
        r"tensor\.concat\s+dim\((?P<dim>\d+)\)\s+(?P<inputs>[^\n:]+?)\s*:\s*"
        r"\((?P<intypes>[^)]+)\)\s*->\s*(?P<outty>tensor<[^>]+>)",
        text,
    )
    if not concat_match:
        return None
    axis = int(concat_match.group("dim"))

    # Verify the concat operands appear among the func args (we don't yet
    # support concats whose inputs come from intermediate ops).
    operand_names = [n.strip().lstrip("%") for n in concat_match.group("inputs").split(",") if n.strip()]
    for name in operand_names:
        if name not in arg_names:
            return None  # op consumes an intermediate; not yet supported

    # Output sanity: along `axis`, sum of input dims = output dim.
    expected_axis = sum(s[axis] for s in arg_shapes)
    if expected_axis != out_shape[axis]:
        return None
    for d in range(len(out_shape)):
        if d == axis:
            continue
        if any(s[d] != out_shape[d] for s in arg_shapes):
            return None

    # Reorder the input shapes to match the actual concat operand order
    # (the op operand list, not the func arg list).
    arg_to_shape = dict(zip(arg_names, arg_shapes))
    ordered_shapes = tuple(arg_to_shape[n] for n in operand_names)
    ordered_input_names = tuple(operand_names)

    return ParsedConcat(
        func_name=sig.group("name"),
        input_arg_names=ordered_input_names,
        input_shapes=ordered_shapes,
        output_shape=out_shape,
        axis=axis,
    )


def lower_concat(parsed: ParsedConcat) -> QnnGraphDesc:
    tensors = []
    for name, shape in zip(parsed.input_arg_names, parsed.input_shapes):
        tensors.append(TensorDesc(name=name, shape=shape, dtype="float32", role="input"))
    tensors.append(TensorDesc(name="output", shape=parsed.output_shape, dtype="float32", role="output"))
    nodes = (
        concat_node(
            name="concat_op",
            input_tensors=parsed.input_arg_names,
            output_tensor="output",
            axis=parsed.axis,
        ),
    )
    return QnnGraphDesc(name=parsed.func_name, tensors=tuple(tensors), nodes=nodes)


# ---------------------------------------------------------------------------
# Reshape recogniser (tensor.collapse_shape / tensor.expand_shape)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ParsedReshape:
    func_name: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


def parse_reshape_mlir(text: str) -> ParsedReshape | None:
    """Recognise a `tensor.collapse_shape` or `tensor.expand_shape` op as
    a reshape from the func's input shape to its output shape. Both ops
    just rearrange dims without changing element count, so we lower to a
    single QNN `Reshape` node regardless of which one the MLIR uses."""
    if "tensor.collapse_shape" not in text and "tensor.expand_shape" not in text:
        return None
    sig = re.search(
        r"func\.func\s+@(?P<name>\w+)\s*\(\s*%[\w\d_]+\s*:\s*"
        r"(?P<in_ty>tensor<[^>]+>)\s*\)\s*->\s*(?P<out_ty>tensor<[^>]+>)",
        text,
    )
    if not sig:
        return None
    in_shape, in_dt = _parse_tensor_shape(sig.group("in_ty"))
    out_shape, out_dt = _parse_tensor_shape(sig.group("out_ty"))
    if in_dt != "f32" or out_dt != "f32":
        return None

    n_in = 1
    for d in in_shape:
        n_in *= d
    n_out = 1
    for d in out_shape:
        n_out *= d
    if n_in != n_out:
        return None  # not a pure reshape

    return ParsedReshape(
        func_name=sig.group("name"),
        input_shape=in_shape,
        output_shape=out_shape,
    )


def lower_reshape(parsed: ParsedReshape) -> QnnGraphDesc:
    tensors = (
        TensorDesc(name="input", shape=parsed.input_shape, dtype="float32", role="input"),
        TensorDesc(name="output", shape=parsed.output_shape, dtype="float32", role="output"),
    )
    nodes = (reshape_node(name="reshape_op", input_tensor="input", output_tensor="output"),)
    return QnnGraphDesc(name=parsed.func_name, tensors=tensors, nodes=nodes)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_mlir(text: str, *, fp_dtype: str = "float32") -> QnnGraphDesc:
    """Multi-pattern dispatcher. Tries each recogniser in priority order;
    returns the QnnGraphDesc from the first one that matches. Raises a
    descriptive error if none match — callers can extend by adding a new
    recogniser to this dispatcher.

    `fp_dtype` selects the on-device floating-point dtype for fp recognisers.
    `"float32"` (default) keeps the legacy behaviour. `"float16"` is the GPU
    path for QRB5165 — Adreno's Conv2D supports FLOAT_16 but not BFLOAT_16.
    Quantized recognisers ignore this parameter (their dtype is fixed by the
    MLIR fixture's qparam attrs).
    """
    # Most-specific first: Conv2D+bias+ReLU has a distinctive op set.
    try:
        return lower_conv2d_relu(parse_conv2d_relu_mlir(text), compute_dtype=fp_dtype)
    except ValueError:
        pass

    # Quantized uint8 Conv2D — the int8 path for HTA (and DSP if/when its
    # finalize is unblocked).
    parsed_qconv = parse_uint8_conv_mlir(text)
    if parsed_qconv is not None:
        return lower_uint8_conv(parsed_qconv)

    parsed_dwconv = parse_depthwise_conv_mlir(text)
    if parsed_dwconv is not None:
        return lower_depthwise_conv(parsed_dwconv)

    parsed_pool = parse_maxpool_mlir(text)
    if parsed_pool is not None:
        return lower_maxpool(parsed_pool)

    parsed_concat = parse_concat_mlir(text)
    if parsed_concat is not None:
        return lower_concat(parsed_concat)

    parsed_reshape = parse_reshape_mlir(text)
    if parsed_reshape is not None:
        return lower_reshape(parsed_reshape)

    parsed_eb = parse_elementwise_binary_mlir(text)
    if parsed_eb is not None:
        return lower_elementwise_binary(parsed_eb)

    parsed_eu = parse_elementwise_unary_mlir(text)
    if parsed_eu is not None:
        return lower_elementwise_unary(parsed_eu)

    raise ValueError(
        "no recogniser matched the MLIR pattern. Supported patterns:\n"
        "  - Conv2D (NHWC×HWCF) + bias-add + ReLU\n"
        "  - DepthwiseConv2D (NHWC×HWC, no bias)\n"
        "  - MaxPool (NHWC, no padding, fp32)\n"
        "  - tensor.concat (>=2 fp32 inputs of same shape except along axis)\n"
        "  - tensor.{collapse,expand}_shape — pure reshape (same elem count)\n"
        "  - elementwise binary (Add/Sub/Mul/Div) over same-shape fp32 tensors\n"
        "  - elementwise unary (Relu/Relu6/Sigmoid/Tanh) over fp32 tensors\n"
        "Add new recognisers to kernels/qnn/emit.py::parse_mlir."
    )


def emit(
    mlir_path: pathlib.Path,
    output_cpp: pathlib.Path,
    *,
    fp_dtype: str = "float32",
) -> QnnGraphDesc:
    """Parse `mlir_path`, lower to a QnnGraphDesc, and write a `.qnn.cpp`
    source file at `output_cpp`. Returns the descriptor for downstream
    callers that want metadata (input/output shapes, etc.).

    `fp_dtype` is forwarded to the recogniser dispatcher; pass `"float16"`
    when targeting QNN GPU on QRB5165 to halve on-device bandwidth."""
    from qnn_ir import emit_qnn_cpp

    text = mlir_path.read_text()
    graph = parse_mlir(text, fp_dtype=fp_dtype)
    cpp = emit_qnn_cpp(graph)
    output_cpp.parent.mkdir(parents=True, exist_ok=True)
    output_cpp.write_text(cpp)
    _LOG.info(
        "emitted %s (%d bytes; graph=%s, %d tensors, %d nodes)",
        output_cpp,
        output_cpp.stat().st_size,
        graph.name,
        len(graph.tensors),
        len(graph.nodes),
    )
    return graph


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlir", type=pathlib.Path, required=True)
    parser.add_argument("--name", required=True, help="Override the graph name (defaults to func.func name)")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="Path to write the emitted .qnn.cpp")
    parser.add_argument(
        "--fp-dtype",
        choices=("float32", "float16"),
        default="float32",
        help="On-device floating-point dtype for fp recognisers. Use "
        "'float16' for the QRB5165 QNN GPU path (Adreno).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    graph = emit(args.mlir, args.output, fp_dtype=args.fp_dtype)
    if graph.name != args.name:
        _LOG.info("note: --name=%s differs from func name %s; using func name", args.name, graph.name)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
