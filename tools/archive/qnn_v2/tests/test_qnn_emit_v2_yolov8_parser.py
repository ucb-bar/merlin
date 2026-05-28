"""Unit tests for the yolov8 NCHW int8 conv DAG recognizer
(`qnn_emit_recognizers/nchw_int8_conv.py`, closing #102).

Covers:
  - Parser DAG walk + parameter extraction (splat and non-splat fixtures)
  - Lowering to a runtime-shaped `QnnGraphDesc` (Transpose / Conv2d /
    Transpose / Dequantize) with correct q-params and tensor inventory
  - OIhw → HWIO byte permutation correctness
  - End-to-end `.qnn.cpp` rendering (no crashes through `emit_qnn_cpp`)

The on-board build (.qnn.cpp → .qnn-ctx via `qnn_build.build_qnn_kernel`)
is exercised by the opt-in test in `test_qnn_emit_v2_yolov8_build.py`.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "kernels"))


def test_parse_yolov8_stem_conv() -> None:
    from iree.compiler import ir
    from qnn_emit_recognizers import nchw_int8_conv as recog

    fixture = REPO_ROOT / "benchmarks/QRB5165/mlir/yolov8_stem_conv_int8.mlir"
    text = fixture.read_text()
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(text, ctx)

    parsed = recog.parse_yolov8_conv(module)
    assert parsed is not None, "parser failed to recognize yolov8 conv DAG"

    assert parsed.func_name == "yolov8_stem_conv_int8"

    # Shapes — NCHW input 1x3x8x8, weight OIhw 16x3x3x3, output 1x16x4x4.
    assert parsed.input_shape == (1, 3, 8, 8)
    assert parsed.padded_input_shape == (1, 3, 10, 10)
    assert parsed.weight_shape == (16, 3, 3, 3)
    assert parsed.output_shape == (1, 16, 4, 4)

    # Conv attributes
    assert parsed.strides == (2, 2)
    assert parsed.dilation == (1, 1)
    assert parsed.pad_low_hw == (1, 1)
    assert parsed.pad_high_hw == (1, 1)

    # Quantization
    assert parsed.input_zero_point == 0
    assert parsed.weight_zero_point == 0
    # Bias scale = input_scale * weight_scale = 0.05 * 0.025 = 0.00125
    assert abs(parsed.bias_scale - 0.00125) < 1e-9
    # Output scale 0.10
    assert abs(parsed.output_scale - 0.10) < 1e-7

    # Static payloads — splat fixture: 16*3*3*3 = 432 bytes of `\x01`,
    # bias is 16 channels of f32 zero (= 64 bytes of `\x00`).
    assert len(parsed.weight_bytes_oihw) == 16 * 3 * 3 * 3
    assert parsed.weight_bytes_oihw == bytes([0x01]) * (16 * 3 * 3 * 3)
    assert len(parsed.bias_bytes_f32) == 16 * 4
    assert parsed.bias_bytes_f32 == bytes(16 * 4)


def test_parser_returns_none_on_unmatching_module() -> None:
    """The parser must return None — not raise — when the input doesn't
    contain the expected DAG. This lets the dispatcher fall through to
    other recognizers without surfacing spurious errors."""
    from iree.compiler import ir
    from qnn_emit_recognizers import nchw_int8_conv as recog

    # An f32 conv+relu fixture has no `linalg.conv_2d_nchw_fchw_q`.
    fixture = REPO_ROOT / "benchmarks/QRB5165/mlir/conv2d_relu_smoke.mlir"
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(fixture.read_text(), ctx)

    assert recog.parse_yolov8_conv(module) is None


def test_try_recognize_lowers_to_graph() -> None:
    """Lowering produces a runtime-shaped `QnnGraphDesc`: 5 nodes
    (Transpose NCHW→NHWC, Conv2d, Transpose NHWC→NCHW, Dequantize) and
    the tensor q-params reflect the parser's bias_scale / output_scale."""
    from iree.compiler import ir
    from qnn_emit_recognizers import nchw_int8_conv as recog

    fixture = REPO_ROOT / "benchmarks/QRB5165/mlir/yolov8_stem_conv_int8.mlir"
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(fixture.read_text(), ctx)
    graph = recog.try_recognize(module)
    assert graph is not None
    assert graph.name == "yolov8_stem_conv_int8"

    # 4 nodes (transpose-in, conv, transpose-out, dequant)
    op_types = [n.op_type for n in graph.nodes]
    assert op_types == ["Transpose", "Conv2d", "Transpose", "Dequantize"]

    # Tensor inventory: input, nhwc_in, weight, bias, nhwc_conv,
    # nchw_quant, output (7 total).
    assert {t.name for t in graph.tensors} == {
        "input",
        "nhwc_in",
        "weight",
        "bias",
        "nhwc_conv",
        "nchw_quant",
        "output",
    }

    by_name = {t.name: t for t in graph.tensors}

    # Input is NCHW i8 with input_qp; weight is HWIO i8.
    assert by_name["input"].shape == (1, 3, 8, 8)
    assert by_name["input"].dtype == "int8"
    assert by_name["weight"].shape == (3, 3, 3, 16)  # HWIO from OIhw
    assert by_name["weight"].dtype == "int8"

    # Bias is sfixed32 with scale = bias_scale = 0.00125.
    assert by_name["bias"].dtype == "sfixed_point_32"
    assert abs(by_name["bias"].quant.scale - 0.00125) < 1e-9

    # Conv intermediate output and nchw_quant share output_qp (scale=0.10).
    assert abs(by_name["nhwc_conv"].quant.scale - 0.10) < 1e-7
    assert abs(by_name["nchw_quant"].quant.scale - 0.10) < 1e-7

    # Final output is NCHW f32 (no quant).
    assert by_name["output"].dtype == "float32"
    assert by_name["output"].shape == (1, 16, 4, 4)
    assert by_name["output"].quant is None

    # Conv strides=2, dilation=1, pad amounts=1 on each H/W edge.
    conv = next(n for n in graph.nodes if n.op_type == "Conv2d")
    pad_param = next(p for p in conv.tensor_params if p.name == "pad_amount")
    assert pad_param.values == (1, 1, 1, 1)
    stride_param = next(p for p in conv.tensor_params if p.name == "stride")
    assert stride_param.values == (2, 2)


def test_oihw_to_hwio_permutation_correctness() -> None:
    """The OIhw → HWIO byte permutation places each weight element at
    the correct flat index. For shape OC=2, IC=1, KH=2, KW=2 with bytes
    1..8 in OIhw flat order, HWIO is computed independently via numpy
    to compare against the implementation.
    """
    import numpy as np
    from qnn_emit_recognizers.nchw_int8_conv import _permute_oihw_to_hwio

    weight_shape = (2, 1, 2, 2)  # OC, IC, KH, KW
    oihw = bytes(range(1, 9))  # 1..8 — distinct so positions are visible
    expected = (
        np.frombuffer(oihw, dtype=np.int8)
        .reshape(weight_shape)
        .transpose(2, 3, 1, 0)  # OIhw → HWIO
        .copy()
        .tobytes()
    )
    actual = _permute_oihw_to_hwio(oihw, weight_shape)
    assert actual == expected
    assert sorted(actual) == sorted(oihw)  # multiset preserved


def test_parse_yolov8_per_element_conv() -> None:
    """The non-splat fixture exercises `dense_to_bytes` element-by-element
    extraction. Validates that:
      - weight_bytes_oihw matches the source dense<[…]> literal exactly
      - bias_bytes_f32 is the per-channel f32 payload [0.1, 0.2]
      - the lowering pipes per-element bytes through the OIhw→HWIO
        permutation (the resulting HWIO weight in the QnnGraphDesc has
        the same multiset of bytes as the source).
    """
    import struct

    from iree.compiler import ir
    from qnn_emit_recognizers import nchw_int8_conv as recog

    fixture = REPO_ROOT / "benchmarks/QRB5165/mlir/yolov8_conv_int8_per_element.mlir"
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(fixture.read_text(), ctx)

    parsed = recog.parse_yolov8_conv(module)
    assert parsed is not None
    # Source weight bytes = 1..8 in OIhw flat order.
    assert parsed.weight_bytes_oihw == bytes(range(1, 9))
    # Source bias bytes = f32 [0.1, 0.2] little-endian.
    assert parsed.bias_bytes_f32 == struct.pack("<2f", 0.1, 0.2)

    # Lowered graph: HWIO weight bytes have the same multiset as OIhw.
    graph = recog.try_recognize(module)
    assert graph is not None
    by_name = {t.name: t for t in graph.tensors}
    hwio_bytes = by_name["weight"].static_data
    assert hwio_bytes is not None
    assert sorted(hwio_bytes) == sorted(parsed.weight_bytes_oihw)
    # And the shape is HWIO (KH, KW, IC, OC) = (2, 2, 1, 2).
    assert by_name["weight"].shape == (2, 2, 1, 2)

    # Bias is i32 quantized: 0.1/0.00125 = 80, 0.2/0.00125 = 160.
    bias_q = struct.unpack("<2i", by_name["bias"].static_data)
    assert bias_q == (80, 160)


@pytest.mark.parametrize(
    "fixture_rel,expected_activation,expected_macro",
    [
        (
            "benchmarks/QRB5165/mlir/yolov8_conv_relu_int8.mlir",
            "Relu",
            "QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU",
        ),
        (
            "benchmarks/QRB5165/mlir/yolov8_conv_sigmoid_int8.mlir",
            "Sigmoid",
            "QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SIGMOID",
        ),
        (
            "benchmarks/QRB5165/mlir/yolov8_conv_tanh_int8.mlir",
            "Tanh",
            "QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH",
        ),
    ],
)
def test_parse_yolov8_conv_with_fused_activation(
    fixture_rel: str, expected_activation: str, expected_macro: str
) -> None:
    """Conv + (Relu | Sigmoid | Tanh) fusion shapes: parser detects the
    trailing single-op activation `linalg.generic` after the dequant
    and sets `fused_activation`. Lowering inserts an
    `ElementWiseNeuron` node carrying the canonical QnnOpDef macro."""
    from iree.compiler import ir
    from qnn_emit_recognizers import nchw_int8_conv as recog

    fixture = REPO_ROOT / fixture_rel
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(fixture.read_text(), ctx)

    parsed = recog.parse_yolov8_conv(module)
    assert parsed is not None
    assert parsed.fused_activation == expected_activation
    # Conv shape unchanged across activation fixtures.
    assert parsed.input_shape == (1, 3, 8, 8)
    assert parsed.weight_shape == (16, 3, 3, 3)
    assert parsed.output_shape == (1, 16, 4, 4)

    # Lowered graph carries the matching ElementWiseNeuron with the
    # right operation macro.
    graph = recog.try_recognize(module)
    assert graph is not None
    op_types = [n.op_type for n in graph.nodes]
    assert op_types == [
        "Transpose",
        "Conv2d",
        "ElementWiseNeuron",
        "Transpose",
        "Dequantize",
    ]
    neuron = next(n for n in graph.nodes if n.op_type == "ElementWiseNeuron")
    assert neuron.scalar_params[0].name == "operation"
    assert neuron.scalar_params[0].value == expected_macro


def test_parse_yolov8_conv_silu_fixture() -> None:
    """SiLU = x * sigmoid(x). The parser detects the sigmoid generic
    *and* a following multiply that takes both the sigmoid output and
    the dequant output as inputs, escalating fused_activation from
    `Sigmoid` to `SiLU`."""
    from iree.compiler import ir
    from qnn_emit_recognizers import nchw_int8_conv as recog

    fixture = REPO_ROOT / "benchmarks/QRB5165/mlir/yolov8_conv_silu_int8.mlir"
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(fixture.read_text(), ctx)

    parsed = recog.parse_yolov8_conv(module)
    assert parsed is not None
    assert parsed.fused_activation == "SiLU"


def test_parse_real_yolov8_silu_with_requantize_roundtrips() -> None:
    """Real IREE-emitted yolov8 IR places `quantize → dequantize` round-
    trips between the conv's f32 dequant and the sigmoid AND between
    the sigmoid and the SiLU multiply. The parser walks past both
    round-trips so the multi-op fusion still recognizes as `SiLU`."""
    from iree.compiler import ir
    from qnn_emit_recognizers import nchw_int8_conv as recog

    fixture = REPO_ROOT / "benchmarks/QRB5165/mlir/yolov8_conv_silu_real_int8.mlir"
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(fixture.read_text(), ctx)
    parsed = recog.parse_yolov8_conv(module)
    assert parsed is not None, (
        "real-yolov8 SiLU shape (with requantize round-trips) failed to "
        "parse — `_strip_requantize_roundtrip` must walk past the "
        "intermediate quantize→dequantize chains."
    )
    assert parsed.fused_activation == "SiLU"

    graph = recog.try_recognize(module)
    assert graph is not None
    op_types = [n.op_type for n in graph.nodes]
    assert op_types == [
        "Transpose",
        "Conv2d",
        "ElementWiseNeuron",
        "ElementWiseMultiply",
        "Transpose",
        "Dequantize",
    ]


def test_lower_conv_silu_emits_sigmoid_plus_multiply() -> None:
    """SiLU lowering is multi-op: ElementWiseNeuron(Sigmoid) +
    ElementWiseMultiply(conv_out, sigmoid_out). The multiply output
    carries scale = output_scale × output_scale (quantized product)
    and feeds the Transpose → Dequantize tail."""
    from iree.compiler import ir
    from qnn_emit_recognizers import nchw_int8_conv as recog

    fixture = REPO_ROOT / "benchmarks/QRB5165/mlir/yolov8_conv_silu_int8.mlir"
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(fixture.read_text(), ctx)
    graph = recog.try_recognize(module)
    assert graph is not None

    op_types = [n.op_type for n in graph.nodes]
    assert op_types == [
        "Transpose",
        "Conv2d",
        "ElementWiseNeuron",
        "ElementWiseMultiply",
        "Transpose",
        "Dequantize",
    ]

    by_name = {t.name: t for t in graph.tensors}
    # Sigmoid output shares conv-output q-params (so HTA could fold
    # Conv→Sigmoid if it had a sigmoid-fold pass).
    assert by_name["nhwc_sig"].quant.scale == by_name["nhwc_conv"].quant.scale
    # Multiply output q-params are output_scale^2 (= 0.10 * 0.10 = 0.01).
    assert abs(by_name["nhwc_act"].quant.scale - 0.01) < 1e-9
    # nchw_quant feeds the Dequantize, so it must carry the same
    # post-multiply q-params (terminal_qp threading).
    assert abs(by_name["nchw_quant"].quant.scale - 0.01) < 1e-9

    sigmoid = next(n for n in graph.nodes if n.op_type == "ElementWiseNeuron")
    assert sigmoid.scalar_params[0].value == "QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SIGMOID"

    mul = next(n for n in graph.nodes if n.op_type == "ElementWiseMultiply")
    # Multiply consumes `nhwc_conv` (= x) and `nhwc_sig` (= sigmoid(x)).
    assert mul.inputs == ("nhwc_conv", "nhwc_sig")
    assert mul.outputs == ("nhwc_act",)


def test_lower_conv_relu_emits_element_wise_neuron() -> None:
    """When fused_activation is set, the lowering inserts an
    ElementWiseNeuron node between the Conv2d and the post-Conv
    Transpose, with output q-params matching the Conv2d's (HTA fold-
    friendly shape)."""
    from iree.compiler import ir
    from qnn_emit_recognizers import nchw_int8_conv as recog

    fixture = REPO_ROOT / "benchmarks/QRB5165/mlir/yolov8_conv_relu_int8.mlir"
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(fixture.read_text(), ctx)
    graph = recog.try_recognize(module)
    assert graph is not None

    op_types = [n.op_type for n in graph.nodes]
    # 5 nodes now: Transpose / Conv2d / ElementWiseNeuron / Transpose / Dequantize
    assert op_types == [
        "Transpose",
        "Conv2d",
        "ElementWiseNeuron",
        "Transpose",
        "Dequantize",
    ]

    by_name = {t.name: t for t in graph.tensors}
    # New tensor `nhwc_act` exists; shares q-params with `nhwc_conv`
    # (HTA's fold_relu_activation_into_conv requires this).
    assert "nhwc_act" in by_name
    assert by_name["nhwc_act"].quant.scale == by_name["nhwc_conv"].quant.scale
    assert by_name["nhwc_act"].quant.offset == by_name["nhwc_conv"].quant.offset

    # The post-Conv Transpose now consumes `nhwc_act`, not `nhwc_conv`.
    transpose_out = next(n for n in graph.nodes if n.name == "nhwc_to_nchw_out")
    assert transpose_out.inputs == ("nhwc_act",)

    # ElementWiseNeuron carries the operation scalar param set to the
    # QnnOpDef macro symbol for RELU.
    neuron = next(n for n in graph.nodes if n.op_type == "ElementWiseNeuron")
    assert neuron.scalar_params[0].name == "operation"
    assert neuron.scalar_params[0].value == "QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU"


def test_emit_qnn_cpp_renders_cleanly() -> None:
    """The lowered graph emits a syntactically valid `.qnn.cpp` source —
    Transpose / Conv2d / Dequantize nodes all render without crashes
    through `qnn_ir.emit_qnn_cpp`. We don't compile it here (board build
    is Phase 2.2 board-correctness gate); just ensure structural rendering."""
    from iree.compiler import ir
    from qnn_emit_recognizers import nchw_int8_conv as recog
    from qnn_ir import emit_qnn_cpp

    fixture = REPO_ROOT / "benchmarks/QRB5165/mlir/yolov8_stem_conv_int8.mlir"
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(fixture.read_text(), ctx)
    graph = recog.try_recognize(module)
    assert graph is not None
    cpp = emit_qnn_cpp(graph)
    # Sanity: header is present, the four ops appear, and the graph name
    # is rendered into composeGraphs.
    assert "QnnModel_composeGraphs" in cpp
    assert "yolov8_stem_conv_int8" in cpp
    assert 'op_type="Transpose"' in cpp or "Transpose" in cpp
    assert "Conv2d" in cpp
    assert "Dequantize" in cpp
