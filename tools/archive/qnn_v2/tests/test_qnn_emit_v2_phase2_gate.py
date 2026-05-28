"""Phase 2 verification gate.

Formalizes the closure criteria from the plan
(`/home/agustin/.claude/plans/i-want-to-enable-rosy-sundae.md`):

  - **Conv coverage on yolov8 IR**: every distinct activation shape that
    real IREE-emitted yolov8 IR produces around a `linalg.conv_2d_nchw_
    fchw_q` is recognized. Activation kinds covered: None, Relu, Relu6,
    Sigmoid, Tanh, SiLU (synthetic), SiLU through requantize round-
    trips (real-yolov8 shape). Each maps to a known QNN op shape in
    the lowering.

  - **HTA fused conv+relu structural match**: the emitter's lowered
    Conv+Relu graph carries the same op sequence and shared output
    q-params as the hand-authored golden kernel
    `benchmarks/QRB5165/kernels/abi/conv2d_relu_int8_fused.qnn.cpp`,
    so HTA's `fold_relu_activation_into_conv` finalize-time pass can
    fuse the pair into one HVX kernel.

These are the exit gates for task #126 / Phase 2. Sub-tasks deferred
to later phases (per-channel weight scales, depthwise int8, standalone
maxpool/concat/reshape/transpose recognizers) are out of scope for the
gate but tracked in the plan's "still pending" list.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "kernels"))


# ----------------------------------------------------------------------
# Conv-shape coverage gate
# ----------------------------------------------------------------------

# Each entry: (fixture_rel, expected_op_sequence). The op sequence is
# deliberately exhaustive — it documents the full Phase 2 lowering
# surface for yolov8's conv variants.
_COVERAGE_FIXTURES = [
    (
        "benchmarks/QRB5165/mlir/yolov8_stem_conv_int8.mlir",
        ["Transpose", "Conv2d", "Transpose", "Dequantize"],
    ),
    (
        "benchmarks/QRB5165/mlir/yolov8_conv_int8_per_element.mlir",
        ["Transpose", "Conv2d", "Transpose", "Dequantize"],
    ),
    (
        "benchmarks/QRB5165/mlir/yolov8_conv_relu_int8.mlir",
        [
            "Transpose",
            "Conv2d",
            "ElementWiseNeuron",
            "Transpose",
            "Dequantize",
        ],
    ),
    (
        "benchmarks/QRB5165/mlir/yolov8_conv_sigmoid_int8.mlir",
        [
            "Transpose",
            "Conv2d",
            "ElementWiseNeuron",
            "Transpose",
            "Dequantize",
        ],
    ),
    (
        "benchmarks/QRB5165/mlir/yolov8_conv_tanh_int8.mlir",
        [
            "Transpose",
            "Conv2d",
            "ElementWiseNeuron",
            "Transpose",
            "Dequantize",
        ],
    ),
    (
        "benchmarks/QRB5165/mlir/yolov8_conv_silu_int8.mlir",
        [
            "Transpose",
            "Conv2d",
            "ElementWiseNeuron",
            "ElementWiseMultiply",
            "Transpose",
            "Dequantize",
        ],
    ),
    (
        "benchmarks/QRB5165/mlir/yolov8_conv_silu_real_int8.mlir",
        [
            "Transpose",
            "Conv2d",
            "ElementWiseNeuron",
            "ElementWiseMultiply",
            "Transpose",
            "Dequantize",
        ],
    ),
]


@pytest.mark.parametrize("fixture_rel,expected_ops", _COVERAGE_FIXTURES)
def test_phase2_conv_shape_recognized(fixture_rel: str, expected_ops) -> None:
    """Every yolov8-shape conv fixture (with all activation variants we
    cover) is recognized and lowers to the expected QNN op sequence."""
    from iree.compiler import ir
    from qnn_emit_recognizers import nchw_int8_conv as recog

    fixture = REPO_ROOT / fixture_rel
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(fixture.read_text(), ctx)
    graph = recog.try_recognize(module)
    assert graph is not None, f"Phase 2 gate: recognizer didn't fire on {fixture_rel}"
    assert [n.op_type for n in graph.nodes] == expected_ops


def test_phase2_conv_coverage_summary() -> None:
    """Sanity-print the recognizer surface so the gate is self-
    documenting. No assertions beyond the existence of every
    coverage fixture; per-shape behavior is asserted in the parametric
    test above."""
    for fixture_rel, _ in _COVERAGE_FIXTURES:
        path = REPO_ROOT / fixture_rel
        assert path.exists(), f"Phase 2 fixture missing: {fixture_rel}"
    # 7 distinct conv-shape fixtures: stem, per-element, +Relu,
    # +Sigmoid, +Tanh, +SiLU (synthetic), +SiLU (real-yolov8 with
    # requantize round-trips).
    assert len(_COVERAGE_FIXTURES) == 7


# ----------------------------------------------------------------------
# Standalone-op recognizers (maxpool / concat / reshape / transpose)
# ----------------------------------------------------------------------

# yolov8 op profile (from `phases/yolov8n_q_int8.1.input.mlir`):
#   3  linalg.pooling_nchw_max
#   17 tensor.concat
#   7  tensor.collapse_shape
#   2  tensor.expand_shape
#   1  linalg.transpose
# Each gets a dedicated recognizer; together with the conv recognizer
# they cover the bulk of yolov8 dispatch shapes (the partitioner in
# Phase 3 splits the multi-op IR into per-dispatch fragments that each
# match one of these).
_STANDALONE_FIXTURES = [
    (
        "benchmarks/QRB5165/mlir/yolov8_maxpool_int8.mlir",
        ["Transpose", "Dequantize", "PoolMax2d", "Transpose"],
    ),
    (
        "benchmarks/QRB5165/mlir/yolov8_concat_int8.mlir",
        [
            "Transpose",
            "Dequantize",
            "Transpose",
            "Dequantize",
            "Concat",
            "Transpose",
        ],
    ),
    (
        "benchmarks/QRB5165/mlir/yolov8_reshape_int8.mlir",
        ["Reshape"],
    ),
    (
        "benchmarks/QRB5165/mlir/yolov8_transpose_int8.mlir",
        ["Transpose"],
    ),
]


@pytest.mark.parametrize("fixture_rel,expected_ops", _STANDALONE_FIXTURES)
def test_phase2_standalone_op_recognized(fixture_rel: str, expected_ops) -> None:
    """Each non-conv yolov8 op class (maxpool, concat, reshape,
    transpose) has a recognizer that fires on its anchored fixture
    and emits the expected QNN op sequence."""
    from qnn_emit_v2 import parse_mlir

    fixture = REPO_ROOT / fixture_rel
    text = fixture.read_text()
    graph = parse_mlir(text)
    assert graph is not None, f"Phase 2 standalone gate failed: {fixture_rel}"
    assert [n.op_type for n in graph.nodes] == expected_ops


# ----------------------------------------------------------------------
# HTA fused conv+relu structural match against golden hand-authored kernel
# ----------------------------------------------------------------------

_GOLDEN_FUSED_FIXTURE = REPO_ROOT / "benchmarks" / "QRB5165" / "kernels" / "abi" / "conv2d_relu_int8_fused.qnn.cpp"


def test_phase2_golden_fused_kernel_present() -> None:
    """The hand-authored Conv+Relu fused HTA kernel is the structural
    reference. It should always be present so the structural-match
    gate has something to compare against."""
    assert _GOLDEN_FUSED_FIXTURE.exists()


def test_phase2_emitter_matches_golden_fused_structure() -> None:
    """The Conv+Relu fusion shape we emit must structurally match the
    hand-authored golden HTA kernel:

      - Same QNN op sequence at the conv core: Conv2D then
        ElementWiseNeuron with operation = RELU
      - Conv2D's output and ElementWiseNeuron's output share q-params
        (this is what enables HTA's `fold_relu_activation_into_conv`
        finalize-time pass to collapse the pair into one HVX kernel)
      - The intermediate quantization scale is preserved across both
        ops (no implicit requantize)

    The emitter's lowering wraps the core in NCHW↔NHWC Transposes plus
    a final Dequantize (because IREE's IR is NCHW and the func returns
    f32); the golden fixture is a direct NHWC kernel because hand-
    authors don't need the layout adapter. We accept that structural
    difference and assert only on the core sub-sequence.
    """
    from iree.compiler import ir
    from qnn_emit_recognizers import nchw_int8_conv as recog

    # Read the golden fixture and verify it contains the canonical
    # macro symbols we expect to match against.
    golden_text = _GOLDEN_FUSED_FIXTURE.read_text()
    assert "QNN_OP_CONV_2D" in golden_text
    assert "QNN_OP_ELEMENT_WISE_NEURON" in golden_text
    assert "QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU" in golden_text

    # Lower our Conv+Relu fixture and assert the core sub-sequence.
    fixture = REPO_ROOT / "benchmarks/QRB5165/mlir/yolov8_conv_relu_int8.mlir"
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(fixture.read_text(), ctx)
    graph = recog.try_recognize(module)
    assert graph is not None

    # Find the conv and the activation node.
    conv = next(n for n in graph.nodes if n.op_type == "Conv2d")
    relu = next(n for n in graph.nodes if n.op_type == "ElementWiseNeuron")
    by_name = {t.name: t for t in graph.tensors}

    # Core property: same q-params across conv output and activation
    # output. This is the HTA fold prerequisite.
    conv_out = by_name[conv.outputs[0]]
    relu_out = by_name[relu.outputs[0]]
    assert conv_out.quant is not None and relu_out.quant is not None
    assert conv_out.quant.scale == relu_out.quant.scale
    assert conv_out.quant.offset == relu_out.quant.offset

    # The activation is connected to the conv (relu's input is the
    # conv's output), matching the golden fixture's direct chain.
    assert relu.inputs == (conv.outputs[0],)

    # Activation operation is RELU (the macro symbol the emitter ships
    # verbatim into the .qnn.cpp).
    assert relu.scalar_params[0].value == "QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU"

    # Render the .qnn.cpp and confirm the canonical macro symbols
    # appear (so the emitted source binds the same QnnOpDef enum
    # values the golden kernel uses).
    from qnn_ir import emit_qnn_cpp

    cpp = emit_qnn_cpp(graph)
    assert "QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU" in cpp
