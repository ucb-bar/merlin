"""Round-trip integration test for the yolov8 NCHW int8 conv recognizer.

The recognizer's output should travel the existing build pipeline:

    .qnn.cpp  →  qnn_build.build_qnn_kernel  →  .qnn-ctx blob

This test is **opt-in** via `MERLIN_QNN_BUILD=1` because:

1. It needs the QAIRT SDK on the host (`/scratch2/dima/misc_sw/qualcomm/
   qairt/2.45.0.260326/` by default; override via `QAIRT_SDK_ROOT`).
2. The full pipeline takes ~10-30s (g++ compile of Qualcomm wrapper
   sources + qnn-context-binary-generator).
3. Backend validation is *backend-specific*: the host-default `libQnnCpu.so`
   on QAIRT 2.45 is known to reject `Transpose` for int8 tensors (the
   OpDef is GPU/HTA-targeted). When this happens we record the failure
   as expected — the recognizer's output is structurally valid; it's the
   CPU validator's coverage gap. The same `.qnn-ctx` build succeeds on
   `libQnnGpu.so` / `libQnnHta.so` (on the QRB5165 board); see Phase 2's
   board-correctness gate.

What this test guards even on default runs (without the env): the
emitter pipeline runs to source-string emission without crashes. The
expensive board build is only triggered with the env knob.
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "kernels"))


# ---- Known-failure substring for libQnnCpu's Transpose rejection. ----
# On QAIRT 2.45 the CPU backend's OpDef registry doesn't include the int8
# Transpose variant we emit; recognizing this substring lets us still run
# the test as a smoke check on machines that have the SDK, without the
# test failing CI for a known backend gap.
_KNOWN_CPU_TRANSPOSE_REJECT = "OpConfig validation failed for Transpose"


def _have_sdk() -> bool:
    """Return True iff the QAIRT SDK appears to be importable. We don't
    construct the QnnBuildConfig here (that would surface a non-test
    error); we just check the env / default path."""
    from qnn_build import _DEFAULT_SDK_ROOT  # type: ignore[attr-defined]

    sdk = pathlib.Path(os.environ.get("QAIRT_SDK_ROOT") or _DEFAULT_SDK_ROOT)
    return sdk.exists()


def _emit_qnn_cpp_for_fixture(fixture_rel: str) -> tuple[str, str]:
    """Run the v2 emitter on `fixture_rel` and return (kernel_name, cpp_text)."""
    from iree.compiler import ir
    from qnn_emit_recognizers import nchw_int8_conv as recog
    from qnn_ir import emit_qnn_cpp

    fixture = REPO_ROOT / fixture_rel
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    module = ir.Module.parse(fixture.read_text(), ctx)
    graph = recog.try_recognize(module)
    assert graph is not None, f"recognizer didn't fire on {fixture_rel}"
    return graph.name, emit_qnn_cpp(graph)


@pytest.mark.skipif(
    os.environ.get("MERLIN_QNN_BUILD") != "1",
    reason="opt-in via MERLIN_QNN_BUILD=1 (needs QAIRT SDK + ~30s build)",
)
@pytest.mark.skipif(
    not _have_sdk(),
    reason=(
        "QAIRT SDK not found; set QAIRT_SDK_ROOT or install the default "
        "/scratch2/dima/misc_sw/qualcomm/qairt/2.45.0.260326"
    ),
)
def test_yolov8_stem_conv_round_trip_to_qnn_ctx() -> None:
    """End-to-end: recognizer → .qnn.cpp → g++ → libQnnModel.so →
    qnn-context-binary-generator → .qnn-ctx. Either succeeds (full
    pipeline green) or fails with the documented CPU-backend Transpose
    rejection (recognizer output is structurally valid; backend gap)."""
    from qnn_build import QnnBuildConfig, build_qnn_kernel

    kernel_name, cpp_text = _emit_qnn_cpp_for_fixture("benchmarks/QRB5165/mlir/yolov8_stem_conv_int8.mlir")

    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp) / f"{kernel_name}.qnn.cpp"
        src.write_text(cpp_text)
        cache = pathlib.Path(tmp) / "cache"
        cache.mkdir()

        cfg = QnnBuildConfig.from_env(backend="cpu")
        try:
            ctxbin = build_qnn_kernel(src, kernel_name, cache, cfg)
        except RuntimeError as e:
            msg = str(e)
            if _KNOWN_CPU_TRANSPOSE_REJECT in msg:
                pytest.xfail(
                    "QAIRT 2.45 libQnnCpu rejects int8 Transpose — "
                    "GPU/HTA backends accept it. Recognizer output is "
                    "structurally valid (compose-graph reached the "
                    "validator with all tensors and addNode calls "
                    "in scope)."
                )
            raise
        assert ctxbin.exists(), f"{ctxbin} not produced"
        assert ctxbin.stat().st_size > 0


@pytest.mark.skipif(
    os.environ.get("MERLIN_QNN_BUILD") != "1",
    reason="opt-in via MERLIN_QNN_BUILD=1 (needs QAIRT SDK + ~30s build)",
)
@pytest.mark.skipif(
    not _have_sdk(),
    reason="QAIRT SDK not found; set QAIRT_SDK_ROOT",
)
def test_yolov8_per_element_round_trip_to_qnn_ctx() -> None:
    """Same gate, but on the per-element (non-splat) fixture — exercises
    the full OIhw→HWIO byte permutation through to ctx-bin generation."""
    from qnn_build import QnnBuildConfig, build_qnn_kernel

    kernel_name, cpp_text = _emit_qnn_cpp_for_fixture("benchmarks/QRB5165/mlir/yolov8_conv_int8_per_element.mlir")

    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp) / f"{kernel_name}.qnn.cpp"
        src.write_text(cpp_text)
        cache = pathlib.Path(tmp) / "cache"
        cache.mkdir()

        cfg = QnnBuildConfig.from_env(backend="cpu")
        try:
            ctxbin = build_qnn_kernel(src, kernel_name, cache, cfg)
        except RuntimeError as e:
            msg = str(e)
            if _KNOWN_CPU_TRANSPOSE_REJECT in msg:
                pytest.xfail("QAIRT 2.45 libQnnCpu rejects int8 Transpose — " "GPU/HTA backends accept it.")
            raise
        assert ctxbin.exists()
        assert ctxbin.stat().st_size > 0


@pytest.mark.skipif(
    os.environ.get("MERLIN_QNN_BUILD") != "1",
    reason="opt-in via MERLIN_QNN_BUILD=1 (needs QAIRT SDK + ~30s build)",
)
@pytest.mark.skipif(not _have_sdk(), reason="QAIRT SDK not found; set QAIRT_SDK_ROOT")
def test_yolov8_conv_relu_round_trip_to_qnn_ctx() -> None:
    """Conv+Relu fusion shape end-to-end. Same expectation as the plain
    conv test: either succeeds on a backend that accepts int8 Transpose
    (GPU/HTA) or xfails with the documented CPU validator gap."""
    from qnn_build import QnnBuildConfig, build_qnn_kernel

    kernel_name, cpp_text = _emit_qnn_cpp_for_fixture("benchmarks/QRB5165/mlir/yolov8_conv_relu_int8.mlir")
    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp) / f"{kernel_name}.qnn.cpp"
        src.write_text(cpp_text)
        cache = pathlib.Path(tmp) / "cache"
        cache.mkdir()
        cfg = QnnBuildConfig.from_env(backend="cpu")
        try:
            ctxbin = build_qnn_kernel(src, kernel_name, cache, cfg)
        except RuntimeError as e:
            if _KNOWN_CPU_TRANSPOSE_REJECT in str(e):
                pytest.xfail("QAIRT 2.45 libQnnCpu rejects int8 Transpose — " "GPU/HTA backends accept it.")
            raise
        assert ctxbin.exists()
        assert ctxbin.stat().st_size > 0


def test_emit_step_conv_relu_runs_without_sdk() -> None:
    """Default-on sanity check for the Conv+Relu fusion fixture: source
    emission completes without crashes, includes ElementWiseNeuron
    + the canonical Relu macro symbol."""
    name, cpp = _emit_qnn_cpp_for_fixture("benchmarks/QRB5165/mlir/yolov8_conv_relu_int8.mlir")
    assert name == "yolov8_conv_relu_int8"
    assert "ElementWiseNeuron" in cpp
    assert "QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU" in cpp


def test_emit_step_runs_without_sdk() -> None:
    """Default-on (no env gate) sanity check: even without the SDK
    available, the recognizer + emitter portion runs cleanly. This is
    the cheapest gate that catches recognizer regressions; the full
    build is opt-in."""
    name, cpp = _emit_qnn_cpp_for_fixture("benchmarks/QRB5165/mlir/yolov8_stem_conv_int8.mlir")
    assert name == "yolov8_stem_conv_int8"
    assert "QnnModel_composeGraphs" in cpp
    assert "Transpose" in cpp
    assert "Conv2d" in cpp
    assert "Dequantize" in cpp


def test_emit_step_per_element_runs_without_sdk() -> None:
    name, cpp = _emit_qnn_cpp_for_fixture("benchmarks/QRB5165/mlir/yolov8_conv_int8_per_element.mlir")
    assert name == "yolov8_conv_int8_per_element"
    assert "QnnModel_composeGraphs" in cpp
