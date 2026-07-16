"""Generalization of the accumulator-resident feature beyond matmul: conv (im2col->matmul) and
attention (batch_matmul), plus the N-tail fix for small-N attention.

These build the workload THROUGH the real RVV compiler (apply_rvv_package -> model.o) and read the
emitted RVV from the structured decoder — they assert the feature forms `vfmacc` on conv/attention
contractions, and that the N-tail-safe variant lets a small-N (N=8) attention batch_matmul vectorize
where the un-clamped variant hits the LLVM-23 masked-transfer_write PipelineError. Auto-skip without
the riscv toolchain. Build-only (no slow whole-model spike boot); bit-exact spike verification is
recorded in output/kernels/ceiling/scalable_gap_result.md.
"""
from __future__ import annotations
from merlin.common.paths import repo_root, merlin_dir

import tempfile
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

REPO = repo_root()


def _toolchain() -> bool:
    # Compiles a whole model via zephyr_model.build_app (Zephyr/spike path) -> needs the Zephyr SW
    # toolchain, not just the asm toolchain. Gate on BOTH so it SKIPS cleanly when Zephyr is absent
    # rather than hard-failing inside build_app with ZephyrModelError (honest fail-closed).
    try:
        from merlin.kernels import build_asm
        from merlin.runtime.backends import zephyr_model
        return build_asm.asm_toolchain_available() and zephyr_model.available()
    except Exception:
        return False


def _build(features, bundle):
    """apply_rvv_package(hand_v0 + features) on a workload bundle -> decoded InsnStream of model.o."""
    from merlin.kernels.decode import rvv
    from merlin.rvvgen.apply import apply_rvv_package
    from merlin.rvvgen.registry import load_rvv_package

    pkg = load_rvv_package(REPO / "out/artifacts/targets" / "rvv" / "hand_v0")
    pkg = replace(pkg, run_id="test_general", compiler_features=list(features))
    work = Path(tempfile.mkdtemp(prefix="test_general_"))
    apply_rvv_package(pkg, bundle, work, board="spike_riscv64", harts=1, arena_mb=64)
    return rvv.decode(str(work / "model.o"))


def _conv_bundle():
    from merlin.rvvgen import workloads
    return workloads.gen_conv2d_as_matmul_f32(tempfile.mkdtemp(), M=64, N=16, K=27)


def _attn_bundle(N: int):
    from merlin.rvvgen import workloads
    return workloads.gen_batch_matmul_f32(tempfile.mkdtemp(), B=4, M=32, N=N, K=32)


def _matmul_bundle(M: int, N: int = 64, K: int = 64):
    from merlin.rvvgen import workloads
    return workloads.gen_matmul_f32(tempfile.mkdtemp(), M=M, N=N, K=K)


@pytest.mark.skipif(not _toolchain(), reason="riscv toolchain missing")
def test_conv_contraction_forms_vfmacc():
    # A conv2d expressed as its im2col matmul: the accumulator-resident feature must form a fused
    # vfmacc chain (baseline emits scalar mul+add). Proves the feature fires on conv, not just GEMM.
    s = _build(["accumulator_resident_microkernel"], _conv_bundle())
    assert s.count("vfmacc", "vmacc") > 0
    assert s.count("vfmul") == 0


@pytest.mark.skipif(not _toolchain(), reason="riscv toolchain missing")
def test_attention_small_n_needs_ntail():
    # The N=8 attention batch_matmul: the UN-clamped accumulator-resident feature hits the LLVM-23
    # masked vector.transfer_write PipelineError (NR=16 > N=8). The N-tail-safe variant clamps
    # NR_bmm<=N so the inner vectorize is full -> it builds AND forms vfmacc.
    bundle = _attn_bundle(8)
    with pytest.raises(Exception):                       # masked-transfer_write PipelineError
        _build(["accumulator_resident_microkernel"], bundle)
    s = _build(["accumulator_resident_ntail"], bundle)   # the N-tail fix
    assert s.count("vfmacc", "vmacc") > 0
    assert s.count("vfmul") == 0


@pytest.mark.skipif(not _toolchain(), reason="riscv toolchain missing")
def test_attention_large_n_unaffected_by_ntail():
    # For N >= NR the N-tail variant still vectorizes (NR_bmm=8 tiles N cleanly) -> general, not a
    # special-case for N=8.
    s = _build(["accumulator_resident_ntail"], _attn_bundle(32))
    assert s.count("vfmacc", "vmacc") > 0


@pytest.mark.skipif(not _toolchain(), reason="riscv toolchain missing")
def test_m1_matmul_needs_mtail():
    # WORK-ITEM 1: the M=1 token-decode matmul. The default accumulator-resident feature (MR=4 over
    # the M=1 tile) hits the LLVM-23 masked vector.transfer_write PipelineError (vector<4xNR> into
    # tensor<1xNR>). The M-tail-safe variant clamps MR_mm<=M so the inner vectorize is full -> it
    # builds AND forms vfmacc.
    bundle = _matmul_bundle(M=1)
    with pytest.raises(Exception):                          # masked-transfer_write PipelineError
        _build(["accumulator_resident_microkernel"], bundle)
    s = _build(["accumulator_resident_mtail"], bundle)      # the M-tail fix
    assert s.count("vfmacc", "vmacc") > 0
    assert s.count("vfmul") == 0


@pytest.mark.skipif(not _toolchain(), reason="riscv toolchain missing")
def test_mtail_general_not_m1_overfit():
    # The M-tail clamp must vectorize NORMAL matmuls too (not a special-case for M=1): a cube and a
    # non-cube both form vfmacc with the M-tail feature -> general.
    for M, N, K in ((64, 64, 64), (96, 48, 160)):
        s = _build(["accumulator_resident_mtail"], _matmul_bundle(M=M, N=N, K=K))
        assert s.count("vfmacc", "vmacc") > 0
        assert s.count("vfmul") == 0


@pytest.mark.skipif(not _toolchain(), reason="riscv toolchain missing")
def test_wholemodel_vectorizes_m1_and_n8_in_one_schedule():
    # WORK-ITEM 2: the composed whole-model-safe feature carries BOTH tail clamps (M-tail + N-tail)
    # in ONE schedule, so it vectorizes an M=1 token-decode matmul AND a small-N (N=8) attention
    # batch_matmul with the SAME feature enabled (no scalar fallback, no vector.mask PipelineError).
    s_m1 = _build(["accumulator_resident_wholemodel"], _matmul_bundle(M=1))
    assert s_m1.count("vfmacc", "vmacc") > 0 and s_m1.count("vfmul") == 0
    s_n8 = _build(["accumulator_resident_wholemodel"], _attn_bundle(8))
    assert s_n8.count("vfmacc", "vmacc") > 0 and s_n8.count("vfmul") == 0
    # and a normal cube still vectorizes -> general
    s_cube = _build(["accumulator_resident_wholemodel"], _matmul_bundle(M=64))
    assert s_cube.count("vfmacc", "vmacc") > 0 and s_cube.count("vfmul") == 0
