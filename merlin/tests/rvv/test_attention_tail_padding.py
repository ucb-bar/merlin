"""Regression coverage for odd attention extents on the accumulator-resident RVV path."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import repo_root


FEATURE = "accumulator_resident_wholemodel_vf_bmmpad"
REPO = repo_root()


def _toolchain() -> bool:
    try:
        from merlin.kernels import build_asm
        from merlin.runtime.backends import zephyr_model

        return build_asm.asm_toolchain_available() and zephyr_model.available()
    except Exception:
        return False


def test_attention_tail_padding_preserves_the_full_tile_shape():
    from merlin.llvmlower import impr_features as features
    from merlin.llvmlower import pipeline

    schedule = features.apply_schedule(
        pipeline.RVV_TRANSFORM_SCHEDULE, frozenset({FEATURE})
    )
    assert "transform.structured.pad %btail pad_to_multiple_of [4, 8]" in schedule
    assert "padding_dimensions = [1, 2]" in schedule
    assert 'copy_back_op = "linalg.copy"' in schedule
    assert "tile_sizes [1, 4, 8, 0]" in schedule
    assert "vector_sizes [1, 4, 8, 1]" in schedule


@pytest.mark.skipif(not _toolchain(), reason="riscv toolchain missing")
def test_113_by_113_attention_lowers_without_a_surviving_vector_mask():
    """The exact smolVLA tail: 113 is indivisible by both MR=4 and NR=8.

    ``lower_model_file`` includes translation to LLVM IR. It therefore fails if a
    multi-op ``vector.mask { vector.contract }`` reaches the translation boundary.
    """
    from merlin.llvmlower.lower import lower_model_file
    from merlin.mining import workloads
    from merlin.runtime.backends import zephyr_model as zm

    root = Path(tempfile.mkdtemp(prefix="test_bmm_tail_113_"))
    bundle = workloads.gen_batch_matmul_f32(root, B=1, M=113, N=113, K=8)
    prepare = root / "prepare"
    prepare.mkdir()
    prepared = zm._prepare_model_mlir(bundle / "model.mlir", prepare)
    lower = root / "lower"
    result = lower_model_file(
        prepared,
        lower,
        targets=("host",),
        textual=True,
        vectorize=True,
        hoist_static_allocs=False,
        features=frozenset({FEATURE}),
    )
    assert result.ll_path.is_file()
    llvm_ir = result.ll_path.read_text(encoding="utf-8")
    assert "vector.mask" not in llvm_ir
    assert "vector.contract" not in llvm_ir
    assert "@llvm.fmuladd.v8f32" in llvm_ir

    from merlin.llvmlower.abi import HostModel

    inputs = np.load(bundle / "inputs.npz")
    a = np.ascontiguousarray(inputs["in0"])
    b = np.ascontiguousarray(inputs["in1"])
    got = np.zeros((1, 113, 113), dtype=np.float32)
    HostModel.load(str(result.host_so), n_args=3)([
        (a.ctypes.data, list(a.shape)),
        (b.ctypes.data, list(b.shape)),
        (got.ctypes.data, list(got.shape)),
    ])
    np.testing.assert_allclose(got, np.load(bundle / "golden.npy"), rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not _toolchain(), reason="riscv toolchain missing")
def test_15_by_113_by_113_attention_survives_the_eight_hart_pipeline():
    """Exercise the exact text-attention batch and out-of-process OMP edge."""
    from merlin.llvmlower.lower import lower_model_file
    from merlin.mining import workloads
    from merlin.runtime.backends import zephyr_model as zm

    root = Path(tempfile.mkdtemp(prefix="test_bmm_tail_113_omp_"))
    bundle = workloads.gen_batch_matmul_f32(root, B=15, M=113, N=113, K=8)
    prepare = root / "prepare"
    prepare.mkdir()
    prepared = zm._prepare_model_mlir(bundle / "model.mlir", prepare)
    lower = root / "lower"
    result = lower_model_file(
        prepared,
        lower,
        targets=(),
        textual=True,
        vectorize=True,
        hoist_static_allocs=False,
        features=frozenset({FEATURE}),
        parallel_harts=8,
    )
    assert result.ll_path.is_file()
    dialect_edge = lower / "model.llvmdialect.mlir"
    assert dialect_edge.is_file()
    text = dialect_edge.read_text(encoding="utf-8")
    assert "vector.mask" not in text
    assert "vector.contract" not in text
    assert "@llvm.fmuladd.v8f32" in result.ll_path.read_text(encoding="utf-8")


@pytest.mark.skipif(not _toolchain(), reason="riscv toolchain missing")
def test_divisible_12_by_1024_attention_is_not_padded_and_still_lowers():
    """Regression for the first pad-all design's invalid allocation scope."""
    from merlin.llvmlower.lower import lower_model_file
    from merlin.mining import workloads
    from merlin.runtime.backends import zephyr_model as zm

    root = Path(tempfile.mkdtemp(prefix="test_bmm_full_1024_omp_"))
    bundle = workloads.gen_batch_matmul_f32(root, B=12, M=1024, N=1024, K=64)
    prepare = root / "prepare"
    prepare.mkdir()
    prepared = zm._prepare_model_mlir(bundle / "model.mlir", prepare)
    lower = root / "lower"
    result = lower_model_file(
        prepared,
        lower,
        targets=(),
        textual=True,
        vectorize=True,
        hoist_static_allocs=False,
        features=frozenset({FEATURE}),
        parallel_harts=8,
    )
    assert result.ll_path.is_file()
    dialect_edge = (lower / "model.llvmdialect.mlir").read_text(encoding="utf-8")
    assert "vector.mask" not in dialect_edge
    assert "vector.contract" not in dialect_edge
    assert "@llvm.fmuladd.v8f32" in result.ll_path.read_text(encoding="utf-8")
    assert "50331648" not in dialect_edge


@pytest.mark.skipif(not _toolchain(), reason="riscv toolchain missing")
def test_flow_denoise_n32_parallel_chunk_keeps_the_nr16_kernel():
    """N=32 must become two 16-wide tasks, not eight masked 4-wide ones."""
    from merlin.llvmlower.lower import lower_model_file
    from merlin.mining import workloads
    from merlin.runtime.backends import zephyr_model as zm

    root = Path(tempfile.mkdtemp(prefix="test_mm_flow_n32_omp_"))
    bundle = workloads.gen_matmul_f32(root, M=50, N=32, K=720)
    prepare = root / "prepare"
    prepare.mkdir()
    prepared = zm._prepare_model_mlir(bundle / "model.mlir", prepare)
    lower = root / "lower"
    result = lower_model_file(
        prepared,
        lower,
        targets=(),
        textual=True,
        vectorize=True,
        hoist_static_allocs=False,
        features=frozenset({FEATURE}),
        parallel_harts=8,
    )
    assert result.ll_path.is_file()
    dialect_edge = (lower / "model.llvmdialect.mlir").read_text(encoding="utf-8")
    assert "vector.mask" not in dialect_edge
    assert "vector.contract" not in dialect_edge
    llvm_ir = result.ll_path.read_text(encoding="utf-8")
    assert "@llvm.fmuladd.v16f32" in llvm_ir
    assert llvm_ir.count("__kmpc_fork_call") <= 2
    parallel_schedule = (lower / "rvv_parallel_schedule.mlir").read_text(encoding="utf-8")
    assert "tile_sizes [0, 16]" in parallel_schedule
    assert "tile_sizes [1, 0, 0]" in parallel_schedule
