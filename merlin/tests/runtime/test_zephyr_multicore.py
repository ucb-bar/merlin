"""Multicore RVV whole-model on Zephyr SMP — the bit-exactness gate.

The multicore image fans each contraction across pinned harts via the OpenMP shim
(``merlin/runtime/c/libomp_zephyr.c``). Two things can go wrong SILENTLY there, and both
produce a plausible-looking run rather than a crash:

  * the hart -> OpenMP-thread-id mapping is wrong, so two harts claim the same output slice
    (the emitted IR feeds ``__kmpc_global_thread_num()`` to the worksharing loop, not the
    outlined region's tid argument), and
  * the Saturn fork's ``arch/riscv/core/v.c`` corrupts vector state across a context switch —
    the documented bug that made yolov8n hash two different wrong values in
    ``samples/merlin_model_runner/prj.conf``.

So the gate here is not "the multicore run is close to the golden" but **1 hart and N harts
must produce BIT-IDENTICAL output**. The lowered dataflow is deterministic and the static
schedule splits only parallel dims (reductions stay serial), so any difference at all is
corruption, not rounding.

Gated on the Zephyr+spike toolchain and ``MERLIN_RUN_SLOW`` (each parametrization lowers a
model and builds a Zephyr image).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from merlin.common.artifacts import recaptures_dir
from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

BUNDLE = "small_llama_int8_consistent"


def _zm():
    from merlin.runtime.backends import zephyr_model
    return zephyr_model


def _bundle():
    b = recaptures_dir() / BUNDLE
    if not (b / "model.mlir").is_file():
        pytest.skip(f"{BUNDLE} not captured")
    return b


def _refs(b):
    refs = {"fp32": np.load(b / "golden.npy")}
    w8 = b / "golden_w8a8.npy"
    if w8.is_file():
        refs["w8a8"] = np.load(w8)
    return refs


slow = pytest.mark.skipif(not os.environ.get("MERLIN_RUN_SLOW"),
                          reason="set MERLIN_RUN_SLOW=1 (builds Zephyr images + runs spike)")


def test_single_hart_image_carries_no_openmp():
    """n_harts=1 must stay byte-identical to the shipping single-worker image."""
    zm = _zm()
    src = zm._main_c(0, n_harts=1)
    assert "libomp_zephyr.h" not in src
    assert "merlin_omp_init" not in src
    cmake = zm._cmakelists(*(4 * [__import__("pathlib").Path("/x")]), omp=False)
    assert "libomp_zephyr.c" not in cmake


def test_multicore_image_links_the_shim_and_sizes_the_soc():
    zm = _zm()
    src = zm._main_c(0, n_harts=4)
    assert "merlin_omp_init(4)" in src
    cmake = zm._cmakelists(*(4 * [__import__("pathlib").Path("/x")]), omp=True)
    assert "libomp_zephyr.c" in cmake
    # the DT overlay must enable exactly the harts the SoC has: the 2-tile sample overlay
    # hard-disabled cpu@2..7 and silently capped every image at 2 harts.
    ov = zm._chipyard_cpu_overlay(4)
    assert 'cpu@4 { status = "disabled"; }' in ov
    assert "cpu@3" not in ov and "cpu@1" not in ov


def test_multicore_requires_the_rvv_backend(tmp_path):
    """The forall is layered under the RVV schedule; scalar would build a serial image."""
    zm = _zm()
    if not zm.available():
        pytest.skip("Zephyr/spike toolchain unavailable")
    with pytest.raises(zm.ZephyrModelError, match="requires backend='rvv'"):
        zm.build_app(_bundle(), tmp_path, backend="scalar", n_harts=4)


@pytest.mark.parametrize("threads", [2, 4, 8])
def test_multicore_is_bit_identical_on_the_k1_board(threads, tmp_path):
    """THE multicore correctness gate, on real silicon.

    Complements the spike gates rather than replacing them: this one exercises the LOWERING
    against the real cross-built libomp on shipping silicon, and the board does the comparison
    in milliseconds. The board never runs ``libomp_zephyr.c``, so it cannot gate the shim —
    only the Zephyr legs can.

    One binary, OMP_NUM_THREADS varied — so the only difference between the compared runs is
    how many cores executed, not what was compiled.
    """
    from merlin.compile_cli import default_package
    from merlin.rvvgen import k1
    from merlin.rvvgen.registry import load_rvv_package

    if not k1.available():
        pytest.skip("K1 board unreachable")
    b = _bundle()
    pkg = load_rvv_package(default_package("int8"))
    binary = k1.build_k1_binary(b, tmp_path, pkg, inputs_npz=b / "inputs.npz",
                                parallel_harts=8)

    def _run(t):
        return k1.run_binary_on_k1(b, tmp_path, pkg, binary, timeout=1800,
                                   env={"OMP_NUM_THREADS": str(t), "OMP_PROC_BIND": "spread"})

    base = _run(1)["outputs"]
    got = _run(threads)["outputs"]
    assert got.shape == base.shape, f"shape changed with threads: {base.shape} -> {got.shape}"
    ndiff = int((got != base).sum())
    assert ndiff == 0, (
        f"{threads} threads differ from 1 thread in {ndiff}/{base.size} elements — an "
        f"overlapping work split or vector-state corruption, NOT rounding (the split touches "
        f"only parallel dims; reductions stay serial, so there is no reordering)")


@pytest.mark.parametrize("n_harts", [2, 4])
def test_zephyr_openmp_shim_is_bit_identical_on_a_kernel(n_harts, tmp_path):
    """The gate for the ZEPHYR SHIM itself (merlin/runtime/c/libomp_zephyr.c).

    Distinct from the K1 test above, which exercises the LOWERING against the real cross-built
    libomp — the board never runs this shim. Only Zephyr does, so only a Zephyr run can gate it.

    Uses a KERNEL-sized synthesized bundle: it is the fast smoke test for the split itself.

    It is NOT sufficient on its own, and the reason is worth keeping. This gate was once the
    only Zephyr-side one, justified by "a whole model costs 25+ minutes at 2 harts". That cost
    was itself the bug — the shim hung on nested parallel regions and then drowned in stall
    printks — and a kernel with a single contraction has no nested regions, so narrowing the
    gate to fit the symptom is exactly why the defect survived. A whole model at 2 harts now
    takes ~12 s, and ``test_multicore_output_is_bit_identical_to_single_hart`` below runs it.
    """
    zm = _zm()
    if not zm.available():
        pytest.skip("Zephyr/spike toolchain unavailable")
    from merlin.rvvgen import workloads

    # N=256 splits cleanly over 2 and 4 harts and stays a multiple of the NR=8 micro-kernel tile.
    b = workloads.gen_matmul_f32(tmp_path / "bundle", M=64, N=256, K=64, seed=0)
    golden = np.load(b / "golden.npy")

    base = zm.build_and_run(b, tmp_path / "h1", board="spike_riscv64", backend="rvv",
                            rvv_hart=0, harts=2, arena_mb=32, n_harts=1,
                            reference=golden, timeout=1800)
    assert base.get("ok"), f"single-hart reference failed its own gate: cos={base.get('cos')}"

    multi = zm.build_and_run(b, tmp_path / f"h{n_harts}", board="spike_riscv64", backend="rvv",
                             rvv_hart=0, harts=n_harts, arena_mb=32, n_harts=n_harts,
                             reference=golden, timeout=1800)
    assert multi["metrics"].get("omp_threads") == n_harts, (
        f"pool clamped to {multi['metrics'].get('omp_threads')} instead of {n_harts}")
    assert multi.get("ok")
    a, c = base["outputs"], multi["outputs"]
    assert np.array_equal(a, c), (
        f"{n_harts}-hart output differs from 1-hart in {(a != c).sum()} elements — the shim's "
        f"work split or its vector-state handling is wrong; both corrupt silently")


@pytest.mark.parametrize("n_harts", [2, 4])
def test_multicore_output_is_bit_identical_to_single_hart(n_harts, tmp_path):
    """THE gate: N harts must agree with 1 hart bit-for-bit, and both must pass the golden.

    WHOLE MODEL on purpose, and deliberately not ``@slow``. Both shim defects fixed in
    bf5052db needed a whole model to appear at all — nested parallel regions exist only in a
    multi-op module, and the worker-stack overflow needed 143 regions of pressure — so a gate
    that only ever ran a single kernel could not have caught either. Measured after the fix:
    ~12 s per parametrization on spike, which is cheap enough to run every time.
    """
    zm = _zm()
    if not zm.available():
        pytest.skip("Zephyr/spike toolchain unavailable")
    b = _bundle()
    refs = _refs(b)

    base = zm.build_and_run(b, tmp_path / "h1", board="spike_riscv64", backend="rvv",
                            rvv_hart=0, harts=2, arena_mb=64, int8_compute=True,
                            n_harts=1, references=refs, timeout=3600)
    assert base.get("ok"), f"single-hart reference failed its own gate: cos={base.get('cos')}"

    multi = zm.build_and_run(b, tmp_path / f"h{n_harts}", board="spike_riscv64", backend="rvv",
                             rvv_hart=0, harts=n_harts, arena_mb=64, int8_compute=True,
                             n_harts=n_harts, references=refs, timeout=3600)
    assert multi["metrics"].get("omp_threads") == n_harts, \
        f"pool clamped to {multi['metrics'].get('omp_threads')} instead of {n_harts}"

    a, c = base["outputs"], multi["outputs"]
    assert a.shape == c.shape, f"shape changed with harts: {a.shape} vs {c.shape}"
    assert np.array_equal(a, c), (
        f"{n_harts}-hart output differs from 1-hart by max "
        f"{np.abs(a - c).max()} — vector-state corruption or an overlapping work split, "
        f"NOT rounding (the split touches only parallel dims)")
    assert multi.get("ok")
