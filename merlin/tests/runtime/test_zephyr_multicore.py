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

from merlin.common.artifacts import artifacts_dir, recaptures_dir
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


def test_scalar_multicore_parallelizes_at_the_linalg_loop_level():
    """Scalar multicore is SUPPORTED, and reaches multicore by a different route than rvv.

    This test used to assert the opposite -- that a scalar multi-hart build was refused because the
    multicore lowering layers its forall under the RVV transform schedule. That reasoning was right
    about the rvv route and wrong about the conclusion: the scalar path has its own OpenMP route
    (convert-linalg-to-parallel-loops + convert-scf-to-openmp, the pipeline the K1 big models use), so
    it only needed routing to rather than refusing.

    It matters because it is the ONLY way to use a hart with no vector unit, and a heterogeneous SoC --
    more cores brought up than vector units attached -- is normal. On a 3-core/2-vector-core chip that
    third core is a third of the machine. Measured on a 3-hart scalar image: 272 fork_call sites, 100+
    outlined parallel regions, zero vector instructions.
    """
    import inspect

    zm = _zm()
    src = inspect.getsource(zm.build_app)
    assert "requires backend='rvv'" not in src
    # The two routes, both wired: scalar via `parallel`, rvv via `parallel_harts`.
    assert 'parallel=(backend != "rvv" and n_harts > 1)' in src
    assert 'parallel_harts=(n_harts if n_harts > 1' in src


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
    from merlin.mining import k1
    from merlin.mining.registry import load_rvv_package

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
    from merlin.mining import workloads

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


# The conv / FFT / recurrent workloads (spectformer, whisper_tiny, lstmnetvit, deepjscc) reach the
# contractions by a different route than an LLM does -- conv arrives as im2col + matmul, an FFT as
# real DFT matmuls -- and their parallel extents are the ones that forced the register block to be
# re-resolved per op class. So the split has to be gated on one of THEM, not only on small_llama:
# a schedule that lowers for a transformer's shapes says nothing about an N=8 frequency-bin matmul.
# 3 harts is deliberate -- it is the Kodiak tapeout's working core count, and 3 divides almost
# nothing, which is exactly when a split that assumed divisibility would drop or double work.
NEW_WORKLOAD_SPLIT_CASES = [
    ("deepjscc_int8_full", 3),        # smallest: conv encoder/decoder, no batch_matmul at all
    ("spectformer_int8_full", 3),     # spectral: matmul N=8 (DFT bins) -> an ADAPTED register block
]


@slow
@pytest.mark.parametrize("bundle_name,n_harts", NEW_WORKLOAD_SPLIT_CASES)
def test_new_workload_multicore_output_is_bit_identical(bundle_name, n_harts, tmp_path):
    """N harts must agree with 1 hart bit-for-bit on a conv/spectral workload.

    Runs through the SHAPE-ADAPTED features, i.e. the same resolution ``merlin-compile`` performs, so
    this also gates that path end to end: spectformer does not even build with the package's pinned
    block (its N=8 matmul needs an adapted one), so a regression in the resolver shows up here as a
    build failure rather than as a wrong number much later.

    Bit-exactness is asserted unconditionally; the golden gate only when the 1-hart run already
    passes it. deepjscc currently carries an open host-vs-board accuracy divergence, and the split
    must be proven correct independently of it -- a wrong-but-identical output still proves the
    parallel decomposition touched nothing, which is the property under test here.
    """
    from merlin.mining.apply import shape_adapted_features
    from merlin.mining.registry import load_rvv_package

    zm = _zm()
    if not zm.available():
        pytest.skip("Zephyr/spike toolchain unavailable")
    b = recaptures_dir() / bundle_name
    if not (b / "model.mlir").is_file():
        pytest.skip(f"{bundle_name} not captured")
    pkg_dir = artifacts_dir() / "targets" / "rvv" / "impr_tuned_wholemodel_vf_int8"
    if not pkg_dir.is_dir():
        pytest.skip("int8 champion package not present")
    pkg = load_rvv_package(pkg_dir)
    feats = frozenset(shape_adapted_features(pkg, b))
    refs = _refs(b)
    common = dict(board="spike_riscv64", backend="rvv", rvv_hart=0, arena_mb=64,
                  int8_compute=True, references=refs, timeout=7200,
                  rvv_schedule=pkg.schedule_text,
                  cflags_override=pkg.cflags + zm._CFLAGS_COMMON, features=feats)

    base = zm.build_and_run(b, tmp_path / "h1", harts=2, n_harts=1, **common)
    multi = zm.build_and_run(b, tmp_path / f"h{n_harts}", harts=n_harts, n_harts=n_harts, **common)

    assert multi["metrics"].get("omp_threads") == n_harts, \
        f"pool clamped to {multi['metrics'].get('omp_threads')} instead of {n_harts}"
    a, c = base["outputs"], multi["outputs"]
    assert a.shape == c.shape, f"shape changed with harts: {a.shape} vs {c.shape}"
    assert np.array_equal(a, c), (
        f"{bundle_name}: {n_harts}-hart output differs from 1-hart by max "
        f"{np.abs(a - c).max()} — an overlapping or lost work split, NOT rounding "
        f"(the split touches only parallel dims)")
    if base.get("ok"):
        assert multi.get("ok"), (
            f"{bundle_name}: 1 hart passed the golden gate but {n_harts} harts did not, "
            f"despite identical output — the gate itself is unstable")
