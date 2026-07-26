"""Multicore RVV: the composed vectorize + OpenMP lowering.

The whole point of ``parallel_harts`` is that vectorization and threading COMPOSE — before
this, ``lower_to_llvm_ir`` picked ``build_rvv_pipeline`` OR ``_parallel_pipeline``, so a build
was either vectorized or multicore and never both. These tests pin the three properties that
make the composition trustworthy:

1. **Default is untouched.** ``par_sched_path=None`` yields the byte-identical shipping
   pipeline string, so every existing package/fork keeps its exact codegen.
2. **The forall carries the parallelism.** This LLVM-23 build has no scf.for->scf.parallel
   pass, so the outer tiling MUST be ``tile_using_forall``; the schedule is checked to tile
   only PARALLEL dims (never the K reduction dim, which would race the accumulator).
3. **Both survive to LLVM IR.** The end-to-end check compiles a matmul through the real
   pipeline and asserts the ``.ll`` carries ``__kmpc_*`` calls AND vector types — the actual
   claim, rather than an inspection of the pass-list string.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from merlin.common.paths import repo_root
from merlin.llvmlower import pipeline as P

MLIR_OPT = repo_root() / "third_party" / "llvm-install" / "bin" / "mlir-opt"
MLIR_TRANSLATE = repo_root() / "third_party" / "llvm-install" / "bin" / "mlir-translate"


def test_default_pipeline_is_byte_identical():
    """No par_sched_path -> the exact shipping serial pipeline (no silent perturbation)."""
    serial = P.build_rvv_pipeline("/tmp/sched.mlir")
    assert P.build_rvv_pipeline("/tmp/sched.mlir", par_sched_path=None) == serial
    for tok in ("scf-forall-to-parallel", "convert-scf-to-openmp", "convert-openmp-to-llvm",
                P.PARALLEL_ENTRY):
        assert tok not in serial, f"{tok} leaked into the serial pipeline"
    assert "func.func(convert-linalg-to-loops)" in serial


def test_parallel_pipeline_adds_the_openmp_stages_and_preloads_both_libraries():
    par = P.build_rvv_pipeline("/tmp/sched.mlir", par_sched_path="/tmp/par.mlir")
    # both libraries merged into ONE preload (the option is a list), parallel entry runs first
    assert "transform-library-paths=/tmp/par.mlir,/tmp/sched.mlir" in par
    # the parallel entry must run BEFORE the package schedule, so the package's match sees
    # the contraction already wrapped in the forall
    assert (par.index(f"transform-interpreter{{entry-point={P.PARALLEL_ENTRY}}}")
            < par.index("transform-interpreter{entry-point=__transform_main}"))
    for tok in ("scf-forall-to-parallel", "convert-scf-to-openmp", "convert-openmp-to-llvm"):
        assert tok in par
    # the serial fallback is REPLACED, not duplicated, or the ops would lower twice
    assert "func.func(convert-linalg-to-loops)" not in par
    assert "func.func(convert-linalg-to-parallel-loops)" in par


def test_parallel_schedule_never_tiles_the_reduction_dim():
    """K is the reduction dim: splitting it across harts races the accumulator."""
    sched = P.parallel_transform_schedule(4)
    assert P.PARALLEL_ENTRY in sched
    assert "tile_using_forall" in sched
    # matmul (M,N,K) defaults to N -> [0, 4]; batch_matmul (B,M,N,K) defaults to B -> [4].
    assert "num_threads [0, 4]" in sched
    assert "num_threads [4]" in sched
    # a 3-entry list on matmul would reach K; the defaults never emit one
    assert "num_threads [0, 0, 4]" not in sched
    with pytest.raises(ValueError):
        P.parallel_transform_schedule(4, matmul_dim="k")
    with pytest.raises(ValueError):
        P.parallel_transform_schedule(1)      # 1 hart is not a parallel build


def test_parallel_harts_requires_vectorize():
    """The outer forall is layered UNDER the RVV schedule; without it there is nothing to layer."""
    with pytest.raises(P.PipelineError, match="requires vectorize"):
        P.lower_to_llvm_ir("module {}", vectorize=False, parallel_harts=4)


@pytest.mark.skipif(not MLIR_OPT.is_file() or not MLIR_TRANSLATE.is_file(),
                    reason="third_party/llvm-install MLIR tools not built")
def test_composed_lowering_emits_both_openmp_and_vectors(tmp_path):
    """End-to-end: a matmul through the composed pipeline yields __kmpc_* AND vector types.

    Runs the pipeline with the standalone LLVM-23 tools (not the m2m venv) so the test is a
    plain toolchain check with no model bundle or board needed.
    """
    (tmp_path / "mm.mlir").write_text("""
func.func @forward(%A: tensor<64x2048xf32>, %B: tensor<2048x512xf32>,
                   %C: tensor<64x512xf32>) -> tensor<64x512xf32> {
  %0 = linalg.matmul ins(%A, %B: tensor<64x2048xf32>, tensor<2048x512xf32>)
                     outs(%C: tensor<64x512xf32>) -> tensor<64x512xf32>
  return %0 : tensor<64x512xf32>
}
""")
    sched = tmp_path / "sched.mlir"
    sched.write_text(P.RVV_TRANSFORM_SCHEDULE)
    par = tmp_path / "par.mlir"
    par.write_text(P.parallel_transform_schedule(4))

    pipe = P.build_rvv_pipeline(sched, par_sched_path=par)
    out = tmp_path / "out.mlir"
    proc = subprocess.run(
        [str(MLIR_OPT), str(tmp_path / "mm.mlir"),
         f"--pass-pipeline=builtin.module({pipe})", "-o", str(out)],
        capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"composed pipeline failed:\n{proc.stderr[-3000:]}"

    ll = tmp_path / "out.ll"
    tproc = subprocess.run([str(MLIR_TRANSLATE), "--mlir-to-llvmir", str(out), "-o", str(ll)],
                           capture_output=True, text=True, timeout=600)
    assert tproc.returncode == 0, f"translate failed:\n{tproc.stderr[-3000:]}"
    text = ll.read_text()

    # multicore: the OpenMP runtime calls the Zephyr shim (merlin/runtime/c/libomp_zephyr.c)
    # implements. This assertion is what pins that shim's required symbol surface.
    assert "@__kmpc_fork_call" in text, "no OpenMP fork -> the build is not multicore"
    assert "@__kmpc_for_static_init" in text, "no worksharing loop -> harts would duplicate work"
    # ...and the RVV vectorization survived being wrapped in the forall
    assert "x float>" in text, "no vector types -> the forall clobbered the RVV schedule"


@pytest.mark.skipif(not MLIR_OPT.is_file(), reason="third_party/llvm-install not built")
def test_llvm23_has_no_scf_for_to_parallel():
    """Pins WHY the outer tiling must be `tile_using_forall`.

    If a future LLVM gains an scf.for->scf.parallel pass this fails, and the simpler
    'parallelize the existing tile loop' design becomes available.
    """
    help_text = subprocess.run([str(MLIR_OPT), "--help"], capture_output=True, text=True,
                               timeout=120).stdout
    assert "--scf-forall-to-parallel" in help_text
    assert "--scf-for-to-parallel" not in help_text
