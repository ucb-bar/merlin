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


# --- The DERIVED per-op split -----------------------------------------------------------------
#
# The class-wide `num_threads` split above is correct but it made the block a function of the hart
# count, so the 8-hart and the 1-hart image were not the same kernel and no scaling number taken
# across them was about threads. Measured on lstmnetvit int8 at 8 harts, in the LINKED ELF: 5 of 37
# matmuls fell out of the block table to scalar loops, 9 more were narrowed, 59,752 -> 97,701 issued
# instructions and 13,767 -> 5,144 vector ops. These tests pin the replacement.

def _leading_op(line: str) -> str:
    """The op name a printed MLIR line starts, read structurally: strip one leading result list
    (``%x = ``) and take the first token. A generic contains substrings of every op name it wraps,
    so a substring test would count nested ops in the enclosing op's line."""
    s = line.strip()
    if " = " in s:
        s = s.split(" = ", 1)[1]
    return s.split()[0] if s.split() else ""


def test_the_derived_split_leaves_the_schedules_own_output_unchanged(tmp_path):
    """THE CLAIM, checked on the IR the package schedule produces rather than on a pass list.

    Same module, same package schedule, 1 hart vs 8: every vector op count must match and the only
    difference is the ``scf.forall`` wrapper. This is the metric that survives inlining -- a static
    count off the object cannot distinguish "devectorized" from "outlined".
    """
    if not MLIR_OPT.is_file():
        pytest.skip("third_party/llvm-install MLIR tools not built")
    from merlin.llvmlower import perop_blocks as pb

    class S:
        op, parallel, reduction = "linalg.matmul", (64, 512), (2048,)

    (tmp_path / "mm.mlir").write_text("""
func.func @forward(%A: tensor<64x2048xf32>, %B: tensor<2048x512xf32>,
                   %C: tensor<64x512xf32>) -> tensor<64x512xf32> {
  %0 = linalg.matmul {merlin.blk_mm_4x16, merlin.par_mm_0_64}
       ins(%A, %B: tensor<64x2048xf32>, tensor<2048x512xf32>)
       outs(%C: tensor<64x512xf32>) -> tensor<64x512xf32>
  return %0 : tensor<64x512xf32>
}
""")
    table = pb.block_table([S()], mr_cap=4, nr_cap=16)
    assert table == {pb.shape_key("linalg.matmul", (64, 512), (2048,)): (4, 16)}
    par = pb.parallel_chunk_table([S()], table, 8)
    arms = pb.distinct_parallel_arms(par)
    assert arms == [("linalg.matmul", (0, 64))], arms

    sched = tmp_path / "sched.mlir"
    sched.write_text(pb.schedule_text(table, 16))

    def _counts(par_path):
        pipe = P.build_rvv_pipeline(sched, par_sched_path=par_path,
                                    perop_parallel=par_path is not None)
        anchor = "transform-interpreter{entry-point=__transform_main},canonicalize,cse"
        prefix = pipe[:pipe.index(anchor) + len(anchor)]
        prefix += ")" * (prefix.count("(") - prefix.count(")"))
        proc = subprocess.run(
            [str(MLIR_OPT), str(tmp_path / "mm.mlir"),
             f"--pass-pipeline=builtin.module({prefix})"],
            capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, proc.stderr[-3000:]
        got = {}
        for tok in ("vector.contract", "vector.transfer_read", "vector.transfer_write",
                    "vector.mask", "linalg.matmul", "scf.forall"):
            got[tok] = sum(1 for l in proc.stdout.splitlines()
                           if _leading_op(l) == tok)
        return got

    parp = tmp_path / "par.mlir"
    parp.write_text(P.parallel_transform_schedule(8, chunks=arms))
    one, eight = _counts(None), _counts(parp)
    assert eight["scf.forall"] == 1, "the parallel wrapper must be there"
    assert one["scf.forall"] == 0
    for tok in ("vector.contract", "vector.transfer_read", "vector.transfer_write",
                "vector.mask", "linalg.matmul"):
        assert one[tok] == eight[tok], (
            f"{tok}: {one[tok]} at 1 hart vs {eight[tok]} at 8 -- the arms are not the same kernel")
    assert eight["vector.mask"] == 0 and eight["linalg.matmul"] == 0


def test_the_derived_split_keeps_the_residue_serial():
    """Everything the schedule did not claim keeps the 1-hart ``convert-linalg-to-loops``.

    Sending it through ``convert-linalg-to-parallel-loops`` makes each leftover op its own outlined
    OpenMP region, and LLVM does not vectorize those: measured on lstmnetvit int8 with the block
    table ALREADY held identical across the arms, ``vle32.v`` 1825 -> 951 and ``vlse32.v`` 652 -> 27.
    So under the derived split the parallelism is the forall arms and nothing else.
    """
    perop = P.build_rvv_pipeline("/tmp/s.mlir", par_sched_path="/tmp/p.mlir", perop_parallel=True)
    assert "func.func(convert-linalg-to-loops)" in perop
    assert "func.func(convert-linalg-to-parallel-loops)" not in perop
    assert "scf-forall-to-parallel" in perop and "convert-scf-to-openmp" in perop
    # the legacy class-wide split is untouched, so no existing multicore build moves
    legacy = P.build_rvv_pipeline("/tmp/s.mlir", par_sched_path="/tmp/p.mlir")
    assert "func.func(convert-linalg-to-parallel-loops)" in legacy


def test_an_empty_split_still_produces_a_valid_schedule():
    """A hart count with nothing splittable must yield a well-formed (empty) library, not a crash
    and not a silently omitted entry point -- the interpreter would then fail the whole build."""
    sched = P.parallel_transform_schedule(8, chunks=[])
    assert P.PARALLEL_ENTRY in sched and "tile_using_forall" not in sched
    assert sched.count("{") == sched.count("}")


def test_harts_reaches_the_compile_only_build():
    """`merlin-compile --run none --harts 8` used to build a SINGLE-CORE binary and report success:
    `--harts` was plumbed only to the run-on-hardware routes. A compile-only multicore A/B then
    compared two identical images."""
    import inspect

    from merlin import compile_cli

    src = inspect.getsource(compile_cli.compile_rvv)
    head = src[:src.index('if run == "k1":')]
    assert head.count("parallel_harts=(harts if harts > 1 else None)") >= 1, (
        "the compile-only build must receive the hart count")
