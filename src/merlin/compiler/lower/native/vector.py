"""Generic upstream fixed-width vector schedule and native pass assembly.

The historical RVV-named API remains generic; target RVV lowering stays OOT."""

from __future__ import annotations

from pathlib import Path

from merlin.compiler.lower.llvm.passes import finalization_passes
from merlin.compiler.lower.native.policy import _fuse_post, _par_residue_parallel, _splice
from merlin.compiler.passes.contract import PipelineError
from merlin.compiler.passes.parallel import PARALLEL_ENTRY

RVV_TRANSFORM_SCHEDULE = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t, %l:3 = transform.structured.tile_using_for %mm tile_sizes [4, 8, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %t vector_sizes [4, 8, 1] : !transform.any_op
    %bm = transform.structured.match ops{["linalg.batch_matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt, %bl:4 = transform.structured.tile_using_for %bm tile_sizes [1, 4, 8, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt vector_sizes [1, 4, 8, 1] : !transform.any_op
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_contraction
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.lower_transpose
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op
    transform.yield
  }
}
"""


VEC_PRE_ENTRY = "__transform_vec_main"


_VEC_PRE_SKELETON = (
    """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @%s(%%arg0: !transform.any_op {transform.readonly}) {
    %%f = transform.structured.match ops{["func.func"]} in %%arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %%f {
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op
    transform.yield
  }
}
"""
    % VEC_PRE_ENTRY
)


def _vec_after_specialize() -> bool:
    """Rebuild the OLD placement of the non-contraction arms (``MERLIN_VEC_AFTER_SPECIALIZE=1``).

    An A/B escape hatch, for the reason :func:`_dealloc_passes` keeps ``MERLIN_NO_DEALLOC``: this
    changes the emitted code for every build that names the lever, and the honest way to defend a
    claim about it is to be able to rebuild both arms."""
    import os

    return bool(os.environ.get("MERLIN_VEC_AFTER_SPECIALIZE"))


def vec_pre_schedule(features: "frozenset[str]") -> "str | None":
    """Pipeline design rationale: see docs/design/llvm_pipeline_rationale.md."""
    from merlin.compiler.optimizations.pointwise import ensure_vec_noncontraction, vec_noncontraction_lanes
    from merlin.compiler.optimizations.registry import apply_schedule

    if _vec_after_specialize():
        return None
    lanes = vec_noncontraction_lanes(features)
    if lanes is None:
        return None
    text = apply_schedule(_VEC_PRE_SKELETON, frozenset({ensure_vec_noncontraction(lanes)}))
    if text == _VEC_PRE_SKELETON:
        # The arms did not splice. Fail closed rather than preload a library whose entry point does
        # nothing and report the lever as placed: that is the "enabled and changed nothing" failure
        # this pipeline keeps re-learning.
        raise PipelineError(
            f"the non-contraction vectorize lever is enabled at {lanes} lanes but no arms spliced "
            f"into the pre-specialization library skeleton; the anchor "
            f'`%f = transform.structured.match ops{{["func.func"]}}` it keys on is gone'
        )
    return text


def build_rvv_pipeline(
    sched_path: "str | Path",
    hoist_static_allocs: bool = True,
    features: "frozenset[str]" = frozenset(),
    par_sched_path: "str | Path | None" = None,
    vec_sched_path: "str | Path | None" = None,
    perop_parallel: bool = False,
) -> str:
    """Whole-module pipeline with the transform vectorization stage spliced in after
    named-op generalization (vectorize on tensors) and before bufferization, plus the
    vector-lowering passes needed to reach LLVM. ``sched_path`` is the preloaded schedule.

    ``par_sched_path`` (default None -> byte-identical to the shipping serial pipeline)
    enables the multicore variant: the parallel library is preloaded alongside the package
    schedule, its entry point runs FIRST (wrapping each contraction in an ``scf.forall``),
    and the loop-generation/LLVM stages gain the forall->parallel->OpenMP conversions.

    ``vec_sched_path`` (default None -> byte-identical to the shipping pipeline) preloads the
    non-contraction vectorize library alongside the package schedule and runs its entry point BEFORE
    ``linalg-specialize-generic-ops``, which is the pass that drops the ``merlin.vec_r{rank}`` tags
    those arms match on. See :func:`vec_pre_schedule`.

    ``hoist_static_allocs=False`` drops the ``hoist-static-allocs`` option of
    buffer-results-to-out-params so static intermediate buffers stay HEAP (memref.alloc) instead
    of being promoted to stack ``alloca``. Needed for the K1 Linux target: big models' lowered
    intermediates otherwise overflow even a multi-GB stack; on the heap they use the board RAM.
    The bare-metal spike/Zephyr targets keep hoisting (default True) — their big linker-reserved
    stack tolerates it and their malloc arena is bounded."""
    brop = (
        "buffer-results-to-out-params{modify-public-functions hoist-static-allocs}"
        if hoist_static_allocs
        else "buffer-results-to-out-params{modify-public-functions}"
    )
    # Multicore: preload the parallel library ALONGSIDE the package schedule (the option is a
    # list) and run its entry point first, so each contraction is already wrapped in an
    # `scf.forall` when the package's `__transform_main` matches and vectorizes it.
    par = par_sched_path is not None
    libs = [str(p) for p in (vec_sched_path, par_sched_path, sched_path) if p is not None]
    preload = f"transform-preload-library{{transform-library-paths={','.join(libs)}}}"
    # THE NON-CONTRACTION ARMS RUN FIRST, before specialization eats their tags. `preload` moves up
    # with them (it only loads libraries; it touches no IR), and specialization then runs on the IR
    # those arms left behind, so the CONTRACTION arms below still see a fully specialized module and
    # run exactly once. With the lever off `vec_sched_path` is None and this whole block is absent --
    # the pass string, and the .ll, are byte-identical to the baseline. See :func:`vec_pre_schedule`.
    vec = vec_sched_path is not None
    specialize = [
        # Recover named contraction ops (matmul/batch_matmul) from the capture's generics so
        # the schedule can match them, THEN vectorize. Do NOT run linalg-fuse-elementwise-ops
        # before this — it folds matmuls into fused generics and the `ops{["linalg.matmul"]}`
        # match then finds nothing (silent 0-vectorization).
        "func.func(linalg-specialize-generic-ops)"
    ]
    head = (
        [preload, f"transform-interpreter{{entry-point={VEC_PRE_ENTRY}}}", "canonicalize", "cse", *specialize]
        if vec
        else [*specialize, preload]
    )
    passes = [
        "canonicalize",
        "cse",
        *head,
        *([f"transform-interpreter{{entry-point={PARALLEL_ENTRY}}}", "canonicalize", "cse"] if par else []),
        "transform-interpreter{entry-point=__transform_main}",
        "canonicalize",
        "cse",
        # POST-matmul elementwise fusion + broadcast folding. Runs HERE, after the schedule has already
        # matched and vectorized the contractions, because fusing earlier folds matmuls into generics and
        # `ops{["linalg.matmul"]}` then matches nothing (silent 0-vectorization).
        #
        # WHY THIS IS THE LOAD-BEARING STAGE, measured on spectformer int8:
        #   * `linalg-specialize-generic-ops` above recovers the contraction NAMES, but it also un-fuses
        #     every elementwise generic: 0 -> 567 `linalg.broadcast` and 1569 -> 1952 `tensor.empty`.
        #     Those broadcasts are 23.8% of on-chip runtime and pure byte traffic -- a 256-element scale
        #     materialized into all 196,608 positions of a 256x768 temporary just to be divided by.
        #   * the model writes 977 MB per inference, 761 MB of it f32, so its arithmetic intensity as
        #     compiled is 1.18 MAC/byte against a machine that balances at 32 -- memory bound by a factor
        #     of 27, on a workload whose own reuse would put it at 115 MAC/byte.
        #   * running these passes on the tagged IR takes broadcast 567 -> 245 and tensor.empty 1952 -> 68.
        #
        # `linalg-fuse-elementwise-ops` collapses the producer->consumer chains so each line is touched
        # once instead of once per op in the chain, and the canonicalize/cse after it is NOT incidental:
        # measured on the tagged IR, fuse alone gives broadcast 277 / empty 1415, and fuse+canonicalize+cse
        # gives 245 / 68 -- most of the temporary collapse is the cleanup, not the fusion.
        #
        # `linalg-fold-into-elementwise` is deliberately NOT here. It reads like exactly the right pass
        # ("fold transpose and broadcast ops into elementwise") and it is MEASURED INERT on this IR:
        # fold+fuse+canonicalize+cse is identical to fuse+canonicalize+cse (245/718/68 either way), and
        # adding it to this stage changed the built image by one instruction. It matches `linalg.elementwise`
        # CATEGORY ops, while this IR carries `linalg.generic`; reaching it would mean running
        # `linalg-named-to-category` first. Left out rather than left in as decoration.
        *(["func.func(linalg-fuse-elementwise-ops)", "canonicalize", "cse"] if _fuse_post() else []),
        "func.func(linalg-generalize-named-ops)",  # remaining (non-vectorized) ops -> loops
        "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
        brop,
        "func.func(buffer-hoisting,buffer-loop-hoisting)",
        "func.func(lower-vector-multi-reduction)",
        "func.func(lower-vector-mask)",
        "convert-vector-to-scf",
        # ``convert-vector-to-scf`` materializes fixed-size ``memref.alloca`` scratch for
        # multi-dimensional vector transfers.  The first hoisting stage above cannot see those
        # allocations because they do not exist yet.  Left in a contraction body, LLVM lowers an
        # alloca on every reduction-loop iteration and the stack grows until ``forward`` returns
        # (observed on a 1x2048 by 2048x1000 int8 contraction: an 8 MiB host stack faults even
        # though the static frame is only a few KiB).  Hoist the newly-created scratch while the
        # loop structure is still SCF; this reuses one scratch object across sequential iterations
        # and does not alter the transform schedule or vector shape.
        # Do not run this hoist inside the parallel route: buffer-hoisting wraps the scratch in
        # ``memref.alloca_scope``, and the subsequent SCF/OpenMP-to-CF lowering gives that scope
        # multiple blocks, which is verifier-invalid. The K1 parallel path therefore keeps the
        # scratch local until a scope-safe late-hoist exists; the serial path retains stack reuse.
        *(["func.func(buffer-hoisting,buffer-loop-hoisting)"] if not par else []),
        # Deallocation goes HERE, not straight after bufferization as in the scalar pipelines.
        # `ownership-based-buffer-deallocation` materializes an i1 ownership constant next to each
        # buffer use, and it walks into `vector.mask` regions -- which accept exactly one masked
        # operation, so the extra constant makes the region fail its own verifier. Running after
        # `lower-vector-mask` (and `convert-vector-to-scf`, which is where the masks actually go
        # away) leaves no such region to walk into. Still before the scf -> cf / openmp conversions,
        # because ownership analysis wants structured control flow.
        "__DEALLOC__",
        # Multicore: the outer `scf.forall` becomes `scf.parallel` and `convert-scf-to-openmp`
        # wraps every `scf.parallel` in `omp.parallel` + `omp.wsloop`. Reduction loops are NOT
        # touched (they stay `scf.for`), so the accumulator is never raced.
        #
        # WHAT HAPPENS TO THE RESIDUE -- the ops the transform schedule did not claim (norms,
        # activations, gathers, every elementwise generic) -- is a SEPARATE decision, and it is the
        # second way the 1-hart and 8-hart images stopped being the same kernel. Sending the residue
        # region, and an outlined OpenMP region is not a loop nest LLVM's own vectorizer will touch:
        # MEASURED on lstmnetvit int8 in the LINKED ELF, with the block table already held identical
        # across the two arms, `vle32.v` 1825 -> 951 and `vlse32.v` 652 -> 27 while total issued
        # instructions went 59,752 -> 98,059. The residue was not parallelized so much as
        # devectorized, once per op, and it is the same conversion that produces the thousands of
        # fork/join events `parallel_grain` exists to suppress.
        #
        # So when the split is the DERIVED per-op one (`perop_parallel`), parallelism is expressed by
        # the `scf.forall` arms merlin chose and by nothing else, and the residue keeps the serial
        # `convert-linalg-to-loops` the 1-hart build gives it. `MERLIN_PAR_RESIDUE_PARALLEL=1`
        # rebuilds the old arm; the named `parallelize_residual_loops_<grain>` family selects the
        # same lowering together with a mandatory small-region cutoff. With the legacy class-wide
        # split (no `chunks`) nothing moves -- that pipeline string is byte-identical to before.
        *(
            [
                "scf-forall-to-parallel",
                *(
                    ["func.func(convert-linalg-to-loops)"]
                    if perop_parallel and not _par_residue_parallel(features)
                    else ["func.func(convert-linalg-to-parallel-loops)"]
                ),
                "convert-scf-to-openmp",
                "canonicalize",
            ]
            if par
            else ["func.func(convert-linalg-to-loops)"]
        ),  # fallback: any op the vectorizer skipped
        "convert-scf-to-cf",
        "expand-strided-metadata",
        "lower-affine",
        *finalization_passes(vector=True, openmp=par),
    ]
    if features:  # default-off impr-fork hooks; empty -> byte-identical
        from merlin.compiler.optimizations.registry import apply_pipeline

        passes = apply_pipeline(passes, features)
    return _splice(passes)
