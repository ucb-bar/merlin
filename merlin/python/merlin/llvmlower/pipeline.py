"""Upstream MLIR lowering + LLVM IR translation, run inside the model2MLIR venv.

The torch-mlir wheel carries the full upstream pass registry (LLVM 23) and the
``translate_module_to_llvmir`` bridge; the lowering executes via a subprocess
in that venv.

Pipeline (whole module, in one shot):
linalg-on-tensors → (fuse elementwise, generalize) → one-shot-bufferize
→ buffer-results-to-out-params{modify-public-functions hoist-static-allocs}
→ convert-linalg-to-loops → scf->cf → llvm-dialect → translate -> .ll

NB: ``buffer-results-to-out-params`` MUST run with ``modify-public-functions`` —
the default silently skips public functions and the entry retains a returned
memref descriptor (heap-allocated). Discovered the hard way; covered by tests.

Vectorization strategy: rely on LLVM's RISC-V auto-vectorizer (clang -O2 -march=rv64gcv)
to vectorize scalar loops — verified to emit vsetvli/RVV instructions. A scalable
vector path (linalg tile→vectorize at vscale) can be layered later without changing
this driver's interface.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .toolchain import m2m_python


def _dealloc_passes(sink: bool | None = None) -> list[str]:
    """Free the buffers ``one-shot-bufferize`` allocates.

    Bufferization turns every ``tensor.empty`` into a ``memref.alloc``; it does NOT insert the
    matching frees — that is a separate pass, and it was missing from all three pipelines here. The
    lowered whole-model ``forward`` therefore called ``malloc`` once per intermediate and ``free``
    never, so a model's heap demand was the SUM of every temporary it ever materialized rather than
    the peak that is live at once. Measured on whisper_tiny: 678 ``@malloc`` calls, 0 ``@free``.

    Small models absorbed it (deepjscc and spectformer fit their cumulative total in the region we
    size for them, which is why they pass). whisper does not: its encoder attention materializes
    6x1500x1500 tensors, i.e. 54 MB apiece, and the run dies partway through the first attention
    block when a 54 MB ``malloc`` returns NULL and the generated code stores through it — observed
    on FireSim as ``mcause=7 Store/AMO access fault`` at op 337 with the destination register zero.
    No region size fixes that on a 512 MB board; the allocations had to stop leaking.

    These are upstream's own deallocation passes — ownership-based deallocation, then the
    simplification and lowering passes that turn ``bufferization.dealloc`` into ``memref.dealloc`` —
    inserted immediately after bufferization as its documentation prescribes.

    Upstream's composite ``buffer-deallocation-pipeline`` is NOT usable here: it interleaves
    ``canonicalize``, and at this point in the RVV pipeline the module still holds ``vector.mask``
    regions that have not been through ``lower-vector-mask``. Canonicalization sinks a constant into
    one of those regions, which then fails its own verifier (``'vector.mask' op expects only one
    operation to mask``) — whisper_tiny hits it, deepjscc does not. So the three passes are named
    individually and the cleanup is left to the ``canonicalize``/``cse`` the pipeline already runs at
    the end, after the vector ops have been lowered.

    ``MERLIN_NO_DEALLOC`` restores the previous behavior for an A/B — it exists because this changes
    generated code for every model, and the honest way to defend "numerics unchanged" is to be able
    to rebuild both.

    ``MERLIN_SINK_DEALLOCS`` (or ``sink=True``) appends the sinking stage described in
    :func:`_sink_dealloc_passes`, which moves each free from the end of the function to just after
    the buffer's last use. Default OFF, so an unflagged build is byte-identical to the frozen
    baseline.
    """
    import os
    if os.environ.get("MERLIN_NO_DEALLOC"):
        return []
    passes = ["ownership-based-buffer-deallocation",
              "buffer-deallocation-simplification",
              "bufferization-lower-deallocations"]
    if sink is None:
        sink = _sink_deallocs()
    if sink:
        passes += _sink_dealloc_passes()
    return passes


def _sink_deallocs() -> bool:
    """Whether to sink each dealloc to its last use. Default OFF (``MERLIN_SINK_DEALLOCS=1``).

    A named predicate rather than an inline environ poke so a test can drive it, and so the flag is
    read per build instead of frozen at import."""
    import os
    return bool(os.environ.get("MERLIN_SINK_DEALLOCS"))


#: The pass that moves a dealloc to its last user. Named once: the runner keys the placement CHECK
#: on finding it in the pass list, so a rename that missed one of the two would silently disarm the
#: check rather than fail.
SINK_DEALLOC_PASS = "func.func(optimize-allocation-liveness)"


def _sink_dealloc_passes() -> list[str]:
    """Move each free from function exit to just after the buffer's last use.

    CORRECTING WHAT THIS COMMENT USED TO SAY. It recorded that
    ``func.func(optimize-allocation-liveness)`` "moved nothing" and concluded that
    ``ownership-based-buffer-deallocation`` "is already placing the frees tightly enough that there is
    no slack to reclaim". The first half was an accurate observation; the second half was the wrong
    explanation, and the emitted IR contradicts it. The frees are not tight — they are ALL at the end
    of ``@forward``: measured on the deepjscc int8 build, the last ``malloc`` is at line 10,530 of a
    12,112-line function and every ``free`` sits after the last compute. The static arena binder
    (:mod:`merlin.llvmlower.arena_bind`), which can only reuse bytes between a free and a later
    malloc, therefore gets 1.00x reuse on all three int8 models, while planning the SAME buffers under
    last-use liveness gives 5.77x / 7.67x / 11.73x. The slack was real; the pass simply could not see
    it.

    WHY IT MOVED NOTHING. ``bufferization-lower-deallocations`` does not emit a bare
    ``memref.dealloc``: it emits the dealloc inside an ``scf.if`` guarded by the buffer's ownership
    flag, which ``buffer-deallocation-simplification`` has already folded to a constant ``true``::

        scf.if %true {
          memref.dealloc %alloc : memref<64x64xf32>
        }

    ``OptimizeAllocationLiveness`` looks for a user of the allocation with a Free memory effect and
    bails unless that user is in the SAME BLOCK as the alloc — deliberately, so it never hoists a
    dealloc out of a conditional (llvm-project ``mlir/lib/Dialect/Bufferization/Transforms/
    OptimizeAllocationLiveness.cpp``, ``deallocOp->getBlock() != allocOp->getBlock()``). The dealloc
    is in the ``scf.if``'s body block, so EVERY allocation is skipped and the pass is a guaranteed
    no-op wherever it is placed after the lowering — and, measured, equally a no-op placed BEFORE it.
    Adding it changed nothing not because the frees were tight but because it could not read them.

    THE FIX is the ``canonicalize`` this pipeline dropped. Upstream's own composite
    ``buffer-deallocation-pipeline`` runs ``cse`` + ``canonicalize`` AFTER ``lower-deallocations``
    (llvm-project ``mlir/lib/Dialect/Bufferization/Pipelines/BufferizationPipelines.cpp:36-38``); this
    pipeline names the three passes individually — for the ``vector.mask`` reason in
    :func:`_dealloc_passes` — and lost that cleanup with them. Canonicalization folds the
    statically-true ``scf.if`` away, leaving the ``memref.dealloc`` a direct user of the alloc in the
    alloc's own block, and the liveness pass then moves every one of them to its last use.

    ORDER IS LOAD-BEARING and is asserted by ``merlin/tests/ir/test_dealloc_sinking.py``:
    ``bufferization-lower-deallocations`` → ``canonicalize`` → ``optimize-allocation-liveness``.
    Placed before the lowering, or after it without the canonicalize, the pass silently moves nothing
    — which is exactly how this was mis-diagnosed the first time.

    The canonicalize is safe HERE for the same reason the composite pipeline was not: it runs where
    ``__DEALLOC__`` is spliced, which in the RVV pipeline is already past ``lower-vector-mask`` and
    ``convert-vector-to-scf`` (no ``vector.mask`` region left to sink a constant into), and in the
    scalar pipelines the module carries no vector ops at all.

    Sinking a free is the one edit in this file that can produce SILENTLY WRONG NUMBERS rather than a
    build failure: a dealloc moved before a use is a use-after-free, and with the arena bound on top
    of it the freed bytes are handed to another buffer. So the flagged path does not merely trust the
    pass — the runner re-checks the placement structurally on the IR the stage produced (every use of
    the allocation and of every value derived from it must precede its dealloc) and FAILS THE BUILD
    on a violation. See :data:`DEALLOC_CHECK_PRELUDE`.
    """
    return ["canonicalize", SINK_DEALLOC_PASS]


_UPSTREAM_PASSES = [
    "canonicalize", "cse",
    "func.func(linalg-fuse-elementwise-ops)",
    "func.func(linalg-generalize-named-ops)",
    "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
    "buffer-results-to-out-params{modify-public-functions hoist-static-allocs}",
    "__DEALLOC__",
    "func.func(buffer-hoisting,buffer-loop-hoisting)",
    "func.func(convert-linalg-to-loops)",
    "convert-scf-to-cf",
    # memref.expand/collapse_shape/subview must be expanded before LLVM conversion.
    "expand-strided-metadata",
    "lower-affine",
    # transcendentals (erf/exp/tanh/...) have no LLVM intrinsic -> libm calls
    # (host libm; riscv newlib libm). Remaining math (fma/abs/...) -> intrinsics.
    "convert-math-to-libm",
    "convert-math-to-llvm",
    "convert-index-to-llvm",
    "convert-arith-to-llvm",
    "finalize-memref-to-llvm",
    "convert-func-to-llvm",
    "convert-cf-to-llvm",
    "reconcile-unrealized-casts",
    "canonicalize", "cse", "symbol-dce",
]


def _splice(passes: list[str]) -> str:
    """Join a pass list, expanding the ``__DEALLOC__`` marker in place.

    The marker names *where* deallocation belongs in each pipeline (immediately after
    bufferization) so the position is visible in the pass list itself rather than computed by an
    index that drifts when the list is edited.

    The generalize/fuse reorder is applied HERE, at the one point every pipeline in this file passes
    through, and AFTER ``apply_pipeline`` has spliced any feature-driven fusion stage -- so the
    reorder reaches the ``fuse_elementwise_post_contraction`` feature's stage as well as the
    ``MERLIN_FUSE_POST`` one, instead of only the copy that happens to be written in a literal list.
    Default off; see :func:`_generalize_before_fuse`."""
    if _generalize_before_fuse():
        passes = _reorder_generalize_before_fuse(passes)
    out: list[str] = []
    for p in passes:
        out.extend(_dealloc_passes() if p == "__DEALLOC__" else [p])
    return ",".join(out)


def _upstream_pipeline() -> str:
    """The scalar whole-module pipeline. A function, not a constant, so ``MERLIN_NO_DEALLOC``
    is read per build rather than frozen at import."""
    return _splice(_UPSTREAM_PASSES)


def _parallel_pipeline() -> str:
    """The scalar pipeline, re-targeted for **multicore** via OpenMP.

    Identical to :data:`UPSTREAM_PIPELINE` except the loop-generation stage: parallel
    iterator dimensions of each linalg op become ``scf.parallel`` (``convert-linalg-to-
    parallel-loops``) which ``convert-scf-to-openmp`` wraps in an ``omp.parallel`` +
    ``omp.wsloop`` (work-shared across threads); reduction dims stay sequential
    ``scf.for`` (correctness preserved). ``mlir-translate`` then emits ``__kmpc_*`` runtime
    calls, satisfied at link time by the cross-built ``libomp.a`` (build_tools/k1_openmp).
    Set ``OMP_NUM_THREADS`` on the board to fan out across the 8 cores.

    This is the K1 big-model path: the diffusion-transformer / large VLAs that run on the
    *scalar* fallback (no fixed-width vectorize) otherwise execute on one core and time out.
    Gated — never on the default flow (``UPSTREAM_PIPELINE`` is untouched)."""
    return _splice([
        "canonicalize", "cse",
        "func.func(linalg-fuse-elementwise-ops)",
        "func.func(linalg-generalize-named-ops)",
        "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
        "buffer-results-to-out-params{modify-public-functions}",  # heap intermediates (K1)
        "__DEALLOC__",
        "func.func(buffer-hoisting,buffer-loop-hoisting)",
        # parallel iterator dims -> scf.parallel; reduction dims stay scf.for (sequential).
        "func.func(convert-linalg-to-parallel-loops)",
        "convert-scf-to-openmp",          # scf.parallel -> omp.parallel{omp.wsloop}
        "canonicalize",
        "convert-scf-to-cf",              # remaining (reduction) scf.for -> cf
        "expand-strided-metadata",
        "lower-affine",
        "convert-openmp-to-llvm",         # omp region operands -> llvm
        "convert-math-to-libm",
        "convert-math-to-llvm",
        "convert-index-to-llvm",
        "convert-arith-to-llvm",
        "finalize-memref-to-llvm",
        "convert-func-to-llvm",
        "convert-cf-to-llvm",
        "reconcile-unrealized-casts",
        "canonicalize", "cse", "symbol-dce",
    ])


# --- Native RVV (fixed-width) vectorization path --------------------------------------
# Instead of relying on clang's RISC-V auto-vectorizer of scalar loops (the scalar path
# above), we bake vector ops into the IR: a transform-dialect schedule vectorizes every
# static-shaped linalg op to fixed-width `vector<Nxf32>` and lowers the structured vector
# ops (contraction/transpose/shape_cast/masked-transfers) to LLVM-convertible primitives.
# clang -march=rv64gcv then emits real RVV (vsetvli/vle32.v/vfmacc.vv) for these vectors
# (verified end-to-end). Ops the vectorizer skips (dynamic shapes) fall through
# `convert-linalg-to-loops` to scalar code, so correctness never depends on every op
# vectorizing. Fixed-width (not scalable `vector<[n]>`) is deliberate: scalable
# transpose/shape_cast lowering is incomplete in this LLVM-23 build; fixed-width lowers
# cleanly and the RISC-V backend still targets RVV.
# We vectorize the CONTRACTION ops — `linalg.matmul` (FFN + projections) and
# `linalg.batch_matmul` (attention QKᵀ / attn·V) — that is where ~all the compute cycles live,
# and they vectorize cleanly. We do NOT blanket-vectorize: a `vectorize_children` on a whole
# transformer scalarizes the transposes/reduces/elementwise generics into tens of thousands of
# `vector.extract`s that don't lower in this LLVM-23 build. The named contraction ops are
# recovered first by `linalg-specialize-generic-ops` (the capture emits attention as
# `linalg.generic`); everything non-contraction stays linalg and goes through
# `convert-linalg-to-loops` (scalar). Tile + vectorize at fixed sizes (matmul [M4,N8,K1];
# batch_matmul [B1,M4,N8,K1]) — uniform sizes that fit any shape (smaller dims masked); proven
# to emit real RVV (vsetvli/vle32.v/vfmacc) and lower end-to-end with no extract explosion.
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


# --- Multicore: an OUTER parallel tiling stage layered under the RVV schedule ----------
# The RVV schedule tiles the contraction with `tile_using_for`, which emits sequential
# `scf.for` — and this LLVM-23 build has no scf.for->scf.parallel pass, so a serial tile
# loop can never become an OpenMP worksharing loop. `scf.forall` IS the parallel carrier
# (`scf-forall-to-parallel` -> `convert-scf-to-openmp`), so multicore has to be introduced
# by an outer `tile_using_forall` BEFORE the package's schedule runs.
#
# This is done as a SEPARATE transform library with its OWN entry point rather than by
# editing the package's schedule text: `transform-preload-library` merges several library
# files (`transform-library-paths=a.mlir,b.mlir`) and the interpreter is then run once per
# entry point. So every existing package schedule — and every impr_features edit, which
# anchors on `__transform_main` — composes unchanged, and no package file is rewritten.
# The package's `structured.match` still finds the contraction after it is wrapped, because
# match walks nested regions.
#
# WHICH DIM: only PARALLEL dims may be split — tiling the K (reduction) dim would race on
# the accumulator. Defaults are chosen so the split does not starve the inner micro-kernel:
#   * matmul (M,N,K) -> N. Splitting M is the obvious choice but is wrong here: LLM shapes
#     are small-M (TinyLlama prefill is M=8), and M/n_harts drops BELOW the MR=4 register
#     block, so each hart runs a masked short tile. N is large (2048/5632) and stays a clean
#     multiple of NR=8.
#   * batch_matmul (B,M,N,K) -> B. Attention carries B=32 heads, far more parallelism than
#     its N (8..64), and the schedule already tiles B by 1 so whole heads split cleanly.


# --- Pass ORDER: generalization vs elementwise fusion ----------------------------------
#: The two passes whose ORDER decides whether fusion can see a named op at all.
#: Named once, because the swap below has to find the same pair in three different pass lists.
_FUSE_ELEMENTWISE = "func.func(linalg-fuse-elementwise-ops)"
_GENERALIZE_NAMED = "func.func(linalg-generalize-named-ops)"


def _generalize_before_fuse() -> bool:
    """Whether ``linalg-generalize-named-ops`` runs BEFORE ``linalg-fuse-elementwise-ops``.

    Default OFF (``MERLIN_GENERALIZE_BEFORE_FUSE=1``), so every unflagged build -- including the
    ``MERLIN_FUSE_POST`` / ``fuse_elementwise_post_contraction`` arm, whose in-tree numbers were all
    measured in the current order -- is byte-identical to the frozen baseline.

    WHY THE ORDER MATTERS, and it is not a style question. Upstream's ``FuseElementwiseOps`` pattern
    is an ``OpRewritePattern<linalg::GenericOp>`` whose producer must ALSO be a ``linalg.GenericOp``
    (``mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp``). A ``linalg.broadcast`` /
    ``linalg.transpose`` / ``linalg.copy`` is therefore INVISIBLE to it: not declined, not counted --
    never matched. So a fusion stage placed in front of the generalization is blind to exactly the ops
    the stage exists to remove.

    MEASURED on the three int8 recaptures, at the point in ``build_rvv_pipeline`` where the stage
    runs: ``linalg-specialize-generic-ops`` turns 0 ``linalg.broadcast`` into 116 (deepjscc), 64
    (small_llama) and 246 (lstmnetvit) -- and every one of them is a named op that the very next
    pass, in the current order, cannot see.

    Deliberately NOT auto-coupled to the fusion lever the way ``vectorize_non_contraction_generics``
    is coupled to its self-copy erase. The self-copy erase CANCELS its lever's payoff, so shipping
    them apart makes the lever inert; this one CHANGES what the fusion stage does, and every number
    recorded for ``fuse_elementwise_post_contraction`` in ``impr_features`` was measured without it.
    Silently re-ordering underneath that feature would invalidate its recorded evidence rather than
    complete it, so the two are measured separately and coupled only once the board says which wins.
    """
    import os
    return bool(os.environ.get("MERLIN_GENERALIZE_BEFORE_FUSE"))


def _reorder_generalize_before_fuse(passes: list[str]) -> list[str]:
    """Move ``linalg-generalize-named-ops`` in front of the ``linalg-fuse-elementwise-ops`` it follows.

    One rule for all three pipelines, because the two spellings differ only in what sits between the
    pair: the scalar/parallel lists have them adjacent, and ``build_rvv_pipeline`` has
    ``fuse, canonicalize, cse, generalize``. Moving the GENERALIZE (rather than swapping in place)
    keeps the ``canonicalize``/``cse`` attached to the fusion they clean up after -- which is not
    incidental: ``impr_features.FUSE_ELEMENTWISE_STAGE`` records that most of the temporary collapse
    is the cleanup, not the fusion.

    A no-op on a list that has no such pair, or where the generalization already runs first, so it is
    safe to apply to any pass list. Never moves a pass across ``transform-interpreter``: a
    generalization hoisted in front of the schedule would leave ``ops{["linalg.matmul"]}`` nothing to
    match (silent 0-vectorization), which is the failure the current order was written to avoid.
    """
    if _GENERALIZE_NAMED not in passes or _FUSE_ELEMENTWISE not in passes:
        return list(passes)
    gen = passes.index(_GENERALIZE_NAMED)
    fuse = next((i for i, p in enumerate(passes) if p == _FUSE_ELEMENTWISE and i < gen), None)
    if fuse is None:
        return list(passes)                       # already generalize-then-fuse
    if any("transform-interpreter" in p for p in passes[fuse:gen]):
        raise ValueError(
            "refusing to move linalg-generalize-named-ops across a transform-interpreter: the "
            "schedule matches contractions by NAME and would then match nothing")
    out = [p for i, p in enumerate(passes) if i != gen]
    return [*out[:fuse], _GENERALIZE_NAMED, *out[fuse:]]


def _fuse_post() -> bool:
    """Whether to run the post-contraction fusion stage.

    Still opt-in (``MERLIN_FUSE_POST``) so the shipping baseline stays byte-identical until the fused
    build is graded bit-exact on hardware -- but a named predicate rather than an inline environ poke,
    so a test can drive it and a build can record it.
    """
    import os
    return bool(os.environ.get("MERLIN_FUSE_POST"))


PARALLEL_ENTRY = "__transform_parallel_main"

_PARALLEL_DIM_NUM_THREADS = {
    # op -> {dim: num_threads list}. A 0 means "do not tile this dim".
    "linalg.matmul": {"m": "[{n}]", "n": "[0, {n}]"},
    "linalg.batch_matmul": {"b": "[{n}]", "m": "[0, {n}]", "n": "[0, 0, {n}]"},
}


def parallel_transform_schedule(n_harts: int, *, matmul_dim: str = "n",
                                batch_matmul_dim: str = "b") -> str:
    """Transform schedule that wraps each contraction in an `scf.forall` over ``n_harts``.

    Runs BEFORE the package's own schedule (separate entry point, see above), so the inner
    tiling/vectorization — and therefore the emitted `vfmacc`/`vwmacc` — is untouched.
    ``matmul_dim`` / ``batch_matmul_dim`` name the parallel dim to split; K is never
    tileable here (it is the reduction dim and splitting it would race).
    """
    if n_harts < 2:
        raise ValueError(f"parallel schedule needs n_harts >= 2, got {n_harts}")
    body = []
    for op, dim in (("linalg.matmul", matmul_dim), ("linalg.batch_matmul", batch_matmul_dim)):
        choices = _PARALLEL_DIM_NUM_THREADS[op]
        if dim not in choices:
            raise ValueError(f"{op}: parallel dim {dim!r} not in {sorted(choices)} "
                             "(the reduction dim K is never tileable)")
        tag = op.split(".")[-1]
        body.append(
            f'    %{tag} = transform.structured.match ops{{["{op}"]}} in %arg0 '
            f': (!transform.any_op) -> !transform.any_op\n'
            f'    %{tag}_loop, %{tag}_tiled = transform.structured.tile_using_forall '
            f'%{tag} num_threads {choices[dim].format(n=n_harts)} '
            f': (!transform.any_op) -> (!transform.any_op, !transform.any_op)')
    return ("module attributes {transform.with_named_sequence} {\n"
            f"  transform.named_sequence @{PARALLEL_ENTRY}"
            "(%arg0: !transform.any_op {transform.readonly}) {\n"
            + "\n".join(body) + "\n"
            "    transform.yield\n"
            "  }\n"
            "}\n")


#: Entry point of the library that runs the NON-CONTRACTION vectorize arms, before specialization.
VEC_PRE_ENTRY = "__transform_vec_main"

#: The skeleton the arms are spliced into. It is deliberately the SAME splice the package schedule
#: gets (``impr_features.apply_schedule``), anchored on the same ``%f = ... ops{["func.func"]}`` line,
#: so the arms in the pre-library and the arms in ``__transform_main`` are generated by one function
#: at one lane width and cannot drift apart.
_VEC_PRE_SKELETON = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @%s(%%arg0: !transform.any_op {transform.readonly}) {
    %%f = transform.structured.match ops{["func.func"]} in %%arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %%f {
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op
    transform.yield
  }
}
""" % VEC_PRE_ENTRY


def _vec_after_specialize() -> bool:
    """Rebuild the OLD placement of the non-contraction arms (``MERLIN_VEC_AFTER_SPECIALIZE=1``).

    An A/B escape hatch, for the reason :func:`_dealloc_passes` keeps ``MERLIN_NO_DEALLOC``: this
    changes the emitted code for every build that names the lever, and the honest way to defend a
    claim about it is to be able to rebuild both arms."""
    import os
    return bool(os.environ.get("MERLIN_VEC_AFTER_SPECIALIZE"))


def vec_pre_schedule(features: "frozenset[str]") -> "str | None":
    """The transform library that vectorizes the tagged non-contraction generics, or None.

    WHY THIS EXISTS -- a COVERAGE bug, measured, not inferred. The prepare pass tags every
    all-parallel ``linalg.generic`` it can vectorize with a discardable ``merlin.vec_r{rank}``
    attribute, and ``vectorize_non_contraction_generics`` matches on it. But
    ``func.func(linalg-specialize-generic-ops)`` -- which must run before the interpreter, so the
    CONTRACTION arms can match named ``linalg.matmul``/``batch_matmul`` -- rewrites most of those
    tagged generics into named ops (``linalg.broadcast`` above all) and a rewrite does not carry a
    DISCARDABLE attribute onto the op it produces. The tag is gone, and with it the match.

    MEASURED, tags present in the module at each stage (int8 recaptures, host lowering). Every tag
    that reaches an arm is consumed -- the count after the interpreter is 0 in both placements -- so
    "reaches" and "vectorized" are the same number here::

        model         tagged   after canonicalize,cse   after specialize   reaching an arm
                      by prep  (where the arms run now) (where they ran)   before -> after
        deepjscc       93       85                       15                 15  ->  85  (16% -> 91%)
        small_llama   107       84                       33                 33  ->  84  (31% -> 79%)
        lstmnetvit    189      160                       36                 36  -> 160  (19% -> 85%)

    So four fifths of the ops the lever was written for never reached an arm, while every tile the
    remaining fifth produced paid the full loop and destination-buffer overhead -- which is the
    coverage half of why the lever measured 1.28x SLOWER (the realization half, a ``memref.copy`` of
    each tile onto itself, was fixed separately and is now implied by the feature).

    The residual (93 -> 85, 107 -> 84, 189 -> 160) is the ``canonicalize``/``cse`` this pipeline
    already ran before any of this: the generic count falls further than the tag count on all three
    models, which is consistent with CSE merging duplicate ops rather than with tags being lost.

    NOT A SPEED CLAIM. What is measured here is coverage and the emitted vector census (vector loads
    108 -> 268 deepjscc, 134 -> 249 small_llama, 285 -> 560 lstmnetvit, with the malloc count FALLING
    on all three). Whether that pays is a board measurement, and 1.28x is the number it has to beat.

    THE FIX IS A PLACEMENT, NOT A PRESERVATION. Restoring the attribute across specialization would
    need op identity to survive a rewrite that is not 1:1 -- specialize DECOMPOSES generics (deepjscc:
    278 generics in, 199 generics + 116 broadcasts + 20 matmuls out), so there is no correspondence to
    key a restore on. Running the arms BEFORE specialization needs no correspondence at all: the tags
    are still on the ops that carry them. It is also the better IR to vectorize -- the pre-specialize
    form is the FUSED one, so an arm tiles one elementwise chain instead of the broadcast + consumer
    pair specialize would have split it into.

    Realized as a SEPARATE library with its own entry point rather than by re-ordering
    ``__transform_main``: ``transform-preload-library`` merges several files and the interpreter runs
    once per entry point, so the package's own schedule -- and every ``impr_features`` edit that
    anchors on ``__transform_main`` -- composes completely unchanged, and the contraction arms still
    run exactly once, after specialization, on exactly the IR they run on today.

    Returns None when the lever is off (then the pipeline string is byte-identical to the baseline),
    when the feature set splices no arms, or under ``MERLIN_VEC_AFTER_SPECIALIZE``.
    """
    from .impr_features import (apply_schedule, ensure_vec_noncontraction,
                                vec_noncontraction_lanes)
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
            f"`%f = transform.structured.match ops{{[\"func.func\"]}}` it keys on is gone")
    return text


def build_rvv_pipeline(sched_path: "str | Path", hoist_static_allocs: bool = True,
                       features: "frozenset[str]" = frozenset(),
                       par_sched_path: "str | Path | None" = None,
                       vec_sched_path: "str | Path | None" = None) -> str:
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
    brop = ("buffer-results-to-out-params{modify-public-functions hoist-static-allocs}"
            if hoist_static_allocs
            else "buffer-results-to-out-params{modify-public-functions}")
    late_hoist: list[str] = []
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
        "func.func(linalg-specialize-generic-ops)"]
    head = ([preload, f"transform-interpreter{{entry-point={VEC_PRE_ENTRY}}}",
             "canonicalize", "cse", *specialize]
            if vec else [*specialize, preload])
    passes = [
        "canonicalize", "cse",
        *head,
        *([f"transform-interpreter{{entry-point={PARALLEL_ENTRY}}}", "canonicalize", "cse"]
          if par else []),
        "transform-interpreter{entry-point=__transform_main}",
        "canonicalize", "cse",
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
        *( ["func.func(linalg-fuse-elementwise-ops)", "canonicalize", "cse"]
           if _fuse_post() else [] ),
        "func.func(linalg-generalize-named-ops)",   # remaining (non-vectorized) ops -> loops
        "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
        brop,
        "func.func(buffer-hoisting,buffer-loop-hoisting)",
        "func.func(lower-vector-multi-reduction)",
        "func.func(lower-vector-mask)",
        "convert-vector-to-scf",
        # Deallocation goes HERE, not straight after bufferization as in the scalar pipelines.
        # `ownership-based-buffer-deallocation` materializes an i1 ownership constant next to each
        # buffer use, and it walks into `vector.mask` regions -- which accept exactly one masked
        # operation, so the extra constant makes the region fail its own verifier. Running after
        # `lower-vector-mask` (and `convert-vector-to-scf`, which is where the masks actually go
        # away) leaves no such region to walk into. Still before the scf -> cf / openmp conversions,
        # because ownership analysis wants structured control flow.
        "__DEALLOC__",
        # Multicore: the outer `scf.forall` becomes `scf.parallel`, and the non-vectorized
        # fallback ops lower to parallel loops too (instead of serial `scf.for`) so the
        # elementwise/norm tail scales with the harts as well. `convert-scf-to-openmp` then
        # wraps every `scf.parallel` in `omp.parallel` + `omp.wsloop`. Reduction loops are
        # NOT touched by convert-linalg-to-parallel-loops (they stay `scf.for`), so the
        # accumulator is never raced.
        *(["scf-forall-to-parallel",
           "func.func(convert-linalg-to-parallel-loops)",
           "convert-scf-to-openmp", "canonicalize"]
          if par else
          ["func.func(convert-linalg-to-loops)"]),   # fallback: any op the vectorizer skipped
        *late_hoist,                            # K1: hoist loop-body alloca temps out of loops
        "convert-scf-to-cf",
        "expand-strided-metadata",
        "lower-affine",
        "convert-vector-to-llvm",
        "convert-ub-to-llvm",
        *(["convert-openmp-to-llvm"] if par else []),
        "convert-math-to-libm",
        "convert-math-to-llvm",
        "convert-index-to-llvm",
        "convert-arith-to-llvm",
        "finalize-memref-to-llvm",
        "convert-func-to-llvm",
        "convert-cf-to-llvm",
        "reconcile-unrealized-casts",
        "canonicalize", "cse", "symbol-dce",
    ]
    if features:                       # default-off impr-fork hooks; empty -> byte-identical
        from .impr_features import apply_pipeline
        passes = apply_pipeline(passes, features)
    return _splice(passes)


# The post-bufferization rewrites, and the argv glue that selects them. `_MID_STAGE_SRC` is spliced
# into EVERY runner variant below (plain / act_poly / scalarize) so a feature-specific runner cannot
# silently drop one -- which is exactly how `erase_self_copy` came to measure as an inert lever: the
# act_poly runner called the PassManager directly, so every fork that also enabled
# `vectorized_transcendental_activation` (the whole-model proposer enables it by default) got the
# self-copy erase requested, reported as applied, and never run.
from .concat_dps import RUNNER_PRELUDE as _CONCAT_DPS_PRELUDE
from .copy_expand import MID_STAGE_SRC as _MID_STAGE_SRC
from .copy_expand import RUNNER_PRELUDE as _COPY_EXPAND_PRELUDE
from .parallel_grain import LATE_STAGE_SRC as _PARALLEL_GRAIN_LATE_SRC
from .parallel_grain import RUNNER_PRELUDE as _PARALLEL_GRAIN_PRELUDE
from .selfcopy import RUNNER_PRELUDE as _SELFCOPY_PRELUDE
from .transpose_fuse import RUNNER_PRELUDE as _TRANSPOSE_FUSE_PRELUDE
from .transpose_maps import RUNNER_PRELUDE as _TRANSPOSE_MAPS_PRELUDE

# --- The use-after-free check that guards the sinking stage ---------------------------------
#
# Sinking a dealloc is the one edit here whose failure mode is WRONG NUMBERS, not a broken build: a
# free moved before a use is a use-after-free, and with the static arena bound on top of it the
# freed bytes are immediately handed to another buffer. `optimize-allocation-liveness` decides where
# a free may go from upstream's `BufferViewFlowAnalysis`; this re-checks the RESULT, on the IR the
# stage actually produced, in terms that do not depend on that analysis being right.
#
# The rule: for every `memref.alloc` whose `memref.dealloc` the stage placed in the alloc's OWN block
# (an allocation still freed inside an `scf.if` was not sunk -- upstream placed it, and it is left
# alone), every use of the allocation AND of every value derived from it must come before that free.
#
# "Derived from" is deliberately OVER-approximated: any op result of memref type, and any memref
# block argument of a region, of an op that consumes an aliasing value joins the alias set. An
# over-approximation can only make the check REFUSE a placement it could have allowed; it cannot let
# a real use-after-free through by narrowing what counts as a use. Anything it cannot read -- a use
# whose ancestor chain does not reach the alloc's block -- is reported, never skipped.
DEALLOC_CHECK_PRELUDE = r"""
def _memref_typed(v):
    return str(v.type).startswith('memref')


def _op_key(o):
    # OpView and Operation both answer `.operation`, and the bindings hand back equal (hashable)
    # objects for the same op however it was reached -- so this is a stable dict key.
    return o.operation


def _dealloc_placement_violations(module):
    # (violations, n_allocs, n_sunk) for a post-deallocation memref module.
    order, block_of, index_of, owner_of_block = [], {}, {}, {}
    counter = [0]

    def visit(block, owner):
        bid = counter[0]
        counter[0] += 1
        owner_of_block[bid] = owner
        for i, op in enumerate(block.operations):
            k = _op_key(op)
            block_of[k], index_of[k] = bid, i
            order.append(op)
            for region in op.regions:
                for b in region.blocks:
                    visit(b, op)

    for op in module.body.operations:
        for region in op.regions:
            for b in region.blocks:
                visit(b, op)

    alias, roots, uses = {}, {}, {}
    for op in order:
        name = op.operation.name
        if name == 'memref.alloc' and len(op.results) == 1:
            rid = len(roots)
            roots[rid] = op
            alias[op.results[0]] = {rid}
            continue
        hit = set()
        for v in op.operands:
            hit |= alias.get(v, set())
        if not hit:
            continue
        for rid in hit:
            uses.setdefault(rid, []).append(op)
        for r in op.results:
            if _memref_typed(r):
                alias.setdefault(r, set()).update(hit)
        for region in op.regions:
            for b in region.blocks:
                for a in b.arguments:
                    if _memref_typed(a):
                        alias.setdefault(a, set()).update(hit)

    def ancestor_in_block(op, bid):
        o = op
        while True:
            k = _op_key(o)
            if k not in block_of:
                return None
            if block_of[k] == bid:
                return o
            parent = owner_of_block[block_of[k]]
            if parent is None:
                return None
            o = parent

    violations, sunk = [], 0
    for rid, alloc in roots.items():
        abid = block_of[_op_key(alloc)]
        here = [u for u in uses.get(rid, [])
                if u.operation.name == 'memref.dealloc' and block_of[_op_key(u)] == abid]
        if not here:
            continue                      # freed in a conditional/another block: not sunk
        sunk += 1
        d = min(here, key=lambda o: index_of[_op_key(o)])
        di = index_of[_op_key(d)]
        ty = str(alloc.results[0].type)
        for u in uses.get(rid, []):
            if _op_key(u) == _op_key(d):
                continue
            anc = ancestor_in_block(u, abid)
            if anc is None:
                violations.append(ty + ': use by ' + u.operation.name +
                                  ' is outside the block its free was placed in')
            elif index_of[_op_key(anc)] > di:
                violations.append(ty + ': freed at #' + str(di) + ' but used at #' +
                                  str(index_of[_op_key(anc)]) + ' by ' + u.operation.name)
    return violations, len(roots), sunk
"""

#: Runner glue: wrap ``_run_stages`` so the placement check runs on the IR the sinking stage
#: produced, in EVERY runner variant, without any of them having to know about it. The wrapper
#: delegates unchanged when the pass list carries no sinking stage.
DEALLOC_CHECK_RUNNER = r"""
_ORIG_RUN_STAGES = _run_stages
_SINK_MARK = 'optimize-allocation-liveness'


def _run_stages(ctx, module, pipeline, erase, mid=(), late=()):
    passes = [p for p in pipeline.split(',') if p]
    k = next((i for i, p in enumerate(passes) if _SINK_MARK in p), -1)
    if k < 0:
        return _ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late)
    head, tail = passes[:k + 1], passes[k + 1:]
    # `erase`/`mid` open their window after buffer-loop-hoisting, and `late` opens its own before
    # convert-scf-to-openmp; hand each to whichever half contains its anchor, or it would be
    # accepted here and silently never run. A `late` whose anchor is in neither half goes to the
    # tail, where the wrapped runner still applies it at the end rather than dropping it.
    hoist_head = any('buffer-loop-hoisting' in p for p in head)
    late_head = any('convert-scf-to-openmp' in p for p in head)
    _ORIG_RUN_STAGES(ctx, module, ','.join(head),
                     erase if hoist_head else 0, mid if hoist_head else (),
                     late if late_head else ())
    bad, n_alloc, n_sunk = _dealloc_placement_violations(module)
    print('OK dealloc_placement', n_alloc, 'allocations', n_sunk, 'sunk', len(bad), 'violations')
    if bad:
        raise RuntimeError('dealloc placement moved a free before a use of the buffer '
                           '(' + str(len(bad)) + ' violations):\n  ' + '\n  '.join(bad[:20]))
    if tail:
        _ORIG_RUN_STAGES(ctx, module, ','.join(tail),
                         0 if hoist_head else erase, () if hoist_head else mid,
                         () if late_head else late)
"""

#: The line the runner prints once the check has run. `lower_to_llvm_ir` REQUIRES it whenever the
#: pipeline carries the sinking stage: a runner variant that drove the PassManager itself and never
#: reached the wrapper would otherwise sink the frees with nothing checking them, and report success.
DEALLOC_CHECK_TOKEN = "OK dealloc_placement"


def apply_passes(mlir_text: str, pipeline: str, timeout: int = 600) -> str:
    """Run ``pipeline`` over ``mlir_text`` in the m2m venv and return the printed module.

    A diagnostic/test seam, not a build path: it drives the SAME pass registry the lowering drives
    (the torch-mlir wheel's), so a test that pins what a stage does to the IR cannot pass against a
    different LLVM than the one that ships.
    """
    work = Path(tempfile.mkdtemp(prefix="merlin_apply_passes_"))
    src, dst = work / "in.mlir", work / "out.mlir"
    src.write_text(mlir_text, encoding="utf-8")
    script = work / "apply.py"
    script.write_text(
        "import sys\n"
        "from torch_mlir import ir\n"
        "from torch_mlir.passmanager import PassManager\n"
        "ctx = ir.Context()\n"
        "with open(sys.argv[1]) as f:\n"
        "    module = ir.Module.parse(f.read(), ctx)\n"
        "PassManager.parse('builtin.module(' + sys.argv[3] + ')', ctx).run(module.operation)\n"
        "with open(sys.argv[2], 'w') as f:\n"
        "    f.write(str(module.operation))\n", encoding="utf-8")
    proc = subprocess.run([str(m2m_python()), str(script), str(src), str(dst), pipeline],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not dst.is_file():
        raise PipelineError(f"apply_passes failed:\n{proc.stdout}\n{proc.stderr}")
    return dst.read_text(encoding="utf-8")


def dealloc_placement_violations(mlir_text: str, timeout: int = 600) -> list[str]:
    """Violations of the placement rule in a post-deallocation memref module.

    Empty list == every allocation whose free was sunk into the alloc's own block is used only
    before that free. Runs the SAME source the build runs, out of process in the m2m venv (the MLIR
    Python bindings live there), so a test and a build cannot disagree about what is checked.
    """
    import json
    work = Path(tempfile.mkdtemp(prefix="merlin_dealloc_check_"))
    src = work / "module.mlir"
    src.write_text(mlir_text, encoding="utf-8")
    script = work / "check.py"
    script.write_text(
        "import json, sys\n"
        "from torch_mlir import ir\n"
        + DEALLOC_CHECK_PRELUDE +
        "ctx = ir.Context()\n"
        "with open(sys.argv[1]) as f:\n"
        "    module = ir.Module.parse(f.read(), ctx)\n"
        "bad, n_alloc, n_sunk = _dealloc_placement_violations(module)\n"
        "print('MERLIN_CHECK ' + json.dumps({'violations': bad, 'allocations': n_alloc, "
        "'sunk': n_sunk}))\n", encoding="utf-8")
    proc = subprocess.run([str(m2m_python()), str(script), str(src)],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise PipelineError(f"dealloc placement check failed:\n{proc.stdout}\n{proc.stderr}")
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("MERLIN_CHECK "))
    return json.loads(line[len("MERLIN_CHECK "):])["violations"]


# How a runner emits its result. The in-process torch-mlir bridge
# (`translate_module_to_llvmir`) is the default and produces .ll directly; it SEGFAULTS on
# OpenMP IR (its OpenMPIRBuilder), so the multicore/parallel paths instead DUMP the
# LLVM-dialect module and translate out-of-process with the standalone mlir-translate.
# Every runner variant carries the `__MERLIN_EMIT__` token so the choice is orthogonal to
# which feature-specific runner is selected — without this, enabling multicore silently
# dropped the feature runners (the accum-v3 SCALARIZE_MARKER would leak into the pipeline
# as an unregistered pass, and erase_self_copy/fuse_transpose_b would stop applying).
EMIT_TRANSLATE = "f.write(str(llvm.translate_module_to_llvmir(module.operation)))"
EMIT_DUMP = "f.write(str(module.operation))"

_RUNNER_SRC = r'''
import sys

from torch_mlir import ir
from torch_mlir.passmanager import PassManager
from torch_mlir.dialects import llvm
''' + _SELFCOPY_PRELUDE + _TRANSPOSE_FUSE_PRELUDE + _TRANSPOSE_MAPS_PRELUDE + _COPY_EXPAND_PRELUDE + _CONCAT_DPS_PRELUDE + _PARALLEL_GRAIN_PRELUDE + _MID_STAGE_SRC + _PARALLEL_GRAIN_LATE_SRC + DEALLOC_CHECK_PRELUDE + DEALLOC_CHECK_RUNNER + r'''
src_path, out_path, pipeline = sys.argv[1], sys.argv[2], sys.argv[3]
ctx = ir.Context()
with open(src_path) as f:
    module = ir.Module.parse(f.read(), ctx)
# fuse_transpose_b (default-off): fold `matmul(A, transpose(B))` into a transpose-b matmul BEFORE
# the pass manager runs, so the (still-named) linalg.matmul carries the transposed-B indexing map
# and the frozen RVV schedule tiles+vectorizes it while the scalar weight transpose disappears.
if _FUSE_TRANSPOSE_B:
    print("OK fuse_transpose_b", _fuse_transpose_b(module, ctx))
# fold_weight_transpose (default-off): the general form of the fold above -- a loop-invariant weight
# transpose sinks into the indexing_maps of EVERY linalg consumer, not just a linalg.matmul's B
# operand. The quantized datapath emits its contraction as a linalg.generic, so the matmul-only fold
# fires zero times on an int8 model; this one folds all 15 of small_llama's weight transposes.
if _FOLD_WEIGHT_TRANSPOSE:
    _fwt_n, _fwt_report = _fold_weight_transposes(module, ctx)
    for _fwt_kind, _fwt_detail in _fwt_report:
        print("OK fold_weight_transpose", _fwt_kind, _fwt_detail)
    print("OK fold_weight_transpose folded", _fwt_n)
# concat_dps (default-off): a `tensor.concat` operand produced by a destination-passing op is
# retargeted to write STRAIGHT INTO the concatenated buffer, so bufferization has no data movement
# left to emit for it. Must run BEFORE the pass manager: after one-shot-bufferize the destination is
# no longer an operand, only a copy. See llvmlower/concat_dps.py.
if _CONCAT_DPS:
    _cd_n, _cd_report = _concat_dps(module, ctx)
    for _cd_kind, _cd_detail in _cd_report:
        print("OK concat_dps", _cd_kind, _cd_detail)
    print("OK concat_dps rewrote", _cd_n)
_run_stages(ctx, module, pipeline, _ERASE_SELF_COPY, _MID_STAGES, _LATE_STAGES)
with open(out_path, "w") as f:
    __MERLIN_EMIT__
print("OK")
'''

_RUNNER = _RUNNER_SRC.replace("__MERLIN_EMIT__", EMIT_TRANSLATE)


# Runner variant for the vectorized_transcendental_activation feature: BEFORE the pass manager runs,
# rewrite every math.exp/erf/tanh into its inline arith polynomial (so the subsequent transform
# schedule vectorizes the activation generic's polynomial to vfmacc chains, instead of
# convert-math-to-libm scalarizing the math op to a libm call loop). The rewriter source is spliced
# in from act_poly.rewrite_source(); `_ir` is the alias the rewriter uses for the MLIR ir module.
# Default-off: only used when the feature is enabled (the baseline keeps the plain _RUNNER above).
_RUNNER_ACT_POLY_HEAD = r'''
import sys

from torch_mlir import ir
from torch_mlir import ir as _ir
from torch_mlir.passmanager import PassManager
from torch_mlir.dialects import llvm
'''

_RUNNER_ACT_POLY_TAIL = (_SELFCOPY_PRELUDE + _TRANSPOSE_FUSE_PRELUDE
                         + _TRANSPOSE_MAPS_PRELUDE + _COPY_EXPAND_PRELUDE
                         + _CONCAT_DPS_PRELUDE + _PARALLEL_GRAIN_PRELUDE + _MID_STAGE_SRC
                         + _PARALLEL_GRAIN_LATE_SRC
                         + DEALLOC_CHECK_PRELUDE + DEALLOC_CHECK_RUNNER + r'''
src_path, out_path, pipeline = sys.argv[1], sys.argv[2], sys.argv[3]
ctx = ir.Context()
with open(src_path) as f:
    module = ir.Module.parse(f.read(), ctx)
# The post-bufferization rewrites are NOT act_poly's business, but they have to run on THIS runner
# too. This tail used to drive the PassManager directly, which meant `erase_self_copy` and
# `fuse_transpose_b` were accepted, threaded through argv, and then never executed whenever
# `vectorized_transcendental_activation` was also enabled. MEASURED on small_llama_int8
# (hand_v0_int8 package, act_poly + erase_self_copy): the 17 in-loop `@memrefCopy` call sites the
# erase removes on the plain runner were all still there. Since the whole-model proposer enables
# act_poly by default, that is every beam fork -- and the erase read as an inert lever.
if _FUSE_TRANSPOSE_B:
    print("OK fuse_transpose_b", _fuse_transpose_b(module, ctx))
# fold_weight_transpose (default-off): the general form of the fold above -- a loop-invariant weight
# transpose sinks into the indexing_maps of EVERY linalg consumer, not just a linalg.matmul's B
# operand. The quantized datapath emits its contraction as a linalg.generic, so the matmul-only fold
# fires zero times on an int8 model; this one folds all 15 of small_llama's weight transposes.
if _FOLD_WEIGHT_TRANSPOSE:
    _fwt_n, _fwt_report = _fold_weight_transposes(module, ctx)
    for _fwt_kind, _fwt_detail in _fwt_report:
        print("OK fold_weight_transpose", _fwt_kind, _fwt_detail)
    print("OK fold_weight_transpose folded", _fwt_n)
# concat_dps (default-off): a `tensor.concat` operand produced by a destination-passing op is
# retargeted to write STRAIGHT INTO the concatenated buffer, so bufferization has no data movement
# left to emit for it. Must run BEFORE the pass manager: after one-shot-bufferize the destination is
# no longer an operand, only a copy. See llvmlower/concat_dps.py.
if _CONCAT_DPS:
    _cd_n, _cd_report = _concat_dps(module, ctx)
    for _cd_kind, _cd_detail in _cd_report:
        print("OK concat_dps", _cd_kind, _cd_detail)
    print("OK concat_dps rewrote", _cd_n)
with ctx, ir.Location.unknown():
    _n = apply_activation_polynomial(module, ctx)
_run_stages(ctx, module, pipeline, _ERASE_SELF_COPY, _MID_STAGES, _LATE_STAGES)
with open(out_path, "w") as f:
    __MERLIN_EMIT__
print("OK act_poly rewrote", _n)
''')


def _accum_microkernel_v3_features() -> frozenset[str]:
    """Names of the accumulator-resident micro-kernel v3 features (the ones whose pipeline splices the
    SCALARIZE_MARKER and need the two-stage A-scalarization runner). Imported lazily so importing
    pipeline never pulls accum_microkernel / impr_features at module load."""
    from .impr_features import ACCUM_RESIDENT_V3_NAMES
    return frozenset(ACCUM_RESIDENT_V3_NAMES)


def _needs_scalarize_runner(pipeline: str, feats: "frozenset[str]") -> bool:
    """Does this lowering need the two-stage A-scalarization runner?

    GROUND TRUTH is the marker's presence in the built pass pipeline — NOT membership in the static
    ACCUM_RESIDENT_V3_NAMES grid. The register block is a continuous beam-tunable knob, so v3 tuning
    points are registered on demand for arbitrary (MR,NR,KC); a static-name check silently missed them,
    the plain runner was used, and SCALARIZE_MARKER leaked into the pipeline as if it were a pass
    ("'__merlin_scalarize_a__' does not refer to a registered pass") — every non-grid MR failed at K2.
    Keying on the marker makes this correct for any registration mechanism."""
    from .accum_microkernel import SCALARIZE_MARKER
    if SCALARIZE_MARKER in pipeline:
        return True
    return any(f in feats for f in _accum_microkernel_v3_features())


def _activation_poly_runner(emit: str = EMIT_TRANSLATE) -> str:
    """The lowering runner with the transcendental->polynomial rewriter spliced in (default-off
    feature). Imported here (not at module top) so importing pipeline never pulls act_poly."""
    from .act_poly import rewrite_source
    return (_RUNNER_ACT_POLY_HEAD + rewrite_source()
            + _RUNNER_ACT_POLY_TAIL.replace("__MERLIN_EMIT__", emit))


def _select_runner(pipeline: str, feats: "frozenset[str]", *, emit: str) -> str:
    """Pick the lowering-runner source for these features and bind how it emits its result.

    The runner variant is chosen by FEATURE (act_poly rewriter / accum-v3 two-stage
    scalarization / plain); ``emit`` is orthogonal and chosen by TRANSPORT (in-process
    translate vs dump-for-mlir-translate). Keeping the two independent is what lets the
    multicore path carry the feature rewrites — see the EMIT_* comment.
    """
    if "vectorized_transcendental_activation" in feats:
        return _activation_poly_runner(emit)
    if _needs_scalarize_runner(pipeline, feats):
        from .accum_microkernel import run_source
        return run_source().replace("__MERLIN_EMIT__", emit)
    return _RUNNER_SRC.replace("__MERLIN_EMIT__", emit)


class PipelineError(RuntimeError):
    pass


def lower_to_llvm_ir(mlir_text: str, workdir: str | Path | None = None,
                     pipeline: str | None = None, timeout: int = 7200,
                     vectorize: bool = False, transform_schedule: str | None = None,
                     hoist_static_allocs: bool = True, parallel: bool = False,
                     features: "frozenset[str] | None" = None,
                     parallel_harts: int | None = None) -> str:
    """Lower upstream-MLIR text to LLVM IR text via the m2m venv. Returns .ll text.

    ``vectorize=True`` selects the native RVV path: writes the transform schedule into
    ``workdir`` and uses :func:`build_rvv_pipeline` so the IR carries fixed-width vector
    ops (real RVV under ``-march=rv64gcv``). ``transform_schedule`` overrides the default
    matmul/batch_matmul schedule (e.g. the elementwise/reduction schedule for the vector
    family) without disturbing it. An explicit ``pipeline`` overrides both defaults.

    ``parallel_harts=N`` (>= 2) additionally makes the RVV path MULTICORE: an outer
    ``scf.forall`` over N chunks is layered under the package schedule and lowered to
    OpenMP, so the emitted object carries both real RVV vectors and ``__kmpc_*`` calls.
    Unlike ``parallel=True`` (the scalar-only OpenMP path used for K1 big models) it
    COMPOSES with ``vectorize`` — vector and threads, not one or the other. Default None
    keeps the pipeline string byte-identical to the shipping serial codegen.
    """
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="merlin_lower_"))
    work.mkdir(parents=True, exist_ok=True)
    from .impr_features import apply_schedule, normalize
    # `normalize` REJECTS an unregistered name, and the lowering runs in forked/child processes that
    # import this module without importing `llvmlower.lower` (where the runner-gated features are
    # registered). Registering here, before the set is validated, is what keeps a feature the caller
    # legitimately asked for from failing as "unknown impr feature" in one process and working in
    # another. Idempotent.
    from .transpose_maps import ensure_registered as _register_fold_weight_transpose
    _register_fold_weight_transpose()
    from .weight_prepack import ensure_registered as _register_prepack_weight_layout
    _register_prepack_weight_layout()
    from .concat_dps import ensure_registered as _register_concat_dps
    _register_concat_dps()
    # The direct-conv arm's REQUEST feature. Registered here for the same reason as the four above:
    # this module is imported by the lowering child processes, and a name registered only in the
    # parent fails `normalize` in the child. It edits nothing here -- the arm is realized by the
    # per-op block table (llvmlower.perop_blocks), which the preparation step derives -- so a build
    # that carries the name lowers identically and the name records that the arm was asked for.
    from .perop_blocks import ensure_registered as _register_conv_arm
    _register_conv_arm()
    # The panel-packed im2col REQUEST, registered for the same reason: the rewrite happens in the
    # preparation step, and by lowering time the name only records that the arm was asked for.
    from .im2col_pack import ensure_registered as _register_im2col_panel_pack
    _register_im2col_panel_pack()
    feats = normalize(features)
    if parallel_harts is not None and not vectorize:
        raise PipelineError(
            "parallel_harts requires vectorize=True (it layers an outer forall UNDER the "
            "RVV transform schedule); for the scalar OpenMP path use parallel=True")
    # OpenMP IR — whatever produced it — must take the out-of-process translate route below.
    omp = parallel or parallel_harts is not None
    if pipeline is None:
        if vectorize:
            sched = work / "rvv_schedule.mlir"
            sched_text = apply_schedule(transform_schedule or RVV_TRANSFORM_SCHEDULE, feats)
            sched.write_text(sched_text, encoding="utf-8")
            par_sched = None
            if parallel_harts is not None:
                par_sched = work / "rvv_parallel_schedule.mlir"
                par_sched.write_text(parallel_transform_schedule(parallel_harts),
                                     encoding="utf-8")
            # The non-contraction arms, as their own preloaded library, so they run while the
            # `merlin.vec_r{rank}` tags they match on are still on the ops. None (and no file) when
            # the lever is off -- see `vec_pre_schedule`.
            vec_sched = None
            vec_text = vec_pre_schedule(feats)
            if vec_text is not None:
                vec_sched = work / "rvv_vec_pre_schedule.mlir"
                vec_sched.write_text(vec_text, encoding="utf-8")
            pipeline = build_rvv_pipeline(sched, hoist_static_allocs=hoist_static_allocs,
                                          features=feats, par_sched_path=par_sched,
                                          vec_sched_path=vec_sched)
        elif parallel:
            pipeline = _parallel_pipeline()   # multicore (OpenMP) scalar path — K1 big models
        else:
            pipeline = _upstream_pipeline()
    src = work / "model.mlir"
    out = work / "model.ll"
    runner = work / "run_lowering.py"
    src.write_text(mlir_text, encoding="utf-8")

    # The vectorized_transcendental_activation feature splices a math.exp/erf/tanh -> arith
    # polynomial rewriter into the runner (run before the pass manager). The accumulator-resident
    # micro-kernel v3 feature splices a two-stage runner that runs the A-operand scalarization rewrite
    # BETWEEN two pass-manager stages (split at the SCALARIZE_MARKER pass name). Default-off; with the
    # feature absent the plain _RUNNER is used and the lowering is byte-identical to the baseline.
    # The SAME selection is used on the OpenMP transport (only `emit` differs), so a multicore build
    # of a v3/act_poly package still gets its rewrites.
    runner_src = _select_runner(pipeline, feats, emit=EMIT_DUMP if omp else EMIT_TRANSLATE)
    runner.write_text(runner_src, encoding="utf-8")
    # argv[4] gates the self-copy erase, so the frozen hand_v0 control keeps its byte-identical
    # lowering unless the feature is explicitly enabled.
    from .selfcopy import FEATURE as _SELFCOPY_FEATURE, with_canonicalize as _with_canon
    from .transpose_fuse import FEATURE as _FUSE_TRANSPOSE_FEATURE
    _erase = "1" if _SELFCOPY_FEATURE in feats else "0"
    if _erase == "1":
        pipeline = _with_canon(pipeline)
    # argv[5] gates fuse_transpose_b (default-off). Every runner variant honors it (the act_poly tail
    # used to drop it along with the self-copy erase); with the feature off the lowering stays
    # byte-identical.
    _fuse_tb = "1" if _FUSE_TRANSPOSE_FEATURE in feats else "0"
    # argv[6] gates expand_memref_copy (default-off): every static `memref.copy` becomes a
    # `linalg.copy`, which the pipeline's own convert-linalg-to-loops turns into an emitted scf
    # load/store nest -- so finalize-memref-to-llvm has no copy left to turn into an @memrefCopy or
    # memcpy call. Runs at the same split point as the erase; see llvmlower/copy_expand.py.
    from .copy_expand import FEATURE as _EXPAND_COPY_FEATURE
    _expand_copy = "1" if _EXPAND_COPY_FEATURE in feats else "0"
    # argv[7] gates fold_weight_transpose (default-off): sink a loop-invariant weight transpose into
    # its consumers' indexing_maps. See llvmlower/transpose_maps.py.
    from .transpose_maps import FEATURE as _FOLD_WEIGHT_TRANSPOSE_FEATURE
    _fold_wt = "1" if _FOLD_WEIGHT_TRANSPOSE_FEATURE in feats else "0"
    # argv[8] gates concat_dps (default-off): a tensor.concat's destination-passing producers write
    # straight into the concatenated buffer. See llvmlower/concat_dps.py.
    from .concat_dps import FEATURE as _CONCAT_DPS_FEATURE
    _concat_dps_gate = "1" if _CONCAT_DPS_FEATURE in feats else "0"
    # argv[9] carries the multicore fork/join GRAIN threshold (default-off: "0"). Every
    # `scf.parallel` cheaper than it becomes a serial `scf.for` nest before convert-scf-to-openmp,
    # so no fork is emitted for it. See llvmlower/parallel_grain.py; 0 -> byte-identical lowering.
    from .parallel_grain import threshold_of as _parallel_grain_threshold
    _grain = _parallel_grain_threshold(feats)
    if _grain is not None and not omp:
        import sys as _sys
        # Say so rather than shipping a build whose named lever cannot fire: without the multicore
        # lowering there is no `scf.parallel` at all, so the grain would report 0 and read as inert.
        print("[parallel_grain] WARNING: a grain threshold is named but this lowering is SERIAL "
              "(no parallel_harts/parallel), so there is no scf.parallel to price; the feature "
              "will serialize nothing.", file=_sys.stderr, flush=True)
    _grain_gate = str(int(_grain)) if _grain is not None else "0"
    # OpenMP transport: the runner DUMPS the LLVM-dialect module and the standalone
    # mlir-translate produces the .ll out-of-process (the in-process torch-mlir bridge
    # segfaults on omp IR). Otherwise the runner writes the .ll directly.
    stage_out = (work / "model.llvmdialect.mlir") if omp else out
    proc = subprocess.run(
        [str(m2m_python()), str(runner), str(src), str(stage_out), pipeline, _erase, _fuse_tb,
         _expand_copy, _fold_wt, _concat_dps_gate, _grain_gate],
        capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not stage_out.is_file():
        raise PipelineError(f"upstream lowering failed:\n{proc.stdout}\n{proc.stderr}")
    if SINK_DEALLOC_PASS in pipeline and DEALLOC_CHECK_TOKEN not in proc.stdout:
        # The sinking stage ran and the use-after-free check did NOT. That is only reachable from a
        # runner variant that drives the PassManager itself instead of going through `_run_stages`
        # (the accumulator-resident v3 runner does, for its first stage), and it would sink every
        # free with nothing verifying the placement -- silently wrong numbers, not a failed build.
        # Refuse the artifact rather than ship an unchecked one.
        raise PipelineError(
            "the dealloc sinking stage ran but its use-after-free check did not report "
            f"({DEALLOC_CHECK_TOKEN!r} absent from the runner's output). This runner variant "
            "bypasses `_run_stages`; do not combine MERLIN_SINK_DEALLOCS with it until the check "
            f"is wired there too.\n{proc.stdout}")
    if omp:
        from .toolchain import mlir_translate
        tproc = subprocess.run(
            [str(mlir_translate()), "--mlir-to-llvmir", str(stage_out), "-o", str(out)],
            capture_output=True, text=True, timeout=timeout)
        if tproc.returncode != 0 or not out.is_file():
            raise PipelineError(f"mlir-translate (parallel) failed:\n{tproc.stdout}\n{tproc.stderr}")
    return _fix_float_literals(out.read_text(encoding="utf-8"))


import struct as _struct

# The MLIR printer emits float immediates in MLIR's own hex spelling, e.g. the real line
#     %9 = fmul float f0x3F4C422A, %8
# from a bf16 gelu capsule's model.ll. LLVM's textual parser has no `f0x` literal, so the
# module fails to parse. LLVM IR wants a bare `0x…`, and a *float* hex literal must carry
# the 64-bit pattern of the double the f32 represents — so an 8-hex (f32) payload is
# widened, a 16-hex (f64) payload only sheds the `f`.
_F0X = "f0x"
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def _is_word_char(ch: str) -> bool:
    """The `\\b` before `f0x`: a preceding identifier character means this is the tail of a
    longer token (e.g. `named_f0x…`), not a float literal. `%` is not one, so `%f0x…` — an
    LLVM local whose name happens to start that way — IS treated as a literal, matching the
    behaviour this replaced."""
    return ch.isalnum() or ch == "_"


def _f0x_payload_to_llvm(hexs: str) -> str:
    if len(hexs) == 16:                       # already a double bit pattern
        return f"0x{hexs.upper()}"
    bits = int(hexs, 16)                      # f32 -> widen to double
    f32 = _struct.unpack("<f", _struct.pack("<I", bits))[0]
    dbits = _struct.unpack("<Q", _struct.pack("<d", f32))[0]
    return f"0x{dbits:016X}"


def _fix_f0x_literals(ll_text: str) -> str:
    """Rewrite every `f0x<8 or 16 hex>` token to its LLVM `0x…` spelling.

    Scanned structurally: find the literal `f0x`, require an identifier boundary before it,
    then take the maximal run of hex digits after it. Only a run of EXACTLY 8 (f32) or 16
    (f64) digits is a float literal — any other length is not something this repair
    understands, so it is left untouched for the LLVM parser to reject loudly rather than
    being half-rewritten into a plausible-looking wrong constant.
    """
    out: list[str] = []
    i = 0
    while True:
        j = ll_text.find(_F0X, i)
        if j < 0:
            out.append(ll_text[i:])
            break
        out.append(ll_text[i:j])
        i = j + len(_F0X)
        if j > 0 and _is_word_char(ll_text[j - 1]):
            out.append(_F0X)                  # inside a longer token — not a literal
            continue
        k = i
        while k < len(ll_text) and ll_text[k] in _HEX_DIGITS:
            k += 1
        if k - i not in (8, 16):
            out.append(_F0X)                  # unknown payload width — leave verbatim
            continue
        out.append(_f0x_payload_to_llvm(ll_text[i:k]))
        i = k
    return "".join(out)


def _fix_float_literals(ll_text: str) -> str:
    """The MLIR LLVM-IR printer emits non-LLVM float literals (bare inf/-inf/nan,
    and `f0x..` f32 hex) that the textual parser rejects — canonicalize them."""
    ll_text = _fix_f0x_literals(ll_text)
    # half (fp16) uses the LLVM `0xH<4hex>` literal form, bfloat (bf16) uses `0xR<4hex>` — the MLIR
    # printer emits bare inf/-inf/nan for these too (e.g. an fp16 causal-mask -inf), which the LLVM
    # textual parser rejects. Order matters: replace the widest names first so `float`↛`bfloat` etc.
    return (ll_text
            .replace("bfloat -inf", "bfloat 0xRFF80")
            .replace("bfloat inf", "bfloat 0xR7F80")
            .replace("bfloat nan", "bfloat 0xR7FC0")
            .replace("half -inf", "half 0xHFC00")
            .replace("half inf", "half 0xH7C00")
            .replace("half nan", "half 0xH7E00")
            .replace("float -inf", "float 0xFFF0000000000000")
            .replace("float inf", "float 0x7FF0000000000000")
            .replace("float nan", "float 0x7FF8000000000000")
            .replace("double -inf", "double 0xFFF0000000000000")
            .replace("double inf", "double 0x7FF0000000000000")
            .replace("double nan", "double 0x7FF8000000000000"))
