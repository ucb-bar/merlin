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


def _dealloc_passes() -> list[str]:
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
    """
    import os
    if os.environ.get("MERLIN_NO_DEALLOC"):
        return []
    return ["ownership-based-buffer-deallocation",
            "buffer-deallocation-simplification",
            "bufferization-lower-deallocations"]
    # NOT here: `func.func(optimize-allocation-liveness)`. It looked like the obvious next lever --
    # sink each dealloc to just after its last user and the peak drops -- but measured on the two
    # models that bracket the range it moved nothing: whisper_tiny stayed at 2184.1 MB and deepjscc at
    # 48.1 MB, byte for byte, while still perturbing the IR. Ownership-based deallocation is already
    # placing the frees tightly enough that there is no slack to reclaim. A pass with no measured
    # effect is not free -- it is another thing that can break a build -- so it stays out until some
    # model demonstrates a delta.


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
    index that drifts when the list is edited."""
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


def build_rvv_pipeline(sched_path: "str | Path", hoist_static_allocs: bool = True,
                       features: "frozenset[str]" = frozenset(),
                       par_sched_path: "str | Path | None" = None) -> str:
    """Whole-module pipeline with the transform vectorization stage spliced in after
    named-op generalization (vectorize on tensors) and before bufferization, plus the
    vector-lowering passes needed to reach LLVM. ``sched_path`` is the preloaded schedule.

    ``par_sched_path`` (default None -> byte-identical to the shipping serial pipeline)
    enables the multicore variant: the parallel library is preloaded alongside the package
    schedule, its entry point runs FIRST (wrapping each contraction in an ``scf.forall``),
    and the loop-generation/LLVM stages gain the forall->parallel->OpenMP conversions.

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
    preload = (f"transform-preload-library{{transform-library-paths={par_sched_path},{sched_path}}}"
               if par else
               f"transform-preload-library{{transform-library-paths={sched_path}}}")
    passes = [
        "canonicalize", "cse",
        # Recover named contraction ops (matmul/batch_matmul) from the capture's generics so
        # the schedule can match them, THEN vectorize. Do NOT run linalg-fuse-elementwise-ops
        # before this — it folds matmuls into fused generics and the `ops{["linalg.matmul"]}`
        # match then finds nothing (silent 0-vectorization).
        "func.func(linalg-specialize-generic-ops)",
        preload,
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
from .copy_expand import MID_STAGE_SRC as _MID_STAGE_SRC
from .copy_expand import RUNNER_PRELUDE as _COPY_EXPAND_PRELUDE
from .selfcopy import RUNNER_PRELUDE as _SELFCOPY_PRELUDE
from .transpose_fuse import RUNNER_PRELUDE as _TRANSPOSE_FUSE_PRELUDE

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
''' + _SELFCOPY_PRELUDE + _TRANSPOSE_FUSE_PRELUDE + _COPY_EXPAND_PRELUDE + _MID_STAGE_SRC + r'''
src_path, out_path, pipeline = sys.argv[1], sys.argv[2], sys.argv[3]
ctx = ir.Context()
with open(src_path) as f:
    module = ir.Module.parse(f.read(), ctx)
# fuse_transpose_b (default-off): fold `matmul(A, transpose(B))` into a transpose-b matmul BEFORE
# the pass manager runs, so the (still-named) linalg.matmul carries the transposed-B indexing map
# and the frozen RVV schedule tiles+vectorizes it while the scalar weight transpose disappears.
if _FUSE_TRANSPOSE_B:
    print("OK fuse_transpose_b", _fuse_transpose_b(module, ctx))
_run_stages(ctx, module, pipeline, _ERASE_SELF_COPY, _MID_STAGES)
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

_RUNNER_ACT_POLY_TAIL = (_SELFCOPY_PRELUDE + _TRANSPOSE_FUSE_PRELUDE + _COPY_EXPAND_PRELUDE + _MID_STAGE_SRC + r'''
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
with ctx, ir.Location.unknown():
    _n = apply_activation_polynomial(module, ctx)
_run_stages(ctx, module, pipeline, _ERASE_SELF_COPY, _MID_STAGES)
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
            pipeline = build_rvv_pipeline(sched, hoist_static_allocs=hoist_static_allocs,
                                          features=feats, par_sched_path=par_sched)
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
    # OpenMP transport: the runner DUMPS the LLVM-dialect module and the standalone
    # mlir-translate produces the .ll out-of-process (the in-process torch-mlir bridge
    # segfaults on omp IR). Otherwise the runner writes the .ll directly.
    stage_out = (work / "model.llvmdialect.mlir") if omp else out
    proc = subprocess.run(
        [str(m2m_python()), str(runner), str(src), str(stage_out), pipeline, _erase, _fuse_tb,
         _expand_copy],
        capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not stage_out.is_file():
        raise PipelineError(f"upstream lowering failed:\n{proc.stdout}\n{proc.stderr}")
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
