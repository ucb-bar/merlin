"""RVV Transform/LLVM pipeline owned by the pinned RVV support provider.

The pass sequence is extracted byte-for-byte from Merlin's proven RVV path. The
upstream MLIR pass registry remains the toolchain; the provider chooses their
ordering and the schedule library. API_VERSION is checked by Merlin's loader.
"""

# ruff: noqa: UP031 -- the extracted Transform skeleton uses literal percent escapes

from __future__ import annotations

import os
from pathlib import Path

API_VERSION = 1
DEFAULT_SCHEDULE = "impr_tuned_wholemodel_vf_int8"
RVV_FLAGS = ("-march=rv64gcv", "-fno-vectorize", "-fno-slp-vectorize")
DEFAULT_CFLAGS = RVV_FLAGS
SPIKE_CFLAGS = (
    "-march=rv64gcv",
    "-mabi=lp64d",
    "-mcmodel=medany",
    "-O2",
    "-ffreestanding",
    "-fno-builtin",
)

# Deallocation and generic pass-list splicing remain shared Merlin infrastructure.
from merlin.llvmlower.compiler_api import (
    PipelineError,
    class_token,
    par_tag_for,
    splice_passes,
)
from merlin.llvmlower.feature_registry import FeatureRegistry, ImprFeature

from .accumulator_grid import register_accumulator_grids
from .active_features import (
    WHOLEMODEL_VF_NAME,
    fused_vfmacc_tiled_schedule,
    wholemodel_pipeline,
    wholemodel_schedule,
    wholemodel_vf_pipeline,
    wholemodel_vf_schedule,
)
from .grid_features import register_grid_features
from .other_vector_features import register_other_vector_features
from .vector_features import (
    ensure_vec_noncontraction,
    vec_noncontraction_lanes,
)
from .vector_features import (
    register_base as register_vec_noncontraction,
)
from .vector_features import (
    register_from_name as register_vec_noncontraction_from_name,
)


def _legacy_feature(name: str) -> bool:
    """Migration bridge for experimental features not yet extracted to this provider."""
    if register_vec_noncontraction_from_name(_FEATURES, name):
        return True
    from .legacy_features import register_existing

    return register_existing(name, _FEATURES)


_FEATURES = FeatureRegistry(lazy_register=_legacy_feature)
register_vec_noncontraction(_FEATURES)
register_grid_features(_FEATURES)
register_accumulator_grids(_FEATURES)
register_other_vector_features(_FEATURES)
_FEATURES.register(
    ImprFeature(
        name="approximate_transcendental_activation",
        action_class="PASS",
        description="Target-neutral activation polynomial rewrite before target lowering.",
    )
)
_FEATURES.register(
    ImprFeature(
        name="erase_self_copy",
        action_class="PASS",
        description="Target-neutral self-copy erase in the shared post-bufferization runner.",
    )
)
_FEATURES.register(
    ImprFeature(
        name="accumulator_resident_wholemodel",
        action_class="PASS",
        description="Whole-model accumulator-resident RVV vector schedule and lowering.",
        edit_pipeline=wholemodel_pipeline,
        edit_schedule=lambda _text: wholemodel_schedule(),
        schedule_replace=True,
    )
)
_FEATURES.register(
    ImprFeature(
        name=WHOLEMODEL_VF_NAME,
        action_class="PASS",
        description="Published whole-model-safe vfmacc.vf accumulator-resident schedule and lowering.",
        edit_pipeline=wholemodel_vf_pipeline,
        edit_schedule=lambda _text: wholemodel_vf_schedule(),
        schedule_replace=True,
    )
)
_FEATURES.register(
    ImprFeature(
        name="prepack_weight_layout",
        action_class="PASS",
        description="Weight-layout preparation selected before RVV pipeline construction.",
    )
)
_FEATURES.register(
    ImprFeature(
        name="fuse_quantize_round_convert",
        action_class="PASS",
        description="Target-neutral exact quantization rewrite performed before target lowering.",
    )
)
_FEATURES.register(
    ImprFeature(
        name="fused_vfmacc_tiled",
        action_class="PASS",
        description="Bounded tiled RVV fused-vfmacc Transform schedule.",
        edit_schedule=lambda _text: fused_vfmacc_tiled_schedule(),
        schedule_replace=True,
    )
)


def normalize_features(features) -> frozenset[str]:
    return _FEATURES.normalize(features)


def accum_microkernel_marker() -> str:
    from .accum_microkernel import SCALARIZE_MARKER

    return SCALARIZE_MARKER


def accum_microkernel_rewrite_source() -> str:
    from .accum_microkernel import rewrite_source

    return rewrite_source()


def accum_microkernel_runner_source(*, tag_bmm_tails: bool = False) -> str:
    from .accum_microkernel import run_source

    return run_source(tag_bmm_tails=tag_bmm_tails)


def apply_schedule(schedule_text: str, features: frozenset[str]) -> str:
    return _FEATURES.apply_schedule(schedule_text, features)


def apply_pipeline(passes: list[str], features: frozenset[str]) -> list[str]:
    return _FEATURES.apply_pipeline(passes, features)


def apply_cflags(cflags: list[str], features: frozenset[str]) -> list[str]:
    return _FEATURES.apply_cflags(cflags, features)


def prepare_build_bundle(
    src: Path | str, work: Path | str, features: frozenset[str] | None
) -> Path:
    """Apply package-owned weight layout preparation before RVV model lowering."""
    from .prepack import prepare_build_bundle as prepare

    return prepare(src, work, normalize_features(features or frozenset()))


def vector_cflags(common: list[str] | tuple[str, ...]) -> list[str]:
    """Add the RVV ISA and controlled-vectorization policy to shared target flags."""
    return [*RVV_FLAGS, *common]


def spike_cflags() -> list[str]:
    """Frozen RVV bare-metal model flags for the Spike backend."""
    return list(SPIKE_CFLAGS)


def spike_harness_cflags(model_flags: list[str]) -> list[str]:
    """Use the selected ISA/ABI with GCC-compatible harness options."""
    flags = spike_cflags()
    for prefix in ("-march=", "-mabi=", "-mcmodel="):
        selected = [flag for flag in model_flags if flag.startswith(prefix)]
        if selected:
            flags = [flag for flag in flags if not flag.startswith(prefix)]
            flags.append(selected[-1])
    return flags


def schedule_path(name: str = DEFAULT_SCHEDULE) -> Path:
    """Resolve a schedule shipped by this exact provider revision."""
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"invalid RVV schedule name {name!r}")
    root = Path(__file__).resolve().parent / "schedules" / name
    for file in ("manifest.yaml", "knobs.yaml", "schedule.mlir"):
        if not (root / file).is_file():
            raise FileNotFoundError(root / file)
    return root


def baseline_schedule_text() -> str:
    """Return the frozen, package-owned schedule used when a caller supplies none."""
    return (schedule_path() / "schedule.mlir").read_text(encoding="utf-8")


def _fuse_post() -> bool:
    return bool(os.environ.get("MERLIN_FUSE_POST"))


def _par_residue_parallel(features: frozenset[str]) -> bool:
    matches = [
        name for name in features if name.startswith("parallelize_residual_loops_")
    ]
    if len(matches) > 1:
        raise ValueError(
            f"multiple residual-loop thresholds selected: {sorted(matches)}"
        )
    if matches:
        threshold = int(matches[0].removeprefix("parallelize_residual_loops_"))
        if threshold < 0:
            raise ValueError("residual-loop threshold must be nonnegative")
        return True
    return bool(os.environ.get("MERLIN_PAR_RESIDUE_PARALLEL"))


PARALLEL_ENTRY = "__transform_parallel_main"

_PARALLEL_DIM_NUM_THREADS = {
    # op -> {dim: num_threads list}. A 0 means "do not tile this dim".
    "linalg.matmul": {"m": "[{n}]", "n": "[0, {n}]"},
    "linalg.batch_matmul": {"b": "[{n}]", "m": "[0, {n}]", "n": "[0, 0, {n}]"},
}


def parallel_transform_schedule(
    n_harts: int,
    *,
    matmul_dim: str = "n",
    batch_matmul_dim: str = "b",
    chunks: list | None = None,
    tile_aligned: bool = False,
) -> str:
    """Transform schedule that wraps each contraction in an `scf.forall` over ``n_harts``.

    Runs BEFORE the package's own schedule (separate entry point, see above), so the inner
    tiling/vectorization — and therefore the emitted `vfmacc`/`vwmacc` — is untouched.
    ``matmul_dim`` / ``batch_matmul_dim`` name the parallel dim to split; K is never
    tileable here (it is the reduction dim and splitting it would race).

    ``chunks`` (from :func:`perop_blocks.distinct_parallel_arms`) selects the PER-OP form, and it is
    the form that keeps the claim in the paragraph above true. The class-wide ``num_threads`` split
    below leaves each hart a ``ceil(dim / n_harts)`` tile the register block then has to mask, so the
    block — and with it the emitted kernel — became a function of the hart count: measured on
    lstmnetvit int8 at 8 harts, 5 matmuls fell out of the block table to scalar loops and the linked
    ELF issued +63.5% instructions with 63% fewer vector ops before any thread was created. With
    ``chunks``, merlin has already derived a tile that divides the dim EXACTLY and that the op's own
    block still lowers on (``perop_blocks.parallel_chunk_table``), one arm per distinct (class, tile),
    matched by the ``merlin.par_*`` tag the same prepare step applies. An op merlin could not split
    that way carries no tag, matches no arm, and stays serial — with the 1-hart kernel intact.

    ``chunks=None`` keeps the class-wide ``num_threads`` schedule byte-identical, for the packages
    that carry no per-op block table for a split to be derived against.
    """
    if n_harts < 2:
        raise ValueError(f"parallel schedule needs n_harts >= 2, got {n_harts}")
    if chunks is not None:
        return _perop_parallel_schedule(chunks)
    if tile_aligned:
        body = [
            (
                '    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op\n'
                "    %matmul_loop, %matmul_tiled = transform.structured.tile_using_forall %matmul tile_sizes [0, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)"
            ),
            (
                '    %batch_matmul = transform.structured.match ops{["linalg.batch_matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op\n'
                "    %batch_matmul_loop, %batch_matmul_tiled = transform.structured.tile_using_forall %batch_matmul tile_sizes [1, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)"
            ),
        ]
        return (
            "module attributes {transform.with_named_sequence} {\n"
            f"  transform.named_sequence @{PARALLEL_ENTRY}"
            "(%arg0: !transform.any_op {transform.readonly}) {\n"
            + "\n".join(body)
            + "\n    transform.yield\n  }\n}\n"
        )
    body = []
    for op, dim in (
        ("linalg.matmul", matmul_dim),
        ("linalg.batch_matmul", batch_matmul_dim),
    ):
        choices = _PARALLEL_DIM_NUM_THREADS[op]
        if dim not in choices:
            raise ValueError(
                f"{op}: parallel dim {dim!r} not in {sorted(choices)} (the reduction dim K is never tileable)"
            )
        tag = op.split(".")[-1]
        body.append(
            f'    %{tag} = transform.structured.match ops{{["{op}"]}} in %arg0 '
            f": (!transform.any_op) -> !transform.any_op\n"
            f"    %{tag}_loop, %{tag}_tiled = transform.structured.tile_using_forall "
            f"%{tag} num_threads {choices[dim].format(n=n_harts)} "
            f": (!transform.any_op) -> (!transform.any_op, !transform.any_op)"
        )
    return (
        "module attributes {transform.with_named_sequence} {\n"
        f"  transform.named_sequence @{PARALLEL_ENTRY}"
        "(%arg0: !transform.any_op {transform.readonly}) {\n" + "\n".join(body) + "\n"
        "    transform.yield\n"
        "  }\n"
        "}\n"
    )


def _perop_parallel_schedule(chunks: list) -> str:
    """The PER-OP multicore library: one ``tile_using_forall`` arm per distinct (class, tile).

    Matched by the ``merlin.par_*`` attribute, not by op name, for the same reason the block arms
    match ``merlin.blk_*``: two contractions of the same class legitimately want different splits,
    and a name match cannot tell them apart. Uses ``tile_sizes`` rather than ``num_threads`` because
    the tile is the thing that has to be exact — merlin chose one that divides the extent, so every
    chunk is STATIC and equal, and the package schedule below sees the same geometry it would have
    seen at 1 hart.

    An empty ``chunks`` (a hart count was asked for, but no contraction has a dim that splits into
    equal block-legal pieces) yields a schedule with no arms: the module reaches the package schedule
    exactly as the serial build does and the image runs on one core. Said rather than silently
    produced -- see :func:`lower_to_llvm_ir`, which prints it.
    """
    body = []
    for i, (op, tiles) in enumerate(chunks):
        tiles = tuple(int(t) for t in tiles)
        h = f"p{i}{class_token(op)}"
        body.append(
            f"    %{h} = transform.structured.match attributes{{{par_tag_for(op, tiles)}}} in %arg0 "
            f": (!transform.any_op) -> !transform.any_op\n"
            f"    %{h}_loop, %{h}_tiled = transform.structured.tile_using_forall "
            f"%{h} tile_sizes [{', '.join(str(t) for t in tiles)}] "
            f": (!transform.any_op) -> (!transform.any_op, !transform.any_op)"
        )
    return (
        "module attributes {transform.with_named_sequence} {\n"
        f"  transform.named_sequence @{PARALLEL_ENTRY}"
        "(%arg0: !transform.any_op {transform.readonly}) {\n"
        + ("\n".join(body) + "\n" if body else "")
        + "    transform.yield\n"
        "  }\n"
        "}\n"
    )


#: Entry point of the library that runs the NON-CONTRACTION vectorize arms, before specialization.
VEC_PRE_ENTRY = "__transform_vec_main"

#: The skeleton the arms are spliced into. It is deliberately the SAME splice the package schedule
#: gets (``impr_features.apply_schedule``), anchored on the same ``%f = ... ops{["func.func"]}`` line,
#: so the arms in the pre-library and the arms in ``__transform_main`` are generated by one function
#: at one lane width and cannot drift apart.
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


def vec_pre_schedule(features: frozenset[str]) -> str | None:
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
    if _vec_after_specialize():
        return None
    lanes = vec_noncontraction_lanes(features)
    if lanes is None:
        return None
    text = apply_schedule(
        _VEC_PRE_SKELETON, frozenset({ensure_vec_noncontraction(_FEATURES, lanes)})
    )
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


def build_pipeline(
    sched_path: str | Path,
    hoist_static_allocs: bool = True,
    features: frozenset[str] = frozenset(),
    par_sched_path: str | Path | None = None,
    vec_sched_path: str | Path | None = None,
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
    libs = [
        str(p) for p in (vec_sched_path, par_sched_path, sched_path) if p is not None
    ]
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
        [
            preload,
            f"transform-interpreter{{entry-point={VEC_PRE_ENTRY}}}",
            "canonicalize",
            "cse",
            *specialize,
        ]
        if vec
        else [*specialize, preload]
    )
    passes = [
        "canonicalize",
        "cse",
        *head,
        *(
            [
                f"transform-interpreter{{entry-point={PARALLEL_ENTRY}}}",
                "canonicalize",
                "cse",
            ]
            if par
            else []
        ),
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
        *(
            ["func.func(linalg-fuse-elementwise-ops)", "canonicalize", "cse"]
            if _fuse_post()
            else []
        ),
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
        "canonicalize",
        "cse",
        "symbol-dce",
    ]
    if features:  # default-off impr-fork hooks; empty -> byte-identical
        passes = apply_pipeline(passes, features)
    return splice_passes(passes)
