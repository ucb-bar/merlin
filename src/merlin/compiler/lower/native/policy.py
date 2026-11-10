"""Selected generic scalar/OpenMP pass ordering and allocation policy."""

from __future__ import annotations

from merlin.compiler.lower.scalar.pipeline import scalar_passes
from merlin.compiler.passes import policy as pass_policy


def _dealloc_passes(sink: bool | None = None) -> list[str]:
    """Retain legacy policy hooks while sharing the provider implementation."""
    import os

    if os.environ.get("MERLIN_NO_DEALLOC"):
        return []
    return pass_policy.deallocation_passes(
        sink=_sink_deallocs() if sink is None else sink,
        split=_split_fresh_deallocs(),
        slice_outputs=_slice_unused_elementwise_outputs(),
        disabled=False,
    )


def _sink_deallocs() -> bool:
    """Whether to sink each dealloc to its last use. Default OFF (``MERLIN_SINK_DEALLOCS=1``).

    A named predicate rather than an inline environ poke so a test can drive it, and so the flag is
    read per build instead of frozen at import."""
    import os

    from merlin.llvmlower.optional_passes import switched

    return switched("sink-deallocs", bool(os.environ.get("MERLIN_SINK_DEALLOCS")))


SPLIT_FRESH_DEALLOCS_MARKER = "__merlin_split_fresh_deallocs__"


def _split_fresh_deallocs() -> bool:
    """Enable the guarded aggregate-dealloc split, with ``0`` as an A/B escape hatch."""
    import os

    return os.environ.get("MERLIN_SPLIT_FRESH_DEALLOCS", "1") == "1"


SLICE_UNUSED_ELEMENTWISE_OUTPUT_MARKER = "__merlin_slice_unused_elementwise_output__"


def _slice_unused_elementwise_outputs() -> bool:
    """Enable the guarded output-slice rewrite, with ``0`` as an A/B escape hatch."""
    import os

    return os.environ.get("MERLIN_SLICE_UNUSED_ELEMENTWISE_OUTPUT", "1") == "1"


SINK_DEALLOC_PASS = "func.func(optimize-allocation-liveness)"


def _sink_dealloc_passes() -> list[str]:
    """Pipeline design rationale: see docs/design/llvm_pipeline_rationale.md."""
    return ["canonicalize", SINK_DEALLOC_PASS]


_UPSTREAM_PASSES = scalar_passes()


def _splice(passes: list[str]) -> str:
    """Expand shared policy with the retained legacy per-build selection hooks."""
    if _generalize_before_fuse():
        passes = _reorder_generalize_before_fuse(passes)
    return pass_policy.splice_passes(
        passes,
        deallocation=_dealloc_passes() if "__DEALLOC__" in passes else [],
        generalize_before_fuse=False,
    )


def _upstream_pipeline(features: "frozenset[str] | None" = None) -> str:
    """The scalar whole-module pipeline, including requested pipeline-edit features.

    A function, not a constant, so ``MERLIN_NO_DEALLOC`` is read per build rather than frozen at
    import. Applying features here is load-bearing: the RVV builder has always called
    :func:`impr_features.apply_pipeline`, but the scalar selection used to normalize and report its
    feature set without ever applying the corresponding pass-list edits. Such an arm compiled the
    baseline while claiming the feature was enabled.
    """
    passes = list(_UPSTREAM_PASSES)
    if features:
        from merlin.compiler.optimizations.registry import apply_pipeline, normalize

        passes = apply_pipeline(passes, normalize(features))
    return _splice(passes)


def _parallel_pipeline(features: "frozenset[str] | None" = None) -> str:
    """The scalar pipeline, re-targeted for **multicore** via OpenMP.

    Identical to :data:`UPSTREAM_PIPELINE` except the loop-generation stage: parallel
    iterator dimensions of each linalg op become ``scf.parallel`` (``convert-linalg-to-
    parallel-loops``) which ``convert-scf-to-openmp`` wraps in an ``omp.parallel`` +
    ``omp.wsloop`` (work-shared across threads); reduction dims stay sequential
    ``scf.for`` (correctness preserved). ``mlir-translate`` then emits ``__kmpc_*`` runtime
    calls, satisfied at link time by the cross-built ``libomp.a`` (build_tools/toolchains/openmp).
    Set ``OMP_NUM_THREADS`` on the board to fan out across the 8 cores.

    This is the K1 big-model path: the diffusion-transformer / large VLAs that run on the
    *scalar* fallback (no fixed-width vectorize) otherwise execute on one core and time out.
    Gated — never on the default flow (``UPSTREAM_PIPELINE`` is untouched)."""
    passes = scalar_passes(parallel=True)
    if features:
        from merlin.compiler.optimizations.registry import apply_pipeline, normalize

        passes = apply_pipeline(passes, normalize(features))
    return _splice(passes)


_FUSE_ELEMENTWISE = "func.func(linalg-fuse-elementwise-ops)"


_GENERALIZE_NAMED = "func.func(linalg-generalize-named-ops)"


def _generalize_before_fuse() -> bool:
    """Pipeline design rationale: see docs/design/llvm_pipeline_rationale.md."""
    import os

    return bool(os.environ.get("MERLIN_GENERALIZE_BEFORE_FUSE"))


def _reorder_generalize_before_fuse(passes: list[str]) -> list[str]:
    """Preserve the legacy entry while sharing the provider ordering check."""
    return pass_policy.reorder_generalize_before_fuse(passes)


def _par_residue_parallel(features=frozenset()) -> bool:
    """Whether residual linalg operations use dependence-aware parallel-loop lowering.

    The named ``parallelize_residual_loops_<grain>`` family is the reproducible path: it also
    implies the matching grain-control feature, which prevents the old unbounded fork explosion.
    ``MERLIN_PAR_RESIDUE_PARALLEL=1`` remains only as an A/B compatibility escape hatch.

    The environment switch is kept for the same reason :func:`_dealloc_passes` keeps
    ``MERLIN_NO_DEALLOC``: this changes the emitted code for every derived-split multicore build,
    and the honest way to defend a claim about it is to be able to rebuild both arms.
    """
    import os

    from merlin.llvmlower.residual_parallel import threshold_of

    return threshold_of(features) is not None or bool(os.environ.get("MERLIN_PAR_RESIDUE_PARALLEL"))


def _fuse_post() -> bool:
    """Whether to run the post-contraction fusion stage.

    Still opt-in (``MERLIN_FUSE_POST``) so the shipping baseline stays byte-identical until the fused
    build is graded bit-exact on hardware -- but a named predicate rather than an inline environ poke,
    so a test can drive it and a build can record it.
    """
    import os

    return bool(os.environ.get("MERLIN_FUSE_POST"))
