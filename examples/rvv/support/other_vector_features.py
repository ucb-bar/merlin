"""Additional RVV Transform schedules for legacy feature names."""

from __future__ import annotations

from merlin.llvmlower.feature_registry import FeatureRegistry, ImprFeature

_VFMACC_SCHEDULE = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %v = transform.structured.vectorize_children_and_apply_patterns %f : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %v {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
      transform.apply_patterns.vector.lower_outerproduct
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.lower_transpose
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op
    transform.yield
  }
}
"""


def _vfmacc_schedule_edit(text: str) -> str:
    """Replace the baseline schedule with the vfmacc-forming recipe (full replacement — the
    structure differs fundamentally from the tile+vectorize baseline)."""
    return _VFMACC_SCHEDULE


_ACT_POLY_SCHEDULE = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t, %l:3 = transform.structured.tile_using_for %mm tile_sizes [4, 8, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %t vector_sizes [4, 8, 1] : !transform.any_op
    %bm = transform.structured.match ops{["linalg.batch_matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt, %bl:4 = transform.structured.tile_using_for %bm tile_sizes [1, 4, 8, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt vector_sizes [1, 4, 8, 1] : !transform.any_op
    // Vectorize PRECISELY the activation generics — and ONLY those — the poly rewriter targeted.
    // The rewriter (act_poly.apply_activation_polynomial) replaces math.erf/exp/tanh with an arith
    // mul/add/fma chain ONLY inside a generic the PROVENANCE marks as an elementwise activation
    // (gelu/silu/sigmoid/tanh), and tags each such generic with a `merlin.act_vectorize` unit attr.
    // We match exactly those tagged generics and vectorize them (bare vectorize = rank-agnostic,
    // sizes inferred from the static iteration space) so the poly chain becomes a vector.fma/vfmacc
    // chain. A softmax-exp / normalization / any other generic is NOT tagged (its exp stayed on
    // libm), so it is NOT matched here and lowers exactly as in the baseline.
    //
    // This drops the previous BLANKET design (foreach over EVERY linalg.generic + failures(suppress)
    // + a hard-coded rank-1 tile). That design (a) rewrote+vectorized the softmax exp too, which
    // amplified the minimax error through the row-sum normalization -> openvla whole-model cos 0.541;
    // (b) used failures(suppress), which HID any vectorize miscompile; and (c) vectorized every
    // generic in the model, blowing clang -O2 compile time to 6+ min/config. Matching only the tagged
    // activation generics fixes all three: correct (softmax stays libm), no suppression (a genuine
    // vectorize failure surfaces), and bounded compile time (only the few activation generics
    // vectorize). No `failures(suppress)` — these tagged generics are static-shape elementwise and
    // always statically vectorize; if one ever didn't, we WANT the error, not a silent scalar fallback.
    %eg = transform.structured.match ops{["linalg.generic"]} attributes{"merlin.act_vectorize"} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.foreach %eg : !transform.any_op {
    ^bb_eg(%one_eg: !transform.any_op):
      transform.structured.vectorize %one_eg : !transform.any_op
    }
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


_VECTORIZE_REDUCTION_BLOCK = """\
    %redr1 = transform.structured.match ops{["linalg.generic"]} attributes{iterator_types = [#linalg.iterator_type<reduction>]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %redr1 : !transform.any_op
    %redr2 = transform.structured.match ops{["linalg.generic"]} attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %redr2 : !transform.any_op
    %redr3 = transform.structured.match ops{["linalg.generic"]} attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %redr3 : !transform.any_op
"""


_REDUCTION_ANCHOR = '    %f = transform.structured.match ops{["func.func"]}'


_VECTORIZE_REDUCTION_STANDALONE = f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
{_VECTORIZE_REDUCTION_BLOCK}\
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.lower_masked_transfers
      transform.apply_patterns.vector.lower_shape_cast
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def vectorize_reduction_schedule(text: str) -> str:
    """Insert the reduction match+vectorize block into the baseline schedule (before its func.func
    lowering-pattern block), keeping the contraction handling intact. Falls back to a reduction-only
    schedule if the anchor is absent (never a silent no-op)."""
    if _REDUCTION_ANCHOR in text:
        return text.replace(
            _REDUCTION_ANCHOR, _VECTORIZE_REDUCTION_BLOCK + _REDUCTION_ANCHOR, 1
        )
    return _VECTORIZE_REDUCTION_STANDALONE


def vectorize_reduction_pipeline(passes: list[str]) -> list[str]:
    """Make the emitted reduction a single-instruction horizontal vector reduce (`vfredusum`/`vredsum`):
    switch `lower-vector-multi-reduction` to the inner-reduction strategy (-> `vector.reduction`) and
    enable `reassociate-fp-reductions` on `convert-vector-to-llvm` (-> unordered `vfredusum.vs`). Only
    runs when this feature is enabled; the baseline pass list is untouched otherwise."""
    out: list[str] = []
    for p in passes:
        if p == "func.func(lower-vector-multi-reduction)":
            out.append(
                "func.func(lower-vector-multi-reduction{lowering-strategy=inner-reduction})"
            )
        elif p == "convert-vector-to-llvm":
            out.append("convert-vector-to-llvm{reassociate-fp-reductions}")
        else:
            out.append(p)
    return out


def register_other_vector_features(registry: FeatureRegistry) -> None:
    registry.register(
        ImprFeature(
            name="fused_vfmacc_contraction",
            action_class="PASS",
            description="RVV full-contraction vfmacc schedule for kernel-sized workloads.",
            edit_schedule=_vfmacc_schedule_edit,
            schedule_replace=True,
        )
    )
    registry.register(
        ImprFeature(
            name="lmul_widen_n",
            action_class="KNOB",
            description="Widen the RVV matmul N tile from 8 to 16.",
            edit_schedule=lambda text: text.replace(
                "tile_sizes [4, 8, 1]", "tile_sizes [4, 16, 1]"
            ).replace("vector_sizes [4, 8, 1]", "vector_sizes [4, 16, 1]"),
        )
    )
    registry.register(
        ImprFeature(
            name="vectorized_transcendental_activation",
            action_class="PASS",
            description="RVV vectorize tagged activation polynomial generics.",
            edit_schedule=lambda _text: _ACT_POLY_SCHEDULE,
            schedule_replace=True,
            implies=frozenset({"approximate_transcendental_activation"}),
        )
    )
    registry.register(
        ImprFeature(
            name="vectorize_reduction",
            action_class="PASS",
            description="RVV horizontal vector reduction schedule and lowering.",
            edit_pipeline=vectorize_reduction_pipeline,
            edit_schedule=vectorize_reduction_schedule,
            schedule_replace=True,
        )
    )
    registry.register(
        ImprFeature(
            name="vectorize_amax_reduction",
            action_class="PASS",
            description="Bounded RVV amax reduction Transform arms; default off after measured regression.",
            edit_schedule=lambda text: splice_reduction_arms(
                text, lanes=8, min_rank=2, max_rank=4
            ),
        )
    )


ATTR_PREFIX = "merlin.vec_red"


def reduction_arms(lanes: int, min_rank: int, max_rank: int) -> str:
    """The bounded tile+vectorize arms for ranks ``min_rank``..``max_rank``.

    GENERATED rather than written out, for the reason `impr_features._vec_rank_arms` records: the
    tile-size list and the loop arity are both functions of the rank, and writing them by hand is
    what fixed the all-parallel family's coverage at rank 4.
    """
    arms = []
    for rank in range(min_rank, max_rank + 1):
        sizes = ", ".join(["1"] * (rank - 1) + [str(lanes)])
        loops = ", ".join(["!transform.any_op"] * (rank + 1))
        arms.append(
            f"    %rd{rank} = transform.structured.match attributes{{{ATTR_PREFIX}{rank}}} in "
            f"%arg0 : (!transform.any_op) -> !transform.any_op\n"
            f"    %rdt{rank}, %rdl{rank}:{rank} = transform.structured.tile_using_for %rd{rank} "
            f"tile_sizes [{sizes}] : (!transform.any_op) -> ({loops})\n"
            f"    transform.structured.vectorize %rdt{rank} vector_sizes [{sizes}] : "
            f"!transform.any_op\n"
        )
    return "".join(arms)


_ANCHOR = '    %f = transform.structured.match ops{["func.func"]}'


def splice_reduction_arms(
    text: str, *, lanes: int, min_rank: int, max_rank: int
) -> str:
    """Insert the arms before the schedule's func anchor. ADDITIVE, so the contraction arms above are
    untouched. A schedule that already carries them, or has no anchor to build on, is returned
    UNCHANGED rather than replaced: silently swapping a caller's tuned schedule for a generic one
    would take away the micro-kernel recipe that schedule exists to carry."""
    if ATTR_PREFIX in text or _ANCHOR not in text:
        return text
    return text.replace(_ANCHOR, reduction_arms(lanes, min_rank, max_rank) + _ANCHOR, 1)
