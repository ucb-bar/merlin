"""Existing stack-promotion, activation and reduction strategy construction."""

from __future__ import annotations

from merlin.compiler.passes.features import (
    PROMOTE_STACK_NAME as PROMOTE_STACK_NAME,
)
from merlin.compiler.passes.features import (
    _promote_buffers_to_stack as _promote_buffers_to_stack,
)
from merlin.compiler.selection.features import ImprFeature


def ensure_promote_stack(nbytes: int) -> str:
    """Register (on demand) a stack-promotion variant whose per-buffer cap is ``nbytes``.

    The cap is a LEVER, not just an overflow guard, and the difference is measured. On small_llama
    int8 (K1, sustained, cos identical on every arm) the SAME feature pays back very differently
    depending only on this number:

        <=  4 KB  4,904,957 ns  1.00x      <= 256 KB  3,649,518 ns  1.34x
        <= 16 KB  4,795,823 ns  1.03x      <=   1 MB  3,579,566 ns  saturated
        <= 64 KB  4,678,494 ns  1.05x

    Monotonic in the cap and flat past 256 KB, which locates the largest intermediate worth promoting
    between 64 KB and 256 KB -- a per-MODEL fact about intermediate sizes, not a constant.

    Why this function exists: the cap was reachable ONLY through ``MERLIN_PROMOTE_STACK_BYTES``, and
    no fork can vary an environment variable. So the beam always built at the 16 KB default, measured
    1.03x, and ranked the lever below levers it should beat -- in the run that prompted this, the
    stack-promotion fork came out SLOWER than its own parent and the search concluded the lever was
    not worth its width. The lever was never weak; the search could not reach the part of it that
    works. Naming the cap in the FEATURE puts it on the channel feature names already travel
    (proposer -> knobs.yaml ``compiler_features`` -> ``normalize`` -> build), so it needs no new
    plumbing and cannot silently disagree with the build the way an ambient env var can.
    """
    from merlin.compiler.optimizations.registry import known, register

    n = int(nbytes)
    if n <= 0:
        raise ValueError(f"{PROMOTE_STACK_NAME}: per-buffer cap must be positive, got {nbytes!r}")
    name = f"{PROMOTE_STACK_NAME}_{n}"
    if name in known():
        return name
    register(
        ImprFeature(
            name=name,
            action_class="PASS",
            description=(
                f"promote small bufferization allocs to the stack, per-buffer cap {n} bytes. "
                f"Same pass as {PROMOTE_STACK_NAME}, with the cap NAMED so the search can vary "
                f"it: the cap is model-dependent (measured 1.03x at 16 KB vs 1.34x at 256 KB on "
                f"the same model and the same feature), and as an env var it was unreachable by "
                f"any fork. Default-off; baseline byte-identical."
            ),
            edit_pipeline=lambda passes, _n=n: _promote_buffers_to_stack(passes, cap=_n),
        )
    )
    return name


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
        return text.replace(_REDUCTION_ANCHOR, _VECTORIZE_REDUCTION_BLOCK + _REDUCTION_ANCHOR, 1)
    return _VECTORIZE_REDUCTION_STANDALONE


def vectorize_reduction_pipeline(passes: list[str]) -> list[str]:
    """Make the emitted reduction a single-instruction horizontal vector reduce (`vfredusum`/`vredsum`):
    switch `lower-vector-multi-reduction` to the inner-reduction strategy (-> `vector.reduction`) and
    enable `reassociate-fp-reductions` on `convert-vector-to-llvm` (-> unordered `vfredusum.vs`). Only
    runs when this feature is enabled; the baseline pass list is untouched otherwise."""
    out: list[str] = []
    for p in passes:
        if p == "func.func(lower-vector-multi-reduction)":
            out.append("func.func(lower-vector-multi-reduction{lowering-strategy=inner-reduction})")
        elif p == "convert-vector-to-llvm":
            out.append("convert-vector-to-llvm{reassociate-fp-reductions}")
        else:
            out.append(p)
    return out
