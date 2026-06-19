"""Fork-scoped, **default-off** compiler-feature registry for ``impr_<target>_vX`` forks.

The baseline RVV compiler (``pipeline.RVV_TRANSFORM_SCHEDULE`` + ``build_rvv_pipeline`` pass list)
is FROZEN — kernel-mining experiments never edit it. KNOB/FLAG/PATTERN improvements ride the
existing ``transform_schedule=``/``cflags_override=`` seams. But PASS- and HEURISTIC-class
improvements need actual compiler code, which still must not perturb the baseline. This registry
is how: each improvement is a NAMED feature with a hook that edits the pipeline pass list and/or
the transform schedule; an ``impr_`` fork's manifest lists the ``compiler_features`` it enables,
threaded through ``build_app`` -> ``lower_*`` -> ``build_rvv_pipeline(features=...)``.

Invariant: with ``features == frozenset()`` (the baseline / any non-impr build), the hooks are
never invoked, so the emitted pipeline string and schedule are **byte-identical** to today
(guarded by ``test_impr_features``). A feature only changes codegen when a fork explicitly enables
it, so it can be measured against the immutable baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ImprFeature:
    """One named, default-off compiler change.

    ``edit_pipeline`` rewrites the ordered MLIR pass list (PASS/HEURISTIC at the pipeline level).
    ``edit_schedule`` rewrites the transform-dialect schedule text (PATTERN at the schedule level
    that goes beyond what a hand-written ``schedule.mlir`` knob expresses). Either may be None.
    Both are pure functions (input -> new value); they must be deterministic.
    """
    name: str
    action_class: str  # "PASS" | "HEURISTIC" | "PATTERN"
    description: str
    edit_pipeline: Callable[[list[str]], list[str]] | None = None
    edit_schedule: Callable[[str], str] | None = None


_REGISTRY: dict[str, ImprFeature] = {}


def register(feature: ImprFeature) -> ImprFeature:
    if feature.name in _REGISTRY:
        raise ValueError(f"duplicate impr feature {feature.name!r}")
    _REGISTRY[feature.name] = feature
    return feature


def get(name: str) -> ImprFeature:
    if name not in _REGISTRY:
        raise KeyError(f"unknown impr feature {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def known() -> list[str]:
    return sorted(_REGISTRY)


def normalize(features) -> frozenset[str]:
    """Accept None / list / set / frozenset -> validated frozenset (every name must be registered)."""
    if not features:
        return frozenset()
    fs = frozenset(features)
    for n in fs:
        get(n)  # raises on unknown
    return fs


def apply_pipeline(passes: list[str], features: frozenset[str]) -> list[str]:
    """Apply each enabled feature's pipeline edit, in a stable (sorted) order. Empty -> unchanged
    list object content (identity), so the joined string is byte-identical to the baseline."""
    if not features:
        return passes
    out = list(passes)
    for name in sorted(features):
        f = get(name)
        if f.edit_pipeline is not None:
            out = f.edit_pipeline(out)
    return out


def apply_schedule(schedule_text: str, features: frozenset[str]) -> str:
    """Apply each enabled feature's schedule edit, in stable order. Empty -> unchanged text."""
    if not features:
        return schedule_text
    out = schedule_text
    for name in sorted(features):
        f = get(name)
        if f.edit_schedule is not None:
            out = f.edit_schedule(out)
    return out


# ---- registered features ------------------------------------------------------------
# Keep this list small and evidence-justified. Each entry corresponds to a typed CompilerAction
# (PASS/HEURISTIC/PATTERN) surfaced by the action catalog from a mined kernel divergence.

# The vfmacc-forming schedule (developed by MLIR-level iteration, R-cont): vectorize the func's
# contraction to a real `vector.contract`, lower it via the OUTERPRODUCT strategy to a chain of
# `vector.outerproduct{kind=add}` (acc + a⊗b — the fused MAC), then `lower_outerproduct` turns each
# into `vector.fma` -> `llvm.intr.fmuladd` -> the RISC-V backend emits `vfmacc`. The baseline's
# [4,8,1] K=1 tiling never forms a contraction (so outerproduct/K-tile/fast-math were all measured
# no-ops); vectorize_children DOES form it. Appropriate for kernel-sized contraction workloads (the
# mining testbed) — NOT a whole transformer (vectorize_children there explodes vector.extract, per
# pipeline.py). Gated to the impr fork; baseline schedule untouched.
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


# Combined / production-scalable variant: the CORRECT, SCALABLE, bounded-code tiled vfmacc.
#
# The earlier `fused_vfmacc_tiled` draft tiled [4,16,0] (K UNTILED) then ran
# `vectorize_children_and_apply_patterns` on the whole func. That had three fatal flaws, all now
# fixed here:
#   (a) K untiled => the [MR,NR,K] contraction read FULL-K vectors (vector<4x64>, vector<64x16>)
#       and fully unrolled the K reduction (64 outerproducts * MR = 256 fma), so .text grew with K
#       (blowing the JAL +-1 MB reach at 128^3) AND the oversized vectors spilled. ROOT CAUSE of the
#       M>=64 `tohost=1337` spike fault: a `trap_store_access_fault` (epc in vprintfmt, tval far
#       outside RAM) — the register allocator spilled those 4 KB vector<64x16>/vector<4x64> temps
#       and at M>=64 the spill overran the stack, corrupting adjacent BSS (printf's byte-count
#       counter loaded as 0x40040030 -> store to 0xc1058630 faulted). It "passed" at 32^3 only
#       because the smaller spill stayed in bounds. Bounding the tile (KC<=16) removes the oversized
#       vectors and the spill, so the fault is gone at 64 AND 128.
#   (b) `vectorize_children` on the WHOLE func scalarizes every non-contraction generic (the
#       transposes / elementwise of a real model) into tens of thousands of `vector.extract`s —
#       the whole-model explosion `pipeline.py` warns about. (Measured: 4120 extracts on a
#       matmul+transpose+add toy vs 84 with the scoped vectorize here.)
#   (c) running form-contract AND lower-contraction/outerproduct in ONE greedy `apply_patterns`
#       block left the contract un-lowered (mul+add, no fma).
#
# The fixed recipe (developed by mlir-opt iteration, verified bit-exact + bounded at 32/64/128):
#   1. TILE matmul [MR=4, NR=16, KC=16] and batch_matmul [1,4,16,16] -> M, N AND K are all scf.for
#      LOOPS. KC>1 (not 1) so a real reduction tile survives (KC=1 collapses to a bare mul+add and
#      no contract can form). KC=16 (over KC=4) cuts K-loop trip count 4x -> ~4x fewer retired
#      instructions on the spike proxy (5.2M->1.33M @ 64^3) while staying bounded; KC=16 keeps the
#      tile vectors small (vector<4x16>, vector<16x16>) so no spill/fault.
#   2. `transform.structured.vectorize %t` (SCOPED to the tiled op handle ONLY — never
#      vectorize_children on the func) => the non-contraction generics are left untouched, so NO
#      vector.extract explosion; they lower scalar via convert-linalg-to-loops as in the baseline.
#   3. `transfer_permutation_patterns` + `reduction_to_contract` rebuild a real `vector.contract`
#      from the scoped-vectorize `mulf`+`multi_reduction` (scoped vectorize alone never forms a
#      contract — that is exactly why the old recipe needed vectorize_children).
#   4. lower_contraction(outerproduct) and lower_outerproduct in SEPARATE apply_patterns blocks
#      -> vector.outerproduct{add} -> vector.fma -> llvm.intr.fmuladd -> vfmacc.
# Result: the inner body is a CONSTANT MR*KC = 64 fma regardless of M/N/K (64 fma @ 32, 64 AND
# 128) — bounded .text => no JAL wall, no register-spill stack-overrun fault, and whole-model-safe
# (84 extracts on the mixed toy vs 4120 before).
_VFMACC_TILED_SCHEDULE = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t, %l:3 = transform.structured.tile_using_for %mm tile_sizes [4, 16, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %t vector_sizes [4, 16, 16] : !transform.any_op
    %bm = transform.structured.match ops{["linalg.batch_matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt, %bl:4 = transform.structured.tile_using_for %bm tile_sizes [1, 4, 16, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt vector_sizes [1, 4, 16, 16] : !transform.any_op
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
    } : !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    } : !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_outerproduct
    } : !transform.any_op
    transform.yield
  }
}
"""


register(ImprFeature(
    name="lmul_widen_n",
    action_class="KNOB",
    description="mined lmul_grouping_policy: widen the matmul N tile/vector 8->16 so the emitted "
                "vector group uses a higher LMUL (m2->m4). Composes with other features in autotune.",
    edit_schedule=lambda t: t.replace("tile_sizes [4, 8, 1]", "tile_sizes [4, 16, 1]").replace(
        "vector_sizes [4, 8, 1]", "vector_sizes [4, 16, 1]"),
))


register(ImprFeature(
    name="fused_vfmacc_tiled",
    action_class="PASS",
    description="CORRECT, SCALABLE, bounded-code tiled vfmacc: tile matmul [MR=4,NR=16,KC=4] and "
                "batch_matmul [1,4,16,4] so M,N,K are all scf.for LOOPS, SCOPED-vectorize the tile "
                "only (no whole-func vectorize_children => whole-model-safe, no vector.extract "
                "explosion), rebuild the contract via reduction_to_contract, then "
                "outerproduct->vector.fma->vfmacc. Inner body = constant MR*KC=16 fma at any "
                "M/N/K => .text bounded (no JAL +-1MB wall) and applicable to whole models. "
                "Benchmarked vs the full-unroll fused_vfmacc_contraction.",
    edit_schedule=lambda _t: _VFMACC_TILED_SCHEDULE,
))


# Same recipe, surfaced under the name the kernel-policy mining task refers to. `fused_vfmacc_tiled`
# is kept for the cross-framework matrix column that already names it; `fused_vfmacc_scalable` is the
# preferred name going forward (it is the whole-model-safe, bounded-code vfmacc).
register(ImprFeature(
    name="fused_vfmacc_scalable",
    action_class="PASS",
    description="Alias of the fixed fused_vfmacc_tiled recipe: bounded-code (constant MR*KC inner "
                "body), K-as-loop, scoped-vectorize (whole-model-safe) tiled vfmacc. Correct + "
                "bit-exact at 32/64/128; the whole-model-safe vfmacc for e2e.",
    edit_schedule=lambda _t: _VFMACC_TILED_SCHEDULE,
))


register(ImprFeature(
    name="fused_vfmacc_contraction",
    action_class="PASS",
    description="mined fma_broadcast_policy: form a real vector.contract -> outerproduct(kind=add) "
                "-> vector.fma -> llvm.fmuladd -> vfmacc (vectorize_children + lower_contraction "
                "outerproduct + lower_outerproduct). Closes the separate-vfmul.vv+vfadd.vv gap. For "
                "kernel-sized contraction workloads (vectorize_children explodes on whole models).",
    edit_schedule=_vfmacc_schedule_edit,
))
