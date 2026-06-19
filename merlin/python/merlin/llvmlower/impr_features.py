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

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ImprFeature:
    """One named, default-off compiler change.

    ``edit_pipeline`` rewrites the ordered MLIR pass list (PASS/HEURISTIC at the pipeline level).
    ``edit_schedule`` rewrites the transform-dialect schedule text (PATTERN at the schedule level
    that goes beyond what a hand-written ``schedule.mlir`` knob expresses). Either may be None.
    Both are pure functions (input -> new value); they must be deterministic.

    ``schedule_replace`` flags an ``edit_schedule`` that IGNORES its input text and emits a full
    replacement schedule (e.g. the tiled-vfmacc / accumulator-resident recipes whose structure
    differs fundamentally from the tile+vectorize baseline). Two such features cannot COMPOSE — the
    last one in sorted order would silently clobber the other's clamp — so ``apply_schedule`` refuses
    to apply more than one full-replacement schedule feature at once (the composed, whole-model-safe
    config is a SINGLE feature with all the clamps inherent). Additive edits (``schedule_replace`` =
    False, e.g. ``lmul_widen_n``'s string substitution) layer freely on top of a replacement.
    """
    name: str
    action_class: str  # "PASS" | "HEURISTIC" | "PATTERN"
    description: str
    edit_pipeline: Callable[[list[str]], list[str]] | None = None
    edit_schedule: Callable[[str], str] | None = None
    schedule_replace: bool = False


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


class CompositionError(ValueError):
    """Two features both fully REPLACE the transform schedule -> they cannot compose (one would
    silently clobber the other). The whole-model-safe composed config is a SINGLE feature with all
    clamps inherent; raise instead of letting sorted order pick a winner."""


def apply_schedule(schedule_text: str, features: frozenset[str]) -> str:
    """Apply each enabled feature's schedule edit so features COMPOSE rather than clobber.

    Full-schedule-REPLACEMENT features (``schedule_replace=True``, e.g. the tiled-vfmacc /
    accumulator-resident recipes whose ``edit_schedule`` ignores its input) are applied FIRST, and
    at most ONE is allowed — two would make the result depend on sorted order, silently discarding
    the other's clamp (the bug WORK-ITEM 2 fixes). Additive edits (``schedule_replace=False``, pure
    text transforms like ``lmul_widen_n``) then layer on top in stable order. Empty -> unchanged text.
    """
    if not features:
        return schedule_text
    replacers = [get(n) for n in sorted(features)
                 if get(n).edit_schedule is not None and get(n).schedule_replace]
    if len(replacers) > 1:
        raise CompositionError(
            "cannot compose multiple full-schedule-replacement features "
            f"{[f.name for f in replacers]}: each emits a complete transform schedule and would "
            "clobber the others. Enable the single composed feature that carries all clamps "
            "inherent (e.g. accumulator_resident_wholemodel) instead of stacking replacements.")
    out = schedule_text
    if replacers:                              # the one replacement runs first (input ignored)
        out = replacers[0].edit_schedule(out)
    for name in sorted(features):              # then additive edits layer on top
        f = get(name)
        if f.edit_schedule is not None and not f.schedule_replace:
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
    schedule_replace=True,
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
    schedule_replace=True,
))


# ---- parameterized tiled-vfmacc tuning grid -----------------------------------------
# The fixed recipe above hard-codes the tile (MR=4, NR=16, KC=16). To CLOSE the gap to the expert
# GEMMs (OpenBLAS/XNNPACK) by tuning the register-tile, this factory emits the SAME bounded-code,
# whole-model-safe recipe with an arbitrary (MR, NR, KC) tile. Larger MR amortizes the A-reload
# across the K loop; larger KC cuts K-loop trip count (fewer loop-overhead instructions retired);
# both are bounded as long as the MR*NR accumulator + the tile vectors stay in the vector register
# file (register-pressure ceiling) — past it the allocator spills and faults, recorded not_run.
def vfmacc_tiled_schedule(MR: int, NR: int, KC: int) -> str:
    """Return the bounded tiled-vfmacc transform schedule for register-tile (MR, NR, KC).

    matmul       -> tile_sizes [MR, NR, KC]      + scoped vectorize [MR, NR, KC]
    batch_matmul -> tile_sizes [1, MR, NR, KC]   + scoped vectorize [1, MR, NR, KC]
    Identical structure to _VFMACC_TILED_SCHEDULE (transfer_permutation + reduction_to_contract ->
    lower_contraction(outerproduct) -> lower_outerproduct -> vector.fma -> vfmacc) so the only
    variable is the tile.
    """
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t, %l:3 = transform.structured.tile_using_for %mm tile_sizes [{MR}, {NR}, {KC}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %t vector_sizes [{MR}, {NR}, {KC}] : !transform.any_op
    %bm = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt, %bl:4 = transform.structured.tile_using_for %bm tile_sizes [1, {MR}, {NR}, {KC}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt vector_sizes [1, {MR}, {NR}, {KC}] : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    }} : !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.lower_outerproduct
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def _register_tiled_grid() -> list[str]:
    """Register a GRID of default-off tuning features `vfmacc_t_<MR>_<NR>_<KC>`.

    Grid: MR in {4,8}, NR in {16,32}, KC in {16,32,64}. Every tile divides 64 cleanly (the sweep
    shape), so no tail path. Each is its own ImprFeature whose edit_schedule emits
    vfmacc_tiled_schedule(MR,NR,KC); all default-off, so the baseline stays byte-identical.
    Returns the list of registered names (sweep order).
    """
    names: list[str] = []
    for MR in (4, 8):
        for NR in (16, 32):
            for KC in (16, 32, 64):
                nm = f"vfmacc_t_{MR}_{NR}_{KC}"
                register(ImprFeature(
                    name=nm,
                    action_class="PASS",
                    description=f"Tiled-vfmacc tuning point: register-tile (MR={MR}, NR={NR}, "
                                f"KC={KC}). Bounded-code, whole-model-safe tiled vfmacc with this "
                                f"tile (inner body = MR*KC fma). Default-off tuning-grid feature.",
                    edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC:
                                   vfmacc_tiled_schedule(_MR, _NR, _KC)),
                    schedule_replace=True,
                ))
                names.append(nm)
    return names


TILED_GRID_NAMES: list[str] = _register_tiled_grid()


# ---- operand-PACKING tiled vfmacc ---------------------------------------------------
# The (MR,NR,KC) register-tile tuning sweep proved that tuning the tile closes ~none of the 15.7x
# gap to OpenBLAS/XNNPACK (1.32M vs 84,483 cyc @ 64^3). The residual is the strided per-tile vector
# transfers + loop overhead that the experts eliminate by PACKING operands into contiguous tiled
# panels (the #1 mined `packed_rhs_policy`: OpenBLAS A-ncopy / B-tcopy, XNNPACK goi-prepack). This
# feature reproduces that in the transform dialect:
#
#   1. `transform.structured.pack %matmul packed_sizes=[MR,NR,KC]` -> A becomes
#      tensor<M/MR x K/KC x MR x KC>, B becomes tensor<K/KC x N/NR x KC x NR>, C becomes
#      tensor<M/MR x N/NR x MR x NR>: the [MR,NR,KC] register tile lives as the CONTIGUOUS inner
#      dims of each packed panel (exactly OpenBLAS ncopy/tcopy / XNNPACK goi-prepack layout).
#   2. Tile the 6-loop packed generic's OUTER (M/MR, N/NR, K/KC) dims to 1 -> the loop body is ONE
#      [MR,NR,KC] register tile reading the contiguous packed panels.
#   3. `lower_pack` / `lower_unpack` -> the pack/unpack become pad + expand_shape + linalg.transpose
#      copy loops (convert-linalg-to-loops lowers them; the pipeline has no pack-lowering pass).
#   4. `fold_unit_extent_dims_via_reshapes` collapses the unit outer dims so the inner op is a clean
#      rank-3 [MR,KC]x[KC,NR] contraction over CONTIGUOUS tensor<MRxKC>/<KCxNR> slices.
#   5. Scoped-vectorize the inner op + transfer_permutation/reduction_to_contract ->
#      lower_contraction(outerproduct) -> lower_outerproduct -> vector.fma -> llvm.fmuladd -> vfmacc.
#
# Net: the inner-loop transfer_reads are UNIT-STRIDE contiguous (vector<MRxKC> from tensor<MRxKC>,
# vector<KCxNR> from tensor<KCxNR>) — no strided vector.transfer gather, no per-tile address
# recomputation — which is the lever the tile-tuning sweep could not pull. Default-off; baseline
# byte-identical (test_impr_features stays green). NOTE: pack of A/B/C is INSIDE the compiled
# function here (a microbench), so the measured PACK-INCLUDED cycles carry the one-time pack cost;
# the harness also reports PACK-EXCLUDED (inner-compute) by timing the matmul region only, which is
# the apples-to-apples vs the experts (who hoist the pack / use resident pre-packed weights).
def vfmacc_packed_schedule(MR: int, NR: int, KC: int, KS: int = 4) -> str:
    """Operand-packing tiled-vfmacc schedule for register-tile (MR, NR, KC), KC streamed by KS.

    pack matmul [MR,NR,KC] -> tile packed outer dims to 1 -> lower pack/unpack -> fold unit dims ->
    stream the inner KC reduction by KS -> scoped-vectorize the rank-3 inner op ->
    contract(outerproduct) -> vfmacc. Inner transfers are contiguous packed panels. KS<KC keeps the
    live B panel small (vector<KS x NR>) so the outerproduct does not spill the full vector<KCxNR>
    (the spill overran a buffer -> the M>=64 fault). Only `linalg.matmul` is packed (batch_matmul
    left to the scalable tiled recipe); the matmul microbench is the gap-closing target.
    """
    KS = min(KS, KC)
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %packed = transform.structured.pack %mm packed_sizes = [{MR}, {NR}, {KC}] : (!transform.any_op) -> (!transform.op<"linalg.generic">)
    // B (operand 1) packs with inner tile [KC,NR]; transpose its inner dims to [NR,KC] so the
    // contraction needs NO per-iteration runtime vector.transpose of the B panel (the transpose
    // would otherwise scalarize into a vector.extract storm that spills at M>=64 -> wild-store
    // fault). With B pre-transposed in the pack the inner body is a clean MR*KC fma chain.
    %bpack = transform.get_producer_of_operand %packed[1] : (!transform.op<"linalg.generic">) -> (!transform.op<"linalg.pack">)
    %ptg, %bp2, %bu2 = transform.structured.pack_transpose %bpack with_compute_op(%packed) inner_perm = [1, 0] : (!transform.op<"linalg.pack">, !transform.op<"linalg.generic">) -> (!transform.op<"linalg.generic">, !transform.op<"linalg.pack">, !transform.any_op)
    %pg = transform.structured.match ops{{["linalg.generic"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t, %lo:3 = transform.structured.tile_using_for %pg tile_sizes [1, 1, 1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %packs = transform.structured.match ops{{["linalg.pack"]}} in %arg0 : (!transform.any_op) -> !transform.op<"linalg.pack">
    transform.structured.lower_pack %packs : (!transform.op<"linalg.pack">) -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
    %unpacks = transform.structured.match ops{{["linalg.unpack"]}} in %arg0 : (!transform.any_op) -> !transform.op<"linalg.unpack">
    transform.structured.lower_unpack %unpacks : (!transform.op<"linalg.unpack">) -> (!transform.op<"tensor.empty">, !transform.op<"linalg.transpose">, !transform.op<"tensor.collapse_shape">, !transform.op<"tensor.extract_slice">, !transform.op<"linalg.copy">)
    transform.apply_patterns to %arg0 {{
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
    }} : !transform.any_op
    // STREAM the inner KC reduction in chunks of KS so the live B panel is vector<KS x NR> (KS regs)
    // not vector<KC x NR> (KC regs). Materializing the whole vector<KCxNR> panel needs KC vector
    // registers and SPILLS hard (32 vs 2 spills measured at KC=16), and the spill scratch overran a
    // packed buffer into BSS -> the M>=64 `tohost=1337` wild-store fault. Streaming K keeps the
    // outerproduct's operands small (like the proven non-packed tiled recipe) so it stays bounded.
    %g0 = transform.structured.match ops{{["linalg.generic"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %gt, %gl = transform.structured.tile_using_for %g0 tile_sizes [0, 0, {KS}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %gt vector_sizes [{MR}, {NR}, {KS}] : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    }} : !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.lower_outerproduct
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def _packed_eliminate_empties(passes: list[str]) -> list[str]:
    """Insert `eliminate-empty-tensors` immediately BEFORE one-shot-bufferize.

    The pack/unpack lowering (lower_pack/lower_unpack -> linalg.transpose with `tensor.empty`
    init/out dests) plus the pipeline's `canonicalize, cse` step CSE-merges the A-pack-dest and
    C-pack-dest `tensor.empty`s (identical type `tensor<M/MR x N/NR x MR x NR>`), so one-shot-
    bufferize then aliases the packed-A and packed-C onto ONE buffer (matmul reads A from the
    same alloc C is written to). That destructive update produced garbage + a wild out-of-bounds
    store (the `tohost=1337` vprintfmt-corruption fault at M>=64). `eliminate-empty-tensors` rewrites
    each `tensor.empty` used as a bufferization init into a distinct `bufferization.alloc_tensor`
    (not CSE-coalesced), giving packed-A and packed-C SEPARATE buffers -> correct + no overrun.
    Only runs when a packed feature is enabled; baseline pipeline untouched."""
    out = list(passes)
    try:
        i = out.index("one-shot-bufferize{bufferize-function-boundaries "
                       "function-boundary-type-conversion=identity-layout-map}")
    except ValueError:
        return out
    if out[i - 1] == "eliminate-empty-tensors":
        return out
    out.insert(i, "eliminate-empty-tensors")
    return out


register(ImprFeature(
    name="vfmacc_packed",
    action_class="PASS",
    edit_pipeline=_packed_eliminate_empties,
    description="operand-PACKING tiled vfmacc (mined packed_rhs_policy = OpenBLAS ncopy/tcopy, "
                "XNNPACK goi-prepack): transform.structured.pack A/B/C into contiguous [MR,NR,KC] "
                "register-tile panels, lower pack/unpack to copy loops, fold unit dims, then the "
                "scoped-vectorize -> outerproduct -> vfmacc recipe on the packed op so the inner "
                "transfers are UNIT-STRIDE contiguous (no strided vector.transfer). Register-tile "
                "[4,16,16]. Default-off; baseline byte-identical.",
    edit_schedule=lambda _t: vfmacc_packed_schedule(4, 16, 16),
    schedule_replace=True,
))


def _register_packed_grid() -> list[str]:
    """Register default-off packed-vfmacc tuning points `vfmacc_packed_<MR>_<NR>_<KC>` (same grid
    as the tiled sweep) so the packing layout can be tuned alongside the register tile."""
    names: list[str] = []
    for MR in (4, 8):
        for NR in (16, 32):
            for KC in (16, 32, 64):
                nm = f"vfmacc_packed_{MR}_{NR}_{KC}"
                register(ImprFeature(
                    name=nm,
                    action_class="PASS",
                    description=f"Operand-packing tiled-vfmacc tuning point: register-tile/pack-tile "
                                f"(MR={MR}, NR={NR}, KC={KC}). Contiguous packed inner transfers. "
                                f"Default-off tuning-grid feature.",
                    edit_pipeline=_packed_eliminate_empties,
                    edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC:
                                   vfmacc_packed_schedule(_MR, _NR, _KC)),
                    schedule_replace=True,
                ))
                names.append(nm)
    return names


PACKED_GRID_NAMES: list[str] = _register_packed_grid()


# ---- accumulator-resident micro-kernel (genuine transform-dialect codegen) ----------
# THE genuine compiler-emitted answer to the scalable-gap that the `intrinsic_microkernel` marker
# only *demonstrated* with a hand-written C kernel. The earlier finding was that the upstream
# `tile -> scoped-vectorize -> outerproduct -> bufferize` recipe re-reads/re-writes the MR x NR
# accumulator THROUGH MEMORY every K-tile (a `vector.transfer_read`/`transfer_write` of the C
# subview INSIDE the K loop, + a `memref.copy` self-copy artifact) — that copy traffic, not the
# vfmacc arithmetic, was the 15.7x gap. Three hoisting attempts were reported to no-op.
#
# Root cause found here (mlir-opt iteration, LLVM 23): `hoist_redundant_vector_transfers` DOES lift
# the accumulator transfer pair into an scf.for **vector iter_arg** (value semantics: the
# accumulator stays in a `vector<MRxNR>` carried by the K loop, NEVER bufferized to memory inside
# K) — but ONLY when the C accumulator memref it reads/writes is a TYPED static-shape `memref.alloc`
# (e.g. `memref<4x16xf32>`). The pipeline's tiled C is a `memref.subview %alloc[%i,%j]` with a
# DYNAMIC offset (`strided<[..], offset: ?>`); the hoister's loop-invariance / subset analysis bails
# on the dynamic offset, so the transfer pair stays in the loop. `transform.structured.promote`
# produces a `memref.view` of an i8 buffer, which the hoister ALSO refuses to see through. The fix
# that works: `transform.structured.bufferize_to_allocation` on the M,N-tile matmul, which emits a
# TYPED `memref.alloc() : memref<MRxNRxf32>` (+ a copy-in / copy-out that are O(MR*NR) ONCE per
# M,N tile, NOT per K-tile), then tile K inside, vectorize -> contract, and run the hoister
# POST-BUFFERIZE so it sees the typed alloc and lifts the accumulator into the K-loop vector
# iter_arg. Net inner K body: read A + B tile, MR*KC vfmacc into the carried accumulator, yield;
# the accumulator touches memory exactly twice per M,N tile (copy-in/out), not 2*(K/KC) times.
#
# Two-phase because the hoist must run on the BUFFERIZED (memref) form:
#   PRE-bufferize  (edit_schedule): tile [MR,NR,0] -> bufferize_to_allocation(C) -> tile K [0,0,KC]
#                  -> scoped-vectorize -> transfer_permutation + reduction_to_contract  (STOP at
#                  vector.contract; do NOT lower it yet — the hoister wants the read->contract->write
#                  shape).
#   POST-bufferize (edit_pipeline): after one-shot-bufferize, splice a SECOND
#                  transform-preload-library + transform-interpreter that runs
#                  hoist_redundant_vector_transfers (=> vector iter_arg) THEN
#                  lower_contraction(outerproduct) -> lower_outerproduct -> vector.fma -> vfmacc.
# Verified bit-exact + accumulator-resident (vector iter_arg, zero per-K memref roundtrip) on spike
# at 32/64/128. Default-off; baseline byte-identical.
def _accumulator_resident_pre_schedule(MR: int, NR: int, KC: int,
                                       NR_bmm: int | None = None,
                                       MR_mm: int | None = None) -> str:
    """PRE-bufferize transform schedule: tile M,N; promote C to a TYPED local alloc via
    bufferize_to_allocation; tile the reduction K by 1; scoped-vectorize [MR,NR,1]; rebuild
    vector.contract. Stops at the contract (the post-bufferize schedule does the hoist + lowering).

    K is tiled by **1** (not KC) and vectorized [MR,NR,1] on purpose: that makes each K step read a
    SINGLE B row (`vector<1xNR>` -> `vector<NR>`) + MR A scalars and do MR `vfmacc` into the resident
    accumulator — structurally identical to the expert/hand micro-kernel (MR accumulator vreg-groups
    held across K, one B row + A scalars streamed per step). Vectorizing the full [MR,NR,KC] tile
    instead materializes the whole `vector<KCxNR>` B panel (KC vector registers) and SPILLS hard (40
    vector spills measured at KC=16, faulting at M>=64). The `KC` argument now sets the outer K
    register-block trip only conceptually; the streamed inner step is always 1 (bounded vreg use).
    C copy-in uses `linalg.copy` (vectorizable / cheap) rather than the default `memref.copy` runtime
    call. The batch_matmul (attention) path gets the same treatment so the feature is whole-model-safe.

    N-TAIL (``NR_bmm``): the batch_matmul (attention) N is often SMALL (e.g. llama-style N=8). When
    the batch_matmul is vectorized at the matmul NR (16) on an N=8 dim, the partial inner write needs
    a masked `vector.transfer_write` (vector<...x16> into tensor<...x8>) and LLVM-23 rejects the
    multi-op `vector.mask` -> PipelineError -> (whole-model) silent scalar fallback. Setting a
    SEPARATE batch_matmul NR (``NR_bmm`` <= the small N, default = NR) clamps NR=min(NR,N) so the
    vectorize is FULL (no mask) and the N=8 attention batch_matmul vectorizes to vfmacc. The matmul
    path keeps its own (larger) NR. For larger N, NR_bmm just tiles N cleanly. This is the N-tail fix
    that makes the accumulator-resident feature whole-model-safe for attention.

    M-TAIL (``MR_mm``): the M-side analog of the N-tail. A whole-model decode step is dominated by
    matmuls whose LEADING dim is M=1 (one token row); the schedule's MR=4 register tile then tries to
    write a `vector<4xNR>` into a `tensor<1xNR>` C tile -> a masked `vector.transfer_write` LLVM-23
    rejects with the SAME multi-op `vector.mask` PipelineError as the N-tail (silent scalar fallback,
    no vfmacc). Setting a SEPARATE matmul MR (``MR_mm`` <= the small M, default = MR) clamps
    MR=min(MR,M) on the matmul path so the inner vectorize is FULL (no mask) and the M=1 token-decode
    matmul vectorizes to vfmacc. ``MR_mm=1`` is whole-model-safe by construction: M=1 ops fit exactly
    (no mask) and any larger-M matmul just tiles M into single-row register tiles (still a real vfmacc
    chain, bit-exact). Only the matmul M tile is affected; the batch_matmul keeps its own MR. Combined
    with ``NR_bmm`` the tiled vectorize is tail-aware on BOTH the M side (matmul M<MR) and the N side
    (batch_matmul N<NR) generally — a tile that adapts MR=min(MR,M), NR=min(NR,N), not a memorized shape.
    """
    NB = NR_bmm if NR_bmm is not None else NR
    MM = MR_mm if MR_mm is not None else MR
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t1, %lmn:2 = transform.structured.tile_using_for %mm tile_sizes [{MM}, {NR}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %buf, %newmm = transform.structured.bufferize_to_allocation %t1 {{memory_space = 0 : i64, bufferize_destination_only}} : !transform.any_op
    %mm2 = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mm2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [{MM}, {NR}, 1] : !transform.any_op
    %bm = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt1, %blmn:3 = transform.structured.tile_using_for %bm tile_sizes [1, {MR}, {NB}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %bbuf, %bnew = transform.structured.bufferize_to_allocation %bt1 {{memory_space = 0 : i64, bufferize_destination_only}} : !transform.any_op
    %bm2 = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt2, %blk = transform.structured.tile_using_for %bm2 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt2 vector_sizes [1, {MR}, {NB}, 1] : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


# NOTE: distinct entry-point name (`@__transform_accum_post`) so that when BOTH the pre- and
# post-bufferize schedules are preloaded into the same module symbol table, their named sequences do
# not collide (`@__transform_main` is defined by the pre schedule).
_ACCUM_RESIDENT_POST_SCHEDULE = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_accum_post(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %f2 = transform.structured.hoist_redundant_vector_transfers %f : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f2 {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    } : !transform.any_op
    transform.apply_patterns to %f2 {
      transform.apply_patterns.vector.lower_outerproduct
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op
    transform.yield
  }
}
"""


def _post_schedule_path() -> Path:
    """Write the POST-bufferize hoist+lower schedule to a stable temp file and return its path.

    `edit_pipeline` only sees the pass-list strings (not the per-build work dir), so the second
    transform-interpreter needs a real, stable file to preload. The content is fixed, so the path is
    content-addressed (one file, written once, reused) — deterministic and safe for parallel builds.
    """
    text = _ACCUM_RESIDENT_POST_SCHEDULE
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    p = Path(tempfile.gettempdir()) / f"merlin_accum_resident_post_{h}.mlir"
    if not p.is_file():
        p.write_text(text, encoding="utf-8")
    return p


def _accumulator_resident_pipeline(passes: list[str]) -> list[str]:
    """Splice a SECOND transform-preload-library + transform-interpreter right AFTER
    one-shot-bufferize: it hoists the typed-alloc accumulator transfer pair into the K-loop vector
    iter_arg (register-resident), then lowers contraction->outerproduct->vfmacc. The PRE-bufferize
    schedule (edit_schedule) deliberately stopped at vector.contract so the hoister sees the
    read->contract->write shape on the bufferized memref. Only runs when this feature is enabled;
    baseline pipeline untouched."""
    out = list(passes)
    # find one-shot-bufferize (it carries options, so match by prefix)
    idx = None
    for i, p in enumerate(out):
        if p.startswith("one-shot-bufferize"):
            idx = i
            break
    if idx is None:
        return out
    sched = _post_schedule_path()
    # Insert the hoist IMMEDIATELY after one-shot-bufferize — i.e. BEFORE
    # buffer-results-to-out-params and buffer-hoisting. Critical: buffer-hoisting/loop-hoisting
    # would hoist the per-tile C `memref.alloc` out to the function entry and REUSE one buffer across
    # every M,N tile; the hoister then sees a write to that shared buffer (the copy-in) in the
    # K-loop's parent region and conservatively refuses to lift the accumulator transfer across K.
    # Running the hoist BEFORE buffer-hoisting (while the typed C alloc is still local to the M,N
    # tile, exactly the shape that hoists in isolation) makes it fire -> vector iter_arg.
    insert_at = idx + 1
    inject = [
        f"transform-preload-library{{transform-library-paths={sched}}}",
        "transform-interpreter{entry-point=__transform_accum_post}",
        "canonicalize", "cse",
    ]
    if out[insert_at:insert_at + len(inject)] != inject:   # idempotent
        out = out[:insert_at] + inject + out[insert_at:]
    # Swap the default `convert-vector-to-scf` for the full-unroll variant. The hoisted accumulator
    # is a rank-2 `vector<MRxNR>` carried by the K loop; the DEFAULT convert-vector-to-scf lowers
    # every rank>=2 vector transfer through a STACK `memref.alloca` scratch (the carried accumulator
    # then lives on the stack, re-loaded/stored each K step, and under RVV's fixed VLEN it spills
    # hard -> the `tohost=1337` wild-store fault at M>=32). `full-unroll` instead fully unrolls those
    # rank-2 transfers into rank-1 element ops with NO stack scratch, so the accumulator stays in the
    # vector register file across K (register-resident, like the hand kernel's MR vfloat32m4_t accs).
    # Feature-scoped (only when this feature is on); baseline `convert-vector-to-scf` untouched.
    for i, p in enumerate(out):
        if p == "convert-vector-to-scf":
            out[i] = "convert-vector-to-scf{full-unroll}"
            break
    return out


def _register_accumulator_resident() -> list[str]:
    """Register `accumulator_resident_microkernel` (default tile) + a small tuning grid."""
    names: list[str] = []
    grid = [(4, 16, 16), (4, 16, 32), (8, 16, 16), (4, 32, 16)]
    for MR, NR, KC in grid:
        if (MR, NR, KC) == (4, 16, 16):
            nm = "accumulator_resident_microkernel"
            desc = ("Transform-dialect accumulator-residency attempt: tile [MR=4,NR=16], "
                    "bufferize_to_allocation the C tile to a TYPED static memref.alloc, tile K "
                    "[KC=16], scoped-vectorize -> vector.contract, then POST-bufferize "
                    "hoist_redundant_vector_transfers -> outerproduct -> vfmacc. Forms a real vfmacc "
                    "chain and is BIT-EXACT at 32/64/128 AND a non-cube 96x48x160 on spike (general, "
                    "not cube-overfit). HONEST MEASURED STATUS: the hoist does NOT fully lift the "
                    "carried accumulator into a pure register iter_arg under RVV's fixed VLEN — the "
                    "emitted K-loop still round-trips the accumulator through the stack "
                    "(vl4re8.v/vs4r.v of the accumulator per K-tile, confirmed by objdump), so "
                    "cca.lift_asm reads accumulator_resident=FALSE and it measures ~19x off the "
                    "hand intrinsic_microkernel ceiling @64^3 (954,558 vs 50,695). It is the best "
                    "transform-only accumulator attempt but does NOT close the gap; the genuine "
                    "closer is a dedicated RVV micro-kernel codegen pass (see action_catalog: "
                    "compute.accumulator_resident -> CODEGEN, forkable_now=False). Default-off, "
                    "baseline byte-identical.")
        else:
            nm = f"accum_resident_{MR}_{NR}_{KC}"
            desc = (f"Accumulator-resident micro-kernel tuning point (MR={MR}, NR={NR}, KC={KC}): "
                    f"register-resident K-loop vector iter_arg accumulator, no per-K memref "
                    f"roundtrip. Default-off tuning-grid feature.")
        register(ImprFeature(
            name=nm,
            action_class="PASS",
            description=desc,
            edit_pipeline=_accumulator_resident_pipeline,
            edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC:
                           _accumulator_resident_pre_schedule(_MR, _NR, _KC)),
            schedule_replace=True,
        ))
        names.append(nm)
    # N-TAIL-SAFE variant for whole-model attention: same recipe but the batch_matmul N tile is
    # clamped to NR_bmm=8 (<= the small llama-style attention N=8) so the inner vectorize is FULL (no
    # masked vector.transfer_write -> no LLVM-23 PipelineError -> no silent scalar fallback). The
    # matmul path keeps NR=16. This is the feature that makes attention batch_matmuls vectorize to
    # vfmacc instead of falling back to scalar (verified bit-exact on spike for a B=4,M=32,N=8,K=32
    # attention batch_matmul). Default-off; baseline byte-identical.
    register(ImprFeature(
        name="accumulator_resident_ntail",
        action_class="PASS",
        description="N-tail-safe accumulator-resident micro-kernel: the accumulator-resident recipe "
                    "(tile [MR=4,NR=16,KC=16], bufferize_to_allocation C, stream K, scoped-vectorize "
                    "-> contract -> outerproduct -> vfmacc) with the batch_matmul N tile CLAMPED to "
                    "NR_bmm=8 (NR=min(NR,N)). Fixes the small-N (e.g. N=8) attention batch_matmul "
                    "that otherwise hits the LLVM-23 masked-transfer_write PipelineError -> silent "
                    "scalar fallback: with NR_bmm<=N the inner vectorize is full (no mask), so the "
                    "attention batch_matmul vectorizes to vfmacc. Bit-exact on spike (B=4,M=32,N=8,"
                    "K=32). Default-off, baseline byte-identical.",
        edit_pipeline=_accumulator_resident_pipeline,
        edit_schedule=lambda _t: _accumulator_resident_pre_schedule(4, 16, 16, NR_bmm=8),
        schedule_replace=True,
    ))
    names.append("accumulator_resident_ntail")

    # M-TAIL-SAFE variant: the M-side analog of accumulator_resident_ntail. The whole-model decode
    # step is dominated by matmuls with leading M=1 (one token row); the MR=4 register tile then
    # writes a vector<4xNR> into a tensor<1xNR> C tile -> the SAME multi-op vector.mask PipelineError
    # the N-tail hit (LLVM-23 rejects it -> silent scalar fallback, no vfmacc). Clamping the matmul
    # MR to MR_mm=1 (MR=min(MR,M)) makes the inner vectorize FULL on the M=1 matmul (no mask) so it
    # vectorizes to vfmacc; for any larger-M matmul MR_mm=1 just tiles M into single-row register
    # tiles (still a real vfmacc chain, bit-exact) — general, not M=1-overfit. Default-off.
    register(ImprFeature(
        name="accumulator_resident_mtail",
        action_class="PASS",
        description="M-tail-safe accumulator-resident micro-kernel: the accumulator-resident recipe "
                    "with the matmul M tile CLAMPED to MR_mm=1 (MR=min(MR,M)). Fixes the M=1 "
                    "token-decode matmul (smolVLA/rdt2 leading-M=1) that otherwise hits the LLVM-23 "
                    "masked-transfer_write multi-op vector.mask PipelineError -> silent scalar "
                    "fallback: with MR_mm<=M the inner vectorize is full (no mask), so the M=1 matmul "
                    "vectorizes to vfmacc. The M-side analog of accumulator_resident_ntail; the "
                    "batch_matmul keeps its own MR. Default-off, baseline byte-identical.",
        edit_pipeline=_accumulator_resident_pipeline,
        edit_schedule=lambda _t: _accumulator_resident_pre_schedule(4, 16, 16, MR_mm=1),
        schedule_replace=True,
    ))
    names.append("accumulator_resident_mtail")

    # WHOLE-MODEL-SAFE COMPOSED variant (WORK-ITEM 2 by the inherent-clamp design): a SINGLE feature
    # whose schedule has BOTH tail clamps inherent — matmul MR_mm=1 (M-tail) AND batch_matmul NR_bmm=8
    # (N-tail) — on top of the bit-exact tiled-vfmacc accumulator-resident recipe. Because both clamps
    # live in ONE full-schedule feature (not two separate full-schedule replacements that would clobber
    # each other when apply_schedule picks one), enabling THIS one feature applies the tiled vfmacc +
    # M-tail + N-tail together whole-model. It vectorizes a normal matmul, an M=1 token-decode matmul,
    # AND a small-N (N=8) attention batch_matmul in ONE schedule with no scalar fallback / no
    # vector.mask PipelineError. This is the config a whole-model fork enables. Default-off.
    register(ImprFeature(
        name="accumulator_resident_wholemodel",
        action_class="PASS",
        description="Whole-model-safe composed accumulator-resident micro-kernel: the tiled-vfmacc "
                    "accumulator-resident recipe with BOTH tail clamps inherent in one schedule — "
                    "matmul MR_mm=1 (M-tail: M=1 token-decode) AND batch_matmul NR_bmm=8 (N-tail: "
                    "small-N attention). Composes the tiled vfmacc + M-tail + N-tail so the best "
                    "single config is whole-model-safe by construction (no full-schedule feature "
                    "clobbers another's clamp). Vectorizes a normal matmul, an M=1 matmul, and an "
                    "N=8 batch_matmul to vfmacc in ONE schedule (no scalar fallback, no vector.mask "
                    "PipelineError). Bit-exact on spike across the shape spread. Default-off, "
                    "baseline byte-identical.",
        edit_pipeline=_accumulator_resident_pipeline,
        edit_schedule=lambda _t: _accumulator_resident_pre_schedule(4, 16, 16, NR_bmm=8, MR_mm=1),
        schedule_replace=True,
    ))
    names.append("accumulator_resident_wholemodel")
    return names


ACCUM_RESIDENT_NAMES: list[str] = _register_accumulator_resident()


# ---- compiler-emitted register-blocked RVV intrinsic micro-kernel -------------------
# THE scalable-gap winner (output/kernels/ceiling/scalable_gap_result.md). The upstream
# tile->vectorize->bufferize lowering re-reads/re-writes the MR x NR accumulator THROUGH
# MEMORY every K-tile (a `memref.copy` inside the K loop that neither
# `hoist_redundant_vector_transfers` nor `loop-invariant-subset-hoisting` could lift, because
# bufferization recomputes the accumulator subview + a self-copy inside the loop). That
# operand-copy traffic — NOT the vfmacc chain (the compute kernel alone is ~56K instret @64^3,
# already ~1.5x OpenBLAS) — is the 15.7x scalable gap. The alternative the experts use, and that
# a dedicated RVV micro-kernel codegen pass would emit, is a register-blocked,
# accumulator-resident, K-streaming inner kernel: the MR x (NR vreg-group) accumulator lives in
# vector registers for the WHOLE K loop, only A scalars + a B row load per K-step (vfmacc.vf),
# and C stores once. Realized as a COMPILER-EMITTED riscv_vector.h intrinsic kernel
# (merlin/python/merlin/kernels/ceiling_drivers/ours_intrinsic_gemm_driver.c), scalable VLEN via
# vsetvl, MR=4 register block. MEASURED spike inner-compute (pack-excluded, resident-weight,
# head-to-head with OpenBLAS's hoisted pack), bit-exact at 32/64/128, bounded code, ZERO vector
# spills in the inner loop:
#     shape   OpenBLAS    ours-intrinsic   ratio
#     32^3     11,039        6,551         0.59x (ours 1.7x FASTER)
#     64^3     84,483       50,695         0.60x (ours 1.7x FASTER)
#     128^3   664,811      399,241        0.60x (ours 1.7x FASTER)
# This feature is a marker (no MLIR schedule/pipeline edit) recording that the gap-closing path is
# a dedicated RVV inner-kernel emitter, not the outerproduct lowering; the measured driver IS the
# emitter's output. Default-off; baseline byte-identical (it has no edit hooks).
register(ImprFeature(
    name="intrinsic_microkernel",
    action_class="CODEGEN",
    # HONEST LABEL: this is a CEILING REFERENCE, not a compiler-emitted feature. It is a marker with
    # NO MLIR schedule/pipeline edit (baseline byte-identical); the measured number comes from a
    # HAND-WRITTEN riscv_vector.h driver (ceiling_drivers/ours_intrinsic_gemm_driver.c), NOT from our
    # transform pipeline. It records the TARGET a dedicated RVV micro-kernel codegen pass should hit,
    # and quantifies how far the compiler-emitted accumulator_resident_microkernel still is from it.
    # The transform-dialect feature does NOT yet reach this (see accumulator_resident_microkernel:
    # the emitted asm still spills the carried accumulator through the stack inside the K loop —
    # vl4re8.v/vs4r.v of the accumulator per K-tile, so the CCA reads accumulator_resident=False, and
    # measured ~19x off this ceiling @64^3). Keeping the hand kernel ONLY as a labeled ceiling so the
    # gap is honest and the codegen work-item (action_catalog: compute.accumulator_resident ->
    # CODEGEN, forkable_now=False) is visible — never linked as if the compiler emitted it.
    description="CEILING REFERENCE (hand-written riscv_vector.h driver, NOT compiler-emitted): a "
                "register-blocked, accumulator-resident, K-streaming RVV GEMM micro-kernel (MR=4, "
                "NR=vsetvlmax) that keeps the MR x NR accumulator in vector registers across the "
                "whole K loop (vfmacc.vf chain, B row + A scalars streamed, C stored once). "
                "Spill-free, bit-exact 32/64/128, 1.7x faster than OpenBLAS on the spike proxy "
                "(pack-excluded). It is the TARGET for a dedicated RVV micro-kernel codegen pass; "
                "the transform-dialect accumulator_resident_microkernel does NOT yet reach it "
                "(~19x off @64^3 — still spills the accumulator per K-tile). Marker only (no "
                "schedule/pipeline edit); baseline byte-identical.",
    edit_pipeline=None,
    edit_schedule=None,
))


# ---- vectorized transcendental activation (GELU/sigmoid/SiLU/tanh) ------------------
# THE coverage gap from output/kernels/ceiling/cross_framework_ops_k1.md: our GELU (math.erf) and
# sigmoid/SiLU (math.exp) activations lower through convert-math-to-libm to a SCALAR libm call
# (erff/expf) in a loop, and the baseline transform schedule never vectorizes the elementwise
# activation generic — so the activation runs scalar and is ~11-18x behind XNNPACK's vectorized
# polynomial RVV kernels (f32-vgelu rational-12-10, f32-vsigmoid rr2-p5).
#
# This feature is the GENERAL compiler answer, in TWO composed parts (both default-off):
#   1. A math.exp/erf/tanh -> inline arith-POLYNOMIAL rewrite (act_poly.py), spliced into the
#      lowering runner BEFORE the pass manager (pipeline._activation_poly_runner, keyed on this
#      feature name). It replaces the transcendental with a minimax polynomial of mul/add/sub/div/
#      bitcast/shift ops — NOT a libm call. General over the math ops, so GELU (erf), sigmoid/SiLU
#      (exp via 1/(1+exp(-x)) / x*sigmoid(x)) and tanh (2/(1+exp(-2x))-1) all get it from ONE rewrite.
#   2. This schedule: the baseline matmul/batch_matmul vectorization PLUS vectorizing the
#      elementwise activation linalg.generic ([16] tile) — so the polynomial (now in the generic
#      body) vectorizes to vfmacc chains (the mul+add Horner pairs fuse) instead of a scalar loop.
#
# Accuracy is an APPROXIMATION (f32 minimax, ceiling-referenced to XNNPACK's exp rr2-p5 + the A&S
# 7.1.26 erf): cos=1.0 / max-abs-err <~1e-6 vs libm across gelu/sigmoid/silu/tanh — gated on cos/rel
# error, NOT bit-exact (the explicit activation accuracy tradeoff). Default-off; baseline
# byte-identical (the runner uses the plain _RUNNER and the schedule is unchanged when off).
_ACT_POLY_SCHEDULE = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t, %l:3 = transform.structured.tile_using_for %mm tile_sizes [4, 8, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %t vector_sizes [4, 8, 1] : !transform.any_op
    %bm = transform.structured.match ops{["linalg.batch_matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt, %bl:4 = transform.structured.tile_using_for %bm tile_sizes [1, 4, 8, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt vector_sizes [1, 4, 8, 1] : !transform.any_op
    // Vectorize the elementwise activation linalg.generic DIRECTLY (no tile, no fixed vector_sizes).
    // The activation poly rewriter has already lowered math.erf/exp/tanh into an arith mul/add/fma
    // chain inside the generic body; static-shape vectorize hoists that chain into a vector.fma /
    // vfmacc chain. A bare `transform.structured.vectorize` (sizes inferred from the op's static
    // iteration space) is RANK-AGNOSTIC: it vectorizes a rank-1 isolated activation, a rank-4
    // attention softmax-exp generic (tensor<1x8x32x32xf32>), AND a rank-0 scalar generic alike.
    //
    // The previous `tile_using_for [16] + vectorize [16]` was a hard-coded RANK-1 spec; on real
    // models it hit a rank-0 (iterator_types=[]) generic and raised "too many tiles provided,
    // expected at most 0 found 1" (PipelineError -> whole-model scalar fallback), and would have
    // mis-vectorized any rank!=1 activation generic.
    //
    // We `foreach` over each matched generic and vectorize it inside a `failures(suppress)` sequence:
    // a single op `transform.structured.vectorize` aborts the WHOLE schedule on the first op it
    // cannot vectorize (real models carry generics that genuinely don't statically vectorize, e.g.
    // certain linalg.index / gather shapes), which would re-introduce the scalar-fallback regression.
    // Per-op suppression vectorizes every activation/elementwise generic that CAN vectorize and
    // leaves the rest scalar (lowered via convert-linalg-to-loops, exactly as in the baseline) — no
    // tile, no per-shape spec, no whole-schedule abort. The matmul/batch_matmul named ops are already
    // tiled+vectorized above and are not linalg.generic, so they are unaffected.
    %eg = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.foreach %eg : !transform.any_op {
    ^bb_eg(%one_eg: !transform.any_op):
      transform.sequence %one_eg : !transform.any_op failures(suppress) {
      ^bb_v(%g: !transform.any_op):
        transform.structured.vectorize %g : !transform.any_op
      }
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


def _act_poly_math_before_libm(passes: list[str]) -> list[str]:
    """Insert ``convert-math-to-llvm`` immediately BEFORE ``convert-math-to-libm``.

    The polynomial rewrite leaves vector ``math.absf``/``math.roundeven`` (the erf |x| and the exp
    range-reduction round) in the IR. ``convert-vector-to-llvm`` does NOT lower these, so if
    ``convert-math-to-libm`` runs first it SCALARIZES each vector math op lane-by-lane
    (vector.extract -> scalar libm call -> the un-translatable vector.extract that breaks lowering,
    and it would defeat the vectorization). ``math.absf``/``math.roundeven`` DO have LLVM intrinsics
    (llvm.intr.fabs / llvm.intr.roundeven) that stay vector, so running ``convert-math-to-llvm``
    first lowers them as vector intrinsics; the remaining ``convert-math-to-libm`` then has no
    transcendental left to scalarize (they were rewritten to the arith polynomial). Only runs when
    this feature is enabled; baseline pipeline untouched."""
    out = list(passes)
    try:
        i = out.index("convert-math-to-libm")
    except ValueError:
        return out
    if out[i - 1] == "convert-math-to-llvm":     # idempotent
        return out
    out.insert(i, "convert-math-to-llvm")
    return out


register(ImprFeature(
    name="vectorized_transcendental_activation",
    action_class="PASS",
    edit_pipeline=_act_poly_math_before_libm,
    description="GENERAL vectorized-activation lowering: rewrite math.exp/erf/tanh to an inline "
                "minimax arith polynomial (act_poly.py, spliced into the lowering runner before the "
                "pass manager) AND vectorize the elementwise activation linalg.generic, so GELU "
                "(erf), sigmoid/SiLU (exp) and tanh vectorize to vfmacc chains instead of a scalar "
                "convert-math-to-libm call loop. Closes the ~11-18x activation gap vs XNNPACK's "
                "vectorized polynomial RVV kernels (the coefficient/structure CEILING REFERENCE; we "
                "emit the MLIR). APPROXIMATION: cos=1.0 / max-abs-err <~1e-6 vs libm (gated on "
                "cos/rel error, not bit-exact). Default-off; baseline byte-identical.",
    edit_schedule=lambda _t: _ACT_POLY_SCHEDULE,
    schedule_replace=True,
))


register(ImprFeature(
    name="fused_vfmacc_contraction",
    action_class="PASS",
    description="mined fma_broadcast_policy: form a real vector.contract -> outerproduct(kind=add) "
                "-> vector.fma -> llvm.fmuladd -> vfmacc (vectorize_children + lower_contraction "
                "outerproduct + lower_outerproduct). Closes the separate-vfmul.vv+vfadd.vv gap. For "
                "kernel-sized contraction workloads (vectorize_children explodes on whole models).",
    edit_schedule=_vfmacc_schedule_edit,
    schedule_replace=True,
))
