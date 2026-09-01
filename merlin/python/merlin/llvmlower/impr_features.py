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

from .selfcopy import FEATURE as _SELF_COPY_FEATURE
from .transpose_fuse import FEATURE as _FUSE_TRANSPOSE_FEATURE
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

    ``implies`` names features this one CANNOT be measured without -- lowering hygiene that is a
    property of the recipe's SHAPE rather than a tuning choice. It exists because a default-off
    feature whose payoff is cancelled by a separate default-off fix is, in practice, an inert lever:
    the beam has to discover the conjunction, and everyone who names the feature directly in
    ``compiler_features`` gets the cancelled version. See ``_tile_epilogue_hygiene``.
    """
    name: str
    action_class: str  # "PASS" | "HEURISTIC" | "PATTERN"
    description: str
    edit_pipeline: Callable[[list[str]], list[str]] | None = None
    edit_schedule: Callable[[str], str] | None = None
    schedule_replace: bool = False
    implies: frozenset[str] = frozenset()


_REGISTRY: dict[str, ImprFeature] = {}


def register(feature: ImprFeature) -> ImprFeature:
    if feature.name in _REGISTRY:
        raise ValueError(f"duplicate impr feature {feature.name!r}")
    _REGISTRY[feature.name] = feature
    return feature


def _try_lazy_register(name: str) -> bool:
    """Auto-register a v3 micro-kernel tuning point from its NAME (accum_resident_v3_<MR>_<NR>_<KC>).

    The register block is a continuous, beam-tunable knob space, but the lowering runs in a SUBPROCESS
    that re-imports this module — so a point registered on demand in the parent is invisible there and
    the feature fails to resolve (observed: every dynamically-registered MR failed at K2 while the
    pre-registered grid worked). Deriving the point from the name makes resolution reproducible in ANY
    process, which is what actually makes the space continuous.

    The same holds for the two other on-demand v3 families, which were previously NOT derivable from
    their name (their prefix is ``accum_resident_v3vl_`` / ``accum_resident_v3p_``, so the
    ``accum_resident_v3_`` test rejected them): a package that names one of those points directly in
    ``compiler_features`` -- rather than reaching it through a ``microkernel`` knob block, which
    registers it as a side effect of resolving -- failed with an "unknown impr feature" KeyError.
    Each family is registered from its own arity, so a name is either derivable or an honest error."""
    parts = name.split("_")
    tails: dict[str, tuple[int, "Callable[..., str]"]] = {
        "accum_resident_v3_": (3, ensure_v3_microkernel),          # MR, NR, KC
        "accum_resident_v3vl_": (3, ensure_v3_scalable_microkernel),  # MR, NR, KC
        "accum_resident_v3p_": (5, ensure_v3_perop_microkernel),   # MR_mm, NR_mm, MR_bmm, NR_bmm, KC
    }
    for prefix, (arity, make) in tails.items():
        if not name.startswith(prefix):
            continue
        args = parts[len(prefix.rstrip("_").split("_")):]
        if len(args) != arity:
            return False
        try:
            make(*(int(a) for a in args))
        except ValueError:
            return False
        return name in _REGISTRY
    return False


def get(name: str) -> ImprFeature:
    if name not in _REGISTRY and not _try_lazy_register(name):
        raise KeyError(f"unknown impr feature {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def known() -> list[str]:
    return sorted(_REGISTRY)


def normalize(features) -> frozenset[str]:
    """Accept None / list / set / frozenset -> validated frozenset (every name must be registered),
    closed under ``ImprFeature.implies``.

    The closure is here, and not at each call site, because ``feats = normalize(...)`` is the single
    point every consumer reads: the runner selection, the schedule edits, and the argv gates for the
    self-copy erase / transpose fusion all key off this one set. Expanding anywhere later would let
    them disagree about which features are on.

    Empty in => empty out, so the frozen baseline stays byte-identical.
    """
    if not features:
        return frozenset()
    fs = frozenset(features)
    for n in fs:
        get(n)  # raises on unknown
    while True:
        grown = fs | frozenset().union(*(get(n).implies for n in fs)) if fs else fs
        if grown == fs:
            return fs
        for n in grown - fs:
            get(n)  # an implied name must be registered too
        fs = grown


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
      transform.apply_patterns.vector.fold_arith_extension
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
      transform.apply_patterns.vector.fold_arith_extension
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
      transform.apply_patterns.vector.fold_arith_extension
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
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


#: The hand-frozen whole-model register block, named ONCE so the registration below and the
#: shape-aware resolver read the same numbers. ``WHOLEMODEL_VF_CAPS`` is ``(MR, NR, KC)``; the two
#: per-class clamps are the tails somebody picked to survive the VLA shapes (matmul M tile 1,
#: batch_matmul N tile 8). They are a SHAPE decision, not a constant of the target, which is why
#: ``mining.apply.shape_adapted_features`` can re-derive them from a workload's own extents --
#: reading the triple as an upper BOUND. On tiny_llama that derivation returns exactly (4, 16) /
#: (4, 8), i.e. this frozen point, so the model this block was tuned on is unaffected.
WHOLEMODEL_VF_NAME = "accumulator_resident_wholemodel_vf"
WHOLEMODEL_VF_CAPS: tuple[int, int, int] = (4, 16, 16)
WHOLEMODEL_VF_MR_MM = 1
WHOLEMODEL_VF_NR_BMM = 8


def _tile_epilogue_hygiene(mr_matmul: int | None) -> frozenset[str]:
    """The lowering hygiene an MR>1 register-block recipe cannot be measured without.

    A recipe that tiles the output and bufferizes per tile leaves, in each tile epilogue, a
    ``memref.copy %x, %x`` -- the destination subview copied onto ITSELF -- which survives
    ``finalize-memref-to-llvm`` as an opaque rank-generic ``@memrefCopy`` runtime call. Erasing it is
    unconditionally value-preserving (identical SSA operand => identical base, offsets and region)
    and is a no-op on a lowering that has none, so ``erase_self_copy`` can only help or do nothing.
    ``mining/from_strategy._with_hygiene`` states exactly this policy for recipes reached through the
    ``microkernel`` knob space; this function is the same policy for the features named DIRECTLY in a
    package's ``compiler_features``, which bypass that space entirely.

    MEASURED, spike, 64^3 matmul, this repo, bit-identical output on every arm:

        int8   MR=1 240,369 cyc | MR=4 491,346 | MR=1+erase 240,369 | MR=4+erase 178,239
        f32    MR=1 231,313 cyc | MR=4 495,241 | MR=1+erase 231,313 | MR=4+erase 181,503

    Read those rows in order. Bare MR=4 is 2.0-2.1x SLOWER than MR=1 even though the register block
    made the compute cheaper -- the PC histogram attributes the whole delta (+250,977 instructions,
    exactly the cycle delta) to ``memrefCopy`` +187,520 / ``memcpy`` +98,304 / ``memset`` +24,896
    against ``forward`` -59,743, i.e. a 1.45x cheaper kernel paying 310,720 instructions of buffer
    copying to get there. With the erase, MR=4 is 1.35x (int8) / 1.27x (f32) FASTER than MR=1. So the
    A-reuse register block was never the losing lever; the un-erased epilogue was.

    MR<=1 returns empty deliberately. Not because the erase would be unsafe there, but because
    MR=1+erase measured BYTE-IDENTICAL (both dtypes, same cycle count to the digit) -- there is no
    self-copy to erase at MR=1, so implying it would move the validated ``MR_mm=1`` control's
    declared feature set for no measured effect. This also explains the standing claim in
    ``from_strategy._with_hygiene`` that the ``accumulator_resident_wholemodel*`` family "has no
    self-copy to erase": that was measured on the MR=1 member, and is false for the MR>1 members.
    """
    return frozenset({_SELF_COPY_FEATURE}) if (mr_matmul or 1) > 1 else frozenset()


def _accumulator_resident_v3_pre_schedule(MR: int, NR: int, KC: int,
                                          NR_bmm: int | None = None,
                                          MR_mm: int | None = None,
                                          skip_mm: bool = False,
                                          skip_bmm: bool = False) -> str:
    """PRE-bufferize schedule for the v3 (vfmacc.vf) micro-kernel — SAME as the v1/v2 pre-schedule
    but WITHOUT ``bufferize_to_allocation``.

    The v1/v2 pre-schedule promoted the C tile to a TYPED local ``memref.alloc`` (via
    ``bufferize_to_allocation``) because v1 ran ``hoist_redundant_vector_transfers`` POST-bufferize
    and that pass needed a typed alloc to (try to) see. v3 runs ``loop-invariant-subset-hoisting`` on
    the TENSOR form (before bufferize), where the K-loop carries the C tile as a value-semantic
    ``tensor<MRxNR>`` iter_arg directly (a ``tensor.extract_slice`` of the output) — no local alloc
    needed. Dropping ``bufferize_to_allocation`` removes the per-M,N-tile C copy-in/copy-out
    (``materialize_in_destination`` -> ``memref.copy``) that otherwise dominated the timed region
    (measured: ~634K of the 692K instret @64^3 was ``memrefCopy``/``memcpy`` from that promotion,
    while the actual ``forward`` compute kernel was only ~57K — at the hand ceiling). Without it the
    accumulator write lands in-place on the output buffer, so the compute kernel stands alone.

    Otherwise identical to the v1 pre-schedule: tile [MR,NR]; tile K by 1; scoped-vectorize [MR,NR,1]
    (one B row + MR A scalars per K step -> the resident-accumulator micro-kernel shape); rebuild
    ``vector.contract``; STOP (the v3 pipeline does subset-hoist + contract-lower + A-scalarize). The
    M-tail (``MR_mm``) / N-tail (``NR_bmm``) clamps carry over unchanged for whole-model safety.

    ``skip_mm`` / ``skip_bmm`` OMIT that op class's arms entirely, leaving those contractions for
    ``convert-linalg-to-loops`` (scalar). Needed because a claim can be impossible: when every block
    legal for a class's extents is 1-lane wide (an N=1 extent forces NR=1), tiling buys no vector
    lanes AND produces a parallel-dim-free ``vector.contract`` that no lowering strategy matches. Not
    claiming the class is then strictly better than claiming it badly — the model builds and runs, and
    the caller reports the class as un-vectorized rather than failing the build.
    """
    NB = NR_bmm if NR_bmm is not None else NR
    MM = MR_mm if MR_mm is not None else MR
    if __import__("os").environ.get("MERLIN_PAD_M"):
        # M-PADDING (default OFF): pad matmul M to multiple of MR=4 BEFORE tiling so small-M (openvla M=17)
        # register-blocks CLEANLY at MR=4 (no masked transfer_write scalar fallback). Measure vs MR=1 vf.
        return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %padded, %pad, %cp = transform.structured.pad %mm pad_to_multiple_of [4] {{padding_values = [0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32], padding_dimensions = [0]}} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %t1, %lmn:2 = transform.structured.tile_using_for %padded tile_sizes [4, {NR}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %mm2 = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mm2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [4, {NR}, 1] : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""
    # EXPERIMENT (env MERLIN_VEC_EW, default OFF -> baseline byte-identical): after matmul/bmm are tiled
    # + vectorized, the remaining linalg.generic ops are the NON-matmul ops (activations/softmax/norm/
    # layout) — which otherwise fall through convert-linalg-to-loops to SCALAR (openvla's 1100ms). Match
    # + scoped-vectorize them here. NOTE: this is the BLUNT version (matches ALL generics incl. reduction
    # softmax/norm — may break cos or explode); the measurement tells us if scoped/tagged is needed.
    # PER-RANK BOUNDED vectorize (env MERLIN_VEC_RANK, default OFF): the non-matmul parallel generics
    # (tagged merlin.vec_rN by the pre-pass, N = loop rank) are vectorized with BOUNDED vector_sizes
    # [1,..,1,8] — innermost dim by 8 lanes (VLEN256/f32), NOT the plain no-sizes vectorize that
    # explodes (vector<17x576>=9792 lanes -> 8725ms). Per rank because openvla generics are rank 2/3/4.
    _ew = ("""
    %g2 = transform.structured.match attributes{merlin.vec_r2} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %g2 vector_sizes [1, 8] : !transform.any_op
    %g3 = transform.structured.match attributes{merlin.vec_r3} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %g3 vector_sizes [1, 1, 8] : !transform.any_op
    %g4 = transform.structured.match attributes{merlin.vec_r4} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %g4 vector_sizes [1, 1, 1, 8] : !transform.any_op"""
           if __import__("os").environ.get("MERLIN_VEC_RANK") else (
    """
    %g = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %g : !transform.any_op"""
           if __import__("os").environ.get("MERLIN_VEC_EW") else ""))
    _mm_arm = "" if skip_mm else f"""
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t1, %lmn:2 = transform.structured.tile_using_for %mm tile_sizes [{MM}, {NR}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %mm2 = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mm2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [{MM}, {NR}, 1] : !transform.any_op"""
    _bmm_arm = "" if skip_bmm else f"""
    %bm = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt1, %blmn:3 = transform.structured.tile_using_for %bm tile_sizes [1, {MR}, {NB}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %bm2 = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt2, %blk = transform.structured.tile_using_for %bm2 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt2 vector_sizes [1, {MR}, {NB}, 1] : !transform.any_op"""
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{{_mm_arm}{_bmm_arm}{_ew}
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def _accumulator_resident_v3_mrpad_pre_schedule(MR: int, NR: int, KC: int,
                                                NR_bmm: int | None = None) -> str:
    """PER-MATMUL MR register block with an M-PAD tail — the general fix for the M=1/M%MR!=0 case.

    ``accumulator_resident_wholemodel_vf`` clamps the matmul M tile to MR_mm=1 (no A-operand reuse:
    1 B-load + 1 A-load per FMA = 2.0 loads/useful-FMA). ``..._vf_mr4`` raises it to MR=4 (A-reuse,
    1.25 loads/FMA) but is ONLY correct where M%4==0: on M not divisible by MR the partial M tile
    needs a masked ``vector.transfer_write`` (``vector<MRxNR>`` into ``tensor<(M%MR)xNR>``) that
    LLVM-23 rejects with a multi-op ``vector.mask`` PipelineError -> (measured) NR=8/non-resident/119
    in-loop spills at M=17, and an outright PipelineError -> silent scalar fallback at M=1.

    This schedule handles the tail the way the hand shim does (``round_up_mr``): it PADS the matmul M
    dimension UP to the next multiple of MR BEFORE tiling (``transform.structured.pad`` on padding
    dimension 0, padding value 0.0). Every matmul then register-blocks CLEANLY at MR — the M tile is
    always FULL (no masked write, no PipelineError), so M=1 pads to MR, M=17 pads to 20, M=64 stays 64,
    each getting the same MR resident-accumulator vfmacc.vf register block. ``transform.structured.pad``
    threads a ``tensor.extract_slice`` back to the original M rows (the ``copy_back`` result), so the
    real output is BIT-EXACT: the padded rows are ``0-row @ B = 0`` and are discarded; only the [0:M]
    slice is written back. This is per-matmul (each op pads to ITS OWN next multiple of MR — a general
    tail rule, not a per-model constant), and it composes with the v3 subset-hoist + A-scalarization
    pipeline unchanged (it still stops at ``vector.contract``).

    The batch_matmul (attention) path is IDENTICAL to ``accumulator_resident_wholemodel_vf`` — MR=4 M
    tile + the ``NR_bmm`` N-tail clamp — so this feature only upgrades the ``linalg.matmul`` path from
    the MR=1 clamp to a padded-MR register block; the proven whole-model attention path is untouched.
    """
    NB = NR_bmm if NR_bmm is not None else NR
    z = "0.000000e+00 : f32"
    # copy_back_op = "linalg.copy" (NOT the default bufferization.materialize_in_destination): the
    # pipeline's canonicalize/cse merges the many identical `linalg.fill 0 -> tensor<MxNxf32>` matmul
    # inits into ONE shared value, and materialize_in_destination copies each padded result back into
    # that SHARED fill buffer -> one-shot-bufferize sees N writes aliasing one buffer -> "cannot avoid
    # RaW conflict" -> whole-model PipelineError -> silent scalar fallback (this is EXACTLY what breaks
    # the pre-existing vf_mr4 on rdt2). A `linalg.copy` copy-back (paired with eliminate-empty-tensors
    # in the edit_pipeline) bufferizes each padded matmul into its own buffer, so the model lowers
    # vectorized. Verified: shared-fill reproducer + rdt2/bitvla whole-model.
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %padded, %pad, %cp = transform.structured.pad %mm pad_to_multiple_of [{MR}] {{padding_values = [{z}, {z}, {z}], padding_dimensions = [0], copy_back_op = "linalg.copy"}} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %t1, %lmn:2 = transform.structured.tile_using_for %padded tile_sizes [{MR}, {NR}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %mm2 = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mm2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [{MR}, {NR}, 1] : !transform.any_op
    %bm = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt1, %blmn:3 = transform.structured.tile_using_for %bm tile_sizes [1, {MR}, {NB}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %bm2 = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt2, %blk = transform.structured.tile_using_for %bm2 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt2 vector_sizes [1, {MR}, {NB}, 1] : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def _accumulator_resident_v3_mrpad_pipeline(passes: list[str]) -> list[str]:
    """The v3 (.vf subset-hoist + A-scalarize) pipeline PLUS ``eliminate-empty-tensors`` before
    one-shot-bufferize — required by the M-pad copy-back.

    The M-pad schedule copies each padded matmul result back with ``linalg.copy`` (not the default
    ``materialize_in_destination``); the pair (linalg.copy copy-back + eliminate-empty-tensors) is what
    lets one-shot-bufferize give each padded matmul its OWN buffer instead of aliasing the CSE-shared
    zero-fill init across every matmul (the "cannot avoid RaW conflict" whole-model PipelineError).
    Reuses ``_packed_eliminate_empties`` (same insertion the packed feature uses). Only runs when this
    feature is enabled; baseline pipeline untouched."""
    return _packed_eliminate_empties(_accumulator_resident_v3_pipeline(passes))


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


# ---- accumulator-resident micro-kernel v2 (PRE-bufferize subset hoist) --------------
# The genuine gap-closer the prior `accumulator_resident_microkernel` could NOT reach. That
# feature ran `hoist_redundant_vector_transfers` POST-bufferize; on the bufferized memref form the
# accumulator transfer pair is on an scf.for **memref iter_arg** (`%arg7`), and BOTH
# `hoist_redundant_vector_transfers` AND `loop-invariant-subset-hoisting` no-op on it (the carried
# memref aliases the per-tile alloc, defeating the loop-invariance / subset analysis) — so the
# emitted K-loop round-tripped the `vector<MRxNR>` accumulator through the stack every K step
# (`vl4re8.v`/`vs4r.v` per K-tile, ~19x off the hand ceiling). MEASURED, with objdump evidence.
#
# Root cause found here (mlir-opt iteration, LLVM 23): the hoist MUST run on the TENSOR form, BEFORE
# one-shot-bufferize, where the K-loop carries the accumulator as a value-semantic
# `tensor<MRxNR>` iter_arg and the accumulator transfer pair reads/writes that tensor iter_arg at
# loop-invariant indices. On THAT form `loop-invariant-subset-hoisting` fires cleanly: it lifts the
# `vector.transfer_read` above the K-loop and the `vector.transfer_write` below it, threading a pure
# `vector<MRxNR>` through the loop as a SECOND iter_arg — the accumulator now lives in SSA vector
# values across K, never bufferized to memory inside the loop (verified in the emitted IR: the
# K-loop carries `iter_args(%acc_tensor, %acc_vector)`, body is read-A + read-B-row + MR `vector.fma`
# into `%acc_vector`, yield; the accumulator transfer pair is GONE from the loop body). After bufferize
# the carried `vector<MRxNR>` lowers to an `!llvm.array<MR x vector<NRxf32>>` loop-carried value (the
# K-loop block argument), which the RISC-V backend keeps in the vector register file across K — the
# accumulator-RESIDENT structure the hand kernel has (MR accumulator vreg-groups held across K).
#
# Because the hoist runs PRE-bufferize, the contract must NOT be lowered until AFTER the hoist (the
# subset analysis wants the read->contract->write shape). So the recipe is:
#   PRE-bufferize  (edit_schedule): the SAME `_accumulator_resident_pre_schedule` — tile [MR,NR]; tile
#                  K by 1; scoped-vectorize [MR,NR,1]; transfer_permutation + reduction_to_contract;
#                  STOP at vector.contract.
#   edit_pipeline: right after the first transform-interpreter (still on TENSORS), splice
#                  `loop-invariant-subset-hoisting` (=> the accumulator becomes a vector iter_arg)
#                  THEN a second transform-interpreter that lowers contraction->outerproduct->fma
#                  (drop-unit-dims first so the B-row/acc extracts are clean). One-shot-bufferize then
#                  sees the value-semantic vector iter_arg and keeps it register-resident.
# NOTE: this still emits `vfmacc.vv` (the A column is read as `vector<MRx1>` -> per-lane `<1xf32>`
# loads + a lane-broadcast the backend builds with a vmv/vslideup ladder), NOT the hand kernel's
# `vfmacc.vf` (A scalar straight from an FP reg). That residual A-broadcast pressure forces a small
# constant number of accumulator spills. So this CLOSES the accumulator-residency structure (the
# documented #1 gap) and a large fraction of the instret gap, but does not fully reach the hand
# ceiling — the remaining delta is the `.vf` A-operand form, recorded honestly. Default-off;
# baseline byte-identical.
#
# These patterns REQUIRE the contract to have at least one parallel dim (outerproduct needs a rank-1
# result to accumulate into). A register block of NR=1 does not give it one: vectorizing a [.., 1, 1]
# tile yields `vector.contract {iterator_types = ["reduction"]} vector<1xi8>, vector<1xi8> into i32`
# — a one-element dot product with no parallel dim, which NO lowering_strategy matches (measured: 24
# such ops survived on whisper_tiny, and an un-lowered vector.contract has no LLVM translation, so
# the build dies late with "missing LLVMTranslationDialectInterface"). That is a selection bug, not a
# lowering gap: a 1-lane vector is not a vectorization. `mining.apply` therefore never selects NR=1 —
# it leaves that op class un-tiled for `convert-linalg-to-loops` instead.
_ACCUM_RESIDENT_V2_LOWER_SCHEDULE = """\
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_accum_v2_lower(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.drop_inner_most_unit_dims_from_xfer_ops
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.vector.drop_unit_dims_with_shape_cast
    } : !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    } : !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.lower_outerproduct
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op
    transform.yield
  }
}
"""


def _accum_v2_lower_schedule_path() -> Path:
    """Stable temp path for the PRE-bufferize contract-lowering schedule (content-addressed)."""
    text = _ACCUM_RESIDENT_V2_LOWER_SCHEDULE
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    p = Path(tempfile.gettempdir()) / f"merlin_accum_v2_lower_{h}.mlir"
    if not p.is_file():
        p.write_text(text, encoding="utf-8")
    return p


def _accumulator_resident_v2_pipeline(passes: list[str]) -> list[str]:
    """Splice the PRE-bufferize subset-hoist + contract-lower right after the FIRST
    transform-interpreter (still on tensors), before one-shot-bufferize.

    The pre-schedule (edit_schedule) stops at `vector.contract`; here we (1) run
    `loop-invariant-subset-hoisting` on the tensor form so the accumulator transfer pair becomes a
    `vector<MRxNR>` scf.for iter_arg (register-resident across K), then (2) run a second
    transform-interpreter that lowers the now-hoisted contraction -> outerproduct -> vector.fma.
    Running BEFORE bufferize is the whole point: post-bufferize the carried accumulator is a memref
    iter_arg the hoist cannot lift (the prior feature's measured blocker). Only runs when this feature
    is enabled; baseline pipeline untouched."""
    out = list(passes)
    # Locate the first transform-interpreter (the pre-schedule's __transform_main) and the
    # canonicalize,cse that follow it; splice after that cse (still on tensors, pre-bufferize).
    ti = None
    for i, p in enumerate(out):
        if p.startswith("transform-interpreter{entry-point=__transform_main}"):
            ti = i
            break
    if ti is None:
        return out
    # after ti come canonicalize, cse (then linalg-generalize-named-ops, one-shot-bufferize)
    insert_at = ti + 1
    # skip the canonicalize, cse pair that the pipeline puts right after the interpreter
    while insert_at < len(out) and out[insert_at] in ("canonicalize", "cse"):
        insert_at += 1
    sched = _accum_v2_lower_schedule_path()
    inject = [
        "loop-invariant-subset-hoisting",
        f"transform-preload-library{{transform-library-paths={sched}}}",
        "transform-interpreter{entry-point=__transform_accum_v2_lower}",
        "canonicalize", "cse",
    ]
    if out[insert_at:insert_at + len(inject)] != inject:   # idempotent
        out = out[:insert_at] + inject + out[insert_at:]
    # Same convert-vector-to-scf{full-unroll} swap as the v1 feature: the hoisted accumulator is a
    # rank-2 `vector<MRxNR>` carried by the K loop; the default convert-vector-to-scf would lower
    # rank>=2 transfers through a stack alloca scratch (re-introducing the per-K stack round-trip).
    # full-unroll lowers them to rank-1 element ops with NO stack scratch, so the carried accumulator
    # stays in the vector register file across K. Feature-scoped; baseline untouched.
    for i, p in enumerate(out):
        if p == "convert-vector-to-scf":
            out[i] = "convert-vector-to-scf{full-unroll}"
            break
    return out


def _register_accumulator_resident_v2() -> list[str]:
    """Register `accumulator_resident_v2` (default tile) + a small tuning grid. Reuses the v1
    PRE-bufferize schedule (forms the contract, stops there); the v2 pipeline does the PRE-bufferize
    subset hoist + contract lowering that makes the accumulator a genuine vector iter_arg."""
    names: list[str] = []
    grid = [(4, 16, 16), (4, 16, 32), (8, 16, 16), (4, 32, 16)]
    for MR, NR, KC in grid:
        if (MR, NR, KC) == (4, 16, 16):
            nm = "accumulator_resident_v2"
            desc = ("PRE-bufferize accumulator-resident micro-kernel (the genuine residency the v1 "
                    "feature could not reach): tile [MR=4,NR=16], tile K by 1, scoped-vectorize -> "
                    "vector.contract, then on the TENSOR form (before bufferize) run "
                    "loop-invariant-subset-hoisting so the accumulator transfer pair becomes a "
                    "vector<MRxNR> scf.for iter_arg (register-resident across K, NO per-K memref "
                    "roundtrip), then lower contraction -> outerproduct -> vfmacc. Verified by "
                    "objdump: the K-loop carries the accumulator as an llvm.array<MR x vector<NR>> "
                    "loop value with NO accumulator load/store inside the loop (unlike v1's per-K "
                    "vl4re8.v/vs4r.v). Bit-exact at 32/64/128 + a non-cube on spike. Residual: emits "
                    "vfmacc.vv (A read as vector<MRx1> -> lane-broadcast) not the hand kernel's "
                    "vfmacc.vf, so a small constant accumulator-spill from A-broadcast pressure "
                    "remains; closes the residency structure + most of the instret gap but not the "
                    "full hand ceiling. Default-off, baseline byte-identical.")
        else:
            nm = f"accum_resident_v2_{MR}_{NR}_{KC}"
            desc = (f"PRE-bufferize accumulator-resident tuning point (MR={MR}, NR={NR}, KC={KC}): "
                    f"tensor-level subset-hoisted vector<MRxNR> K-loop iter_arg accumulator (no "
                    f"per-K memref roundtrip). Default-off tuning-grid feature.")
        register(ImprFeature(
            name=nm,
            action_class="PASS",
            description=desc,
            edit_pipeline=_accumulator_resident_v2_pipeline,
            edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC:
                           _accumulator_resident_pre_schedule(_MR, _NR, _KC)),
            schedule_replace=True,
        ))
        names.append(nm)
    return names


ACCUM_RESIDENT_V2_NAMES: list[str] = _register_accumulator_resident_v2()


# ---- accumulator-resident micro-kernel v3 (resident accumulator + vfmacc.vf) --------
# v2 made the accumulator register-resident (PRE-bufferize subset hoist -> vector iter_arg) but the
# emitted K-loop still used `vfmacc.vv`: the A operand was read as `vector<MRx1xf32>` and each row
# extracted `[i,0]:f32`, and the RISC-V backend cannot cheaply move a vector LANE into the `.vf`
# scalar FP operand, so it rebuilt the broadcast with a vmv/vslideup ladder (that ladder, NOT a
# spill, was the residual instret — v2 measured ~19x off the hand ceiling even though the
# accumulator was resident). v3 adds the A-operand SCALARIZATION rewrite (accum_microkernel.py):
# after the contraction is lowered to `vector.fma` with f32 A-extracts (but before bufferize), each
# `vector.transfer_read -> vector<MRx1xf32>` whose only uses are `vector.extract [i,0]:f32` is
# replaced with per-row scalar `tensor.extract`/`memref.load` (the SAME `a[i]` scalar the hand kernel
# loads). clang-23 then selects the clean `vfmacc.vf` (flw -> vfmacc.vf), and the emitted K-loop is
# the hand kernel's structure exactly: ONE B-row vle32, MR scalar A flw, MR `vfmacc.vf` into the
# resident accumulator, C stored once — 0 in-loop accumulator spills, 0 vfmacc.vv (verified by
# objdump). Numerically identical to v2 (scalar load of [i,0] == lane [i,0] of the vector read), so
# BIT-EXACT. The rewrite runs via a two-stage runner (pipeline._accum_microkernel_v3_features); the
# pipeline edit splices the SCALARIZE_MARKER where the split happens. Default-off; baseline
# byte-identical.
def _accumulator_resident_v3_pipeline(passes: list[str]) -> list[str]:
    """v2 pipeline (PRE-bufferize subset hoist + contract lower) PLUS the SCALARIZE_MARKER sentinel
    spliced immediately AFTER the contract-lowering transform-interpreter (still on tensors, before
    one-shot-bufferize). The two-stage runner splits the pipeline at the marker and runs the
    A-scalarization rewrite there. Only runs when a v3 feature is enabled; baseline untouched."""
    from .accum_microkernel import SCALARIZE_MARKER
    out = _accumulator_resident_v2_pipeline(passes)
    # The v2 edit spliced: loop-invariant-subset-hoisting, transform-preload-library{...v2_lower},
    # transform-interpreter{entry-point=__transform_accum_v2_lower}, canonicalize, cse. Put the
    # marker right AFTER that interpreter's canonicalize,cse (contract is now vector.fma w/ f32
    # A-extracts; bufferize has not run).
    idx = None
    for i, p in enumerate(out):
        if p.startswith("transform-interpreter{entry-point=__transform_accum_v2_lower}"):
            idx = i
            break
    if idx is None:
        return out
    insert_at = idx + 1
    while insert_at < len(out) and out[insert_at] in ("canonicalize", "cse"):
        insert_at += 1
    if SCALARIZE_MARKER not in out:
        out = out[:insert_at] + [SCALARIZE_MARKER] + out[insert_at:]
    return out


def _accumulator_resident_v3_kblocked_pre_schedule(MR: int, NR: int, KC: int) -> str:
    """v3 pre-schedule with a REAL K-blocking tile (the ``KC`` lever, which the plain v3 recipe
    silently ignores — it tiles K by 1 and never uses KC, so every KC produced the same schedule).

    Why it matters (measured): the emitted inner loop advances B by the full row stride each K step
    (``addi a0,a0,512`` for N=128 f32), touching a NEW cache line every iteration and re-walking them
    for every (M,N) tile. Instruction count matches XNNPACK's kernel almost exactly, yet we run ~5x
    slower — i.e. we are memory-stalled, not instruction-bound. Blocking K by KC keeps a B panel
    resident across the M/N tiles so those lines are reused before eviction (classic GEMM cache
    blocking), which is the standard fix for exactly this access pattern.

    Structure: tile [MR, NR] (registers) -> tile K by KC (CACHE) -> tile K by 1 (register-resident
    accumulation) -> scoped-vectorize [MR, NR, 1]. Pure code generation; no hand ukernel."""
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t1, %lmn:2 = transform.structured.tile_using_for %mm tile_sizes [{MR}, {NR}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %mmk = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tkc, %lkc = transform.structured.tile_using_for %mmk tile_sizes [0, 0, {KC}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %mm2 = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mm2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [{MR}, {NR}, 1] : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def ensure_v3_kblocked_microkernel(MR: int, NR: int, KC: int) -> str:
    """Register (on demand) the K-BLOCKED v3 tuning point — makes the KC lever real (cache blocking)."""
    name = f"accum_resident_v3kb_{MR}_{NR}_{KC}"
    if name in known():
        return name
    register(ImprFeature(
        name=name,
        action_class="PASS",
        description=(f"Accumulator-resident micro-kernel with REAL K-blocking (MR={MR}, NR={NR}, "
                     f"KC={KC}): K tiled by KC for cache reuse of the B panel, then by 1 for the "
                     f"register-resident accumulation. Compiler-emitted (no ukernel). Default-off."),
        edit_pipeline=_accumulator_resident_v3_pipeline,
        edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC:
                       _accumulator_resident_v3_kblocked_pre_schedule(_MR, _NR, _KC)),
        schedule_replace=True,
        implies=_tile_epilogue_hygiene(MR),
    ))
    return name


def _accumulator_resident_v3_unrolled_pre_schedule(MR: int, NR: int, KC: int) -> str:
    """v3 pre-schedule with M held as MR INDEPENDENT accumulators (the `unroll_m` axis).

    The plain v3 recipe tiles M by MR and vectorizes to a 2-D ``vector<MRxNR>``, so MR must be
    vectorization-friendly — measured on K1: MR 3/5/6/7 collapse to 193-279x off XNNPACK while MR=4
    is 5.0x. An expert micro-kernel instead keeps MR SEPARATE accumulator registers (M fully
    unrolled), which is shape-agnostic and is why XNNPACK can pick MR=7.

    Recipe: tile M by 1 (each row is its own ``vector<NR>`` accumulator) x N by NR, tile K by 1,
    scoped-vectorize the [1, NR, 1] body, then UNROLL the M loop by MR -> MR independent accumulators
    carried as separate iter_args. Pure code generation; no hand ukernel."""
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t1, %lmn:2 = transform.structured.tile_using_for %mm tile_sizes [1, {NR}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %mm2 = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mm2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [1, {NR}, 1] : !transform.any_op
    transform.loop.unroll %lmn#0 {{ factor = {MR} }} : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def ensure_v3_unrolled_microkernel(MR: int, NR: int, KC: int) -> str:
    """Register (on demand) the M-UNROLLED v3 tuning point — MR independent accumulators, any MR."""
    name = f"accum_resident_v3u_{MR}_{NR}_{KC}"
    if name in known():
        return name
    register(ImprFeature(
        name=name,
        action_class="PASS",
        description=(f"Accumulator-resident micro-kernel with M UNROLLED into {MR} independent "
                     f"accumulators (NR={NR}, KC={KC}) — shape-agnostic in MR, unlike the 2-D "
                     f"vector<MRxNR> formulation. Compiler-emitted (no ukernel). Default-off."),
        edit_pipeline=_accumulator_resident_v3_pipeline,
        edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC:
                       _accumulator_resident_v3_unrolled_pre_schedule(_MR, _NR, _KC)),
        schedule_replace=True,
        implies=_tile_epilogue_hygiene(MR),
    ))
    return name


#: RVV lowering fact (not a board constant): LLVM's RISC-V backend defines ``vscale = VLEN / 64``,
#: so an MLIR ``vector<[k]xT>`` holds ``k * VLEN/64`` elements and occupies ``k * VLEN/64 * sizeof(T)``
#: bytes of vector register file, i.e. ``LMUL = k * sizeof(T) / 8``. Two consequences we rely on:
#: the LMUL of a scalable type is FIXED by the type (never widened to cover a worst-case VLEN, which
#: is exactly the defect the ``_zvl`` pin worked around), and the element count at the RVV MINIMUM
#: VLEN of 128 bits is ``2 * k`` for EVERY element type. So ``k = NR // 2`` makes the ``NR`` knob mean
#: "lanes at the minimum VLEN" under VL_DYNAMIC, with the real lane count scaling with the hardware.
_VSCALE_LANES_PER_128B = 2


def scalable_lanes(NR: int) -> int:
    """The scalable multiplier ``k`` of ``vector<[k]xT>`` for a register block ``NR`` lanes wide at
    the RVV MINIMUM VLEN (128 bits). Dtype-independent — see :data:`_VSCALE_LANES_PER_128B`."""
    if NR % _VSCALE_LANES_PER_128B:
        raise ValueError(
            f"vl_strategy='dynamic' needs an even NR (NR counts lanes at the RVV minimum VLEN of "
            f"128 bits, and a scalable vector<[k]xT> holds 2k of them); got NR={NR}.")
    return NR // _VSCALE_LANES_PER_128B


def _accumulator_resident_v3_scalable_pre_schedule(MR: int, NR: int, KC: int) -> str:
    """v3 pre-schedule emitting a VL-AGNOSTIC (scalable) register block — the ``vl_strategy=dynamic``
    axis. Same recipe as :func:`_accumulator_resident_v3_pre_schedule` with the N tile made SCALABLE.

    Why this is the general fix. Our fixed-width codegen asks for ``vector<NRxf32>``, and the backend
    must size that register group for the worst VLEN the ``march`` string admits (``rv64gcv`` promises
    only the 128-bit minimum), so on a VLEN=256 part every value got DOUBLE the LMUL it needed and
    ``vl`` sat at half ``VLMAX``. Pinning ``_zvl256b`` fixes it for ONE board and miscompiles on a
    narrower one. A scalable ``vector<[k]xf32>`` has no such worst case: its LMUL is fixed by the type
    and its lane count is whatever the hardware reports, which is what the expert kernels do by
    calling ``__riscv_vsetvl_e32m4`` at run time. ``NR`` is reinterpreted as lanes at the MINIMUM
    VLEN (see :func:`scalable_lanes`), so ``NR`` still names the register-block width — it just names
    the LMUL rather than a lane count that only one board would honour.

    Two things the naive scalable schedule gets wrong, plus one PRECONDITION it inherits (all
    reproduced against LLVM/MLIR 23, see docs/design/vl_agnostic_codegen.md):

      * The N loop trip count is no longer a compile-time multiple of the tile, so ``vectorize``
        MASKS every transfer. Masked scalable transfers (a) block
        ``loop-invariant-subset-hoisting``, which is what makes the accumulator a register-resident
        ``scf.for`` iter_arg — without it the K loop round-trips C through memory, the exact defect
        v3 exists to remove — and (b) lower through a ``vector.transpose`` on
        ``vector<MRx1x[k]xf32>`` that has NO scalable lowering and survives to the LLVM edge
        (``error: Dialect 'vector' not found for custom op 'vector.transpose'``). THIS transpose,
        not the ``ub.poison`` the old note named, is the real residue of the "incomplete lowering".
        FIX: PEEL the N loop (``transform.loop.peel``) so the main loop's tile is exactly
        ``[k] * vscale`` wide, then vectorize it with ``assume_dynamic_dims_match_vec_sizes`` — the
        peel is what MAKES that assumption true. No masks, no transpose; the subset hoist fires
        exactly as in the fixed-width recipe.
      * The peeled N remainder keeps its ``linalg.matmul`` and falls through
        ``convert-linalg-to-loops`` to a scalar tail — correct for any N, zero-trip whenever the
        hardware VL divides N.
      * PRECONDITION ``MR | M`` (inherited, not introduced). ``assume_dynamic_dims_match_vec_sizes``
        asserts EVERY dynamic tile dim equals its vector size, INCLUDING M. The static ``MR`` M-tile
        has a partial last iteration when ``MR`` does not divide M, and the flag then makes the body
        write ``MR`` rows into a shorter tile (measured on the K1 at 130^3: ``malloc(): corrupted
        top size``; 128^3 and 100^3, where MR|M, are bit-exact). This is the SAME "MR must divide M"
        constraint the fixed 2-D ``vector<MRxNR>`` v3 recipe already carries (see
        expert_gap_attribution.md — small-M falls off a cliff). It is NOT fixed here by peeling M:
        ``transform.loop.peel`` FAILS on a statically-divisible loop, so an M peel would break the
        common MR|M case (M=128) to protect the MR∤M one. The M-tail is an orthogonal axis
        (pad/peel, shared with the fixed recipe); until it lands, MR∤M is fail-closed (the harness
        records not_run on the crash, never a false timing), not silently wrong.

    ``KC`` is carried for naming parity with the other v3 points; like the plain v3 recipe it does
    not tile the reduction (``k_block`` is the lever that does)."""
    k = scalable_lanes(NR)
    return f"""\
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t1, %lm, %ln = transform.structured.tile_using_for %mm tile_sizes [{MR}, [{k}], 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %lnf = transform.cast %ln : !transform.any_op to !transform.op<"scf.for">
    %main, %tail = transform.loop.peel %lnf : (!transform.op<"scf.for">) -> (!transform.any_op, !transform.any_op)
    %mms = transform.structured.match ops{{["linalg.matmul"]}} in %main : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mms tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [{MR}, [{k}], 1] {{assume_dynamic_dims_match_vec_sizes}} : !transform.any_op
    %f = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {{
      transform.apply_patterns.vector.transfer_permutation_patterns
      transform.apply_patterns.vector.reduction_to_contract
      transform.apply_patterns.vector.fold_arith_extension
      transform.apply_patterns.vector.reduction_to_contract
    }} : !transform.any_op
    transform.yield
  }}
}}
"""


def ensure_v3_scalable_microkernel(MR: int, NR: int, KC: int) -> str:
    """Register (on demand) the VL-AGNOSTIC v3 tuning point — a scalable (``vsetvli``-sized) N block.

    This is the realization of ``MicrokernelSpec.vl_strategy = VL_DYNAMIC``: the emitted inner loop
    sizes itself to the vector length the hardware reports, so it needs no ``_zvl`` march pin and is
    correct (and full-width) on ANY RVV part."""
    k = scalable_lanes(NR)
    name = f"accum_resident_v3vl_{MR}_{NR}_{KC}"
    if name in known():
        return name
    register(ImprFeature(
        name=name,
        action_class="PASS",
        description=(f"VL-AGNOSTIC accumulator-resident micro-kernel (MR={MR}, NR={NR} lanes at the "
                     f"RVV minimum VLEN -> vector<[{k}]xT>, KC={KC}): the N register block is a "
                     f"SCALABLE vector, so the emitted loop sizes to the hardware's runtime vector "
                     f"length (vsetvli against VLMAX) instead of a compile-time width that the "
                     f"backend must widen for the worst-case VLEN. Needs no _zvl march pin. N loop "
                     f"is peeled so the main body is unmasked; the remainder is a scalar tail. "
                     f"Compiler-emitted (no ukernel). Default-off."),
        edit_pipeline=_accumulator_resident_v3_pipeline,
        edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC:
                       _accumulator_resident_v3_scalable_pre_schedule(_MR, _NR, _KC)),
        schedule_replace=True,
        implies=_tile_epilogue_hygiene(MR),
    ))
    return name


def ensure_v3_microkernel(MR: int, NR: int, KC: int) -> str:
    """Register (on demand) and return the name of the v3 accumulator-resident micro-kernel tuning
    point for ANY (MR, NR, KC) — turning the fixed 4-point grid into a CONTINUOUS, beam-tunable space.

    This is how the compiler reaches expert-kernel granularity by CODE GENERATION alone: the v3 recipe
    is fully compiler-emitted (tile [MR,NR] + K-by-1, scoped-vectorize -> vector.contract, PRE-bufferize
    subset-hoisting makes the accumulator a register-resident scf.for iter_arg, contraction ->
    outerproduct -> vector.fma, then A-scalarization -> vfmacc.vf). NO hand ukernel is involved — the
    intrinsic driver remains a ceiling REFERENCE only. Registering a point is idempotent."""
    name = ("accumulator_resident_microkernel_v3" if (MR, NR, KC) == (4, 16, 16)
            else f"accum_resident_v3_{MR}_{NR}_{KC}")
    if name in known():
        return name
    register(ImprFeature(
        name=name,
        action_class="PASS",
        description=(f"Accumulator-resident vfmacc.vf micro-kernel tuning point (MR={MR}, NR={NR}, "
                     f"KC={KC}), registered on demand so the beam can tune the register block "
                     f"continuously. Compiler-emitted (no ukernel). Default-off."),
        edit_pipeline=_accumulator_resident_v3_pipeline,
        edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC:
                       _accumulator_resident_v3_pre_schedule(_MR, _NR, _KC)),
        schedule_replace=True,
        implies=_tile_epilogue_hygiene(MR),
    ))
    return name


def ensure_v3_perop_microkernel(MR_mm: int | None, NR_mm: int | None,
                                MR_bmm: int | None, NR_bmm: int | None,
                                KC: int) -> str:
    """Register (on demand) a v3 tuning point with an INDEPENDENT register block per op class.

    ``_accumulator_resident_v3_pre_schedule`` already emits four separate tile factors — the matmul
    arm uses ``[MR_mm, NR_mm]`` and the batch_matmul arm ``[1, MR_bmm, NR_bmm]`` — but until now the
    only way to reach them was the hand-frozen ``accumulator_resident_wholemodel_vf`` point, whose
    clamps (MR_mm=1, NR_bmm=8) are constants somebody picked once. They are a SHAPE decision: a
    model's matmuls and its attention batch_matmuls have different extents, so the largest
    non-masking block differs per class. This registrar makes that decision expressible, so a
    shape-aware policy can DERIVE the four factors from the workload instead of pinning them.

    ``accumulator_resident_wholemodel_vf`` == ``ensure_v3_perop_microkernel(1, 16, 4, 8, 16)``; that
    point keeps its own name (and its measured history) rather than being aliased here.

    A class's block may be ``None``, meaning DO NOT CLAIM this op class — its contractions are left
    to ``convert-linalg-to-loops``. That is the honest answer when no block wider than one lane is
    legal for the class's extents (see ``_accumulator_resident_v3_pre_schedule``); the alternative is
    a build that fails outright. Both classes ``None`` would vectorize nothing, so it is rejected."""
    skip_mm = MR_mm is None or NR_mm is None
    skip_bmm = MR_bmm is None or NR_bmm is None
    if skip_mm and skip_bmm:
        raise ValueError("ensure_v3_perop_microkernel: at least one op class must be claimed "
                         "(both blocks None vectorizes nothing — use the scalar backend instead)")
    tag_mm = "x_x" if skip_mm else f"{MR_mm}_{NR_mm}"
    tag_bmm = "x_x" if skip_bmm else f"{MR_bmm}_{NR_bmm}"
    name = f"accum_resident_v3p_{tag_mm}_{tag_bmm}_{KC}"
    if name in known():
        return name
    _mm_desc = "linalg.matmul UNCLAIMED (scalar)" if skip_mm else f"linalg.matmul [{MR_mm}, {NR_mm}]"
    _bmm_desc = ("linalg.batch_matmul UNCLAIMED (scalar)" if skip_bmm
                 else f"linalg.batch_matmul [1, {MR_bmm}, {NR_bmm}]")
    register(ImprFeature(
        name=name,
        action_class="PASS",
        description=(f"Accumulator-resident vfmacc.vf micro-kernel with a per-op-class register "
                     f"block ({_mm_desc}, {_bmm_desc}, KC={KC}), registered on demand so a "
                     f"shape-aware policy can pick a blocking that masks no parallel dim. "
                     f"Compiler-emitted (no ukernel). Default-off."),
        edit_pipeline=_accumulator_resident_v3_pipeline,
        edit_schedule=(lambda _t, _mm=MR_mm, _nm=NR_mm, _mb=MR_bmm, _nb=NR_bmm, _KC=KC,
                       _sm=skip_mm, _sb=skip_bmm:
                       _accumulator_resident_v3_pre_schedule(_mb or 1, _nm or 1, _KC,
                                                             NR_bmm=_nb or 1, MR_mm=_mm or 1,
                                                             skip_mm=_sm, skip_bmm=_sb)),
        schedule_replace=True,
        implies=_tile_epilogue_hygiene(None if skip_mm else MR_mm),
    ))
    return name


#: Op-class order encoded in a ``accum_resident_v3p_*`` name (two factors each, then KC).
_V3P_PREFIX = "accum_resident_v3p_"
_V3P_CLASS_ORDER = ("linalg.matmul", "linalg.batch_matmul")


def unclaimed_op_classes(feature: str) -> list[str]:
    """Contraction op classes a ``accum_resident_v3p_*`` point does NOT tile/vectorize.

    Lives next to :func:`ensure_v3_perop_microkernel` so the ``x_x`` spelling has ONE definition:
    a caller that wants to report "this class runs scalar" must not re-derive the name format.
    Returns [] for any other feature (including the hand-frozen points, which claim both classes).
    """
    if not feature.startswith(_V3P_PREFIX):
        return []
    parts = feature[len(_V3P_PREFIX):].split("_")
    if len(parts) != 2 * len(_V3P_CLASS_ORDER) + 1:
        return []
    return [cls for i, cls in enumerate(_V3P_CLASS_ORDER) if parts[2 * i] == "x"]


#: The non-contraction tail: every op the contraction-only schedule never matches (elementwise,
#: layout, im2col gather, pad) falls through convert-linalg-to-loops to SCALAR code on one core. It is
#: 86-89% of the linalg ops of every workload measured, which is why whole-model MAC/cycle sits at a
#: few percent of the datapath's peak while the matmul kernel itself looks fine.
VEC_NONCONTRACTION_NAME = "vectorize_non_contraction_generics"
#: Default lane count for the bare feature name. The lane width is a KNOB SPACE, not a
#: constant: `ensure_vec_noncontraction(lanes)` registers a point per width so a search can
#: pick it. Measured on deepjscc: 8 lanes emits 4.9x more vector instructions and runs 1.28x
#: SLOWER (per-8-element loop overhead beats the vector win), so the width matters and the
#: default must not be assumed good.
VEC_NONCONTRACTION_LANES = 8

#: Bounded per-rank vectorize of the tagged all-parallel generics. BOUNDED on purpose: a plain
#: no-sizes `vectorize` on a whole model explodes (vector<17x576> = 9792 lanes, measured 8725 ms), and
#: `vectorize_children` on the func scalarizes every generic into tens of thousands of vector.extracts
#: that this LLVM build does not lower. Innermost 8 lanes, one arm per loop rank, matched by the
#: `merlin.vec_r{rank}` attribute the prepare pass sets.
#: Each arm TILES to the vector width before vectorizing, exactly as the contraction arms do.
#: ``structured.vectorize`` does NOT tile: its sizes have to cover the iteration space, so vectorizing
#: an untiled 1x64 relu with [1, 8] fails the whole pipeline with "Attempted to vectorize, but failed"
#: (measured on deepjscc). Tiling first also means the vector shape is exactly the tile, so nothing is
#: masked -- the tagging predicate only admits extents that are whole multiples of the lane count.
def _vec_rank_arms(lanes: int) -> str:
    """The per-rank tile+vectorize arms at ``lanes`` innermost lanes."""
    return f"""\
    %g2 = transform.structured.match attributes{{merlin.vec_r2}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %gt2, %gl2:2 = transform.structured.tile_using_for %g2 tile_sizes [1, {lanes}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %gt2 vector_sizes [1, {lanes}] : !transform.any_op
    %g3 = transform.structured.match attributes{{merlin.vec_r3}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %gt3, %gl3:3 = transform.structured.tile_using_for %g3 tile_sizes [1, 1, {lanes}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %gt3 vector_sizes [1, 1, {lanes}] : !transform.any_op
    %g4 = transform.structured.match attributes{{merlin.vec_r4}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %gt4, %gl4:4 = transform.structured.tile_using_for %g4 tile_sizes [1, 1, 1, {lanes}] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.structured.vectorize %gt4 vector_sizes [1, 1, 1, {lanes}] : !transform.any_op
    %vecf = transform.structured.match ops{{["func.func"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %vecf {{
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.vector.drop_unit_dims_with_shape_cast
      transform.apply_patterns.vector.lower_shape_cast
    }} : !transform.any_op
"""



def _splice_vec_rank_arms(text: str, lanes: int = VEC_NONCONTRACTION_LANES) -> str:
    """Insert the per-rank vectorize arms just before the schedule's func-level pattern block.

    ADDITIVE (not schedule_replace), so it layers on whatever micro-kernel recipe is in play: the
    contraction arms above it are untouched and only the previously-unmatched generics gain arms.
    A schedule that already carries them, or has no func anchor, is returned unchanged.
    """
    anchor = '    %f = transform.structured.match ops{["func.func"]}'
    if "merlin.vec_r" in text or anchor not in text:
        return text
    return text.replace(anchor, _vec_rank_arms(lanes) + anchor, 1)


register(ImprFeature(
    name=VEC_NONCONTRACTION_NAME,
    action_class="PASS",
    description=(
        "Bounded per-rank vectorize of the NON-CONTRACTION all-parallel linalg.generics "
        "(elementwise / layout / im2col gather / pad) that the contraction-only schedule leaves for "
        "convert-linalg-to-loops to emit as scalar loops. Measured share of the loss: 86-89% of the "
        "linalg ops of every captured workload, against whole-model MAC/cycle of 0.40 (spectformer), "
        "0.22 (deepjscc), 0.067 (lstmnetvit) versus ~8 for a VLEN=128 int8 vwmacc datapath. Needs the "
        "prepare pass's merlin.vec_r{rank} tags, which build_app enables when this feature is on. "
        "MEASURED on deepjscc int8 (spike): emits 394 -> 1945 vector instructions (4.9x, so the "
        "lever is NOT inert), output BIT-IDENTICAL, and 484,690,000 -> 621,555,001 cycles, i.e. "
        "1.28x SLOWER. Flat at 0.78x across 8/16/32 lanes, and memrefCopy/malloc counts barely "
        "move, so the cost is the tile-and-vectorize-on-tensors realization (per-tile destructive "
        "update) rather than the vector width. Keep default-off until a realization that fuses "
        "the elementwise into the contraction epilogue (or uses a vsetvlmax-width loop) beats the "
        "scalar baseline. Recorded so nobody enables it expecting a win."),
    edit_schedule=_splice_vec_rank_arms,
))


def vec_noncontraction_lanes(features) -> int | None:
    """Innermost lane count the enabled non-contraction-vectorize feature asks for, else None.

    The TAGGING predicate (which admits an op only when its innermost extent is a whole multiple of the
    lane count, so nothing is masked) and the SCHEDULE arms must agree on this number; deriving it from
    the feature name is what keeps them from drifting apart.
    """
    for f in features or ():
        if f == VEC_NONCONTRACTION_NAME:
            return VEC_NONCONTRACTION_LANES
        if f.startswith(f"{VEC_NONCONTRACTION_NAME}_l"):
            tail = f.rsplit("_l", 1)[1]
            if tail.isdigit():
                return int(tail)
    return None


# ---------------------------------------------------------------------------------------------------
# `linalg.broadcast`, PRICED AND DECLINED. It surfaced as 8.95% of the profiled accelerator device leg
# -- third behind linalg.generic (62.48%) and linalg.transpose (12.52%), and a lever nobody had named --
# so it looked like a candidate for the vectorize lever right below. It is not, and the reason is the
# same reason that lever measured inert.
#
# Priced from the completed profiled leg (5,032 PROF lines joined against its op_profile_table.json,
# resolved coverage 1.00000, 300,868,459 of 3,359,861,701 ticks across 567 distinct ops), then read
# back out of the instrumented module. Every one is a RANK-EXPANDING materialization of a small vector
# into a large tensor -- `256xf32 -> 1x196x256xf32` (x70), `2048xf32 -> 2048x14xf32` (x64),
# `196xf32 -> 196x256xf32` (x48) -- i.e. a scale/bias/mask vector written out at full tensor size so
# the next elementwise op can read it index-for-index. Across all 567: 3.3 MiB read, 171.1 MiB
# WRITTEN, a 51.9x amplification, and ZERO arithmetic.
#
# So it is pure memory traffic, and "vectorize the broadcast loop" cannot be the fix: wider stores do
# not reduce the number of bytes stored. That is exactly what `vectorize_non_contraction_generics`
# measured -- 4.9x more vector instructions, bit-identical output, 1.28x SLOWER, flat across 8/16/32
# lanes -- because the cost was never vector width. The only real fix is to NOT materialize the tensor:
# have the consumer read the small operand through a broadcasting indexing map and drop the broadcast
# entirely. That is fusion / indexing-map rewriting, which is the lean-runtime fusion program and is
# explicitly out of scope here. Recorded so the next reader does not re-derive it and reach for the
# lever below.
# ---------------------------------------------------------------------------------------------------


def ensure_vec_noncontraction(lanes: int) -> str:
    """Register (on demand) the non-contraction vectorize point at ``lanes`` innermost lanes.

    The lane width is the knob the measurement says matters: at 8 lanes on deepjscc the lever emits
    4.9x more vector instructions, keeps the output BIT-IDENTICAL, and still runs 1.28x slower, because
    a [1, 8] tile pays loop overhead per 8 elements. Exposing the width as a registered point per value
    is what lets a search find one that pays instead of a human guessing.
    """
    if lanes == VEC_NONCONTRACTION_LANES:
        return VEC_NONCONTRACTION_NAME
    name = f"{VEC_NONCONTRACTION_NAME}_l{lanes}"
    if name in known():
        return name
    register(ImprFeature(
        name=name, action_class="PASS",
        description=(f"Bounded per-rank vectorize of the non-contraction all-parallel generics at "
                     f"{lanes} innermost lanes (see {VEC_NONCONTRACTION_NAME}). Default-off; the width "
                     f"must be chosen by measurement, not assumed."),
        edit_schedule=lambda t, _l=lanes: _splice_vec_rank_arms(t, _l),
    ))
    return name


#: Per-op register blocking: each contraction is tiled at the block legal for ITS OWN extents, instead
#: of one block per op CLASS. The class-wide decision is one too coarse -- whisper_tiny's single N=1
#: decode step forces its whole batch_matmul class (34% of the model's MACs) off the vector path. See
#: llvmlower/perop_blocks.py for why the tag has to be applied after specialization.
PEROP_BLOCK_NAME = "perop_register_block"


def _perop_sentinel_unresolved(_passes):
    """Reached only if the sentinel survived to lowering, which is a BUG -- so raise, loudly.

    ``perop_register_block`` is a REQUEST, not a lowering edit: the block table can only be derived
    from the PREPARED module, so ``zephyr_model.prepare_for_lowering`` derives it, tags the IR, and
    swaps this name for the concrete ``ensure_perop_block(...)`` feature. If the sentinel is still
    enabled at ``apply_pipeline`` time, the caller skipped that step -- and the consequence is
    invisible: nothing tagged the IR, so no schedule arm matches and EVERY contraction falls to
    ``convert-linalg-to-loops`` while the build reports success and the numbers stay correct. That is
    the measured deepjscc "2.56x regression that looks like a bad block but is an untagged build".
    Failing here is the only way that gets noticed.
    """
    raise RuntimeError(
        f"{PEROP_BLOCK_NAME!r} reached the lowering pipeline unresolved. It must be consumed by "
        "runtime.backends.zephyr_model.prepare_for_lowering, which derives the per-op block table "
        "from the prepared IR, tags the contractions, and replaces this sentinel with the concrete "
        "ensure_perop_block(...) feature. Lowering with it still set would leave every contraction "
        "untagged and silently scalar.")


PEROP_NR_FILL_NAME = "perop_nr_fill_register"


def _perop_nr_fill_unresolved(_passes):
    """Same contract as the block sentinel: a REQUEST consumed at preparation time, never a pass."""
    raise RuntimeError(
        f"{PEROP_NR_FILL_NAME!r} reached the lowering pipeline unresolved. It must be consumed by "
        "runtime.backends.zephyr_model.prepare_for_lowering, which is where the per-op block table is "
        "derived and is therefore the only place the board's vector length can widen an N cap.")


# A SEARCH KNOB, deliberately not a default, because its SIGN depends on the model. NR is an element
# count, so one number is a different fraction of the register file at each element width; this asks
# block_table to widen each contraction's N cap until its narrowest element fills a whole vector
# register (perop_blocks.nr_cap_for_dtypes). MEASURED on the live K1 (VLEN=256, whole model, per-op
# blocking on both arms, interleaved same-session arms, n=3, min-of-n, cos identical per model):
#
#   spectformer_int8_full        2,066,414,196 -> 1,781,260,404 cyc   1.160x FASTER
#   small_llama_int8_consistent      8,822,088 ->    10,547,988 cyc   1.196x SLOWER
#
# Both are far outside the board's 2.6% band, and they point OPPOSITE ways. The mechanism is visible in
# the emitted object and explains why: the accumulator is i32, not i8, so at VLEN=256 an NR=16 tile is
# already LMUL m4 and NR=32 pushes it to m8 -- the whole register file in one group. Decoded at 128^3
# int8 MR=4: NR=16 emits `e32,m4` with ZERO accumulator spill ops; NR=32 emits `e32,m8` with SIX
# (vs8r.v x3 + vl8re8.v x3), i.e. the carried accumulator round-trips through the stack every K-tile.
# Whether the wider tile wins is therefore a per-model question about how much of the model is wide
# enough to pay for that -- exactly the kind of question the beam exists to answer and a hand-picked
# default cannot. So it ships forkable and off.
#
# `implies` the block sentinel because it has no meaning without per-op blocking: there is no per-op N
# cap to widen otherwise. schedule_replace stays False -- it changes the TABLE, not the schedule shape,
# and the replacement schedule comes from the block feature it implies.
register(ImprFeature(
    name=PEROP_NR_FILL_NAME,
    action_class="KNOB",
    description="widen each contraction's per-op N cap until its NARROWEST element fills a whole "
                "vector register at the board's VLEN, instead of every op sharing one element count. "
                "MEASURED model-dependent on the K1: 1.160x faster on spectformer int8, 1.196x slower "
                "on small_llama int8 -- because the i32 accumulator is what sets LMUL, so a wider N "
                "tile can push it from m4 to m8 and spill (decoded: 0 -> 6 accumulator spill ops). A "
                "search knob, not a default. Default-off; baseline byte-identical.",
    edit_pipeline=_perop_nr_fill_unresolved,
    implies=frozenset({PEROP_BLOCK_NAME}),
))


# Registered so the SEARCH can reach it. The beam composes candidate feature sets through
# `impr_features.get`/`normalize`, so an unregistered name is not "rejected" -- it is silently never
# proposed (`wholemodel_proposer._composes` catches the KeyError and returns False). An unregisterable
# lever is an unsearchable one, which is the exact failure this whole line of work is about.
# `schedule_replace=True` is honest: what it resolves TO emits a complete transform schedule, so the
# composition rule must refuse stacking it with another replacement.
register(ImprFeature(
    name=PEROP_BLOCK_NAME,
    action_class="PASS",
    description="request PER-CONTRACTION register blocking: derive the widest block legal for each "
                "contraction's OWN extents (and its own narrowest element width) from the prepared "
                "IR, tag each contraction, and emit one tile+vectorize arm per distinct block. "
                "Resolved by prepare_for_lowering into a concrete, table-specific feature; a "
                "sentinel, so it must never reach lowering itself. Replaces the class-wide clamps "
                "(one degenerate extent in a class otherwise forces the whole class off the vector "
                "path). Default-off; baseline byte-identical.",
    edit_pipeline=_perop_sentinel_unresolved,
    schedule_replace=True,
))


def ensure_perop_block(table, kc: int) -> str:
    """Register (on demand) the per-op-blocked schedule for THIS model's block table.

    The schedule text is a function of the table (one tile+vectorize arm per distinct block), so the
    feature is registered per distinct table -- named by a hash of it, the same on-demand pattern
    ``ensure_v3_perop_microkernel`` uses for the register-block knob space.
    """
    import hashlib

    from . import perop_blocks as _pb

    blocks = _pb.distinct_blocks(table)
    key = hashlib.sha1(repr(sorted(table.items())).encode()).hexdigest()[:12]
    name = f"{PEROP_BLOCK_NAME}_{len(blocks)}b_{kc}_{key}"
    if name in known():
        return name
    text = _pb.schedule_text(table, kc)
    register(ImprFeature(
        name=name,
        action_class="PASS",
        description=(f"Per-op register blocking: {len(blocks)} distinct blocks over "
                     f"{len(table)} contraction geometries, KC={kc}. Each contraction is tiled at the "
                     f"widest block legal for its own extents, matched by the merlin.blk_<MR>x<NR> tag "
                     f"the prepare step applies after specialization. Replaces the per-op-CLASS block, "
                     f"whose smallest member otherwise clamps the whole class (measured: whisper_tiny "
                     f"claims 65.9% of its MACs per class vs 100% per op)."),
        edit_pipeline=_accumulator_resident_v3_pipeline,
        edit_schedule=lambda _t, _text=text: _text,
        schedule_replace=True,
    ))
    return name


def _register_accumulator_resident_v3() -> list[str]:
    """Register `accumulator_resident_microkernel_v3` (default tile) + a small tuning grid. Reuses
    the v1 PRE-bufferize schedule (forms the contract); the v3 pipeline does the v2 PRE-bufferize
    subset hoist + contract lowering AND splices the SCALARIZE_MARKER so the runner scalarizes the A
    operand -> the K-loop emits the hand kernel's accumulator-resident vfmacc.vf structure."""
    names: list[str] = []
    grid = [(4, 16, 16), (4, 16, 32), (8, 16, 16), (4, 32, 16)]
    for MR, NR, KC in grid:
        if (MR, NR, KC) == (4, 16, 16):
            nm = "accumulator_resident_microkernel_v3"
            desc = ("COMPILER-EMITTED accumulator-resident, register-blocked, vfmacc.vf RVV GEMM "
                    "micro-kernel — the genuine answer to the #1 scalable-gap the transform-only "
                    "v1/v2 features could not reach. Recipe: tile [MR=4,NR=16], tile K by 1, "
                    "scoped-vectorize -> vector.contract; PRE-bufferize loop-invariant-subset-"
                    "hoisting makes the accumulator a vector<MRxNR> scf.for iter_arg (register-"
                    "resident across K); lower contraction -> outerproduct -> vector.fma; then the "
                    "A-operand scalarization rewrite (accum_microkernel.py) replaces the A "
                    "vector<MRx1> read + extract:f32 with per-row scalar loads so the backend emits "
                    "vfmacc.vf (flw) not vfmacc.vv (vmv/vslideup). Emitted K-loop (objdump): ONE B "
                    "vle32 + MR A flw + MR vfmacc.vf into the resident accumulator + C stored once, "
                    "0 in-loop accumulator spills, 0 vfmacc.vv — the hand ceiling's structure, "
                    "compiler-emitted. BIT-EXACT vs scalar ref at 32/64/128 + non-cube on spike "
                    "(scalar load of A[i,0] == lane [i,0]). Default-off, baseline byte-identical.")
        else:
            nm = f"accum_resident_v3_{MR}_{NR}_{KC}"
            desc = (f"Accumulator-resident vfmacc.vf micro-kernel tuning point (MR={MR}, NR={NR}, "
                    f"KC={KC}): resident vector<MRxNR> K-loop iter_arg accumulator + scalar-A "
                    f"vfmacc.vf. Default-off tuning-grid feature.")
        register(ImprFeature(
            name=nm,
            action_class="PASS",
            description=desc,
            edit_pipeline=_accumulator_resident_v3_pipeline,
            edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC:
                           _accumulator_resident_v3_pre_schedule(_MR, _NR, _KC)),
            schedule_replace=True,
            implies=_tile_epilogue_hygiene(MR),
        ))
        names.append(nm)

    # WHOLE-MODEL-SAFE vfmacc.vf composed variant (this iteration's gap-closer). The
    # `accumulator_resident_wholemodel` feature carries BOTH tail clamps (matmul MR_mm=1, batch_matmul
    # NR_bmm=8) so the resident-accumulator micro-kernel SURVIVES the small-M openvla/rdt2 matmuls
    # (M=17/20/28) and small-N attention — but it rides the v1 POST-bufferize hoist path, which still
    # emits `vfmacc.vv` + a per-K vslideup/vmv broadcast ladder (~20 inner-loop insns/FMA; the
    # measured openvla/rdt2 residual, see output/kernels/ceiling/kernel_breakdown.md). The
    # `accumulator_resident_microkernel_v3` feature emits the clean `vfmacc.vf` (~3 insns/FMA, at the
    # hand ceiling) via the PRE-bufferize subset hoist + A-scalarization — but its bare MR=4 M-tiling
    # trips the LLVM-23 masked-transfer_write PipelineError on small-M (M=17), degrading to
    # NR=8/non-resident/118-spill (the `ours_v3 @ openvla 17×192×576` breakdown row).
    #
    # THIS feature is the merge of the two proven pieces: the v3 PRE-bufferize subset-hoist +
    # contract-lower + A-scalarization (`_accumulator_resident_v3_pipeline`, emits vfmacc.vf) on top of
    # the v3 PRE-bufferize schedule WITH the wholemodel tail clamps inherent (MR_mm=1 + NR_bmm=8). The
    # clamps make MR=min(MR,M)=1 on the matmul path, so the small-M matmul vectorizes FULL (no masked
    # write -> no PipelineError -> no scalar fallback) AT NR=32, accumulator-resident; the A-scalarize
    # then turns the A `vector<MRx1>` lane-rebuild into per-row scalar loads so the K-loop emits
    # `vfmacc.vf` not `vfmacc.vv`. Net: the whole-model kernel's K-loop goes from ~20 ops/FMA
    # (vfmacc.vv + broadcast ladder) to ~3 ops/FMA (flw + vle32 + vfmacc.vf) while KEEPING the NR=32 +
    # residency it already had — closing the dominant share of the openvla/rdt2 gap. Both pieces are
    # proven separately (wholemodel = small-M survival at NR=32; v3 = vfmacc.vf at the hand ceiling);
    # this composes them in ONE schedule. Default-off; baseline byte-identical.
    register(ImprFeature(
        name=WHOLEMODEL_VF_NAME,
        action_class="PASS",
        description="Whole-model-safe vfmacc.vf accumulator-resident micro-kernel: the v3 "
                    "PRE-bufferize subset-hoist + A-scalarization recipe (emits vfmacc.vf, ~3 "
                    "inner-loop insns/FMA, at the hand ceiling) WITH the wholemodel tail clamps "
                    "inherent (matmul MR_mm=1, batch_matmul NR_bmm=8). The clamps make the small-M "
                    "openvla/rdt2 matmuls (M=17/20/28) vectorize FULL at NR=32 (no masked "
                    "transfer_write -> no LLVM-23 PipelineError -> no scalar fallback, unlike bare "
                    "v3 which degrades to NR=8/non-resident), and the A-scalarization turns the A "
                    "lane-rebuild into per-row scalar loads so the K-loop emits vfmacc.vf not "
                    "vfmacc.vv + the ~20-insn vslideup/vmv broadcast ladder. Net: takes the "
                    "whole-model kernel's K-loop from ~20 to ~3 ops/FMA while keeping NR=32 + "
                    "accumulator residency it already had — the openvla/rdt2 gap-closer "
                    "(output/kernels/ceiling/kernel_breakdown.md). Bit-exact (scalar A[i,0] == lane "
                    "[i,0]). Default-off; baseline byte-identical.",
        edit_pipeline=_accumulator_resident_v3_pipeline,
        edit_schedule=lambda _t: _accumulator_resident_v3_pre_schedule(
            *WHOLEMODEL_VF_CAPS, NR_bmm=WHOLEMODEL_VF_NR_BMM, MR_mm=WHOLEMODEL_VF_MR_MM),
        schedule_replace=True,
    ))
    names.append(WHOLEMODEL_VF_NAME)

    # ITERATION-3 (packing/memory residual): MR>1 register-block variant of the vf kernel for
    # A-OPERAND REUSE — the OpenBLAS lever. The memory-traffic decode (output/kernels/ceiling/
    # packing_residual.md) established that the iteration-2 `accumulator_resident_wholemodel_vf`
    # kernel ALREADY ties XNNPACK's inner-loop DATA MOVEMENT at every openvla/rdt2 shape: 1
    # unit-stride B load + 1 scalar A load = 2.0 loads/useful-FMA, unit_stride_only, 0 broadcast
    # ladder (the .vv vslideup/vmv ladder is gone). So vs XNNPACK there is NO per-FMA memory residual
    # left. The ONE remaining data-movement lever is the OpenBLAS MR>1 register block: holding MR
    # output rows in MR accumulator vreg-groups so ONE B-row load is shared across MR FMAs, dropping
    # loads/FMA from 2.0 (MR=1) toward 1+1/MR. This feature is that register block: matmul MR_mm=MR
    # (A-reuse) on the v3 vfmacc.vf path, with the batch_matmul NR_bmm=8 N-tail clamp retained.
    #
    # MEASURED (decode, this iteration): on a LARGE-M, MR-divisible matmul (M=20=5*4, M=64, M=128) the
    # MR=4 kernel emits MR=4 vfmacc.vf sharing 1 unit-stride B load => loads/FMA = 1.25 (1 B-load + 4
    # A-scalars over 4 FMAs), 0 spills, accumulator-resident — the OpenBLAS A-reuse shape, bit-exact.
    #
    # HONEST WHOLE-MODEL SCOPE (the reason this is a SEPARATE feature, not folded into wholemodel_vf):
    # MR>1 needs M >= MR with a CLEAN M-tile. The openvla/rdt2 matmuls are ALL small-M (the token/
    # batch dim: openvla M in {16,17,20}, rdt2 M in {1,28}); they have NO large-M matmul. On those
    # shapes MR=4 either has no clean tile (M=17,1 not divisible by 4 -> the M-tail trips the LLVM-23
    # masked-transfer_write PipelineError -> NR=8/non-resident/118-spill, MEASURED) or silently
    # scalar-falls-back (M=16,28 -> 0 vfmacc emitted, MEASURED). So MR>1 is NOT whole-model-safe for
    # the VLA decode/prefill matmuls and would REGRESS them — it is correct + a genuine A-reuse win
    # only on large-M GEMM (M>=MR, M%MR==0). It is therefore DEFAULT-OFF and intended for a large-M
    # workload; for the small-M openvla/rdt2 whole-model the safe config remains
    # `accumulator_resident_wholemodel_vf` (MR=1, already at XNNPACK's per-FMA traffic floor). The
    # residual A-reuse the VLAs leave on the table is a STRUCTURAL property of their small token dim,
    # not a matmul-kernel defect — closing it would need a dispatch-level layout/batching pass (group
    # multiple small-M matmuls into one large-M GEMM), out of scope for the matmul-kernel feature.
    register(ImprFeature(
        name="accumulator_resident_wholemodel_vf_mr4",
        action_class="PASS",
        description="MR=4 register-block variant of accumulator_resident_wholemodel_vf for A-operand "
                    "REUSE (the OpenBLAS MR>1 lever): matmul MR_mm=4 so ONE unit-stride B-row load is "
                    "shared across 4 vfmacc.vf into 4 resident accumulators, dropping K-loop "
                    "loads/useful-FMA from 2.0 (MR=1) to 1.25 (1 B-load + 4 A-scalars / 4 FMAs) — "
                    "MEASURED by the memory-traffic decode on large-M cube/M=20. batch_matmul NR_bmm=8 "
                    "N-tail clamp retained. CORRECT + bit-exact + A-reuse ONLY on large-M GEMM "
                    "(M>=MR and M%MR==0); on the small-M openvla/rdt2 matmuls (token dim 1-28) it has "
                    "no clean M-tile (M=17,1) -> LLVM-23 masked-write PipelineError, or scalar-falls-"
                    "back (M=16,28) -> would regress the whole model, so it is NOT whole-model-safe "
                    "for VLAs (use wholemodel_vf, already at XNNPACK's per-FMA traffic floor, there). "
                    "The openvla/rdt2 A-reuse residual is structural (small token dim), not a "
                    "matmul-kernel defect; closing it needs a dispatch-level large-M batching/layout "
                    "pass (output/kernels/ceiling/packing_residual.md). Default-off; baseline "
                    "byte-identical.",
        edit_pipeline=_accumulator_resident_v3_pipeline,
        edit_schedule=lambda _t: _accumulator_resident_v3_pre_schedule(4, 16, 16,
                                                                       NR_bmm=8, MR_mm=4),
        schedule_replace=True,
        implies=_tile_epilogue_hygiene(4),
    ))
    names.append("accumulator_resident_wholemodel_vf_mr4")

    # PER-MATMUL MR + M-PAD TAIL — the whole-model-safe MR>1 register block (this iteration).
    # `..._vf_mr4` proved the A-reuse register block (MR=4 -> loads/FMA 2.0->1.25) but is correct ONLY
    # where M%MR==0: on the small-/odd-M matmuls that dominate VLA decode (rdt2 M=1, openvla M=17) its
    # bare MR=4 M-tile trips the LLVM-23 masked-transfer_write PipelineError -> NR=8/non-resident/119
    # spills (M=17) or an outright scalar fallback (M=1) — MEASURED via the decoder. So it is NOT
    # whole-model-safe and would REGRESS the mixed-M models. This feature makes MR>1 whole-model-safe by
    # handling the M-tail the way the hand shim does (`round_up_mr`): PAD each matmul's M up to the next
    # multiple of MR before tiling, so EVERY matmul register-blocks cleanly at MR regardless of M (M=1
    # pads to 4, M=17 to 20, M=64 stays 64). Bit-exact — the padded rows are 0-row@B=0 and are sliced
    # off (transform.structured.pad's extract_slice copy_back), only the real [0:M] rows are written.
    # Per-matmul (each op pads to its own next multiple of MR — a general tail rule), so a model that
    # MIXES M%4==0 and M=1 matmuls (rdt2) gets the MR register block on ALL of them in ONE schedule.
    # The batch_matmul path is identical to `..._vf` (MR=4 + NR_bmm=8), so only the matmul path changes
    # from the MR=1 clamp to the padded MR register block. Default-off; baseline byte-identical.
    register(ImprFeature(
        name="accumulator_resident_wholemodel_vf_mrpad",
        action_class="PASS",
        description="Per-matmul MR>1 register block with an M-PAD tail — the whole-model-safe A-operand "
                    "reuse lever. Same v3 vfmacc.vf subset-hoist + A-scalarization recipe as "
                    "accumulator_resident_wholemodel_vf, but the matmul M tile is a padded MR=4 register "
                    "block instead of the MR=1 clamp: transform.structured.pad rounds each matmul's M up "
                    "to a multiple of MR (padding value 0) BEFORE tiling, so every matmul — INCLUDING "
                    "the M=1/M=17/M=28 VLA-decode matmuls that make bare vf_mr4 trip the LLVM-23 "
                    "masked-transfer_write PipelineError (M=1 scalar fallback, M=17 NR=8/119-spill) — "
                    "register-blocks cleanly at MR: ONE unit-stride B-row load shared across MR "
                    "vfmacc.vf into MR resident accumulators (loads/useful-FMA 2.0 -> 1.25). Bit-exact "
                    "(padded rows are 0-row@B=0, sliced off by pad's copy_back extract_slice; only real "
                    "[0:M] rows written). Per-matmul (each op pads to its own next MR multiple — general "
                    "tail rule, not a per-model constant), so a MIXED-M model (rdt2 M in {1,28}) gets "
                    "the MR block on every matmul in ONE schedule. batch_matmul path identical to _vf "
                    "(MR=4 + NR_bmm=8). Default-off; baseline byte-identical.",
        edit_pipeline=_accumulator_resident_v3_mrpad_pipeline,
        implies=_tile_epilogue_hygiene(4),
        edit_schedule=lambda _t: _accumulator_resident_v3_mrpad_pre_schedule(4, 16, 16, NR_bmm=8),
        schedule_replace=True,
    ))
    names.append("accumulator_resident_wholemodel_vf_mrpad")
    return names


ACCUM_RESIDENT_V3_NAMES: list[str] = _register_accumulator_resident_v3()


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


# NOTE: the feature no longer edits the pass list. The earlier `_act_poly_math_before_libm` inserted
# `convert-math-to-llvm` BEFORE `convert-math-to-libm` because the OLD poly emitted vector
# `math.fma`/`math.absf`/`math.roundeven` that libm would scalarize. But that same pass ALSO converted
# the (correctly un-rewritten) softmax `math.exp` to `llvm.intr.exp` (`llvm.exp.f32`), which the
# freestanding spike/RVV runtime cannot legalize -> a wild instruction -> the openvla whole-model
# "bad syscall" CRASH (run produced no OUT/DONE -> certify status=fail). The activation poly is now
# PURE ARITH (act_poly._ap_fma = arith mul+add, _ap_absf = max(x,-x), _ap_roundeven = add/sub-magic),
# so it carries NO math.* op; the only remaining transcendental is the softmax exp, which takes the
# baseline `convert-math-to-libm` -> scalar `expf` path (exact, crash-free). No pipeline edit needed.


register(ImprFeature(
    name="vectorized_transcendental_activation",
    action_class="PASS",
    description="GENERAL vectorized-activation lowering, PRECISELY TARGETED by provenance: the "
                "act_poly rewriter (spliced into the lowering runner before the pass manager) "
                "replaces math.erf/exp/tanh with an inline minimax ARITH polynomial ONLY inside a "
                "linalg.generic the provenance marks as an elementwise ACTIVATION (gelu/silu/sigmoid/"
                "tanh) — NOT a softmax/normalization (whose exp stays on the exact libm path; "
                "blanket-rewriting it drove openvla whole-model cos to 0.541). It TAGS each targeted "
                "generic (merlin.act_vectorize) and the schedule vectorizes exactly those (no blanket "
                "foreach over every generic, no failures(suppress) -> no masked miscompile, no "
                "6+min/config compile blowup). So GELU (erf) and sigmoid/SiLU (exp) vectorize to a "
                "vector fmul/fadd (vfmacc) chain while softmax stays correct. Closes the activation "
                "gap vs XNNPACK's polynomial RVV kernels (coefficient/structure CEILING REFERENCE; we "
                "emit the MLIR). APPROXIMATION: cos>0.999 / max-abs-err <~6e-7 vs libm on REALISTIC "
                "ranges (gated on cos/rel error, not bit-exact). Default-off; baseline byte-identical.",
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


# Erase the per-tile `memref.copy %x, %x` bufferization leaves behind. Pure lowering hygiene: it
# changes no schedule, only removes a runtime call that moves a buffer onto itself. Default-off so
# the frozen hand_v0 control stays byte-identical; the beam turns it on to close an
# `envelope.runtime_calls` divergence (see kernels/action_catalog.py).
#
# MEASURED, K1 f32 GEMM 128^3, kernel region, bit-exact:
#   instructions 1,710,650 -> 475,899 (3.59x)   ticks 41,195 -> 21,882 (1.88x)
#   vs XNNPACK   3.57x -> 1.90x
# The mechanism and why nothing upstream folds it: see llvmlower/selfcopy.py.
register(ImprFeature(
    name=_SELF_COPY_FEATURE,
    action_class="PASS",
    description="erase `memref.copy %x, %x` (a buffer copied onto itself) after bufferization/cse "
                "and before finalize-memref-to-llvm, where it would otherwise survive as an opaque "
                "@memrefCopy rank-generic runtime call costing ~79 retired instructions per OUTPUT "
                "ELEMENT. Removes the tile-epilogue copy; emits no new code.",
))


# Fold a `linalg.transpose` of a matmul's B operand INTO the matmul's access pattern (transpose-b
# GEMM). WHOLE-MODEL, cross-op: the whole-model profiler measured `linalg.transpose` at 393 ms = 57%
# of openvla -- more than every matmul combined -- and SCALAR (convert-linalg-to-loops). Every
# openvla matmul is a transposed-B addmm fed by a standalone weight transpose. Folding it eliminates
# the scalar transpose op AND its materialized buffer; the op stays `linalg.matmul` (transposed-B
# indexing_map) so the frozen RVV schedule still vectorizes it. Default-off; baseline byte-identical.
# The rewrite runs in the lowering runner (gated by argv[5]); see llvmlower/transpose_fuse.py.
register(ImprFeature(
    name=_FUSE_TRANSPOSE_FEATURE,
    action_class="PASS",
    description="fuse `matmul(A, transpose(B, [1,0]))` into a transpose-b `linalg.matmul` (repoint "
                "the B operand to the un-transposed weight + permute its indexing_map (k,n)->(n,k), "
                "then erase the dead transpose). Eliminates the standalone SCALAR weight transpose "
                "(393 ms / 57% of openvla) and its DRAM buffer with no op materialized; the matmul "
                "stays vectorized by the frozen schedule and reads B contiguously along k. "
                "Whole-model cross-op fusion; default-off, baseline byte-identical.",
))


# ---- vectorize a standalone reduction -> vfredusum/vredsum (the compute.reduction_form lever) ------
# The baseline RVV schedule vectorizes ONLY the contraction ops (linalg.matmul / batch_matmul); a
# standalone reduction (softmax max/sum, LayerNorm/RMSNorm mean/var, a `linalg.reduce`) is left as a
# `linalg.generic` with a `reduction` iterator and falls through `convert-linalg-to-loops` to a SCALAR
# accumulate loop. That is the 2nd-biggest byte-traffic op family (softmax ~3.85% of the census) going
# unvectorized -- and the `compute.reduction_form` CCA lever had NO route (a bijection orphan), so the
# beam could not act on it. This feature is that route.
#
# It vectorizes the reduction generic (matched by its `reduction` iterator, at the ranks that occur in
# practice -- pure reduce, softmax/norm row-reduce, batched 3-D) and lets the reduction lower to a
# HARDWARE vector reduction. Two pipeline knobs make that land as `vfredusum`/`vredsum` rather than a
# scalar tail or a lane-parallel add tree:
#   * `lower-vector-multi-reduction{lowering-strategy=inner-reduction}` lowers the
#     `vector.multi_reduction` (that `transform.structured.vectorize` produces) to a `vector.reduction`
#     (the single-instruction horizontal reduce) instead of the default inner-PARALLEL add tree.
#   * `convert-vector-to-llvm{reassociate-fp-reductions}` sets `reassoc` on the emitted
#     `llvm.intr.vector.reduce.fadd`, so the RISC-V backend selects the UNORDERED `vfredusum.vs`
#     (detectable as a real reduction by cca.lift_asm) instead of the ordered `vfredosum.vs`. Integer
#     reductions emit `vredsum.vs` directly (integer add is associative).
#
# The contraction blocks of the baseline schedule are LEFT INTACT (this edit only INSERTS the reduction
# match+vectorize before the func-level lowering-pattern block), so enabling the feature on a whole
# model does NOT regress the matmuls to scalar -- MEASURED: the emitted matmul object is byte-identical
# with and without this feature (65 vec ops, 4 vfmul, 4 vfadd, 0 vfmacc either way). Empty matches are
# harmless (vectorize on an empty handle is a no-op), so a reduction rank the model does not contain
# just does nothing.
#
# PROVEN on the EMITTED CODE (lower a gen_reduce_f32 / gen_softmax_f32 workload with the real
# RVV_CFLAGS `-fno-vectorize -fno-slp-vectorize`, so clang's own auto-vectorizer is OFF and every
# vector reduction is MLIR-emitted, then decode the objdump):
#   reduce_f32 64x256 :  vfredusum.vs x64 (baseline: 0 vector ops)   cca.reduction_form=vredsum_tree
#   softmax_f32 64x256:  vfredmax.vs x64 + vfredusum.vs x64          cca.reduction_form=vredsum_tree
# APPROXIMATION (not bit-exact): `reassociate-fp-reductions` reorders the fp sum, the standard tree /
# unordered reduction XNNPACK et al. also use -- gated on cos/rel error, never claimed bit-exact.
# Default-off; empty features => pipeline + schedule byte-identical (guarded by test_impr_features).

# The reduction match+vectorize block, inserted before the baseline schedule's func.func lowering-
# pattern block. One match per reduction rank we support (pure-reduction, [parallel, reduction],
# [parallel, parallel, reduction]); an arity absent from the module matches nothing and is skipped.
_VECTORIZE_REDUCTION_BLOCK = """\
    %redr1 = transform.structured.match ops{["linalg.generic"]} attributes{iterator_types = [#linalg.iterator_type<reduction>]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %redr1 : !transform.any_op
    %redr2 = transform.structured.match ops{["linalg.generic"]} attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %redr2 : !transform.any_op
    %redr3 = transform.structured.match ops{["linalg.generic"]} attributes{iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %redr3 : !transform.any_op
"""

# Anchor: the baseline schedule's func-level lowering-pattern match. Insert the reduction block just
# before it so the lowering patterns (lower_masked_transfers/transpose/shape_cast) clean up afterward.
_REDUCTION_ANCHOR = '    %f = transform.structured.match ops{["func.func"]}'

# A standalone reduction-only schedule, used when the input schedule has no func.func anchor to build
# on (so the feature is never a silent no-op — it always emits the reduction vectorization).
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
        return text.replace(_REDUCTION_ANCHOR,
                            _VECTORIZE_REDUCTION_BLOCK + _REDUCTION_ANCHOR, 1)
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


register(ImprFeature(
    name="vectorize_reduction",
    action_class="PASS",
    description="Vectorize a standalone reduction (softmax/norm row-reduce, `linalg.reduce`) so it "
                "lowers to a HARDWARE vector reduction (`vfredusum.vs` for fp / `vredsum.vs` for int) "
                "instead of the scalar convert-linalg-to-loops accumulate the baseline emits. Matches "
                "the reduction `linalg.generic` by its reduction iterator (ranks 1/2/3), vectorizes it "
                "to `vector.multi_reduction`, lowers that via the inner-reduction strategy to "
                "`vector.reduction`, and reassociates the fp reduce so the backend picks the unordered "
                "vfredusum. The contraction schedule is left intact (matmul object byte-identical), so "
                "it is whole-model-safe. The route for the compute.reduction_form CCA lever (previously "
                "a bijection orphan). APPROXIMATION (fp reassociation, cos-gated). Default-off; baseline "
                "byte-identical.",
    edit_pipeline=vectorize_reduction_pipeline,
    edit_schedule=vectorize_reduction_schedule,
    schedule_replace=True,
))


# ---- matrix-unit routing ------------------------------------------------------------
# Unlike every feature above, this one changes NEITHER the pipeline pass list NOR the transform
# schedule. It is an IR-level rewrite: `passes_opu.rewrite_prepared_file` replaces selected int8
# contractions with calls to the certified matrix microkernel, so those contractions leave the vector
# path entirely and the schedule's `linalg.matmul` arms simply match fewer ops.
#
# It is registered here anyway, with both hooks None, because this registry is the one place a build
# says which non-baseline compiler behaviour it wants -- `build_app` threads `compiler_features` through
# to the prepare step, and a feature that lived outside the registry would need a second, parallel way
# to be requested. Both hooks being None also means the byte-identity invariant is structural rather
# than merely tested: there is no edit to apply.
OPU_MATMUL_NAME = "opu_matmul"

register(ImprFeature(
    name=OPU_MATMUL_NAME,
    action_class="PASS",
    description=(
        "Route int8 rank-2 contractions with a zero accumulator init to the certified outer-product "
        "matrix microkernel, as calls to a generated translation unit that transposes the left operand "
        "K-major and reads its extents from the memref descriptors. NOT a schedule or pipeline edit: "
        "the rewrite happens on the prepared IR (llvmlower/passes_opu), which is why both hooks are "
        "None. Selection is a separate decision -- which contractions move is answered by the cost "
        "model / e-graph and passed in, so enabling this feature without a selector routes nothing. "
        "Coverage on spectformer int8: 90 of 106 contractions are legal (the 16 batch_matmuls are "
        "gapped by a matmul-only contract), and a tile-filling selector at edge 32 moves 41 of them, "
        # target-ok: names the hardware_pins.yaml entry this feature requires at build time — a pin
        # reference in prose, not a target this code routes on (selection is passed in, see above).
        "which is the shapes carrying ~88% of the arithmetic. Requires the pinned saturn revision "
        "carrying the unit (hardware_pins.yaml: saturn_opu_int8) at build time, because the "
        "instruction encodings are derived from its RTL rather than written down. Default-off."),
))
