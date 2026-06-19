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
                ))
                names.append(nm)
    return names


PACKED_GRID_NAMES: list[str] = _register_packed_grid()


register(ImprFeature(
    name="fused_vfmacc_contraction",
    action_class="PASS",
    description="mined fma_broadcast_policy: form a real vector.contract -> outerproduct(kind=add) "
                "-> vector.fma -> llvm.fmuladd -> vfmacc (vectorize_children + lower_contraction "
                "outerproduct + lower_outerproduct). Closes the separate-vfmul.vv+vfadd.vv gap. For "
                "kernel-sized contraction workloads (vectorize_children explodes on whole models).",
    edit_schedule=_vfmacc_schedule_edit,
))
