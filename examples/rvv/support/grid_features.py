"""RVV register tile, operand packing, and LMUL feature families."""

from __future__ import annotations

from merlin.llvmlower.feature_registry import FeatureRegistry, ImprFeature

from . import lmul_group
from .active_features import fused_vfmacc_tiled_schedule

LMUL_GROUP_PREFIX = "lmul_group_m"
LMUL_GROUP_SENTINEL = "lmul_register_group"


def lmul_group_feature(lmul: int) -> str:
    """The name of the default-off feature that pins the auto-vectorizer's register group to ``lmul``.

    Every whole-register width is registered eagerly (there are four), so a name resolves in the
    lowering SUBPROCESS as well as in the parent -- the same reproducibility requirement
    :func:`_try_lazy_register` exists for.
    """
    from .lmul_group import LMUL_LADDER

    if int(lmul) not in LMUL_LADDER:
        from .lmul_group import LmulDerivationError

        raise LmulDerivationError(f"LMUL={lmul!r} is not one of {LMUL_LADDER}")
    return f"{LMUL_GROUP_PREFIX}{int(lmul)}"


def ensure_lmul_group(
    *,
    operand_bits: int,
    acc_bits: int,
    vlen: int | None = None,
    max_group_elems: int | None = None,
) -> str:
    """DERIVE the register-group width from the datapath and return the feature that pins it.

    The width is ``lmul_group.group_lmul`` -- the smallest whole-register group at which the
    narrowest operand's group stops being a fraction, capped by what the target's VLEN can usefully
    hold. Nobody types a 4: an ``i8 x i8 -> i32`` contraction derives 4 (which is what the expert
    XNNPACK qd8 kernel runs at), an ``f32`` one derives 1, a ``bf16 -> f32`` one derives 2.
    """
    from .lmul_group import group_lmul

    return lmul_group_feature(
        group_lmul(
            operand_bits=operand_bits,
            acc_bits=acc_bits,
            vlen=vlen,
            max_group_elems=max_group_elems,
        )
    )


def ensure_lmul_group_for_elem_types(
    a: str,
    b: str,
    c: str,
    *,
    vlen: int | None = None,
    max_group_elems: int | None = None,
) -> str:
    """:func:`ensure_lmul_group` for a contraction named by its MLIR element types ``a x b -> c``."""
    from .lmul_group import group_lmul_for_elem_types

    return lmul_group_feature(
        group_lmul_for_elem_types(a, b, c, vlen=vlen, max_group_elems=max_group_elems)
    )


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
        i = out.index(
            "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}"
        )
    except ValueError:
        return out
    if out[i - 1] == "eliminate-empty-tensors":
        return out
    out.insert(i, "eliminate-empty-tensors")
    return out


def register_grid_features(registry: FeatureRegistry) -> None:
    for lmul in lmul_group.LMUL_LADDER:
        registry.register(
            ImprFeature(
                name=f"{LMUL_GROUP_PREFIX}{lmul}",
                action_class="KNOB",
                description=f"Pin RVV auto-vectorizer register-group width to LMUL={lmul}.",
                edit_cflags=lambda flags, _lmul=lmul: [
                    *flags,
                    *lmul_group.lmul_cflags(_lmul),
                ],
            )
        )
    registry.register(
        ImprFeature(
            name="fused_vfmacc_scalable",
            action_class="PASS",
            description="Alias of the bounded fused_vfmacc_tiled schedule.",
            edit_schedule=lambda _text: fused_vfmacc_tiled_schedule(),
            schedule_replace=True,
        )
    )
    for mr in (4, 8):
        for nr in (16, 32):
            for kc in (16, 32, 64):
                registry.register(
                    ImprFeature(
                        name=f"vfmacc_t_{mr}_{nr}_{kc}",
                        action_class="PASS",
                        description=f"RVV tiled vfmacc register tile {mr}x{nr}x{kc}.",
                        edit_schedule=lambda _text, _mr=mr, _nr=nr, _kc=kc: (
                            vfmacc_tiled_schedule(_mr, _nr, _kc)
                        ),
                        schedule_replace=True,
                    )
                )
    registry.register(
        ImprFeature(
            name="vfmacc_packed",
            action_class="PASS",
            description="RVV operand packing tiled vfmacc.",
            edit_pipeline=_packed_eliminate_empties,
            edit_schedule=lambda _text: vfmacc_packed_schedule(4, 16, 16),
            schedule_replace=True,
        )
    )
    for mr in (4, 8):
        for nr in (16, 32):
            for kc in (16, 32, 64):
                registry.register(
                    ImprFeature(
                        name=f"vfmacc_packed_{mr}_{nr}_{kc}",
                        action_class="PASS",
                        description=f"RVV packed vfmacc register tile {mr}x{nr}x{kc}.",
                        edit_pipeline=_packed_eliminate_empties,
                        edit_schedule=lambda _text, _mr=mr, _nr=nr, _kc=kc: (
                            vfmacc_packed_schedule(_mr, _nr, _kc)
                        ),
                        schedule_replace=True,
                    )
                )
