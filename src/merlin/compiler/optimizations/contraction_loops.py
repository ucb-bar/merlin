"""Generic upstream contraction tiling and packed schedule construction."""

from __future__ import annotations

from merlin.compiler.optimizations.accumulator_residency import (
    WHOLEMODEL_VF_BMMPAD_NAME,
    WHOLEMODEL_VF_CAPS,
    WHOLEMODEL_VF_MR_MM,
    WHOLEMODEL_VF_NAME,
    WHOLEMODEL_VF_NR_BMM,
    _accumulator_resident_v3_pipeline,
    _tile_epilogue_hygiene,
)
from merlin.compiler.selection.features import ImprFeature

# ---- contraction -------------------------------------------------------------------------------

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
    from merlin.compiler.optimizations.registry import register

    names: list[str] = []
    for MR in (4, 8):
        for NR in (16, 32):
            for KC in (16, 32, 64):
                nm = f"vfmacc_t_{MR}_{NR}_{KC}"
                register(
                    ImprFeature(
                        name=nm,
                        action_class="PASS",
                        description=f"Tiled-vfmacc tuning point: register-tile (MR={MR}, NR={NR}, "
                        f"KC={KC}). Bounded-code, whole-model-safe tiled vfmacc with this "
                        f"tile (inner body = MR*KC fma). Default-off tuning-grid feature.",
                        edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC: vfmacc_tiled_schedule(_MR, _NR, _KC)),
                        schedule_replace=True,
                    )
                )
                names.append(nm)
    return names


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


def _register_packed_grid() -> list[str]:
    """Register default-off packed-vfmacc tuning points `vfmacc_packed_<MR>_<NR>_<KC>` (same grid
    as the tiled sweep) so the packing layout can be tuned alongside the register tile."""
    from merlin.compiler.optimizations.registry import register

    names: list[str] = []
    for MR in (4, 8):
        for NR in (16, 32):
            for KC in (16, 32, 64):
                nm = f"vfmacc_packed_{MR}_{NR}_{KC}"
                register(
                    ImprFeature(
                        name=nm,
                        action_class="PASS",
                        description=f"Operand-packing tiled-vfmacc tuning point: register-tile/pack-tile "
                        f"(MR={MR}, NR={NR}, KC={KC}). Contiguous packed inner transfers. "
                        f"Default-off tuning-grid feature.",
                        edit_pipeline=_packed_eliminate_empties,
                        edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC: vfmacc_packed_schedule(_MR, _NR, _KC)),
                        schedule_replace=True,
                    )
                )
                names.append(nm)
    return names


# ---- loops -------------------------------------------------------------------------------------
# Parameterized generic vector contraction schedules and loop variants.


def _accumulator_resident_v3_pre_schedule(
    MR: int,
    NR: int,
    KC: int,
    NR_bmm: int | None = None,
    MR_mm: int | None = None,
    skip_mm: bool = False,
    skip_bmm: bool = False,
    pad_bmm: bool = False,
) -> str:
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
    _ew = (
        """
    %g2 = transform.structured.match attributes{merlin.vec_r2} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %g2 vector_sizes [1, 8] : !transform.any_op
    %g3 = transform.structured.match attributes{merlin.vec_r3} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %g3 vector_sizes [1, 1, 8] : !transform.any_op
    %g4 = transform.structured.match attributes{merlin.vec_r4} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %g4 vector_sizes [1, 1, 1, 8] : !transform.any_op"""
        if __import__("os").environ.get("MERLIN_VEC_RANK")
        else (
            """
    %g = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %g : !transform.any_op"""
            if __import__("os").environ.get("MERLIN_VEC_EW")
            else ""
        )
    )
    _mm_arm = f"""
    %mm = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t1, %lmn:2 = transform.structured.tile_using_for %mm tile_sizes [{MM}, {NR}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %mm2 = transform.structured.match ops{{["linalg.matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %t2, %lk = transform.structured.tile_using_for %mm2 tile_sizes [0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %t2 vector_sizes [{MM}, {NR}, 1] : !transform.any_op"""
    if skip_mm:
        _mm_arm = ""
    _bmm_arm = f"""
    %bm = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt1, %blmn:3 = transform.structured.tile_using_for %bm tile_sizes [1, {MR}, {NB}, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %bm2 = transform.structured.match ops{{["linalg.batch_matmul"]}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bt2, %blk = transform.structured.tile_using_for %bm2 tile_sizes [0, 0, 0, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.vectorize %bt2 vector_sizes [1, {MR}, {NB}, 1] : !transform.any_op"""
    if skip_bmm:
        _bmm_arm = ""
    _pad_sequence = (
        ""
        if not pad_bmm
        else f"""
  transform.named_sequence @__transform_bmm_pad(%arg0: !transform.any_op {{transform.readonly}}) {{
    %btail = transform.structured.match attributes{{merlin.bmm_pad_tail}} in %arg0 : (!transform.any_op) -> !transform.any_op
    %bpadded, %bpad, %bcp = transform.structured.pad %btail pad_to_multiple_of [{MR}, {NB}] {{padding_dimensions = [1, 2], copy_back_op = "linalg.copy"}} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }}
"""
    )
    return f"""\
module attributes {{transform.with_named_sequence}} {{
{_pad_sequence}
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


def _zero_attr(t: str) -> str:
    """The MLIR zero literal for element type ``t``, derived from the type rather than tabulated.

    Integer types take ``0 : iN``; floating types take the f-literal form. A tabulated map would go
    stale the first time a new element type appears, and the failure is silent (scalar fallback).
    """
    t = str(t).strip()
    if t.startswith("i") and t[1:].isdigit():
        return f"0 : {t}"
    if t.startswith(("f", "bf")):
        return f"0.000000e+00 : {t}"
    raise ValueError(
        f"no zero literal derivable for element type {t!r}; add the rule rather than "
        "letting the schedule pad with a wrongly-typed value"
    )


def _accumulator_resident_v3_mrpad_pre_schedule(
    MR: int, NR: int, KC: int, NR_bmm: int | None = None, elem_types: tuple[str, str, str] | None = None
) -> str:
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
    # The padding value must have the ELEMENT TYPE OF THE OPERAND IT PADS. This was a single f32
    # literal reused for all three, which is correct only while every contraction is f32 -- and
    # `transform.structured.pad` does not warn, it hard-errors:
    #   'transform.structured.pad' op expects a padding value of type 'i8', got 0.000000e+00 : f32
    # A transform-interpreter error is a PipelineError, which the lowering catches as a whole-model
    # SILENT SCALAR FALLBACK -- the failure mode is "mysteriously slow", not "build failed".
    # It has never fired because the int8 datapath rewrites every linalg.matmul into a
    # linalg.generic, so no i8 matmul reaches this schedule at all (see the module docstring note on
    # named-op erasure). It fires the moment that is fixed, which is why it is fixed first.
    za, zb, zc = (_zero_attr(t) for t in (elem_types or ("f32", "f32", "f32")))
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
    %padded, %pad, %cp = transform.structured.pad %mm pad_to_multiple_of [{MR}] {{padding_values = [{za}, {zb}, {zc}], padding_dimensions = [0], copy_back_op = "linalg.copy"}} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
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


def _accumulator_resident_v3_bmmpad_pipeline(passes: list[str]) -> list[str]:
    """Pad odd BMM extents before the parallel transform, then run v3 lowering.

    Padding inside an ``scf.forall`` leaves its temporary allocation inside an
    allocation scope that becomes multi-block during CFG lowering.  Running the
    dedicated pad entry before ``__transform_parallel_main`` keeps the allocation
    outside the worker region; the later main entry only tiles/vectorizes.
    """
    out = _accumulator_resident_v3_mrpad_pipeline(passes)
    anchor = next(i for i, p in enumerate(out) if "transform-interpreter{entry-point=__transform_main}" in p)
    parallel = next((i for i, p in enumerate(out) if "entry-point=__transform_parallel_main" in p), anchor)
    out.insert(parallel, "transform-interpreter{entry-point=__transform_bmm_pad}")
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


_VSCALE_LANES_PER_128B = 2


def scalable_lanes(NR: int) -> int:
    """The scalable multiplier ``k`` of ``vector<[k]xT>`` for a register block ``NR`` lanes wide at
    the RVV MINIMUM VLEN (128 bits). Dtype-independent — see :data:`_VSCALE_LANES_PER_128B`."""
    if NR % _VSCALE_LANES_PER_128B:
        raise ValueError(
            f"vl_strategy='dynamic' needs an even NR (NR counts lanes at the RVV minimum VLEN of "
            f"128 bits, and a scalable vector<[k]xT> holds 2k of them); got NR={NR}."
        )
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


# ---- variants ----------------------------------------------------------------------------------
# Existing composed whole-model contraction strategy declarations.


def _register_accumulator_resident_v3() -> list[str]:
    """Register `accumulator_resident_microkernel_v3` (default tile) + a small tuning grid. Reuses
    the v1 PRE-bufferize schedule (forms the contract); the v3 pipeline does the v2 PRE-bufferize
    subset hoist + contract lowering AND splices the SCALARIZE_MARKER so the runner scalarizes the A
    operand -> the K-loop emits the hand kernel's accumulator-resident vfmacc.vf structure."""
    from merlin.compiler.optimizations.registry import register

    names: list[str] = []
    grid = [(4, 16, 16), (4, 16, 32), (8, 16, 16), (4, 32, 16)]
    for MR, NR, KC in grid:
        if (MR, NR, KC) == (4, 16, 16):
            nm = "accumulator_resident_microkernel_v3"
            desc = (
                "COMPILER-EMITTED accumulator-resident, register-blocked, vfmacc.vf RVV GEMM "
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
                "(scalar load of A[i,0] == lane [i,0]). Default-off, baseline byte-identical."
            )
        else:
            nm = f"accum_resident_v3_{MR}_{NR}_{KC}"
            desc = (
                f"Accumulator-resident vfmacc.vf micro-kernel tuning point (MR={MR}, NR={NR}, "
                f"KC={KC}): resident vector<MRxNR> K-loop iter_arg accumulator + scalar-A "
                f"vfmacc.vf. Default-off tuning-grid feature."
            )
        register(
            ImprFeature(
                name=nm,
                action_class="PASS",
                description=desc,
                edit_pipeline=_accumulator_resident_v3_pipeline,
                edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC: _accumulator_resident_v3_pre_schedule(_MR, _NR, _KC)),
                schedule_replace=True,
                implies=_tile_epilogue_hygiene(MR),
            )
        )
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
    register(
        ImprFeature(
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
                *WHOLEMODEL_VF_CAPS, NR_bmm=WHOLEMODEL_VF_NR_BMM, MR_mm=WHOLEMODEL_VF_MR_MM
            ),
            schedule_replace=True,
        )
    )
    names.append(WHOLEMODEL_VF_NAME)

    # Odd attention extents need BOTH M and N tail handling.  smolVLA's language-side attention
    # uses 113x113 score/value contractions, so the ordinary MR=4/NR=8 schedule produces a single
    # multi-op vector.mask around each vector.contract.  LLVM 23 cannot lower that form.  Padding
    # the two PARALLEL iteration dimensions to their tile multiples leaves the 4x8 main kernel
    # unchanged, fills only zero rows/columns, and copies the original result slice back.  Omitting
    # padding_values is intentional: transform.structured.pad infers the additive zero from each
    # contraction's own element types, so one schedule safely handles i8*i8->i32 score BMMs and
    # bf16*bf16->f32 softmax/value BMMs in the same graph.
    register(
        ImprFeature(
            name=WHOLEMODEL_VF_BMMPAD_NAME,
            action_class="PASS",
            description="Whole-model vfmacc/vwmacc accumulator-resident kernel with batch-matmul M/N "
            "padding. Pads attention's parallel M and N extents to MR=4/NR=8, preserving "
            "the full 4x8 vector tile while eliminating unlowerable vector.mask-wrapped "
            "contracts on odd extents such as smolVLA's 113x113 attention. Zero padding is "
            "inferred per operand dtype and the original result slice is copied back. "
            "Default-off; the published wholemodel_vf feature is unchanged.",
            edit_pipeline=_accumulator_resident_v3_bmmpad_pipeline,
            edit_schedule=lambda _t: _accumulator_resident_v3_pre_schedule(
                *WHOLEMODEL_VF_CAPS, NR_bmm=WHOLEMODEL_VF_NR_BMM, MR_mm=WHOLEMODEL_VF_MR_MM, pad_bmm=True
            ),
            schedule_replace=True,
        )
    )
    names.append(WHOLEMODEL_VF_BMMPAD_NAME)

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
    register(
        ImprFeature(
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
            edit_schedule=lambda _t: _accumulator_resident_v3_pre_schedule(4, 16, 16, NR_bmm=8, MR_mm=4),
            schedule_replace=True,
            implies=_tile_epilogue_hygiene(4),
        )
    )
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
    register(
        ImprFeature(
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
        )
    )
    names.append("accumulator_resident_wholemodel_vf_mrpad")
    return names


MRPAD_NAME = "accumulator_resident_wholemodel_vf_mrpad"


def ensure_mrpad_for_elem_types(a: str, b: str, c: str, MR: int = 4, NR: int = 16, NR_bmm: int = 8) -> str:
    """Register (once) an M-pad register block whose padding values match the OPERAND TYPES.

    The default ``accumulator_resident_wholemodel_vf_mrpad`` pads with an f32 zero for all three
    operands, which is right only for an f32 contraction. On any other element type
    ``transform.structured.pad`` hard-errors, the transform interpreter raises, and the lowering
    catches that as a whole-model scalar fallback -- so the lever reads as "applied but slow" rather
    than "rejected". Naming the types makes the variant selectable by the search instead of being a
    property of whichever dtype the default happened to be written for.

    Returns the feature name to put in a feature set.
    """
    from merlin.compiler.optimizations.registry import known, register

    types = (str(a), str(b), str(c))
    MR, NR, NR_bmm = int(MR), int(NR), int(NR_bmm)
    if min(MR, NR, NR_bmm) <= 0:
        raise ValueError(f"MR/NR/NR_bmm must be positive, got {(MR, NR, NR_bmm)}")
    # The TILE is part of the identity. MR=4/NR=16 were chosen against f32; at one VLEN an i8 lane
    # holds four times the elements, so the f32 tile is not the int8 tile and a variant that hides
    # its shape behind the dtype name would silently pin the wrong one.
    name = f"{MRPAD_NAME}_" + "_".join(types) + f"_mr{MR}_nr{NR}_nb{NR_bmm}"
    if name in known():
        return name
    for t in types:  # fail closed at REGISTRATION, not inside the interpreter
        _zero_attr(t)
    register(
        ImprFeature(
            name=name,
            action_class="PASS",
            description=(
                f"{MRPAD_NAME} at MR={MR}/NR={NR}/NR_bmm={NR_bmm} with padding values typed "
                f"for a {types[0]}x{types[1]}->{types[2]} "
                "contraction. Same recipe and same tail rule; only the pad literals differ, "
                "because a wrongly-typed pad value is a transform-interpreter error and "
                "therefore a silent whole-model scalar fallback."
            ),
            edit_pipeline=_accumulator_resident_v3_mrpad_pipeline,
            implies=_tile_epilogue_hygiene(4),
            edit_schedule=(
                lambda _t, _ty=types, _mr=MR, _nr=NR, _nb=NR_bmm: _accumulator_resident_v3_mrpad_pre_schedule(
                    _mr, _nr, 16, NR_bmm=_nb, elem_types=_ty
                )
            ),
            schedule_replace=True,
        )
    )
    return name
