"""Existing composed whole-model contraction strategy declarations."""

from __future__ import annotations

from merlin.compiler.optimizations.hoisting import _accumulator_resident_v3_pipeline
from merlin.compiler.optimizations.loops import (
    _accumulator_resident_v3_bmmpad_pipeline,
    _accumulator_resident_v3_mrpad_pipeline,
    _accumulator_resident_v3_mrpad_pre_schedule,
    _accumulator_resident_v3_pre_schedule,
    _zero_attr,
)
from merlin.compiler.optimizations.resident import (
    WHOLEMODEL_VF_BMMPAD_NAME,
    WHOLEMODEL_VF_CAPS,
    WHOLEMODEL_VF_MR_MM,
    WHOLEMODEL_VF_NAME,
    WHOLEMODEL_VF_NR_BMM,
    _tile_epilogue_hygiene,
)
from merlin.compiler.selection.features import ImprFeature


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
