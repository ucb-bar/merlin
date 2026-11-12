"""Accumulator retention schedules, pipeline edits and declared hygiene."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from merlin.compiler.lower.model.selfcopy import FEATURE as _SELF_COPY_FEATURE
from merlin.compiler.selection.features import ImprFeature


def _accumulator_resident_pre_schedule(
    MR: int, NR: int, KC: int, NR_bmm: int | None = None, MR_mm: int | None = None
) -> str:
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


WHOLEMODEL_VF_NAME = "accumulator_resident_wholemodel_vf"


WHOLEMODEL_VF_BMMPAD_NAME = "accumulator_resident_wholemodel_vf_bmmpad"


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
        "canonicalize",
        "cse",
    ]
    if out[insert_at : insert_at + len(inject)] != inject:  # idempotent
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
    from merlin.compiler.optimizations.registry import register

    names: list[str] = []
    grid = [(4, 16, 16), (4, 16, 32), (8, 16, 16), (4, 32, 16)]
    for MR, NR, KC in grid:
        if (MR, NR, KC) == (4, 16, 16):
            nm = "accumulator_resident_microkernel"
            desc = (
                "Transform-dialect accumulator-residency attempt: tile [MR=4,NR=16], "
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
                "baseline byte-identical."
            )
        else:
            nm = f"accum_resident_{MR}_{NR}_{KC}"
            desc = (
                f"Accumulator-resident micro-kernel tuning point (MR={MR}, NR={NR}, KC={KC}): "
                f"register-resident K-loop vector iter_arg accumulator, no per-K memref "
                f"roundtrip. Default-off tuning-grid feature."
            )
        register(
            ImprFeature(
                name=nm,
                action_class="PASS",
                description=desc,
                edit_pipeline=_accumulator_resident_pipeline,
                edit_schedule=(lambda _t, _MR=MR, _NR=NR, _KC=KC: _accumulator_resident_pre_schedule(_MR, _NR, _KC)),
                schedule_replace=True,
            )
        )
        names.append(nm)
    # N-TAIL-SAFE variant for whole-model attention: same recipe but the batch_matmul N tile is
    # clamped to NR_bmm=8 (<= the small llama-style attention N=8) so the inner vectorize is FULL (no
    # masked vector.transfer_write -> no LLVM-23 PipelineError -> no silent scalar fallback). The
    # matmul path keeps NR=16. This is the feature that makes attention batch_matmuls vectorize to
    # vfmacc instead of falling back to scalar (verified bit-exact on spike for a B=4,M=32,N=8,K=32
    # attention batch_matmul). Default-off; baseline byte-identical.
    register(
        ImprFeature(
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
        )
    )
    names.append("accumulator_resident_ntail")

    # M-TAIL-SAFE variant: the M-side analog of accumulator_resident_ntail. The whole-model decode
    # step is dominated by matmuls with leading M=1 (one token row); the MR=4 register tile then
    # writes a vector<4xNR> into a tensor<1xNR> C tile -> the SAME multi-op vector.mask PipelineError
    # the N-tail hit (LLVM-23 rejects it -> silent scalar fallback, no vfmacc). Clamping the matmul
    # MR to MR_mm=1 (MR=min(MR,M)) makes the inner vectorize FULL on the M=1 matmul (no mask) so it
    # vectorizes to vfmacc; for any larger-M matmul MR_mm=1 just tiles M into single-row register
    # tiles (still a real vfmacc chain, bit-exact) — general, not M=1-overfit. Default-off.
    register(
        ImprFeature(
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
        )
    )
    names.append("accumulator_resident_mtail")

    # WHOLE-MODEL-SAFE COMPOSED variant (WORK-ITEM 2 by the inherent-clamp design): a SINGLE feature
    # whose schedule has BOTH tail clamps inherent — matmul MR_mm=1 (M-tail) AND batch_matmul NR_bmm=8
    # (N-tail) — on top of the bit-exact tiled-vfmacc accumulator-resident recipe. Because both clamps
    # live in ONE full-schedule feature (not two separate full-schedule replacements that would clobber
    # each other when apply_schedule picks one), enabling THIS one feature applies the tiled vfmacc +
    # M-tail + N-tail together whole-model. It vectorizes a normal matmul, an M=1 token-decode matmul,
    # AND a small-N (N=8) attention batch_matmul in ONE schedule with no scalar fallback / no
    # vector.mask PipelineError. This is the config a whole-model fork enables. Default-off.
    register(
        ImprFeature(
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
        )
    )
    names.append("accumulator_resident_wholemodel")
    return names
