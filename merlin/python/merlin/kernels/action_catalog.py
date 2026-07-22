"""Typed compiler-action catalog — the actionable "what can we change in the compiler" output.

Each CCA ``Divergence`` (from ``cca_compare``) routes to a typed ``CompilerAction`` tagged with a
**class** so we know structurally what to change:
- ``FLAG``      — a cflag / pass option (e.g. ``-ffp-contract=fast``, march features).
- ``HEURISTIC`` — a selection rule in a pass (tile size, LMUL choice, fuse-or-not).
- ``PASS``      — a new/modified MLIR pass or lowering pattern (an ``impr_features`` hook).
- ``KNOB``      — a transform-schedule parameter (forkable today via ``schedule.mlir``).

``target_seam`` names the concrete place to make the change; ``forkable_now`` says whether an
``impr_`` fork can express it today (schedule knob / cflag / a registered ``impr_features`` hook)
or it is a deferred work-item. Supersedes the knob-only ``rvv_knobs`` gap-router. Routes are keyed
by ``(backend, axis)`` so non-RVV targets add their own rows without disturbing RVV.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .cca_compare import Divergence


@dataclass
class CompilerAction:
    divergence_axis: str
    action_class: str         # FLAG | KNOB | HEURISTIC | PASS | CODEGEN (the escalation ladder)
    target_seam: str          # concrete place: "impr_features:<name>" | "schedule:<knob>" | "cflag:<f>" | "pass:<name>"
    change: str               # human-readable description of the change
    forkable_now: bool
    expected_effect: str
    backend: str
    evidence: list[str] = field(default_factory=list)
    # MACHINE-READABLE promise: the CCA facet(s) this action claims to make ours achieve, so the loop
    # can CHECK whether a fork that applied it actually delivered (intended-vs-achieved), e.g.
    # {"compute.accumulator_resident": True} or {"compute.register_block": 7} (MR>= semantics). Used by
    # `achieved_residual` + `route_escalated`. None = no machine-checkable promise (prose-only).
    intended_facet: dict[str, Any] | None = None
    # The shape REGIMES (bench_ceiling.shape_regime vocabulary: square_large/skinny/...) this action is
    # most beneficial for — so a big matmul and a small-M decode matmul get DIFFERENT optimizations.
    # Empty = shape-agnostic (applies to all regimes). Generalizes on regimes, never exact (M,N,K).
    shape_regimes: tuple[str, ...] = ()


# A route: predicate over a Divergence -> a CompilerAction template (filled with evidence/backend).
@dataclass
class _Route:
    axis: str
    when: Callable[[Divergence], bool]
    action_class: str
    target_seam: str
    change: str
    forkable_now: bool
    expected_effect: str
    intended_facet: dict[str, Any] | None = None
    shape_regimes: tuple[str, ...] = ()   # regimes this route targets (empty = all); see CompilerAction


def _is_higher(d: Divergence) -> bool:
    try:
        return float(d.expert) > float(d.ours)
    except (TypeError, ValueError):
        return False


# RVV routes. Keyed implicitly by backend "rvv" (see route()). Each maps a mined divergence to the
# concrete compiler lever — and whether an impr_ fork can do it today.
_RVV_ROUTES: list[_Route] = [
    _Route(
        axis="compute.contraction_form",
        when=lambda d: d.expert == "fused_fma" and d.ours in ("mul_add", None),
        action_class="PASS", target_seam="impr_features:fused_vfmacc_contraction",
        change="form a real vector.contract -> outerproduct(kind=add) -> vector.fma -> "
               "llvm.fmuladd -> vfmacc (vectorize_children + lower_contraction outerproduct + "
               "lower_outerproduct), instead of separate vfmul.vv+vfadd.vv",
        # EVIDENCE-DRIVEN, RESOLVED: 4 knob/flag attempts (outerproduct-only, K=4 tile, +fp-contract,
        # +ffast-math) ALL measured vfmacc=0 -> demoted to deferred PASS. The PASS was then
        # implemented (impr_features.fused_vfmacc_contraction) and MEASURED: impr_rvv_v5 certifies
        # correct on spike AND decodes to vfmacc=8065, vfmul=0, vfadd=0 -> gap CLOSED. Re-promoted to
        # forkable_now=True (a registered, certified feature now expresses it).
        forkable_now=True,
        expected_effect="vfmacc replaces vfmul+vfadd; MEASURED 7.9x faster on K1 silicon (64^3 f32 "
                        "matmul, N=5, cos=1.0) vs the frozen baseline",
        intended_facet={"compute.contraction_form": "fused_fma"}),
    _Route(
        axis="compute.accumulator_resident",
        when=lambda d: bool(d.expert) and d.ours is False,
        action_class="PASS",
        target_seam="impr_features:accumulator_resident_microkernel",
        change="carry the MR x NR C-accumulator across the K reduction in scf.for vector iter_args "
               "(value semantics — stays in the vector register file, never bufferizes to memory "
               "per K-tile) and commit C once after the loop, instead of the per-K-tile "
               "transfer_read/transfer_write+memref.copy accumulator roundtrip the tile->vectorize-> "
               "bufferize lowering emits. The general accumulator-residency capability behind the "
               "experts' MR vfmacc.vf accumulator-vreg block.",
        # HONEST forkable status: the transform-dialect feature (accumulator_resident_microkernel)
        # forms vfmacc and is bit-exact, but the EMITTED asm still spills the carried accumulator
        # through the stack inside the K loop (whole-register vsNr/vlNre measured in-loop), so the
        # CCA reads it back as accumulator_resident=False — the transform path does NOT yet fully
        # close the gap. The genuine register-resident closer is the dedicated micro-kernel codegen
        # (intrinsic_microkernel, CODEGEN action below). Recorded as a deferred codegen work-item.
        forkable_now=False,
        expected_effect="accumulator never touches memory across the reduction; removes the "
                        "accumulator/result memref.copy traffic that is the measured ~15.7x "
                        "scalable gap (compute kernel alone is already ~1.5x OpenBLAS)",
        intended_facet={"compute.accumulator_resident": True}),
    _Route(
        axis="compute.accumulator_resident",
        # CODEGEN closer: when no impr PASS expresses residency, route to the dedicated RVV
        # micro-kernel emitter that the intrinsic_microkernel marker + driver demonstrate (the only
        # path measured to actually keep the accumulator register-resident across K).
        when=lambda d: bool(d.expert) and d.ours is None,
        action_class="CODEGEN",
        target_seam="pass:rvv-microkernel-emitter (intrinsic_microkernel ceiling)",
        change="lower the inner MR x NR x K block to a register-blocked, accumulator-resident, "
               "K-streaming RVV micro-kernel (MR accumulator vreg-groups held across K, A scalars + "
               "B row streamed via vfmacc.vf, C stored once) — what a dedicated micro-kernel codegen "
               "pass would emit; demonstrated bit-exact + spill-free by intrinsic_microkernel.",
        forkable_now=False,
        expected_effect="register-resident accumulator with zero per-K memref roundtrip; measured "
                        "1.7x faster than OpenBLAS pack-excluded on the spike proxy (hand ceiling)",
        intended_facet={"compute.accumulator_resident": True}),
    # --- envelope: the code AROUND the loop. Added after a K1 measurement showed the entire f32 GEMM
    # gap lived here and NOWHERE in the tiling space: our hot loop is better per-FMA than XNNPACK's
    # (3.0 vs 3.82 ins/FMA) while a per-tile memrefCopy cost ~79 instructions per OUTPUT ELEMENT,
    # ~77% of everything retired at N=128. A full MR/NR/KC/unroll_m sweep could not move it, because
    # no tiling choice removes a call the epilogue emits regardless.
    _Route(
        axis="envelope.runtime_calls",
        # Fire when the expert calls no runtime helper and we call at least one: a codegen ESCAPE.
        when=lambda d: not d.expert and bool(d.ours),
        action_class="PASS",
        target_seam="impr_features:erase_self_copy",
        change="erase the `memref.copy %x, %x` bufferization leaves in the tile epilogue. The tile "
               "result is already in place from the vector.transfer_write; cse collapses the two "
               "identical destination subviews into one SSA value, and the resulting self-copy is a "
               "no-op that nothing upstream folds, so it survives as an opaque @memrefCopy call.",
        forkable_now=True,
        expected_effect="MEASURED on K1 (f32 GEMM 128^3, bit-exact): retired instructions "
                        "1,710,650 -> 475,899 (3.59x) and ticks 41,195 -> 21,882 (1.88x), moving "
                        "us from 3.57x to 1.90x of XNNPACK",
        intended_facet={"envelope.runtime_calls": ()}),
    _Route(
        axis="envelope.calls_in_loop",
        # The symbol-free form of the same divergence: a call in a loop body is per-iteration
        # overhead whatever it calls, so this fires even without an object symbol table.
        when=lambda d: bool(d.ours) and not d.expert,
        action_class="HEURISTIC",
        target_seam="schedule:hoist-loop-invariant-call / enlarge tile (amortize the call)",
        change="hoist a loop-invariant call out of the tile loop, or enlarge the tile so the "
               "per-tile cost is amortized over more outputs — the cheap mitigation to try before "
               "the PASS that removes the call outright.",
        forkable_now=False,
        expected_effect="reduces, but does not eliminate, per-tile overhead. NOT forkable-now: no "
                        "builder implements this seam, so the proposer honestly demotes it to a "
                        "work-item. The envelope.runtime_calls PASS is the route that actually "
                        "closes the gap; this rung stays as the symbol-free record of the same "
                        "divergence.",
        intended_facet={"envelope.calls_in_loop": 0}),
    # --- layout: a WHOLE-MODEL, cross-op data-movement divergence. The whole-model per-op profiler
    # measured linalg.transpose at 393 ms = 57% of openvla (more than every matmul combined), emitted
    # SCALAR. Every openvla/bitvla matmul is a transposed-B GEMM fed by a STANDALONE weight transpose:
    # a full transposed copy materialized in DRAM each forward, then read back. The experts (BLAS /
    # XNNPACK) never materialize it — they read B transposed via the GEMM's own access pattern
    # (transpose-b kernel). The divergence: expert has NO standalone transpose feeding the contraction
    # ("folded"); we do ("materialized").
    _Route(
        axis="layout.transpose_materialized",
        # Fire when the expert folds the transpose into the consumer's access ("folded"/None) and we
        # materialize a standalone transpose op+buffer feeding a matmul B operand.
        when=lambda d: d.expert in ("folded", None) and d.ours == "materialized",
        action_class="PASS",
        target_seam="impr_features:fuse_transpose_b",
        change="fold a `linalg.transpose` of a matmul's B operand INTO the matmul: repoint B to the "
               "un-transposed weight and permute its indexing_map (k,n)->(n,k), then erase the dead "
               "transpose. The op stays `linalg.matmul` (transposed-B maps) so the frozen RVV schedule "
               "still tiles+vectorizes it, while the scalar transpose op AND its DRAM buffer vanish — "
               "exactly the transpose-b access-pattern the experts use.",
        forkable_now=True,
        expected_effect="MEASURED on K1 (openvla fp32, whole-model, cos 0.9999999 / per-element rel "
                        "9.6e-7): transpose bucket 390 ms -> ~0, whole-model wall 5995.7 -> 5604.3 ms "
                        "(-6.5%, well over the 1.9% noise floor); the matmul is unchanged (5602 -> "
                        "5561 ms, within noise). Whole-model cross-op fusion, not a kernel micro-win.",
        intended_facet={"layout.transpose_materialized": "folded"}),
    _Route(
        axis="compute.nr_is_vsetvlmax",
        when=lambda d: bool(d.expert) and not d.ours,
        action_class="HEURISTIC",
        target_seam="schedule:NR=vsetvlmax (VL-adaptive output tile + N-tail clamp)",
        change="set the inner output width NR = vsetvlmax (the runtime VLEN lane count) via a "
               "polymorphic vsetvli VL-loop with an N-tail (clamp NR=min(NR,N)), instead of a "
               "compile-time-fixed NR tile — VL-adaptive like the XNNPACK 1xNv / OpenBLAS scalable "
               "kernels, and the lever that lets a small-N (e.g. N=8 attention) batch_matmul "
               "vectorize instead of falling back to scalar.",
        forkable_now=True,
        expected_effect="one kernel adapts NR to any VLEN; small-N contractions vectorize (N-tail) "
                        "instead of hitting the masked-transfer_write fallback to scalar",
        shape_regimes=("skinny", "square_small")),   # small-N attention / skinny contractions
    _Route(
        axis="compute.mr_adapts_to_m",
        # The M-side analog of nr_is_vsetvlmax. A whole-model decode step is dominated by matmuls
        # whose LEADING dim is M=1 (one token row); our fixed MR=4 register tile does NOT adapt to
        # M<MR, so it writes a vector<4xNR> into a tensor<1xNR> C tile -> a masked vector.transfer_write
        # LLVM-23 rejects (multi-op vector.mask PipelineError) -> silent scalar fallback (no vfmacc).
        # The expert kernels clamp the register block to the actual M (MR=min(MR,M)); ours did not.
        when=lambda d: bool(d.expert) and not d.ours,
        action_class="HEURISTIC",
        target_seam="schedule:MR=min(MR,M) (matmul M-tail clamp; impr_features:accumulator_resident_mtail)",
        change="clamp the matmul register-block MR to the actual leading dim, MR=min(MR,M) (M-tail), "
               "so an M=1 token-decode matmul vectorizes FULL (no masked transfer_write) instead of "
               "MR=4 over the M=1 tile -> the LLVM-23 multi-op vector.mask PipelineError -> scalar "
               "fallback. The M-side analog of the batch_matmul N-tail clamp (NR=min(NR,N)); both "
               "compose in accumulator_resident_wholemodel so a whole model with mixed M (M=1 decode "
               "+ larger-M prefill) and small-N attention vectorizes in ONE schedule.",
        # EVIDENCE-DRIVEN, RESOLVED: the M=1 matmul reproduced the masked-transfer_write PipelineError
        # on spike (vector<4x16> into tensor<1x16>); the MR=min(MR,M) clamp (accumulator_resident_mtail)
        # builds + is bit-exact (cos~1.0) and forms vfmacc on M=1, cube AND non-cube matmuls; the
        # composed accumulator_resident_wholemodel adds the N-tail and vectorizes M=1 + cube + non-cube
        # + N=8 attention in one schedule (all vfmacc>0, vfmul=0, gate_ok). A registered feature
        # expresses it -> forkable_now=True.
        forkable_now=True,
        expected_effect="the M=1 token-decode matmul (smolVLA/rdt2 leading-M=1) vectorizes to vfmacc "
                        "instead of the masked-transfer_write scalar fallback; larger-M matmuls "
                        "unaffected (tile into single-row register tiles, still vfmacc, bit-exact)",
        shape_regimes=("vector", "square_small", "skinny")),   # small-/skinny-M decode matmuls
    _Route(
        axis="compute.register_block",
        # The #1 GEMM data-movement decision and the one the policy-built expert CCA used to hide
        # (register_block=null). The experts ship register blocking up to MR=7 (XNNPACK 7x4v) / MR=16
        # (OpenBLAS 16x8) and SELECT the high MR; our baseline is unblocked (MR=1, one accumulator,
        # 2.0 loads/useful-FMA). Raising MR reuses one loaded B-row across MR broadcast-FMA
        # accumulators (loads/FMA -> 1+1/MR) AND gives MR independent accumulator chains to hide the
        # vfmacc latency. Now a LEARNED divergence (expert MR > ours MR), not a hand-known knob.
        when=lambda d: bool(d.expert),
        action_class="PASS",
        target_seam="impr_features:accumulator_resident_wholemodel_vf_mrpad (per-op MR + M-pad tail)",
        change="raise the matmul register-block MR toward the expert MR so one streamed B-row feeds "
               "MR resident accumulators (the OpenBLAS/XNNPACK A-row-reuse lever), instead of the "
               "unblocked MR=1 baseline. Realized WHOLE-MODEL-SAFELY by the per-op MR feature: each "
               "matmul's M is padded up to a multiple of MR before the register-blocked vfmacc.vf tile, "
               "so a matmul with M%MR==0 gets the MR block and one with M<MR (M=1 decode) or M%MR!=0 "
               "pads cleanly (no masked transfer_write PipelineError / scalar fallback) — no hand kernel.",
        forkable_now=True,
        # MEASURED. The #1 GEMM data-movement lever, now realized whole-model:
        #  - DIAGNOSTIC CEILING (ours_board hand microkernel, MR-configurable, k1_kernel_speedup_*.json):
        #    the matmul bucket falls ~5x as MR 1->7 (bitvla 11.2x->2.2x, openvla 24x->4.8x, rdt2 45x->9x
        #    vs XNNPACK 7x4v) -> register blocking IS the dominant compute lever.
        #  - COMPILER-EMITTED, whole-model-safe (accumulator_resident_wholemodel_vf_mrpad): the earlier
        #    vf_mr4 realized the MR=4 block but ONLY on M%4==0 matmuls; on the small-/odd-M matmuls that
        #    dominate VLA decode (rdt2 M=1, openvla M=17) its bare MR=4 M-tile tripped the LLVM-23
        #    masked-transfer_write PipelineError -> whole-model scalar fallback (it never lowered rdt2).
        #    The mrpad feature PADS each matmul's M to a multiple of MR (transform.structured.pad, sliced
        #    back bit-exact) so EVERY matmul register-blocks cleanly: decoded rdt2 whole-model emits MR=4
        #    on the 28 M%4==0 matmuls and MR=1 on the 3 M=1 tails, 0 vfmacc.vv, in ONE schedule (per-op
        #    MR). It is the whole-model realization of this lever; the MR=7 ceiling still needs a wider
        #    register-block codegen, but MR=4 A-reuse (loads/FMA 2.0->1.25) now applies to the mixed-M
        #    VLA models that vf_mr4 could not lower.
        expected_effect="register blocking is the dominant compute lever (diagnostic: matmul ~5x as "
                        "MR 1->7); the per-op MR + M-pad feature emits the MR=4 A-reuse block on every "
                        "matmul WHOLE-MODEL-SAFELY (padding the M-tail so M=1/M%MR!=0 no longer scalar-"
                        "fall-back), unlike bare vf_mr4 which could not lower the mixed-M VLA models. "
                        "Realizes MR=4 (loads/FMA 2.0->1.25); the MR=7 ceiling still needs wider codegen.",
        # Proposed for matmuls large enough in M that an MR>1 block pays; a pure-M=1 GEMV divergence
        # routes to the M-clamp (mr_adapts_to_m) instead — padding M=1 up to MR wastes MR-1 rows. (The
        # mrpad feature still safely handles any M=1 matmuls PRESENT in a model it lowers; this is only
        # which optimization the beam PROPOSES for a shape.)
        shape_regimes=("square_large", "square_medium", "rectangular")),
    _Route(
        axis="vector.lmul",
        when=_is_higher,
        action_class="KNOB", target_seam="schedule:vector_sizes (widen N to raise LMUL)",
        change="widen the N tile/vector so the emitted vector group uses a higher LMUL",
        forkable_now=True,
        expected_effect="larger vector groups -> fewer vset/loop iterations per output tile"),
    _Route(
        axis="vector.vl_strategy",
        when=lambda d: d.expert == "vsetvl_loop" and d.ours == "vsetivli_fixed",
        action_class="PASS", target_seam="pass:vl-polymorphic-tail (emit vsetvli loop)",
        change="emit a VL-agnostic vsetvli loop with mask/vl tail instead of fixed vsetivli "
               "unrolling (matches the expert vl_agnostic_loop_policy)",
        forkable_now=False,  # needs a scalable/VL-loop lowering — deferred work-item
        expected_effect="one kernel handles any VLEN; smaller code; no fixed-width tail waste"),
    _Route(
        axis="compute.widening",
        when=lambda d: bool(d.expert) and not d.ours,
        action_class="KNOB", target_seam="schedule:dtype_strategy=int8_w8a8 (vwmacc datapath)",
        change="route the i8 matmul through the widening vwmacc i8xi8->i32 datapath",
        forkable_now=True,
        expected_effect="i32-accumulating widening MAC instead of dequantize-to-f32"),
    _Route(
        axis="compute.accumulator_dtype",
        # The accumulate width the expert holds the reduction in (i32 for i8xi8, f32 for bf16/f16
        # inputs). Same compiler lever as widening: the dtype strategy picks the datapath and thus the
        # accumulator type (int8_w8a8 -> i32 acc; bf16 -> f32 acc via lower_bf16_matmul_f32acc). A
        # distinct CCA axis from `widening` (which only asks whether the MAC widens at all).
        when=lambda d: bool(d.expert) and d.expert != d.ours,
        action_class="KNOB", target_seam="schedule:dtype_strategy (accumulate-width datapath)",
        change="select the dtype strategy whose datapath accumulates in the expert's accumulator type "
               "(int8_w8a8 -> i32 via vwmacc; bf16/f16 -> f32 via the widening-float / f32-acc lowering) "
               "instead of the baseline's accumulator width",
        forkable_now=True,
        expected_effect="the reduction accumulates in the intended width (no precision loss / no "
                        "dequantize-to-f32 detour) — the accumulator-type half of the dtype datapath"),
    _Route(
        axis="vector.sew",
        # The element width the expert vectorizes at. SEW is a projection of the element datatype, so
        # its lever is the SAME mixed-precision dtype knob (a narrower SEW = the int8/f16 datapath, a
        # wider SEW = the f32 datapath). Registration of the existing seam under a distinct axis, not a
        # new pass.
        when=lambda d: bool(d.expert) and d.expert != d.ours,
        action_class="KNOB", target_seam="schedule:dtype_strategy (element-width datapath)",
        change="vectorize at the expert's element width by selecting the matching dtype strategy "
               "(e.g. e8 int8 / e16 f16 datapath instead of an e32 f32 lowering)",
        forkable_now=True,
        expected_effect="the emitted vector ops use the intended SEW (element datapath), the "
                        "element-width half of the dtype datapath decision"),
    _Route(
        axis="compute.activation_vectorization",
        # The mined divergence: the expert activation (GELU/sigmoid/SiLU/tanh) evaluates the
        # transcendental as a VECTORIZED polynomial (XNNPACK f32-vgelu rational-12-10 / f32-vsigmoid
        # rr2-p5 — vfmacc chains, no libm call), while OURS lowers math.erf/math.exp through
        # convert-math-to-libm to a SCALAR libm call loop (the elementwise activation generic is
        # never vectorized by the baseline schedule). Honest, general: keyed on the structural value
        # "vectorized_polynomial" vs "scalar_libm_call", not a shape or a single activation.
        when=lambda d: d.expert == "vectorized_polynomial" and d.ours in ("scalar_libm_call", None),
        action_class="PASS",
        target_seam="impr_features:vectorized_transcendental_activation",
        change="rewrite math.exp/erf/tanh to an inline minimax arith polynomial (range-reduced exp "
               "+ A&S erf, mul/add/bitcast/shift) BEFORE vectorization AND vectorize the elementwise "
               "activation linalg.generic, so the activation lowers to vfmacc chains instead of a "
               "scalar convert-math-to-libm call loop. GENERAL over the math ops: GELU (erf), "
               "sigmoid/SiLU (exp) and tanh all vectorize from the one rewrite. The XNNPACK "
               "vectorized-polynomial kernel is the coefficient/structure CEILING REFERENCE only — "
               "the COMPILER emits the polynomial MLIR, no hand kernel is linked.",
        # EVIDENCE-DRIVEN: the gap is the measured divergence in cross_framework_ops_k1.md (GELU
        # ~11-18x, sigmoid ~11-12x slower than XNNPACK, ours-scalar within ~6% of ours-"vectorized"
        # because BOTH are scalar libm). The PASS (vectorized_transcendental_activation) was then
        # implemented + MEASURED on spike: the activation vectorizes (vfmacc>0, no scalar libm call
        # loop) and is accurate (cos=1.0, max-abs-err <~1e-6 vs libm) for GELU/sigmoid/SiLU. A
        # registered, certified feature expresses it -> forkable_now=True. APPROXIMATION (not
        # bit-exact): the honest activation accuracy tradeoff, gated on cos/rel error.
        forkable_now=True,
        expected_effect="the activation vectorizes to vfmacc chains (no per-element libm call); "
                        "closes most of the ~11-18x scalar-libm gap; approximation accuracy cos=1.0 "
                        "/ max-abs-err <~1e-6 vs the libm reference"),
    _Route(
        # The #2 byte-traffic op family (softmax ~3.85% of the census) and the one CCA lever that had
        # no route (a bijection orphan). The baseline vectorizes only the contraction ops, so a
        # standalone reduction (softmax max/sum, LayerNorm/RMSNorm mean/var, a linalg.reduce) stays a
        # SCALAR accumulate loop while the expert uses a hardware horizontal reduce. Route it to the
        # vectorize_reduction PASS: match the reduction generic, vectorize it, lower the
        # multi_reduction via inner-reduction -> vector.reduction, reassociate the fp reduce ->
        # vfredusum.vs (int: vredsum.vs). PROVEN on emitted code (gen_reduce_f32/gen_softmax_f32
        # decoded under the real -fno-vectorize cflags): vfredusum present, cca reduction_form set.
        axis="compute.reduction_form",
        # Fire when the expert vectorizes the reduction (any non-null reduction_form) and we do not
        # (None / "none" / a scalar form). The reduction_form the lifter reports for a vfred*/vred*
        # kernel is "vredsum_tree".
        when=lambda d: bool(d.expert) and d.ours in (None, "none", "scalar"),
        action_class="PASS",
        target_seam="impr_features:vectorize_reduction",
        change="vectorize the standalone reduction (softmax/norm row-reduce, linalg.reduce) and lower "
               "vector.multi_reduction -> vector.reduction -> a hardware horizontal reduce "
               "(vfredusum.vs for fp via reassociate-fp-reductions, vredsum.vs for int), instead of "
               "the scalar convert-linalg-to-loops accumulate the baseline leaves. Contraction "
               "schedule untouched (whole-model-safe).",
        # EVIDENCE: emitted-code proof (not schedule text) — lowering gen_reduce_f32 (64x256) emits 64x
        # vfredusum.vs where the baseline emits ZERO vector ops (RVV_CFLAGS -fno-vectorize, so it is
        # MLIR-emitted, not clang autovec); gen_softmax_f32 emits vfredmax.vs + vfredusum.vs; both lift
        # to cca.reduction_form="vredsum_tree". APPROXIMATION: fp reassociation (cos-gated, not bit-exact).
        forkable_now=True,
        expected_effect="the reduction runs as a hardware vector reduce (vfredusum/vredsum) instead of "
                        "a scalar accumulate loop; vectorizes the softmax/norm reduction family the "
                        "baseline left scalar. Approximation: fp reassociation (cos-gated).",
        intended_facet={"compute.reduction_form": "vredsum_tree"}),
    _Route(
        axis="compute.epilogue",
        when=lambda d: d.expert == "requant_narrow" and d.ours in ("none", None),
        action_class="PASS", target_seam="pass:fuse-requant-narrowing-store",
        change="fuse the requantize + narrowing (vnclip/vfncvt) into the store epilogue",
        forkable_now=False,
        expected_effect="single narrowing store; no separate requant pass over the tile"),
    _Route(
        # THE #1 expert data-movement lever (BB1b): decode.memory lifts memory.access_pattern into the
        # CCA but it was orphaned (no route). Route it to the existing vfmacc_packed feature so the beam
        # can propose operand pre-packing — the dimension gap_analysis flags as the residual expert gap.
        axis="memory.access_pattern",
        when=lambda d: d.expert in ("unit_stride", "packed") and d.ours not in ("unit_stride", "packed", None),
        action_class="PASS",
        target_seam="impr_features:vfmacc_packed (operand pre-packing + layout assignment)",
        change="pre-pack the streamed operand into a unit-stride panel (goi-prepacked, like XNNPACK's "
               "prepacked RHS / OpenBLAS ncopy-tcopy) so the inner loop does unit-stride vector loads "
               "instead of strided/gathered access — the #1 expert data-movement lever.",
        forkable_now=True,
        expected_effect="the contraction's operand loads become unit-stride (packed panel), cutting the "
                        "loads-per-FMA + strided-access stalls the residual expert gap is attributed to",
        intended_facet={"memory.access_pattern": "unit_stride"}),
]

# RVV is the in-tree reference backend (its content is this module). Every OTHER backend's routes are
# backend-SPECIFIC content that a pluggable backend plugin registers into this agnostic router via
# ``register_route`` — the plugin is generated / beam-searched and lives OUT of tree (see
# targetgen/gemmini_plugin.py for the gemmini reference). The core never hardcodes a non-RVV backend.
_ROUTES: dict[str, list[_Route]] = {"rvv": _RVV_ROUTES}


def register_route(backend: str, route: _Route) -> None:
    """Plug one backend-specific route into the agnostic router (idempotent per (backend, axis, seam)).
    Backend plugins (generated/OOT) call this at load time so the core stays backend-agnostic."""
    routes = _ROUTES.setdefault(backend, [])
    if any(r.axis == route.axis and r.target_seam == route.target_seam and
           r.action_class == route.action_class for r in routes):
        return
    routes.append(route)


# The action-class escalation ladder: cheapest/weakest -> strongest. When an action's intended facet
# is NOT achieved by the emitted code, the loop escalates to the next-stronger class for the same axis.
_CLASS_ORDER = {"FLAG": 0, "KNOB": 1, "HEURISTIC": 2, "PASS": 3, "CODEGEN": 4}


def _action_from_route(r: _Route, divergence: Divergence) -> CompilerAction:
    # The promise is the route's static intended_facet, EXCEPT numeric "match-the-expert" axes whose
    # target is the expert's own value (register_block MR, lmul) — derive those from the divergence so
    # the achieved check is "did we reach what THIS expert does", not a hardcoded constant.
    intended = dict(r.intended_facet) if r.intended_facet else None
    if intended is None:
        if divergence.axis == "compute.register_block":
            ev = divergence.expert
            mr = ev[0] if isinstance(ev, (tuple, list)) and ev and isinstance(ev[0], int) else None
            if isinstance(mr, int):
                intended = {"compute.register_block": mr}
        elif divergence.axis == "vector.lmul" and isinstance(divergence.expert, (int, float)):
            intended = {"vector.lmul": float(divergence.expert)}
    return CompilerAction(
        divergence_axis=divergence.axis, action_class=r.action_class,
        target_seam=r.target_seam, change=r.change, forkable_now=r.forkable_now,
        expected_effect=r.expected_effect, backend=divergence.backend,
        evidence=list(divergence.evidence), intended_facet=intended,
        shape_regimes=r.shape_regimes)


def route(divergence: Divergence) -> CompilerAction | None:
    """Map one Divergence to a typed CompilerAction (or None if no route — surfaced as 'unrouted'
    so it is never silently dropped). When several routes match the axis (a class ladder), picks the
    CHEAPEST (weakest) class first — escalation walks up from there via :func:`route_escalated`."""
    cands = [r for r in _ROUTES.get(divergence.backend, [])
             if r.axis == divergence.axis and r.when(divergence)]
    if not cands:
        return None
    return _action_from_route(min(cands, key=lambda r: _CLASS_ORDER.get(r.action_class, 99)),
                              divergence)


def route_escalated(divergence: Divergence, prior_class: str) -> CompilerAction | None:
    """The next-stronger action for ``divergence`` after ``prior_class`` was insufficient (its intended
    facet was not achieved by the emitted code). Returns the cheapest route whose class is strictly
    above ``prior_class``, or None when the ladder is exhausted (no stronger lever exists yet)."""
    floor = _CLASS_ORDER.get(prior_class, -1)
    # The measured residual already PROVES the gap on this axis, so escalation matches on axis + a
    # stronger class + the expert having the property — NOT the original `when` predicate (which may
    # gate on the BASELINE's value, e.g. ours-is-None, that no longer holds after a fork was applied).
    cands = [r for r in _ROUTES.get(divergence.backend, [])
             if r.axis == divergence.axis and bool(divergence.expert)
             and _CLASS_ORDER.get(r.action_class, 99) > floor]
    if not cands:
        return None
    return _action_from_route(min(cands, key=lambda r: _CLASS_ORDER.get(r.action_class, 99)),
                              divergence)


def _facet_value(cca, axis: str):
    """Read ``facet.field`` off a CCA (e.g. 'compute.accumulator_resident'). register_block collapses
    to its MR (None -> 1, the unblocked floor) so the >= check below is meaningful."""
    facet_name, _, field_name = axis.partition(".")
    facet = getattr(cca, facet_name, None) if cca else None
    if facet is None:
        return None
    val = getattr(facet, field_name, None)
    if field_name == "register_block":
        if isinstance(val, (tuple, list)) and val and isinstance(val[0], int):
            return val[0]
        return 1
    return val


def achieved_residual(action: CompilerAction, achieved_cca) -> list[str]:
    """Given an action's machine-readable promise (``intended_facet``) and the CCA lifted from the
    fork's EMITTED asm, return the axes the action PROMISED but the emitted code did NOT achieve — the
    residual that should escalate. Empty list == the action delivered (or made no checkable promise)."""
    if not action.intended_facet:
        return []
    residual: list[str] = []
    for axis, want in action.intended_facet.items():
        got = _facet_value(achieved_cca, axis)
        if axis.endswith("register_block"):           # MR: achieved must be >= promised
            ok = isinstance(got, int) and isinstance(want, int) and got >= want
        else:                                          # everything else: exact match
            ok = got == want
        if not ok:
            residual.append(axis)
    return residual


def shape_regime_of(op: str, m: int, n: int, k: int) -> str:
    """The shape regime (square_large/skinny/...) for an (op, M, N, K) — reuses the canonical
    bench_ceiling classifier (regimes, never exact shapes)."""
    from .bench_ceiling import shape_regime
    return shape_regime(op, m, n, k)


def applies_to_shape(action: CompilerAction, regime: str) -> bool:
    """Whether a shape-conditional optimization applies to a shape regime. Shape-agnostic actions
    (empty ``shape_regimes``) apply everywhere; shape-specific ones only to their regimes — so a big
    matmul and a small-M decode matmul get DIFFERENT optimizations."""
    return not action.shape_regimes or regime in action.shape_regimes


def route_for_shape(divergence: Divergence, op: str, m: int, n: int, k: int) -> CompilerAction | None:
    """Route a divergence, but ONLY if the chosen action applies to this shape's regime — the
    shape-conditional selection. Returns None if the cheapest route is not for this regime."""
    a = route(divergence)
    if a is None:
        return None
    return a if applies_to_shape(a, shape_regime_of(op, m, n, k)) else None


def build_catalog(divergences: list[Divergence]) -> tuple[list[CompilerAction], list[Divergence]]:
    """Return (typed actions, unrouted divergences). Unrouted are reported, never dropped."""
    actions, unrouted = [], []
    for d in divergences:
        a = route(d)
        (actions if a is not None else unrouted).append(a if a is not None else d)
    return actions, unrouted


# ---- "which section of the compiler do I modify" surface -----------------------------
# Each target_seam prefix names a CONCRETE place to edit. ``needs_new_code`` distinguishes editing an
# existing seam (a knob/flag/registered feature — a fork can express it today) from writing a NEW pass
# module. The map is PLUGGABLE and BACKEND-SCOPED so the middle-end can be modified ad-hoc: register a
# new seam at runtime with :func:`register_seam`, and each backend resolves its OWN seams. For an OOT
# target (gemmini), seams are expressed relative to the GENERATED OOT PACKAGE the agent authors (a
# ``<oot_package>`` placeholder), with the in-tree file named only as a reference — so the "where do I
# modify the compiler" answer points at the pluggable OOT middle-end, never couples the agent to our
# in-tree core.
_SeamSpec = tuple[str, str, bool]   # (file, kind, needs_new_code)

# Shared / RVV seams (our in-tree compiler is the artifact for RVV).
SEAM_FILES: dict[str, _SeamSpec] = {
    "impr_features": ("merlin/python/merlin/llvmlower/impr_features.py",
                      "registered PASS/HEURISTIC/PATTERN feature hook (default-off)", False),
    "schedule": ("merlin/python/merlin/rvvgen/from_strategy.py (+ the package knobs.yaml / schedule.mlir)",
                 "transform-schedule knob (forkable via schedule.mlir today)", False),
    "quant": ("merlin/python/merlin/llvmlower/quant_passes.py",
              "int8 quant-pass registry (register/toggle a QuantPass; reached via dtype_strategy=int8_w8a8"
              " -> pkg.is_int8 -> int8_compute -> apply_quant)", False),
    "cflag": ("merlin/python/merlin/runtime/backends/zephyr_model.py (RVV cflags)",
              "compiler flag / march feature", False),
    "pass": ("merlin/python/merlin/llvmlower/ (NEW pass module — write it, then register as an impr feature)",
             "new MLIR pass / lowering", True),
}

# Backend-scoped seams, populated at runtime by backend plugins via ``register_seam`` (the core holds
# NO non-RVV backend content). A backend's plugin registers OOT-package-relative seams (a
# ``<oot_package>`` placeholder, filled by seam_location(oot_package=)) so the agent is pointed at its
# own generated middle-end, never our in-tree core. See targetgen/gemmini_plugin.py.
_BACKEND_SEAM_FILES: dict[str, dict[str, _SeamSpec]] = {}


def register_seam(prefix: str, seam_file: str, seam_kind: str, needs_new_code: bool,
                  *, backend: str | None = None) -> None:
    """Plug a new middle-end seam at runtime (ad-hoc, no core edit). ``backend=None`` registers a
    shared/RVV seam; a backend name registers a backend-scoped seam (e.g. an OOT package's own seam)."""
    spec = (seam_file, seam_kind, needs_new_code)
    if backend is None:
        SEAM_FILES[prefix] = spec
    else:
        _BACKEND_SEAM_FILES.setdefault(backend, {})[prefix] = spec


def _resolve_seam(prefix: str, backend: str | None) -> _SeamSpec:
    """Backend-scoped seam lookup: the backend's own map wins; then the shared map; then any backend
    that defines the prefix (so a no-backend call still resolves a backend-specific seam)."""
    if backend and prefix in _BACKEND_SEAM_FILES.get(backend, {}):
        return _BACKEND_SEAM_FILES[backend][prefix]
    if prefix in SEAM_FILES:
        return SEAM_FILES[prefix]
    for bmap in _BACKEND_SEAM_FILES.values():
        if prefix in bmap:
            return bmap[prefix]
    return ("(unknown seam)", "unknown", True)


def seam_location(target_seam: str, *, backend: str | None = None,
                  oot_package: str | None = None) -> dict:
    """Resolve a route's ``target_seam`` to the concrete file + kind + whether new code is needed.

    ``backend`` selects backend-scoped seams; ``oot_package`` fills the ``<oot_package>`` placeholder in
    an OOT seam with the agent's generated-package root (left as a placeholder if not given, and then
    treated as needing new code since the target location is not yet materialized)."""
    prefix = target_seam.split(":", 1)[0].strip()
    file, kind, needs_new = _resolve_seam(prefix, backend)
    if "<oot_package>" in file:
        if oot_package:
            file = file.replace("<oot_package>", str(oot_package).rstrip("/"))
        else:
            needs_new = True  # OOT target location not materialized yet
    return {"prefix": prefix, "target_seam": target_seam, "seam_file": file,
            "seam_kind": kind, "needs_new_code": needs_new}


def escalation_ladder(axis: str, backend: str = "rvv", *, oot_package: str | None = None) -> list[dict]:
    """The full FLAG->KNOB->HEURISTIC->PASS->CODEGEN ladder for one axis: every route weakest->strongest,
    each annotated with the concrete (backend-scoped, OOT-relative) seam to edit and whether it is
    forkable today. This is the "which section to modify, and what's the next stronger lever" answer."""
    rs = sorted((r for r in _ROUTES.get(backend, []) if r.axis == axis),
                key=lambda r: _CLASS_ORDER.get(r.action_class, 99))
    out = []
    for r in rs:
        loc = seam_location(r.target_seam, backend=backend, oot_package=oot_package)
        out.append({"action_class": r.action_class, "target_seam": r.target_seam,
                    "forkable_now": r.forkable_now, "seam_file": loc["seam_file"],
                    "seam_kind": loc["seam_kind"], "needs_new_code": loc["needs_new_code"]})
    return out
