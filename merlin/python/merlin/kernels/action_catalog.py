"""Typed compiler-action catalog — the actionable "what can we change in the compiler" output.

Each CCA ``Divergence`` (from ``cca_compare``) routes to a typed ``CompilerAction`` tagged with a
**class** so we know structurally what to change:
- ``FLAG``      — a cflag / pass option (e.g. ``-ffp-contract=fast``, march features).
- ``HEURISTIC`` — a selection rule in a pass (tile size, LMUL choice, fuse-or-not).
- ``PASS``      — a new/modified MLIR pass or lowering pattern (an ``impr_features`` hook).
- ``KNOB``      — a transform-schedule parameter (forkable today via ``schedule.mlir``).

``target_seam`` names the concrete place to make the change; ``forkable_now`` says whether an
``impr_`` fork can express it today (schedule knob / cflag / a registered ``impr_features`` hook)
or it is a deferred work-item. Supersedes the knob-only ``knobs`` gap-router. Routes are keyed
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
    # --- what it takes to apply this, and what it cannot be applied WITH -----------------------
    # Conditions that must hold for the action to be LEGAL, in prose the planner can read and a
    # human can check. Legality was previously only the route's `when` predicate over a divergence,
    # which cannot express a capacity or dependency constraint ("the working set must fit the
    # discovered accumulator"), so those were enforced nowhere or by the fork failing to build.
    preconditions: tuple[str, ...] = ()
    # Other actions that must be applied WITH this one, and ones it cannot compose with. Composition
    # was previously an ad-hoc rule inside one proposer ("two full-schedule replacements clobber"),
    # so any other caller composing actions had no way to know.
    requires: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    # What must be rebuilt for this to take effect; see REBUILD_SCOPES. The dominant cost term of a
    # search step, and previously inferable only from `seam_location()["needs_new_code"]`.
    rebuild_scope: str = "schedule"
    # For a tunable action: the domain to search, as an explicit set/range rather than a seam name.
    # None = not a numeric action. The local tuner searches THIS; the planner does not enumerate it.
    parameter_domain: Any | None = None
    # A family key, so a beam can keep its candidates diverse by KIND rather than by temperature —
    # six variants of one tile change are one idea, not six.
    action_family: str = ""
    # Prior belief this action helps, from measured outcomes (see kernels.space.corpus_prior and the
    # transform ledger: 1509 attempts, 13.45% improved). None = no prior, which is NOT 0.5.
    evidence_prior: float | None = None
    #: Whether the empty ``shape_regimes`` above is a CLAIM or merely silence. An empty tuple makes
    #: the action apply to every regime, so an undeclared action asserts universality nobody
    #: established -- and a policy applied to a shape regime it was never validated on is precisely
    #: the guess that a one-shot compile cannot afford to make silently. True = deliberately
    #: shape-agnostic; False with empty regimes = UNSPECIFIED, reported by :func:`unvalidated_scope`.
    shape_agnostic: bool = False
    #: How the emitted value is compared against the promise: "exact" (default) or "at_least".
    #: Direction is a property of the AXIS'S MEANING and cannot be inferred from the value's type — a
    #: bigger register block is better, while a bigger vector.sew is worse (wider elements, fewer
    #: lanes). It was previously a name test on the axis string, which silently gave every other
    #: numeric axis exact-match semantics whether or not that was right.
    promise_comparison: str = "exact"


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
    # Carried onto the CompilerAction this route builds; see CompilerAction for what each means.
    preconditions: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    rebuild_scope: str = "schedule"
    parameter_domain: Any | None = None
    action_family: str = ""
    evidence_prior: float | None = None
    promise_comparison: str = "exact"
    shape_agnostic: bool = False


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
        # MR is a magnitude: reaching MORE blocking than the expert keeps the promise.
        promise_comparison="at_least",
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
        # THE NEXT RUNG on the same axis, so a detected scalar-math loss no longer exhausts the ladder
        # in silence. The PASS below rewrites math.exp/erf/tanh -- the GELU/softmax family. It does NOT
        # cover the two families that dominate a transformer's non-contraction tail: the ALGEBRAIC
        # rsqrt of RMSNorm and the TRIGONOMETRIC sin/cos of RoPE. Before this rung existed, a model
        # paying for those routed to a pass that cannot help, measured no gain, and the escalation
        # ladder ended -- with no work-item recording that the lever was missing rather than useless.
        #
        # MEASURED on small_llama int8 (model symbols only, static): 16.63% of the binary is scalar
        # FLOAT on an INT8 model, and 36 model symbols are entirely scalar, including __ieee754_sqrt,
        # __kernel_sinf, __kernel_cosf, __kernel_rem_pio2f and __extendbfsf2. cca._MATH_* now classifies
        # those structurally, so the divergence is finally OBSERVABLE; this is where it escalates to.
        axis="compute.activation_vectorization",
        # The SAME predicate as the cheaper rung, deliberately. Which rung applies is decided by
        # MEASUREMENT, not by a predicate hand-coded here: `route()` returns the cheapest matching
        # class (the PASS), and `route_escalated` only reaches this one after the PASS failed to
        # achieve the intended facet -- which is exactly what happens on a model whose math ops the
        # polynomial emitter does not cover. Encoding "is it RoPE or GELU?" in the predicate instead
        # would put the answer in the router, where nothing can check it, rather than in the emitted
        # code, where the facet check already does.
        when=lambda d: d.expert == "vectorized_polynomial" and d.ours in ("scalar_libm_call", None),
        action_class="CODEGEN",
        target_seam="pass:llvmlower/act_poly.py (extend the polynomial emitter's op coverage)",
        change="add vector lowerings for the math families the minimax-polynomial pass does not "
               "cover: (a) ALGEBRAIC rsqrt/sqrt -- a Newton-Raphson iteration on the initial estimate, "
               "which is the RMSNorm normaliser; (b) TRIGONOMETRIC sin/cos with range reduction -- the "
               "RoPE rotation, which currently pays glibc's __kernel_rem_pio2f argument reduction as a "
               "scalar call of its own; (c) the soft-float conversion helpers (__extendbfsf2 and "
               "friends), which a widened vector datapath removes outright rather than vectorises. "
               "NEEDS NEW CODE: the existing pass emits exp/erf/tanh polynomials only, so this is an "
               "op-coverage extension of a proven emitter, not a new mechanism -- which is exactly the "
               "shape of task a constrained, oracle-graded capsule can hold.",
        forkable_now=False,
        expected_effect="the RMSNorm and RoPE generics lower to vfmacc chains instead of per-element "
                        "scalar math calls, removing the scalar-float share from an int8 model's "
                        "instruction mix; sized by that mix, not asserted",
        intended_facet={"compute.activation_vectorization": "vectorized_polynomial"}),
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
    # ---- coverage (WHOLE-MODEL): the losses a per-kernel CCA structurally cannot see -------------
    # Every route above compares OUR kernel to an EXPERT kernel. That can find nothing wrong while the
    # model runs at a few percent of peak, because the biggest measured losses are graph-level: an
    # entire contraction class left for convert-linalg-to-loops, and the ~88% of linalg ops that are
    # not contractions at all. Without these routes the mining loop could not PROPOSE the fix, because
    # nothing in the abstraction reported the loss -- the optimization was undiscoverable by
    # construction, not merely unfound.
    _Route(
        axis="coverage.unclaimed_op_classes",
        # ours leaves a contraction class unclaimed; an expert claims every class it computes.
        when=lambda d: bool(d.ours) and not d.expert,
        action_class="PASS",
        target_seam="impr_features:perop_register_block (tag each contraction, match by attribute)",
        change="choose the register block PER CONTRACTION instead of per op class: a pre-pass tags each "
               "contraction with the largest block legal for ITS OWN extents and the schedule emits one "
               "tile+vectorize arm per distinct block, matched by attribute (the "
               "transform.structured.match attributes{...} form the elementwise arms already use). "
               "Today one degenerate extent in a class forces the whole class off the vector path.",
        # WAS forkable_now=False as a deferred work-item; that is stale documentation, not a blocker.
        # The machinery is wired end to end: `llvmlower/perop_blocks.block_table` derives the table from
        # the PREPARED IR, `tag_prepared_mlir` tags it, and `zephyr_model.prepare_for_lowering` swaps
        # the sentinel for the concrete `ensure_perop_block` feature. What was actually broken
        # was this line's SEAM NAME: it read `per_op_register_block`, and the sentinel is
        # `perop_register_block` -- `fork_from_action` splits the seam and puts that string straight into
        # `compiler_features`, so a fork minted from this route died with "unknown impr feature" instead
        # of running. A dead seam string and a False flag are the same failure wearing two hats, and the
        # honesty test in merlin/tests/kernels/test_action_catalog.py now fails the build on either.
        forkable_now=True,
        expected_effect="every contraction is vectorized at its own legal width instead of the class "
                        "being clamped by its smallest member — measured loss on whisper_tiny: its N=1 "
                        "decode step drops the 1500-wide encoder attention to scalar, 34% of all MACs",
        intended_facet={"coverage.unclaimed_op_classes": ()}),
    _Route(
        axis="coverage.claimed_mac_fraction",
        when=lambda d: isinstance(d.ours, (int, float)) and d.ours < 1.0,
        action_class="KNOB",
        target_seam="schedule:per-op-class register block (mining.apply.shape_adapted_features)",
        change="re-derive the register block per op class against the workload's real extents (and, for "
               "a multicore build, against the per-hart tile) so a class is claimed at a legal width "
               "rather than declined outright",
        forkable_now=True,
        expected_effect="the claimed share of the model's MACs rises toward 1.0 without changing the "
                        "emitted kernel for classes that already fit",
        intended_facet={"coverage.claimed_mac_fraction": 1.0}),
    _Route(
        axis="coverage.non_contraction_op_fraction",
        # the expert's kernel family vectorizes its elementwise/layout work; ours leaves it scalar.
        when=lambda d: isinstance(d.ours, (int, float)) and d.ours > 0.5,
        action_class="PASS",
        target_seam="impr_features:vectorize_non_contraction_generics (MERLIN_VEC_RANK)",
        change="scoped-vectorize the NON-contraction linalg.generics (elementwise, layout, im2col "
               "gather, pad) at a bounded per-rank vector width, instead of letting "
               "convert-linalg-to-loops emit scalar loops for them. The bounded per-rank form exists "
               "as a default-off experiment (MERLIN_VEC_RANK); promoting it to a registered feature is "
               "what makes it selectable by the beam. NOT the blunt whole-func vectorize_children, "
               "which explodes into vector.extracts on a whole model.",
        forkable_now=True,
        expected_effect="the elementwise/layout tail stops running scalar on one core — it is the "
                        "DOMINANT structural loss on every workload (86-89% of linalg ops are "
                        "non-contractions; spectformer 0.40 MAC/cycle, deepjscc 0.22, lstmnetvit 0.067 "
                        "against ~8 for a VLEN=128 int8 vwmacc datapath). MEASURED so far: the "
                        "registered feature does emit 4.9x more vector instructions with bit-identical "
                        "output, but runs 1.28x SLOWER on deepjscc at every lane width (8/16/32), so the "
                        "current realization does NOT pay. The action stays routed because the loop must "
                        "be able to try and REJECT it; a fork that enables it has to clear the baseline "
                        "on cycles, not on vector count.",
        intended_facet=None),
]

# RVV is the in-tree reference backend (its content is this module). Every OTHER backend's routes are
# registered into this agnostic router via ``register_route`` by the generic, derivation-driven backend
# (targetgen/rtl_backend.py), which DERIVES them from mlc RTL discovery for any target — no per-target
# code. The core never hardcodes a non-RVV backend.
_ROUTES: dict[str, list[_Route]] = {"rvv": _RVV_ROUTES}

# Lazy backend derivers: callables (backend -> None) that DERIVE + register a non-RVV backend's routes on
# first use (e.g. the RTL-derived spatial levers). The derivation-driven backend (targetgen/rtl_backend)
# self-registers one at import; `ensure_backend` runs them so a seam-menu call for a fresh backend
# populates itself. Kept here (not imported eagerly) so the core stays backend-agnostic + import-cycle-free.
_DERIVERS: list[Callable[[str], None]] = []


def register_deriver(fn: Callable[[str], None]) -> None:
    """Register a lazy route-deriver (idempotent by identity). Called by the derivation-driven backend."""
    if fn not in _DERIVERS:
        _DERIVERS.append(fn)


def ensure_backend(backend: str) -> None:
    """Make sure ``backend``'s routes are derived+registered before they are read. No-op for the in-tree
    RVV reference and for a backend already populated. Runs each deriver GUARDED — a backend with no RTL
    access (e.g. the non-CIRCT arm) or no mlc simply yields no routes, never an exception, so the seam
    menu degrades to empty instead of crashing. The moat is preserved: derivers key on RTL facts, which
    only the CIRCT arm can read."""
    if backend == "rvv" or _ROUTES.get(backend):
        return
    if not _DERIVERS:
        try:                       # lazy, guarded: make the derivation-driven backend self-register
            import merlin.targetgen.rtl_backend  # noqa: F401
        except Exception:          # noqa: BLE001 — targetgen unavailable in this sandbox -> no derivers
            return
    for fn in list(_DERIVERS):
        try:
            fn(backend)
        except Exception:          # noqa: BLE001 — a deriver with no RTL access is a no-op
            continue


def backends() -> tuple[str, ...]:
    """Every backend that has registered routes — the discovery seam the checkers iterate.

    Exists so ``check_regions``/``check_categories`` stop being scoped to the literal ``"rvv"``. Those
    invariants were true by construction for every other target: a second backend could register a
    lever axis with no governing region or no improvement category and nothing would look. Reads only
    what is already registered and never triggers a derivation, so asking the question cannot change
    the answer.
    """
    return tuple(sorted(b for b, routes in _ROUTES.items() if routes))


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
#
# RUNTIME sits at the top, and what that ordering means is worth stating because it is easy to
# over-read. It is a COST ordering for escalation -- try the cheaper intervention first -- not a
# claim that a runtime change subsumes a codegen one. They act on different subsystems: CODEGEN
# changes what instructions are emitted, RUNTIME changes how the emitted work is orchestrated
# (command-buffer batching, launch grouping, DMA schedule, fences, engine overlap). Most runtime
# axes have no compile-time route at all, so a route is registered as RUNTIME directly and no
# escalation walks through CODEGEN to reach it.
#
# It exists because these were otherwise expressed as knobs wearing a disguise: `categories` already
# reserved a "runtime-sync" bucket and recorded that it had no lever axis, which is what a missing
# action class looks like from the other side.
_CLASS_ORDER = {"FLAG": 0, "KNOB": 1, "HEURISTIC": 2, "PASS": 3, "CODEGEN": 4, "RUNTIME": 5}

#: Rebuild scope of an action — what has to be rebuilt for it to take effect. Ordered cheapest first,
#: because it is the main cost term in a search step and the loop should prefer a cheap probe.
REBUILD_SCOPES: tuple[str, ...] = (
    "none", "schedule", "target-package", "compiler", "runtime", "full")


def _promised_value(expert_value):
    """The value an action promises the emitted code will reach, from the EXPERT's value on that axis.

    None means no checkable promise, which is the fail-closed answer for an expert value that is
    absent or of a shape the audit cannot compare. A tuple collapses to its leading element, matching
    :func:`_facet_value`'s register-block treatment -- the audit compares MR, so the promise must be
    the MR too, not the whole tuple.
    """
    if expert_value is None or isinstance(expert_value, bool):
        return expert_value if isinstance(expert_value, bool) else None
    if isinstance(expert_value, (int, float, str)):
        return expert_value
    if isinstance(expert_value, (tuple, list)):
        if expert_value and isinstance(expert_value[0], int):
            return expert_value[0]
        return tuple(expert_value)
    return None


def _action_from_route(r: _Route, divergence: Divergence) -> CompilerAction:
    # The promise is the route's static intended_facet, EXCEPT "match-the-expert" axes whose target is
    # the expert's OWN value — derive those from the divergence so the achieved check is "did we reach
    # what THIS expert does", not a hardcoded constant.
    #
    # This derivation used to name two axes (compute.register_block, vector.lmul) as literals, which
    # left every other match-the-expert axis with NO machine-checkable promise: applying such an action
    # was unverifiable from the emitted code, so confirming it needed an execution. That is the
    # expensive direction. A route that exists to close an axis toward the expert is, by definition,
    # promising the expert's value on that axis, so derive it for any axis whose expert value is
    # present and checkable -- and fail closed (no promise) when it is absent, since "the expert does
    # not exhibit this property" names no target to reach.
    intended = dict(r.intended_facet) if r.intended_facet else None
    if intended is None:
        target = _promised_value(divergence.expert)
        if target is not None:
            intended = {divergence.axis: target}
    return CompilerAction(
        divergence_axis=divergence.axis, action_class=r.action_class,
        target_seam=r.target_seam, change=r.change, forkable_now=r.forkable_now,
        expected_effect=r.expected_effect, backend=divergence.backend,
        evidence=list(divergence.evidence), intended_facet=intended,
        shape_regimes=r.shape_regimes,
        preconditions=r.preconditions, requires=r.requires,
        conflicts=r.conflicts, rebuild_scope=r.rebuild_scope,
        parameter_domain=r.parameter_domain, action_family=r.action_family,
        evidence_prior=r.evidence_prior, promise_comparison=r.promise_comparison,
        shape_agnostic=r.shape_agnostic)


def route(divergence: Divergence) -> CompilerAction | None:
    """Map one Divergence to a typed CompilerAction (or None if no route — surfaced as 'unrouted'
    so it is never silently dropped). When several routes match the axis (a class ladder), picks the
    CHEAPEST (weakest) class first — escalation walks up from there via :func:`route_escalated`."""
    ensure_backend(divergence.backend)
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
    ensure_backend(divergence.backend)
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
    at_least = action.promise_comparison == "at_least"
    for axis, want in action.intended_facet.items():
        got = _facet_value(achieved_cca, axis)
        if at_least:
            ok = (isinstance(got, (int, float)) and not isinstance(got, bool)
                  and isinstance(want, (int, float)) and not isinstance(want, bool)
                  and got >= want)
        else:
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


def shape_scope(action: CompilerAction) -> str:
    """``regimes`` | ``agnostic`` | ``unspecified`` -- how far this action's applicability was
    established. ``unspecified`` still APPLIES everywhere (the behaviour is unchanged); it just stops
    that permissiveness from reading as a validated claim."""
    if action.shape_regimes:
        return "regimes"
    return "agnostic" if action.shape_agnostic else "unspecified"


def unvalidated_scope(actions, regime: str) -> tuple[CompilerAction, ...]:
    """Of ``actions`` that would fire on ``regime``, the ones whose scope was never established.

    This is the hardware-budget question for a one-shot compile: applying a policy to a shape regime
    nobody validated it on is a guess, and this names exactly which guesses are being made. Where the
    set is empty the emission rests on validated scope; where it is not, that is where a measurement
    would buy the most.
    """
    return tuple(a for a in actions
                 if applies_to_shape(a, regime) and shape_scope(a) == "unspecified")


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
    "schedule": ("merlin/python/merlin/mining/from_strategy.py (+ the package knobs.yaml / schedule.mlir)",
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
# own generated middle-end, never our in-tree core. See targetgen/rtl_backend.py.
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
    ensure_backend(backend)
    rs = sorted((r for r in _ROUTES.get(backend, []) if r.axis == axis),
                key=lambda r: _CLASS_ORDER.get(r.action_class, 99))
    out = []
    for r in rs:
        loc = seam_location(r.target_seam, backend=backend, oot_package=oot_package)
        out.append({"action_class": r.action_class, "target_seam": r.target_seam,
                    "forkable_now": r.forkable_now, "seam_file": loc["seam_file"],
                    "seam_kind": loc["seam_kind"], "needs_new_code": loc["needs_new_code"]})
    return out


def composition_problems(actions) -> tuple[str, ...]:
    """Why this SET of actions cannot be applied together. Empty when the bundle is legal.

    Composition legality lived in one proposer as a single ad-hoc rule -- "two full-schedule
    replacement features clobber" -- so every other caller that bundled actions had no way to learn
    it. A bundle is how interaction effects get tested, and the beam applies bundles, so the rule
    belongs on the action.

    Three checks, each naming what would otherwise be silently wrong:

    * a declared CONFLICT between two actions in the bundle;
    * a declared REQUIREMENT that the bundle does not satisfy -- an action applied without its
      precondition tends to build fine and do nothing, which the intended-facet audit then reports as
      an unachieved promise, sending the loop escalating for a reason that is not the real one;
    * two actions writing the SAME seam, which is the general form of the clobber rule: the second
      overwrites the first and the result is credited to both.
    """
    problems: list[str] = []
    families = {a.action_family for a in actions if a.action_family}
    names = {a.action_family or a.target_seam for a in actions}

    for a in actions:
        for c in a.conflicts:
            if c in names or c in families:
                problems.append(
                    f"{a.divergence_axis}: declares a conflict with {c!r}, which is in this bundle")
        for req in a.requires:
            if req not in names and req not in families:
                problems.append(
                    f"{a.divergence_axis}: requires {req!r}, which the bundle does not apply — an "
                    f"action without its requirement usually builds and does nothing")

    by_seam: dict[str, list[str]] = {}
    for a in actions:
        by_seam.setdefault(a.target_seam, []).append(a.divergence_axis)
    for seam, axes in sorted(by_seam.items()):
        if len(axes) > 1:
            problems.append(
                f"{len(axes)} actions write the same seam {seam!r} ({', '.join(sorted(axes))}): the "
                f"later one overwrites the earlier and the result would be credited to both")
    return tuple(problems)


def composable(actions) -> bool:
    return not composition_problems(actions)


def lineage_problems(applied, candidate) -> tuple[str, ...]:
    """Why ``candidate`` cannot be applied on top of the ``applied`` actions of its parent.

    This is the SEQUENTIAL sibling of :func:`composition_problems`, and it deliberately checks less.
    A bundle applies its actions together, so two actions writing one seam make the credit ambiguous
    and that is a real problem. A beam applies them one generation at a time: the child overwrites the
    parent's seam, the measured delta is parent-to-child, and the credit is unambiguous — that is
    ordinary refinement, not a clobber. Running the bundle rule over a lineage would therefore reject
    every deepening step, which is the opposite of what the beam is for.

    What DOES survive the sequential case is what the declarations say about the actions themselves:
    a conflict is a conflict however it was reached, and an unmet requirement still produces an action
    that builds and does nothing.
    """
    problems: list[str] = []
    names = {a.action_family or a.target_seam for a in applied}
    names |= {a.action_family for a in applied if a.action_family}
    for c in candidate.conflicts:
        if c in names:
            problems.append(
                f"{candidate.divergence_axis}: declares a conflict with {c!r}, already applied on "
                f"this parent")
    self_names = {candidate.action_family or candidate.target_seam}
    if candidate.action_family:
        self_names.add(candidate.action_family)
    for req in candidate.requires:
        if req not in names and req not in self_names:
            problems.append(
                f"{candidate.divergence_axis}: requires {req!r}, which neither this action nor its "
                f"parent lineage applies — an action without its requirement usually builds and does "
                f"nothing, and the facet audit then blames the action")
    return tuple(problems)
