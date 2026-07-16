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
        action_class="KNOB",
        target_seam="schedule:register-block MR (impr_features:accumulator_resident_wholemodel_vf_mr4 "
                    "/ the vfmacc_t_<MR>_<NR>_<KC> tuning grid)",
        change="raise the matmul register-block MR toward the expert MR so one streamed B-row feeds "
               "MR resident accumulators (the OpenBLAS/XNNPACK A-row-reuse lever), instead of the "
               "unblocked MR=1 baseline. Compiler-emitted via the existing register-blocked vfmacc.vf "
               "transform schedule — no hand kernel.",
        forkable_now=True,
        # MEASURED, with an HONEST split between the ideal lever and what the compiler realizes today:
        #  - DIAGNOSTIC CEILING (ours_board hand microkernel, MR-configurable, k1_kernel_speedup_*.json):
        #    the matmul bucket falls ~5x as MR 1->7 (bitvla 11.2x->2.2x, openvla 24x->4.8x, rdt2 45x->9x
        #    vs XNNPACK 7x4v) -> register blocking IS the dominant compute lever.
        #  - COMPILER-EMITTED (inlined transform schedule, the real product): the pipeline ALREADY emits
        #    MR=4/NR=16 (accumulator_resident_microkernel_v3 = the bitVLA winner, 147ms). Pushing the
        #    EXISTING grid knobs higher REGRESSES whole-model (MR=8 ->178ms, NR=32 ->165ms): the
        #    transform-schedule codegen does NOT yet realize the diagnostic headroom. So this KNOB pays
        #    off only up to MR=4 today; capturing the MR=7 ceiling needs a register-block CODEGEN
        #    improvement (a PASS), not just the knob.
        #  - Small-M / GEMV (rdt2 M=1, openvla M=16-20): padding-limited regardless -> route to the
        #    dispatch-level small-M batching pass (group token-dim matmuls into one large-M GEMM) first.
        expected_effect="register blocking is the dominant compute lever (diagnostic: matmul ~5x as "
                        "MR 1->7); compiler emits MR=4 today (bitVLA sweet spot), but the existing grid "
                        "knobs above MR=4 REGRESS whole-model -> the MR=7 ceiling needs better register-"
                        "block codegen (PASS), and small-M/GEMV needs batching first. Knob pays to MR=4.",
        shape_regimes=("square_large", "square_medium", "rectangular")),   # big blocks pay off when large
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
        axis="compute.epilogue",
        when=lambda d: d.expert == "requant_narrow" and d.ours in ("none", None),
        action_class="PASS", target_seam="pass:fuse-requant-narrowing-store",
        change="fuse the requantize + narrowing (vnclip/vfncvt) into the store epilogue",
        forkable_now=False,
        expected_effect="single narrowing store; no separate requant pass over the tile"),
]

_ROUTES: dict[str, list[_Route]] = {"rvv": _RVV_ROUTES}


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
# Each target_seam prefix names a CONCRETE file to edit. ``needs_new_code`` distinguishes editing an
# existing seam (a knob/flag/registered feature — a fork can express it today) from writing a NEW pass
# module (a ``pass:`` route not yet backed by an impr_features hook). This is the map the CLI prints so
# an engineer knows exactly where FLAG/KNOB/HEURISTIC/PASS/CODEGEN each live.
SEAM_FILES: dict[str, tuple[str, str, bool]] = {
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


def seam_location(target_seam: str) -> dict:
    """Resolve a route's ``target_seam`` to the concrete file + kind + whether new code is needed."""
    prefix = target_seam.split(":", 1)[0].strip()
    file, kind, needs_new = SEAM_FILES.get(prefix, ("(unknown seam)", "unknown", True))
    return {"prefix": prefix, "target_seam": target_seam, "seam_file": file,
            "seam_kind": kind, "needs_new_code": needs_new}


def escalation_ladder(axis: str, backend: str = "rvv") -> list[dict]:
    """The full FLAG->KNOB->HEURISTIC->PASS->CODEGEN ladder for one axis: every route weakest->strongest,
    each annotated with the concrete seam file to edit and whether it is forkable today. This is the
    "which section to modify, and what's the next stronger lever if this one doesn't land it" answer."""
    rs = sorted((r for r in _ROUTES.get(backend, []) if r.axis == axis),
                key=lambda r: _CLASS_ORDER.get(r.action_class, 99))
    out = []
    for r in rs:
        loc = seam_location(r.target_seam)
        out.append({"action_class": r.action_class, "target_seam": r.target_seam,
                    "forkable_now": r.forkable_now, "seam_file": loc["seam_file"],
                    "seam_kind": loc["seam_kind"], "needs_new_code": loc["needs_new_code"]})
    return out
