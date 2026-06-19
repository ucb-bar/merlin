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
    action_class: str         # FLAG | HEURISTIC | PASS | KNOB
    target_seam: str          # concrete place: "impr_features:<name>" | "schedule:<knob>" | "cflag:<f>" | "pass:<name>"
    change: str               # human-readable description of the change
    forkable_now: bool
    expected_effect: str
    backend: str
    evidence: list[str] = field(default_factory=list)


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
                        "matmul, N=5, cos=1.0) vs the frozen baseline"),
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
                        "scalable gap (compute kernel alone is already ~1.5x OpenBLAS)"),
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
                        "1.7x faster than OpenBLAS pack-excluded on the spike proxy (hand ceiling)"),
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
                        "instead of hitting the masked-transfer_write fallback to scalar"),
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
        axis="compute.epilogue",
        when=lambda d: d.expert == "requant_narrow" and d.ours in ("none", None),
        action_class="PASS", target_seam="pass:fuse-requant-narrowing-store",
        change="fuse the requantize + narrowing (vnclip/vfncvt) into the store epilogue",
        forkable_now=False,
        expected_effect="single narrowing store; no separate requant pass over the tile"),
]

_ROUTES: dict[str, list[_Route]] = {"rvv": _RVV_ROUTES}


def route(divergence: Divergence) -> CompilerAction | None:
    """Map one Divergence to a typed CompilerAction (or None if no route — surfaced as 'unrouted'
    so it is never silently dropped)."""
    for r in _ROUTES.get(divergence.backend, []):
        if r.axis == divergence.axis and r.when(divergence):
            return CompilerAction(
                divergence_axis=divergence.axis, action_class=r.action_class,
                target_seam=r.target_seam, change=r.change, forkable_now=r.forkable_now,
                expected_effect=r.expected_effect, backend=divergence.backend,
                evidence=list(divergence.evidence))
    return None


def build_catalog(divergences: list[Divergence]) -> tuple[list[CompilerAction], list[Divergence]]:
    """Return (typed actions, unrouted divergences). Unrouted are reported, never dropped."""
    actions, unrouted = [], []
    for d in divergences:
        a = route(d)
        (actions if a is not None else unrouted).append(a if a is not None else d)
    return actions, unrouted
