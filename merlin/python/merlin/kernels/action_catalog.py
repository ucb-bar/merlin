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
        action_class="PASS", target_seam="pass:vector.fma-forming contraction lowering",
        change="emit a vector.fma / llvm.fmuladd for the contraction so it lowers to fused vfmacc "
               "instead of separate vfmul.vv+vfadd.vv",
        # EVIDENCE-DRIVEN: certified+decoded 3 attempts (outerproduct lowering; K=4-vector tile;
        # K-tile + -ffp-contract=fast) — ALL measured as no-op (vfmacc still 0). So no current
        # fork expresses this; it is a genuine deferred PASS work-item (needs a pattern that
        # builds vector.fma in the IR, not a schedule knob/flag). The loop demoted it from the
        # hypothesised forkable=True after measuring.
        forkable_now=False,
        expected_effect="vfmacc replaces vfmul+vfadd; fewer ops, higher FLOP/insn"),
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
