"""CCA-native fork proposer — ``action_catalog`` (the CCA<->lever router) as the SINGLE source of truth
for what to change in the compiler.

Historically the beam had two parallel routers: ``kernels.action_catalog`` (CCA-axis-keyed, the one the
bijection contract enforces) and ``kernels.knobs`` (motif-string-keyed). This module derives the
beam's proposer from ``action_catalog`` so there is one source of truth. ``propose_forks_from_cca`` takes
typed CCA ``Divergence`` objects (from ``cca_compare.compare``), routes each via ``action_catalog.route``,
and maps the typed ``CompilerAction`` to a ``ForkProposal`` the beam can mint + certify.

Honesty: it emits ONLY overrides the schedule actually honors — ``compiler_features`` (an impr_features
hook), or the known schedule knobs (``op_match`` / ``dtype_strategy`` via ``from_strategy``). Anything the
router marks ``forkable_now=False`` (a deferred PASS/CODEGEN work-item), or a forkable axis with no
auto-knob builder yet, becomes a ``forkable=False`` proposal — recorded, never faked into a knob.

Drop-in for ``beam.run_beam(proposer=...)``: same ``(divergences, knobs) -> [ForkProposal]`` contract as
``knobs.propose_forks``, but consuming CCA Divergences instead of motif strings. (``knobs`` is
retained for the existing motif-string beam path until the beam is cut over to CCA divergences in WS-D.)
"""
from __future__ import annotations

from typing import Any

from ..kernels.action_catalog import CompilerAction, route
from ..kernels.cca_compare import Divergence
from ..kernels.knobs import ForkProposal

# HEURISTIC axes whose (schedule:) seam is IMPLEMENTED by a registered impr_features feature — the
# router marks them forkable_now=True but the proposer had no builder, silently demoting them to
# work-items. Map each to its feature so the beam actually mints the fork (BB1a: "propose more").
_AXIS_FEATURE = {
    "compute.mr_adapts_to_m": "accumulator_resident_mtail",   # M-tail clamp: M=1 decode matmul -> vfmacc
    "compute.nr_is_vsetvlmax": "fused_vfmacc_scalable",       # scalable NR=vsetvlmax (small-N attention)
}


def action_to_fork(action: CompilerAction, knobs: dict[str, Any]) -> ForkProposal:
    """Map one typed CompilerAction to a ForkProposal, emitting only knob keys the schedule honors."""
    seam = action.target_seam
    intended = action.intended_facet or {}
    axis = action.divergence_axis
    ev = list(action.evidence)

    # 1) an impr_features hook -> a compiler_features override (forkable iff the router says so).
    #    `vector.lmul` arrives here too: its seam names the `lmul_register_group` SENTINEL, which
    #    `zephyr_model.prepare_for_lowering` resolves against the prepared IR's element widths and the
    #    board VLEN and swaps for the concrete `lmul_group_m<N>`. That is deliberate -- the proposer
    #    has a knobs dict, the IR has the arithmetic, and only one of them actually knows.
    if seam.startswith("impr_features:"):
        feat = seam.split(":", 1)[1].split()[0]
        return ForkProposal(overrides={"compiler_features": [feat]}, lever="feature",
                            targets=axis, evidence=ev, forkable=action.forkable_now, note=action.change,
                            action=action)

    # 2) a forkable schedule KNOB/HEURISTIC -> a concrete, KNOWN knob override.
    if action.forkable_now and not seam.startswith("pass:"):
        overrides: dict | None = None
        if "int8_w8a8" in seam or intended.get("compute.widening"):
            overrides = {"dtype_strategy": "int8_w8a8"}
        elif axis in _AXIS_FEATURE:
            # a HEURISTIC schedule-seam implemented by a registered feature (mtail / scalable NR).
            overrides = {"compiler_features": [_AXIS_FEATURE[axis]]}
        elif "dtype_strategy" in seam:
            # the accumulate-width / element-width datapath axes (accumulator_dtype, vector.sew) reach
            # the same int8 datapath knob as widening (i32-accum + i8-sew via vwmacc).
            overrides = {"dtype_strategy": "int8_w8a8"}
        if overrides is not None:
            return ForkProposal(overrides=overrides, lever="knob", targets=axis, evidence=ev,
                                forkable=True, note=action.change, action=action)

    # 3) deferred PASS/CODEGEN, or a forkable axis with no auto-knob builder yet -> HONEST work-item.
    return ForkProposal(overrides={}, lever="work_item", targets=axis, evidence=ev,
                        forkable=False, note=action.change, action=action)


def propose_forks_from_cca(divergences: list[Divergence], knobs: dict[str, Any]) -> list[ForkProposal]:
    """Route each CCA Divergence via action_catalog and map to a ForkProposal. Unrouted divergences are
    skipped here (``build_catalog`` surfaces them as 'unrouted' separately — never silently dropped)."""
    out: list[ForkProposal] = []
    for d in divergences:
        a = route(d)
        if a is not None:
            out.append(action_to_fork(a, knobs))
    return out
