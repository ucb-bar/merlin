"""CCA-native fork proposer — ``action_catalog`` (the CCA<->lever router) as the SINGLE source of truth
for what to change in the compiler.

Historically the beam had two parallel routers: ``kernels.action_catalog`` (CCA-axis-keyed, the one the
bijection contract enforces) and ``kernels.rvv_knobs`` (motif-string-keyed). This module derives the
beam's proposer from ``action_catalog`` so there is one source of truth. ``propose_forks_from_cca`` takes
typed CCA ``Divergence`` objects (from ``cca_compare.compare``), routes each via ``action_catalog.route``,
and maps the typed ``CompilerAction`` to a ``ForkProposal`` the beam can mint + certify.

Honesty: it emits ONLY overrides the schedule actually honors — ``compiler_features`` (an impr_features
hook), or the known schedule knobs (``op_match`` / ``dtype_strategy`` via ``from_strategy``). Anything the
router marks ``forkable_now=False`` (a deferred PASS/CODEGEN work-item), or a forkable axis with no
auto-knob builder yet, becomes a ``forkable=False`` proposal — recorded, never faked into a knob.

Drop-in for ``beam.run_beam(proposer=...)``: same ``(divergences, knobs) -> [ForkProposal]`` contract as
``rvv_knobs.propose_forks``, but consuming CCA Divergences instead of motif strings. (``rvv_knobs`` is
retained for the existing motif-string beam path until the beam is cut over to CCA divergences in WS-D.)
"""
from __future__ import annotations

from typing import Any

from ..kernels.action_catalog import CompilerAction, route
from ..kernels.cca_compare import Divergence
from ..kernels.rvv_knobs import ForkProposal, _wider_n_overrides


def _set_mr_overrides(knobs: dict, mr: int) -> dict:
    """Set the matmul register-block MR (the leading M tile dim) toward the expert's MR. M is the
    leading contraction dim: tile [M,N,K] (matmul) / [B,M,N,K] (batch), i.e. tile[-3]."""
    new = []
    for m in knobs.get("op_match", []):
        tile, vec = list(m["tile"]), list(m["vector"])
        if len(tile) >= 3:
            tile[-3] = mr
            vec[-3] = mr
        new.append({"op": m["op"], "tile": tile, "vector": vec})
    return {"op_match": new}


def action_to_fork(action: CompilerAction, knobs: dict[str, Any]) -> ForkProposal:
    """Map one typed CompilerAction to a ForkProposal, emitting only knob keys the schedule honors."""
    seam = action.target_seam
    intended = action.intended_facet or {}
    axis = action.divergence_axis
    ev = list(action.evidence)

    # 1) an impr_features hook -> a compiler_features override (forkable iff the router says so).
    if seam.startswith("impr_features:"):
        feat = seam.split(":", 1)[1].split()[0]
        return ForkProposal(overrides={"compiler_features": [feat]}, lever="feature",
                            targets=axis, evidence=ev, forkable=action.forkable_now, note=action.change)

    # 2) a forkable schedule KNOB/HEURISTIC -> a concrete, KNOWN knob override.
    if action.forkable_now and not seam.startswith("pass:"):
        overrides: dict | None = None
        if "int8_w8a8" in seam or intended.get("compute.widening"):
            overrides = {"dtype_strategy": "int8_w8a8"}
        elif axis == "vector.lmul":
            overrides = _wider_n_overrides(knobs, 2)
        elif axis == "compute.register_block" and isinstance(intended.get("compute.register_block"), int):
            overrides = _set_mr_overrides(knobs, intended["compute.register_block"])
        if overrides is not None:
            return ForkProposal(overrides=overrides, lever="knob", targets=axis, evidence=ev,
                                forkable=True, note=action.change)

    # 3) deferred PASS/CODEGEN, or a forkable axis with no auto-knob builder yet -> HONEST work-item.
    return ForkProposal(overrides={}, lever="work_item", targets=axis, evidence=ev,
                        forkable=False, note=action.change)


def propose_forks_from_cca(divergences: list[Divergence], knobs: dict[str, Any]) -> list[ForkProposal]:
    """Route each CCA Divergence via action_catalog and map to a ForkProposal. Unrouted divergences are
    skipped here (``build_catalog`` surfaces them as 'unrouted' separately — never silently dropped)."""
    out: list[ForkProposal] = []
    for d in divergences:
        a = route(d)
        if a is not None:
            out.append(action_to_fork(a, knobs))
    return out
