"""Carry compiler observations into model results before admission decisions."""

from __future__ import annotations

from merlin.evaluation.capsule.lanes import lane_report
from merlin.evaluation.capsule.lanes_evidence import dispatch_boundary_report


def carry_model_evidence(capsule, result, out):
    if out.get("routing_plan") is not None:  # per-op mesh routing for the target
        result["routing_plan"] = out["routing_plan"]
    if out.get("coverage_certificate") is not None:  # ARR certificate (numerator×independent oracle)
        result["coverage_certificate"] = out["coverage_certificate"]
    if out.get("placement_census") is not None:  # whole-module population, including unmatched operations
        result["placement_census"] = out["placement_census"]
    # THE PLACEMENT SURFACE, carried into the verdict rather than left in the compile output. It runs in
    # SHADOW -- routing is the authority -- so this is reported, not gated. But two facts only it can
    # state were computed and read by nothing:
    #
    #   `n_emulated`  -- ops the host took in a format it cannot natively carry. The numeric gate hides
    #                    these completely: emulated arithmetic gets the right answer slowly, so a capsule
    #                    passes while the model runs on a datapath nobody chose.
    #   `divergence`  -- where placement and routing disagree. Flipping placement to authority is only
    #                    defensible once that set is empty, and until it is reported nobody can tell.
    if out.get("placement") is not None:
        _pl = out["placement"]
        result["placement"] = _pl
        _n_emul = _pl.get("n_emulated")
        if isinstance(_n_emul, int) and _n_emul > 0:
            result.setdefault("advisories", []).append(
                {
                    "plane": "placement",
                    "category": "EMULATED_ON_HOST",
                    "n": _n_emul,
                    "detail": f"{_n_emul} op(s) were placed on the host in a format it cannot natively "
                    f"carry; the numeric gate cannot see this, because emulated arithmetic is "
                    f"correct and merely slow",
                }
            )
    if out.get("mesh_execution") is not None:  # THIS MODEL's layers: on-mesh vs host
        result["mesh_execution"] = out["mesh_execution"]
    if out.get("mesh_tile_verification") is not None:  # synthesized tiles of the routed shapes
        result["mesh_tile_verification"] = out["mesh_tile_verification"]
    if result.get("mesh_execution") is not None:
        result["boundary_execution"] = dispatch_boundary_report(result["mesh_execution"])
        _lane = lane_report(capsule, result.get("routing_plan"), result["mesh_execution"])
        if _lane is not None:
            result["lane_report"] = _lane
