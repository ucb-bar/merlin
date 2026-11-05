"""Development priorities from complete, validated observations of one source.

This projection reports potential time savings, not reachable performance or
timing qualification. Measured whole-member cycles include host work; absent
host bounds and evaluation costs remain explicit unknowns.
"""

from __future__ import annotations

import math

from .contracts import StageGateError


def _positive(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _target(row):
    macs, rate = row.get("declared_macs"), row.get("achievable_macs_per_cycle")
    if _positive(macs) and _positive(rate):
        target = macs / rate
        if _positive(target):
            return target, "matched empirical throughput; attainable improvement unproven", False
    roofline = row.get("roofline") or {}
    floor = roofline.get("roofline_cycles")
    if roofline.get("status") == "derived" and _positive(floor):
        return floor, "hardware lower bound; gap is an upper bound on possible savings", True
    return None, "no matched throughput target or unrefuted hardware bound", False


def measured_priorities(members, weights):
    """Use the ranking objective's exact weights and every observed repeat.

    The caller owns feedback/source/corpus validation. A missing or inconsistent
    bound cannot remove a member's measured time from the objective. Costs are
    not inferred from guest cycles or wall time of an incomplete evaluation.
    """
    if (
        not members
        or set(members) != set(weights)
        or any(not _positive(value) for value in weights.values())
        or not math.isclose(sum(weights.values()), 1.0)
    ):
        raise StageGateError("search priorities require the exact complete ranking weights")
    rows = []
    for identity, repeats in sorted(members.items()):
        if not repeats or any(not _positive(row.get("candidate_gsim_cycles")) for row in repeats):
            raise StageGateError("search priorities require every complete measured repeat")
        weight = weights[identity]
        cycles = [row["candidate_gsim_cycles"] for row in repeats]
        targets = [_target(row) for row in repeats]
        target, basis, lower_bound = targets[0]
        if any(pair != targets[0] for pair in targets):
            target, basis = None, "repeat observations disagree on their bound or throughput target"
        if target is not None and lower_bound and target > min(cycles):
            target, basis = None, "observed timing refutes the supplied hardware bound"
        rows.append(
            {
                "family": identity[0],
                "capsule": identity[1],
                "weight": weight,
                "weighted_cycles": {"lo": weight * min(cycles), "hi": weight * max(cycles)},
                "target_cycles": target,
                "potential_weighted_savings": (
                    {"lo": weight * max(0, min(cycles) - target), "hi": weight * max(0, max(cycles) - target)}
                    if target is not None
                    else None
                ),
                "basis": basis,
                "evaluation_cost_seconds": None,
                "savings_per_evaluation_second": None,
            }
        )
    priced = [row for row in rows if row["potential_weighted_savings"] is not None]
    unknown = [row for row in rows if row["potential_weighted_savings"] is None]
    return {
        "ranked": sorted(
            priced, key=lambda row: (-row["potential_weighted_savings"]["lo"], row["family"], row["capsule"])
        ),
        "unknown": unknown,
        "weighted_cycles": {key: sum(row["weighted_cycles"][key] for row in rows) for key in ("lo", "hi")},
        "unpriced_weighted_cycles": {key: sum(row["weighted_cycles"][key] for row in unknown) for key in ("lo", "hi")},
        "evaluation_cost_status": "UNKNOWN: complete matched evaluation costs have not been observed",
        "scope": "complete generated roster; potential savings are guidance, never promotion or a stop proof",
    }
