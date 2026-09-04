"""Per-capsule performance verdict: is this member done, better, or still owed cycles?

The harness measures a cycle count for every capsule and computes its share of the achievable rate,
then discards both into agent-authored prose.  Nothing compares the number to anything, so "success"
per capsule has meant only "still correct, and the oracle produced a number".  A capsule at 3% of
what this machine has been shown to reach and one at 100% read identically.

This module supplies the missing comparison.  Every input is already derived elsewhere and is
carried in the tuning feedback document; nothing new is measured here, and no threshold is invented:

    ideal   = declared_macs / achievable_rate     cycles this work would take at the best rate
                                                  anything on this machine has actually reached
    share   = ideal / cycles                      1.0 means "at the achievable ceiling"
    closed  = (baseline - candidate) / (baseline - ideal)

WHY THE ACHIEVABLE RATE AND NOT THE STRUCTURAL PEAK.  The structural peak is what the array could
retire if nothing ever stalled; no measured program reaches it (31.3% on the target this was written
against), so scoring against it would mark every capsule a failure forever and rank none of them.
The achievable rate is falsified against every sample it was built from, so it is a rate something
demonstrably ran at.

THE TOLERANCE IS MEASURED, NOT DECLARED.  "At the ceiling" is `share >= 1 - dispersion`, where
dispersion is the spread of the achievable rate across the points that established it.  A capsule is
not called finished because a constant somebody chose says so.  Likewise "improved" is a strict
inequality beyond the oracle's own replicate dispersion -- which is zero for a deterministic
cycle-accurate simulator, so any real cycle saved counts and no saving is averaged away.

Fails closed: an underivable input yields REFUSED with the reason, never a substituted number.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

NO_HEADROOM = "no_headroom"
IMPROVED = "improved"
HEADROOM_OPEN = "headroom_open"
REGRESSED = "regressed"
REFUSED = "refused"


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def ceiling_dispersion(points: Sequence[Mapping[str, Any]]) -> float | None:
    """Spread of the achievable rate across the points that established it.

    The ceiling is `max(macs/cycles)` over measured points.  How tightly the best few cluster is a
    measurement of how repeatable that rate is, and it is what "at the ceiling" should tolerate.
    Returns None when fewer than two points price, so the caller refuses rather than assuming 0.
    """
    rates = [float(p["macs"]) / float(p["cycles"])
             for p in points
             if isinstance(p, Mapping) and _positive(p.get("macs")) and _positive(p.get("cycles"))]
    if len(rates) < 2:
        return None
    best = max(rates)
    if best <= 0:
        return None
    # dispersion of the TOP decile against the best: how much the fastest observations disagree.
    top = sorted(rates, reverse=True)[:max(2, len(rates) // 10)]
    return (best - min(top)) / best


def capsule_verdict(*, capsule: str, declared_macs: Any, achievable_rate: Any,
                    baseline_cycles: Any, candidate_cycles: Any,
                    dispersion: Any, replicate_dispersion: float = 0.0) -> dict[str, Any]:
    """Decide one capsule, from evidence the feedback document already carries."""
    row: dict[str, Any] = {"capsule": capsule}
    for name, value in (("declared_macs", declared_macs), ("achievable_rate", achievable_rate),
                        ("baseline_cycles", baseline_cycles)):
        if not _positive(value):
            return {**row, "verdict": REFUSED,
                    "reason": f"{name} is not a positive quantity, so no share can be derived"}
    if not _positive(dispersion) and dispersion != 0:
        return {**row, "verdict": REFUSED,
                "reason": ("the achievable rate's dispersion could not be measured, so "
                           "\"at the ceiling\" has no derived tolerance")}

    ideal = float(declared_macs) / float(achievable_rate)
    baseline_share = ideal / float(baseline_cycles)
    row.update({"ideal_cycles_at_achievable": ideal,
                "baseline_share_of_achievable": baseline_share,
                "dispersion": float(dispersion)})

    if baseline_share >= 1.0 - float(dispersion):
        return {**row, "verdict": NO_HEADROOM,
                "reason": (f"the baseline already runs at {baseline_share:.3f} of the achievable "
                           f"rate, within the measured dispersion {float(dispersion):.3f}; nothing "
                           f"on this machine has been shown to run this work faster")}

    factor = float(baseline_cycles) / ideal
    row["factor_to_achievable"] = factor
    if not _positive(candidate_cycles):
        return {**row, "verdict": HEADROOM_OPEN,
                "reason": (f"no candidate measurement; the baseline leaves {factor:.2f}x to the "
                           f"achievable rate")}

    row["candidate_share_of_achievable"] = ideal / float(candidate_cycles)
    saved = float(baseline_cycles) - float(candidate_cycles)
    gap = float(baseline_cycles) - ideal
    row["cycles_saved"] = saved
    if gap > 0:
        row["gap_closed"] = saved / gap
    if saved > float(replicate_dispersion):
        return {**row, "verdict": IMPROVED,
                "reason": (f"{saved:.0f} cycles saved, closing {row.get('gap_closed', 0.0):.1%} of "
                           f"the gap to the achievable rate")}
    if saved < -float(replicate_dispersion):
        return {**row, "verdict": REGRESSED,
                "reason": f"the candidate spends {-saved:.0f} more cycles than the baseline"}
    return {**row, "verdict": HEADROOM_OPEN,
            "reason": (f"no cycle change beyond the oracle's replicate dispersion, and "
                       f"{factor:.2f}x remains to the achievable rate")}


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Roll per-capsule verdicts up without letting a refusal read as a pass."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("verdict"))] = counts.get(str(row.get("verdict")), 0) + 1
    decided = [r for r in rows if r.get("verdict") in (NO_HEADROOM, IMPROVED, HEADROOM_OPEN, REGRESSED)]
    return {"n_capsules": len(rows), "by_verdict": counts, "n_decided": len(decided),
            "n_refused": counts.get(REFUSED, 0),
            "worst_first": [r.get("capsule") for r in
                            sorted((r for r in decided if "factor_to_achievable" in r),
                                   key=lambda r: -float(r["factor_to_achievable"]))][:10]}
