"""Optional stopping goals over the coordinator's complete measured incumbent.

This arithmetic projection issues no timing, numerical or final qualification.
The coordinator supplies the incumbent from its live original evaluation pool;
partial diagnostics and predictions never enter that ranking.
"""

from __future__ import annotations

from .roofline_priority import CONVERGED


def progress(incumbent, *, maximum_ratio=None, roofline_attainment=None):
    """Require every configured goal, retaining the original worst repeat ranges.

    Ratio uses the selection's exact family/member weights. Roofline attainment
    instead requires every member independently, through roofline_priority's
    existing complete-floor, unresolved-term and subfloor-refutation checks.
    """
    configured = maximum_ratio is not None or roofline_attainment is not None
    result = {
        "status": "not_configured" if not configured else "unknown",
        "maximum_ratio": maximum_ratio,
        "roofline_attainment_min": roofline_attainment,
        "source_sha256": None if incumbent is None else incumbent["source_sha256"],
        "conditions": {},
        "scope": "complete measured development roster; no whole-model or final acceptance claim",
    }
    if not configured:
        return result
    if incumbent is None:
        result["reason"] = "incumbent has no complete correct measured observation"
        return result
    conditions = result["conditions"]
    if maximum_ratio is not None:
        worst = incumbent["ratio_hi"]
        conditions["cycle_ratio"] = {"worst": worst, "reached": worst <= maximum_ratio}
    if roofline_attainment is not None:
        roofline = incumbent.get("roofline_progress")
        conditions["roofline"] = {
            "status": None if roofline is None else roofline["status"],
            "reached": bool(roofline and roofline["status"] == CONVERGED),
            "completion_blocked": not roofline or roofline["completion_blocked"],
        }
    if all(row["reached"] for row in conditions.values()):
        result["status"] = "reached"
    elif any(row.get("completion_blocked") for row in conditions.values()):
        result["status"] = "unknown"
    else:
        result["status"] = "continue"
    return result
