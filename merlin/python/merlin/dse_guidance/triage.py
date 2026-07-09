"""DSE axis triage: rank axes by measured/trace-derived target-gap closure.

The question that drives the ranking:

    For a workload W and target T, if DSE optimizes axis X, how much of the measured/
    trace-derived target gap can X actually close?

Definitions::

    target_gap            = baseline_total - target_total
    intervention_benefit  = baseline_total - intervention_total   (= Σ component reductions)
    gap_closure           = intervention_benefit / target_gap
    priority_score        = gap_closure * confidence * legality / max(cost_tier, 1)

Edge handling, done explicitly rather than papered over:

  * No target supplied -> ``target_gap`` and ``gap_closure`` are ``None``; we report the axis's
    benefit as a *share of the baseline total* instead, and ``priority_score`` is ``None``.
  * ``target_gap <= 0`` (baseline already meets the target) -> no gap exists; ``gap_closure`` is
    ``None`` and we flag ``no_target_gap`` rather than inventing a score.
  * ``gap_closure`` can exceed 1 (an axis that closes more than the gap); the raw value is kept
    in ``gap_closure_raw`` and the score uses the value clamped to ``[0, 1]``.
"""
from __future__ import annotations

from merlin.dse_guidance import evidence as E
from merlin.dse_guidance.axes import AxisResult, evaluate_axes
from merlin.dse_guidance.baseline_cost import BaselineCost
from merlin.dse_guidance.representation import Representation

# Column order for axis_triage.csv (also the canonical row-field order).
TRIAGE_COLUMNS = [
    "workload", "representation", "axis", "family",
    "baseline_total", "target_total", "intervention_total",
    "gap_closure", "gap_closure_raw", "baseline_share",
    "affected_components", "evidence_type", "confidence", "legality",
    "cost_tier", "priority_score", "reason",
]


def _row(workload: str, representation: str, baseline_total: float,
         target_total: float | None, target_gap: float | None,
         ar: AxisResult) -> dict:
    confidence = E.confidence_for(ar.evidence_type)
    intervention_total = baseline_total - ar.benefit_ms
    baseline_share = (ar.benefit_ms / baseline_total) if baseline_total > 0 else None

    gap_closure_raw: float | None
    gap_closure: float | None
    priority_score: float | None
    note = ar.reason

    if ar.legality and not ar.quantified:
        # Structurally legal, but the benefit cannot be grounded -> no fabricated magnitude.
        gap_closure_raw = gap_closure = None
        priority_score = None
        baseline_share = None
    elif target_gap is None:
        gap_closure_raw = gap_closure = None
        priority_score = None
        if target_total is None:
            note = f"{ar.reason} (no target provided; reporting baseline share only)"
    elif target_gap <= 0:
        gap_closure_raw = gap_closure = None
        priority_score = None
        note = f"{ar.reason} (baseline already meets target; no gap to close)"
    else:
        gap_closure_raw = ar.benefit_ms / target_gap
        gap_closure = max(0.0, min(gap_closure_raw, 1.0))
        priority_score = gap_closure * confidence * ar.legality / max(ar.cost_tier, 1)

    return {
        "workload": workload,
        "representation": representation,
        "axis": ar.axis,
        "family": ar.family,
        "baseline_total": round(baseline_total, 6),
        "target_total": target_total,
        "intervention_total": round(intervention_total, 6),
        "gap_closure": None if gap_closure is None else round(gap_closure, 6),
        "gap_closure_raw": None if gap_closure_raw is None else round(gap_closure_raw, 6),
        "baseline_share": None if baseline_share is None else round(baseline_share, 6),
        "affected_components": ar.affected_components,
        "evidence_type": ar.evidence_type,
        "confidence": confidence,
        "legality": ar.legality,
        "cost_tier": ar.cost_tier,
        "priority_score": None if priority_score is None else round(priority_score, 6),
        "reason": note,
        "could_be_wrong_if": ar.could_be_wrong_if,
        "benefit_ms": round(ar.benefit_ms, 6),
    }


def _sort_key(row: dict):
    # Sort by priority_score desc; rows without a score (no gap) fall back to baseline share.
    ps = row["priority_score"]
    if ps is not None:
        return (1, ps)
    share = row["baseline_share"] or 0.0
    return (0, share)


def triage(representation: Representation, baseline: BaselineCost,
           coupling_per_replan: dict | None = None) -> dict:
    """Rank all axes for one representation. Returns a ``dse_axis_triage``-shaped dict."""
    baseline_total = baseline.baseline_total_ms
    target_total = baseline.target_total_ms
    target_gap = baseline.target_gap_ms

    results = evaluate_axes(representation.facts, baseline, coupling_per_replan)
    rows = [_row(baseline.workload, representation.name, baseline_total,
                 target_total, target_gap, ar) for ar in results]
    rows.sort(key=_sort_key, reverse=True)

    from merlin.common.schemas import validate_or_raise
    out = {
        "workload": baseline.workload,
        "representation": representation.name,
        "baseline_total_ms": baseline_total,
        "target_total_ms": target_total,
        "target_gap_ms": target_gap,
        "axes": rows,
    }
    validate_or_raise(out, "dse_axis_triage")   # schemas/ rule: if it lives here, code validates it
    return out
