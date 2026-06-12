"""Metric: reuse (state / persistence pressure).

Captures how strongly an immutable operand is reused across a region — the pressure that
justifies a ``resident_packed_tensor`` contract.
"""
from __future__ import annotations

from merlin.design_pressure import region as R


def metric_reuse(region: dict) -> dict:
    """Reuse facts for the weight/RHS of ``region``."""
    count = R.rhs_reuse_count(region)
    return {
        "rhs_reuse_count": count,
        "rhs_mutable": R.rhs_mutable(region),
        # In a flat action loop each step re-touches the same W, so the reuse distance in
        # dispatches equals the reuse count. A refined (interleaved) view is M2.
        "reuse_distance": count,
        "distinct_weights": R.distinct_weights(region),
    }
