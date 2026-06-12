"""Metric: layout (layout pressure).

Counts layout conversions implied by repacking the weight each use. In M1 this mirrors the
baseline pack count (each repack is a layout conversion); richer layout-ping-pong accounting
across heterogeneous consumers is M2.
"""
from __future__ import annotations

from merlin.design_pressure import region as R


def metric_layout(region: dict) -> dict:
    roles = R.classify_tensors(region)
    reuse = R.rhs_reuse_count(region)
    conversions = reuse if roles["rhs"] else 0
    return {
        "layout_conversions": conversions,
        "layout_convert_resident": 1 if roles["rhs"] else 0,
    }
