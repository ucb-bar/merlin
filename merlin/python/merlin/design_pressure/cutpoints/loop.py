"""Cut point: loop.

At the loop view the region's iteration structure (the action-chunk loop) is explicit, so
layout-conversion and lifetime pressure across iterations become visible.
"""
from __future__ import annotations

from merlin.design_pressure.metrics.layout import metric_layout
from merlin.design_pressure.metrics.lifetimes import metric_lifetimes


def cut_loop(region: dict) -> dict:
    """Metrics observable at the loop cutpoint."""
    out: dict = {}
    out.update(metric_layout(region))
    out.update(metric_lifetimes(region))
    return out
