"""Cut point: dispatch.

At the dispatch view each op becomes a command; dispatch counts, per-dispatch work, and
DMA/compute overlap pressure are observable here.
"""
from __future__ import annotations

from merlin.design_pressure.metrics.dispatch import metric_dispatch
from merlin.design_pressure.metrics.synchronization import metric_synchronization


def cut_dispatch(region: dict) -> dict:
    """Metrics observable at the dispatch cutpoint."""
    out: dict = {}
    out.update(metric_dispatch(region))
    out.update(metric_synchronization(region))
    return out
