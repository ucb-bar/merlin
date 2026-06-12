"""Cut point: linalg.

At the linalg view we can read op kinds, shapes, dtypes, reuse and lifetimes directly from
the region. This cutpoint therefore surfaces the shape/reuse/lifetime/packing/memory metrics.
"""
from __future__ import annotations

from merlin.design_pressure.metrics.lifetimes import metric_lifetimes
from merlin.design_pressure.metrics.memory import metric_memory
from merlin.design_pressure.metrics.packing import metric_packing
from merlin.design_pressure.metrics.reuse import metric_reuse
from merlin.design_pressure.metrics.shapes import metric_shapes


def cut_linalg(region: dict) -> dict:
    """Metrics observable at the linalg cutpoint."""
    out: dict = {}
    out.update(metric_shapes(region))
    out.update(metric_reuse(region))
    out.update(metric_lifetimes(region))
    out.update(metric_packing(region))
    out.update(metric_memory(region))
    return out
