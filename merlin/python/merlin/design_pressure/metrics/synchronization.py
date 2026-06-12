"""Metric: synchronization (overlap / sync pressure).

In M1 this reports only a boolean: whether DMA/compute overlap is beneficial (a reused
weight plus per-step activation transfer gives a copy/compute overlap window). Full
event-count / barrier / pipeline-bubble accounting needs the trace cutpoint and is M2.
"""
from __future__ import annotations

from merlin.design_pressure import region as R


def metric_synchronization(region: dict) -> dict:
    roles = R.classify_tensors(region)
    reuse = R.rhs_reuse_count(region)
    # Overlap is beneficial when there is repeated compute (reuse>1) with per-step input DMA
    # that can be staged ahead of compute.
    overlap_beneficial = bool(roles["lhs"]) and reuse > 1
    return {
        "dma_compute_overlap_beneficial": overlap_beneficial,
        # Placeholder until the trace cutpoint lands (M2).
        "sync_event_count": None,
    }
