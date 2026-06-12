"""Metric: dispatch (dispatch pressure).

Estimates how many command dispatches the region issues and how much useful work each
carries. A batch-1 action loop split into many tiny dependent dispatches is launch-overhead
dominated — the pressure that motivates command batching / persistent regions (M2 contracts).
"""
from __future__ import annotations

from merlin.design_pressure import region as R


def metric_dispatch(region: dict) -> dict:
    reuse = R.rhs_reuse_count(region)
    seq_len = max(len(R.op_sequence(region)), 1)
    mnk = R.mnk(region)
    M, K, N = mnk["M"] or 0, mnk["K"] or 0, mnk["N"] or 0
    macs = M * K * N

    # One dispatch per op per step (the unfused, per-op baseline granularity).
    steps = max(reuse, 1)
    dispatch_count = seq_len * steps
    work_per_dispatch = (macs // seq_len) if seq_len else macs
    return {
        "dispatch_count": dispatch_count,
        "work_per_dispatch": work_per_dispatch,
        "steps": steps,
    }
