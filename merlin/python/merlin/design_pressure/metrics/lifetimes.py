"""Metric: lifetimes (state pressure + accumulator-commit legality fact).

Determines whether a contraction's accumulator stays live across a fused epilogue (the fact
``accumulator_commit_policy`` keys on) and the per-step persistent-state footprint.
"""
from __future__ import annotations

from merlin.design_pressure import region as R


def metric_lifetimes(region: dict) -> dict:
    op = R.contraction_op(region)
    epilogue = R.has_epilogue(region)
    # The accumulator is live across the epilogue iff there is a contraction whose result is
    # consumed by an in-place bias/requant/activation chain before being committed.
    acc_live = bool(op) and epilogue

    ts = R.tensors(region)
    roles = R.classify_tensors(region)
    state_bytes = 0
    if roles["rhs"]:
        w = ts[roles["rhs"]]
        state_bytes = _prod(w.get("shape", [])) * R.dtype_bytes(w.get("dtype"))

    return {
        "accumulator_live_across_epilogue": acc_live,
        "state_bytes_per_step": state_bytes,
        "weight_immutable": not R.rhs_mutable(region),
    }


def _prod(xs) -> int:
    out = 1
    for x in xs or []:
        out *= int(x)
    return out
