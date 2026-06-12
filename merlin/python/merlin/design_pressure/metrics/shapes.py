"""Metric: shapes (compute pressure + the op facts the policies key on).

Emits the matmul (M, K, N), the op kind, the dtype distribution, and whether a fused
epilogue is present. ``K`` is surfaced here but, per the design decision, is used only for
*policy endorsement* of ``resident_packed_tensor`` (the K>=256 mined condition), not for its
structural legality — see ``design_pressure/synthesize.py``.
"""
from __future__ import annotations

from collections import Counter

from merlin.design_pressure import region as R


def metric_shapes(region: dict) -> dict:
    mnk = R.mnk(region)
    ts = R.tensors(region)
    dtype_dist = dict(Counter(str(t.get("dtype", "unknown")).lower() for t in ts.values()))
    M, K, N = mnk["M"], mnk["K"], mnk["N"]
    macs = (M * K * N) if (M and K and N) else 0
    return {
        "op": R.contraction_op(region) or (region.get("ops") or ["unknown"])[0],
        "op_mix": dict(Counter(str(o).lower() for o in R.op_sequence(region))),
        "M": M,
        "K": K,
        "N": N,
        "macs": macs,
        "dtype_dist": dtype_dist,
        "has_epilogue": R.has_epilogue(region),
    }
