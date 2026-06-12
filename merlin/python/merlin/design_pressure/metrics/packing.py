"""Metric: packing (layout pressure).

Counts how often the immutable weight is packed and the bytes moved doing so. Repeated
packing of a reused immutable weight is the waste a ``resident_packed_tensor`` removes.
"""
from __future__ import annotations

from merlin.design_pressure import region as R


def metric_packing(region: dict) -> dict:
    ts = R.tensors(region)
    roles = R.classify_tensors(region)
    reuse = R.rhs_reuse_count(region)

    pack_bytes_once = 0
    if roles["rhs"]:
        w = ts[roles["rhs"]]
        pack_bytes_once = _prod(w.get("shape", [])) * R.dtype_bytes(w.get("dtype"))

    # Baseline packs the weight once per use (per reuse / per dispatch); residency packs once.
    pack_count_baseline = reuse if roles["rhs"] else 0
    return {
        "pack_bytes": pack_bytes_once,
        "pack_count_baseline": pack_count_baseline,
        "pack_count_resident": 1 if roles["rhs"] else 0,
        "pack_bytes_baseline": pack_bytes_once * pack_count_baseline,
    }


def _prod(xs) -> int:
    out = 1
    for x in xs or []:
        out *= int(x)
    return out
