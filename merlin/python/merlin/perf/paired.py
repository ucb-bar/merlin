"""Confidence bounds for paired latency measurements.

The experimental unit is a session, not an individual inner-loop sample: samples from one
process share caches, allocator state, and board conditions.  Callers reduce each arm to one
session statistic before using this module, then pair sessions measured next to each other.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["paired_speedup_confidence", "shared_accuracy_verdict"]


def shared_accuracy_verdict(gate: dict[str, Any], bar: dict[str, Any]) -> dict[str, Any]:
    """Apply one FP32-relative accuracy bar to a complete output from either runtime.

    ``zephyr_model._gate`` also reports a stricter W8A8 reproduction tier.  That remains useful
    compiler diagnostics, but it is not comparable with an external runtime scored against FP32.
    Certification therefore reads only the explicitly named ``fp32_*`` fields and refuses a
    partial tensor rather than allowing a good prefix to stand for the model's answer.
    """
    cos = gate.get("fp32_cos")
    rel = gate.get("fp32_rel")
    cos_threshold = bar.get("cos_threshold")
    rel_threshold = bar.get("rel_threshold")
    result = {
        "basis": bar.get("basis"),
        "comparison_complete": gate.get("comparison_complete"),
        "cos": cos,
        "rel": rel,
        "cos_threshold": cos_threshold,
        "rel_threshold": rel_threshold,
        "cos_passes": None,
        "rel_passes": None,
        "passes": False,
        "reason": "",
    }
    if gate.get("comparison_complete") is not True:
        result["reason"] = "full-output comparison is not complete"
        return result
    if cos is None or rel is None:
        result["reason"] = "independent fp32 comparison is unavailable"
        return result
    if cos_threshold is None or rel_threshold is None:
        result["reason"] = "shared accuracy thresholds are unavailable"
        return result
    result["cos_passes"] = bool(float(cos) > float(cos_threshold))
    result["rel_passes"] = bool(float(rel) < float(rel_threshold))
    result["passes"] = bool(result["cos_passes"] and result["rel_passes"])
    failed = []
    if not result["cos_passes"]:
        failed.append(f"cosine {float(cos):.6g} <= {float(cos_threshold):.6g}")
    if not result["rel_passes"]:
        failed.append(f"relative error {float(rel):.6g} >= {float(rel_threshold):.6g}")
    result["reason"] = "pass" if not failed else "; ".join(failed)
    return result


def paired_speedup_confidence(
    ours_ns: Sequence[float],
    reference_ns: Sequence[float],
    *,
    confidence: float = 0.95,
    margin: float = 1.05,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Return a one-sided bootstrap bound for paired ``reference / ours`` speedup.

    Ratios are averaged in log space so exchanging the arms inverts the estimate and so a
    10% regression offsets a 10% improvement symmetrically.  The bootstrap resamples whole
    pairs, preserving the local board-condition cancellation that pairing was meant to obtain.
    """
    ours = np.asarray(tuple(ours_ns), dtype=np.float64)
    reference = np.asarray(tuple(reference_ns), dtype=np.float64)
    if ours.size != reference.size:
        raise ValueError("paired arms must contain the same number of session statistics")
    if ours.size < 2:
        raise ValueError("a confidence bound needs at least two paired sessions")
    if not np.isfinite(ours).all() or not np.isfinite(reference).all():
        raise ValueError("paired latency values must be finite")
    if (ours <= 0).any() or (reference <= 0).any():
        raise ValueError("paired latency values must be positive")
    if not 0.5 < float(confidence) < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    if float(margin) <= 0:
        raise ValueError("margin must be positive")
    if int(resamples) < 1:
        raise ValueError("resamples must be positive")

    log_ratios = np.log(reference / ours)
    observed = math.exp(float(np.mean(log_ratios)))
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, log_ratios.size, size=(int(resamples), log_ratios.size))
    boot = np.mean(log_ratios[indices], axis=1)
    alpha = 1.0 - float(confidence)
    lower = math.exp(float(np.quantile(boot, alpha)))
    upper = math.exp(float(np.quantile(boot, 1.0 - alpha)))
    return {
        "method": "paired_percentile_bootstrap_mean_log_speedup",
        "n_pairs": int(log_ratios.size),
        "confidence": float(confidence),
        "resamples": int(resamples),
        "seed": int(seed),
        "margin": float(margin),
        "geometric_speedup": observed,
        "lower_confidence_bound": lower,
        "upper_confidence_bound": upper,
        "passes": bool(lower >= float(margin)),
        "paired_speedups": [float(v) for v in np.exp(log_ratios)],
    }
