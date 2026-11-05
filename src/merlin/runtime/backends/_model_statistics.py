"""Output hashes and steady-state summaries for model execution records."""

from __future__ import annotations

from typing import Any
import numpy as np


def out_hash_fnv1a32(a: np.ndarray) -> int:
    """The digest the generated harness prints, recomputed host-side over the same bytes.

    Kept beside the C that emits it so the two cannot drift apart: the image hashes
    ``MERLIN_OUT_ELEMS * 4`` bytes of its little-endian f32 output buffer in memory order, and so
    does this. It answers one question -- is the WHOLE tensor bit-identical -- for about twenty
    bytes of console, where dumping the tensor itself costs tens of kilobytes and a run cap.
    """
    b = np.ascontiguousarray(a, dtype="<f4").tobytes()
    h = 2166136261
    for byte in b:
        h = ((h ^ byte) * 16777619) & 0xFFFFFFFF
    return h


def _sustained_stats(cycles: list[int]) -> dict[str, Any]:
    """Steady-state summary of a sustained run.

    Reports the MEDIAN as the headline: the K1 measurement noise floor is >= 1.9%, so a
    single sample (or a mean skewed by one outlier) is not a result. ``drift`` compares the
    median of the last third against the first third — a growing per-iteration cost is the
    signature of arena growth / allocator churn across invocations, which a single-shot run
    cannot see at all.
    """
    xs = sorted(cycles)
    n = len(xs)

    def _pct(p: float) -> int:
        if n == 1:
            return xs[0]
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return xs[idx]

    med = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) // 2
    third = max(1, n // 3)
    early = sorted(cycles[:third])[third // 2]
    late = sorted(cycles[-third:])[third // 2]
    return {
        "n": n,
        "min": xs[0],
        "median": med,
        "p95": _pct(0.95),
        "max": xs[-1],
        "early_median": early,
        "late_median": late,
        "drift": (late - early) / early if early else 0.0,
    }
