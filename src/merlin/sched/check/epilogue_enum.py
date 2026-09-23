"""Exhaustive epilogue enumeration: where do two readouts disagree, over EVERY reachable accumulator?

An elementwise readout is a function of one integer accumulator value (for a fixed scale). The set of
values an accumulator can take is bounded by the reduction length and the operand ranges, and for the
shapes that matter it is small enough to enumerate outright (a K=4608 int8 reduction reaches about
1.5e8 values). So instead of sampling a stimulus -- which is how a readout that drops a sign or rounds
ties the wrong way has passed before -- this gate evaluates both readouts on every reachable value and
reports each one where they differ.

Uses: certify that a device readout matches a contract's golden at one site (0 flips), or measure how
far a device epilogue is from a host reference (the flip set IS the bound). Neither readout is trusted
over the other here; the report only says where they differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Readout = Callable[[np.ndarray, float], np.ndarray]


@dataclass(frozen=True)
class FlipReport:
    """The outcome of one exhaustive comparison."""

    lo: int
    hi: int
    scale: float
    evaluated: int
    flips: int
    #: Up to ``max_examples`` disagreements as ``(accumulator, value_a, value_b)``.
    examples: tuple[tuple[int, int, int], ...]

    @property
    def exact(self) -> bool:
        return self.flips == 0 and self.evaluated == self.hi - self.lo + 1


def reachable_accumulator_range(
    k: int, a_range: tuple[int, int], b_range: tuple[int, int], bias_range: tuple[int, int] = (0, 0)
) -> tuple[int, int]:
    """Bounds of ``sum_{i<k} a_i * b_i + bias`` for operands in the given inclusive ranges."""
    if k <= 0:
        raise ValueError("reduction length must be positive")
    corners = [a * b for a in a_range for b in b_range]
    return k * min(corners) + bias_range[0], k * max(corners) + bias_range[1]


def enumerate_readout_flips(
    readout_a: Readout,
    readout_b: Readout,
    *,
    scale: float,
    lo: int,
    hi: int,
    chunk: int = 1 << 22,
    max_examples: int = 16,
) -> FlipReport:
    """Evaluate both readouts on every integer in ``[lo, hi]`` and count disagreements."""
    if hi < lo:
        raise ValueError("empty accumulator range")
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    flips = 0
    evaluated = 0
    examples: list[tuple[int, int, int]] = []
    start = lo
    while start <= hi:
        stop = min(hi, start + chunk - 1)
        acc = np.arange(start, stop + 1, dtype=np.int64)
        va = np.asarray(readout_a(acc, scale)).astype(np.int64)
        vb = np.asarray(readout_b(acc, scale)).astype(np.int64)
        if va.shape != acc.shape or vb.shape != acc.shape:
            raise ValueError("a readout changed the shape of its input")
        diff = np.nonzero(va != vb)[0]
        flips += int(diff.size)
        for idx in diff[: max(0, max_examples - len(examples))]:
            examples.append((int(acc[idx]), int(va[idx]), int(vb[idx])))
        evaluated += acc.size
        start = stop + 1
    return FlipReport(lo=lo, hi=hi, scale=float(scale), evaluated=evaluated, flips=flips, examples=tuple(examples))
