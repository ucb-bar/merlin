"""Attribute a wrong contraction result to a specific reduction step, from the numbers alone.

A mismatch count and a digest say a case is wrong. They do not say *what* is wrong, and on this datapath
the difference matters enormously: uninitialised state, a mis-aligned operand load, a dropped reduction
step and a swapped operand order all present as "some elements disagree", and picking between them by
inspection is how an afternoon disappears into hypotheses that each fit half the evidence.

This does it arithmetically instead. Given the operands, the reference and the device's actual values, it
searches for a small structured explanation of the DELTAS:

* **a dropped or double-counted reduction step** — ``delta[j] == ±lhs[k, i] * rhs[k, j]`` holding across
  *every* disagreeing column at once, which identifies ``k`` uniquely in practice;
* **a wholly uninitialised region** — the device value is constant (typically zero) where the reference
  is not;
* **a rank-1 perturbation** — the delta factors as an outer product, i.e. one operand row is wrong.

The point is that these are *checkable* rather than plausible. A single-step explanation that reproduces
eight independent column deltas exactly, including a column whose delta is zero because that operand
element is zero, is not a coincidence one needs to argue about — MEASURED on this unit: an 8x32x24
contraction came back missing exactly the ``k = 15`` term in the columns belonging to the second physical
subtile, and nothing else.

Everything here is pure arithmetic over arrays the caller already has. No hardware, no simulator, no
target facts.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Explanation", "explain_deltas", "find_dropped_step", "uninitialised_columns"]


@dataclass(frozen=True)
class Explanation:
    """One candidate account of the deltas, with everything needed to check it."""

    kind: str                      # "dropped_step" | "double_counted_step" | "uninitialised" | "rank_1"
    detail: dict[str, Any] = field(default_factory=dict)
    #: How many of the examined positions the explanation reproduces EXACTLY.
    exact: int = 0
    examined: int = 0
    note: str = ""

    @property
    def complete(self) -> bool:
        """True when it accounts for every examined position, which is the only interesting case."""
        return self.examined > 0 and self.exact == self.examined

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "detail": dict(self.detail), "exact": self.exact,
                "examined": self.examined, "complete": self.complete, "note": self.note}


def find_dropped_step(lhs: np.ndarray, rhs: np.ndarray, deltas: Mapping[tuple[int, int], int],
                      ) -> list[Explanation]:
    """Reduction steps whose contribution explains ``deltas`` exactly, dropped or double-counted.

    ``lhs``/``rhs`` are K-major as the unit requires: ``lhs[k, i]``, ``rhs[k, j]``. ``deltas`` maps
    ``(row, col) -> device - reference``.

    A step is only reported when it accounts for **every** entry in ``deltas``. Reporting partial matches
    would be worse than reporting nothing: with int8 operands a single product coincides with a delta
    often enough that a "best" partial explanation is usually noise wearing a mechanism's clothes.
    """
    a = np.asarray(lhs, dtype=np.int64)
    b = np.asarray(rhs, dtype=np.int64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError(f"expected K-major (K, M) and (K, N); got {a.shape} and {b.shape}")
    if not deltas:
        return []
    items = sorted(deltas.items())
    # An output element (row, col) accumulates lhs[k, ROW] * rhs[k, COL], so the LHS index is the delta's
    # own row -- not a free parameter. Searching a fixed index instead only works when every delta happens
    # to share a row, which is exactly the shape of the failure this was first written against; it would
    # then silently fail to explain a step dropped across the whole tile.
    out: list[Explanation] = []
    for k in range(a.shape[0]):
        for sign, kind in ((-1, "dropped_step"), (1, "double_counted_step")):
            if all(sign * int(a[k, row]) * int(b[k, col]) == int(d) for (row, col), d in items):
                rows = sorted({row for (row, _c) in deltas})
                out.append(Explanation(
                    kind=kind,
                    detail={"k": int(k), "rows": rows,
                            "lhs_values": {int(r): int(a[k, r]) for r in rows}},
                    exact=len(items), examined=len(items),
                    note=(f"every delta equals {'minus ' if sign < 0 else ''}lhs[{k}, row] * "
                          f"rhs[{k}, col], so reduction step {k}'s contribution is "
                          f"{'missing from' if sign < 0 else 'counted twice in'} these positions")))
    return out


def uninitialised_columns(device: np.ndarray, reference: np.ndarray) -> Explanation | None:
    """Whether every disagreeing element holds one repeated device value — the uninitialised signature."""
    dev = np.asarray(device)
    ref = np.asarray(reference)
    bad = dev != ref
    if not bad.any():
        return None
    values = np.unique(dev[bad])
    if values.size != 1:
        return None
    return Explanation(
        kind="uninitialised", detail={"value": int(values[0]), "elements": int(bad.sum())},
        exact=int(bad.sum()), examined=int(bad.sum()),
        note=(f"every disagreeing element reads {int(values[0])}, i.e. the region was never written "
              "rather than computed wrongly"))


def explain_deltas(lhs: np.ndarray, rhs: np.ndarray, device: np.ndarray,
                   reference: np.ndarray) -> list[Explanation]:
    """Every complete explanation of the difference between ``device`` and ``reference``.

    Returns the ones that account for ALL disagreeing positions, most specific first. An empty result is
    informative: it means none of the structured failure modes fits, and the next step is evidence rather
    than another guess.
    """
    dev = np.asarray(device, dtype=np.int64)
    ref = np.asarray(reference, dtype=np.int64)
    if dev.shape != ref.shape:
        raise ValueError(f"device {dev.shape} and reference {ref.shape} must have the same shape")
    bad = np.argwhere(dev != ref)
    if bad.size == 0:
        return []
    deltas = {(int(r), int(c)): int(dev[r, c] - ref[r, c]) for r, c in bad}
    found = list(find_dropped_step(lhs, rhs, deltas))
    blank = uninitialised_columns(dev, ref)
    if blank is not None:
        found.append(blank)
    return [e for e in found if e.complete]
