"""The corpus as a SEARCH SPACE — axes, levels, observed cells, and the cells nobody tried.

A mined corpus is usually read as a set of good kernels. It is also a factorial design that somebody
already ran, and the interesting half is the part they did NOT run: an expert corpus is a sample of a
space, and the unobserved cells are where a compiler could look next.

The design is DERIVED from the corpus rather than declared, because a declared axis list goes stale the
moment someone adds a kernel. Two sources, in order of trust:

* the GENERATOR that produced the corpus, when there is one -- reading the script that emitted the grid
  gives the real axes and their intended levels, rather than a reconstruction from filenames;
* the kernel records themselves, whose shape/dtype/variant fields are the axes by construction.

⚠️ A sampled corpus is not a full grid, and an irregular sample is the normal case rather than an
error: a measured 16-cell sweep covered 32x32x{32,64,128,256}, 64x64x{32,64,128}, 128x{32,64,128},
96x96x96 and 160x160x160 -- deliberately uneven. So ``unobserved`` is computed against the CROSS
PRODUCT and reported with its size, never silently truncated: "here are the 200 cells nobody tried" is
a finding, and a top-N list of them presented as the whole is a lie by omission.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

__all__ = ["DesignSpace", "Cell", "space_from_records", "merge_ledger", "to_dict"]


@dataclass(frozen=True)
class Cell:
    """One point of the design, and what is known about it."""

    coords: tuple[tuple[str, Any], ...]      # (axis, level) pairs, sorted by axis
    observed: bool = False
    n_kernels: int = 0
    #: Measured outcomes for this cell, when a ledger supplied them. Absent is NOT zero.
    improved: int = 0
    regressed: int = 0
    failed: int = 0

    @property
    def key(self) -> str:
        return " ".join(f"{a}={v}" for a, v in self.coords)

    @property
    def attempted(self) -> int:
        return self.improved + self.regressed + self.failed


@dataclass
class DesignSpace:
    target: str
    axes: dict[str, tuple] = field(default_factory=dict)     # axis -> the levels seen, sorted
    cells: dict[str, Cell] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def observed(self) -> list[Cell]:
        return [c for c in self.cells.values() if c.observed]

    def unobserved(self) -> list[Cell]:
        """Every cell of the cross product nobody ran. Complete, never truncated."""
        if not self.axes:
            return []
        names = sorted(self.axes)
        out: list[Cell] = []
        for combo in itertools.product(*(self.axes[n] for n in names)):
            coords = tuple(zip(names, combo))
            key = " ".join(f"{a}={v}" for a, v in coords)
            if key not in self.cells:
                out.append(Cell(coords=coords))
        return out

    @property
    def coverage(self) -> float:
        """Observed cells / cross-product size. Low is informative, not a defect."""
        total = 1
        for levels in self.axes.values():
            total *= max(1, len(levels))
        return (len(self.observed) / total) if total else 0.0

    def to_dict(self) -> dict:
        un = self.unobserved()
        return {
            "target": self.target,
            "axes": {k: list(v) for k, v in sorted(self.axes.items())},
            "n_cells_observed": len(self.observed),
            "n_cells_unobserved": len(un),
            "coverage": round(self.coverage, 4),
            "observed": [{"cell": c.key, "n_kernels": c.n_kernels,
                          "improved": c.improved, "regressed": c.regressed, "failed": c.failed}
                         for c in sorted(self.observed, key=lambda c: c.key)],
            # Complete. A truncated list read as the whole space is how a sweep gets reported as
            # covered when most of it was never run.
            "unobserved": [c.key for c in sorted(un, key=lambda c: c.key)],
            "notes": list(self.notes),
        }


def _levels_of(records, axis: str) -> tuple:
    seen = []
    for r in records:
        v = r.get(axis) if isinstance(r, dict) else getattr(r, axis, None)
        if v is not None and v not in seen:
            seen.append(v)
    return tuple(sorted(seen, key=lambda x: (isinstance(x, str), x)))


def space_from_records(records, *, axes, target: str) -> DesignSpace:
    """Derive the design from corpus records over the named ``axes``.

    An axis whose records give it only ONE level is dropped with a note: a constant is not a dimension,
    and leaving it in inflates the cross product so coverage reads far worse than it is.
    """
    recs = list(records)
    space = DesignSpace(target=target)
    for axis in axes:
        levels = _levels_of(recs, axis)
        if len(levels) > 1:
            space.axes[axis] = levels
        elif levels:
            space.notes.append(f"axis {axis!r} is constant at {levels[0]!r} across the corpus — not a "
                               f"dimension here, so it is excluded from the cross product")
        else:
            space.notes.append(f"axis {axis!r} is absent from every record — nothing to vary")

    names = sorted(space.axes)
    for r in recs:
        get = (lambda k: r.get(k)) if isinstance(r, dict) else (lambda k: getattr(r, k, None))
        if any(get(n) is None for n in names):
            continue                      # a record that does not place in the design is not a cell
        coords = tuple((n, get(n)) for n in names)
        key = " ".join(f"{a}={v}" for a, v in coords)
        prev = space.cells.get(key)
        space.cells[key] = Cell(coords=coords, observed=True,
                                n_kernels=(prev.n_kernels if prev else 0) + 1)
    return space


def merge_ledger(space: DesignSpace, rows, *, axis_of=None) -> DesignSpace:
    """Fold a transform ledger's ATTEMPTS into the design — including the ones that failed.

    A mining loop that sees only winners over-proposes: the corpus records what an expert kept, and a
    ledger of attempts is the only record of what was tried and discarded. Measured on one target, a
    1509-attempt ledger holds 285 compile errors, 36 regressions and 50 improvements — so the base rate
    for a proposed transform is far below half, and a loop calibrated on winners alone will keep
    proposing what somebody already refuted.

    An attempt that lands on a cell the corpus never kept is still recorded, with ``observed=False``:
    "tried and rejected" and "never tried" are different states and the search should not confuse them.
    """
    place = axis_of or (lambda row: row.get("cell"))
    for row in rows:
        key = place(row)
        if not key:
            continue
        cur = space.cells.get(key)
        if cur is None:
            coords = tuple(tuple(part.split("=", 1)) for part in str(key).split() if "=" in part)
            cur = Cell(coords=coords, observed=False)
        outcome = str(row.get("outcome") or "").lower()
        space.cells[key] = Cell(
            coords=cur.coords, observed=cur.observed, n_kernels=cur.n_kernels,
            improved=cur.improved + (1 if outcome == "improved" else 0),
            regressed=cur.regressed + (1 if outcome == "regressed" else 0),
            failed=cur.failed + (1 if outcome in ("failed", "compile_error", "error") else 0))
    return space


def to_dict(space: DesignSpace) -> dict:
    return space.to_dict()


def candidates(space: DesignSpace, *, limit: int | None = None) -> list[dict]:
    """Unobserved cells as DSE candidates, nearest-to-observed first.

    Ordered by how many axes a cell shares with something already run: a cell one step from a known
    point is a cheaper and more interpretable probe than one that varies everything at once. When
    ``limit`` truncates the list, the caller is told how many were dropped -- a silently capped
    candidate list reads as "this is the space", which it is not.
    """
    obs = space.observed
    def _distance(c: Cell) -> int:
        if not obs:
            return 0
        return min(sum(1 for (a, v) in c.coords if dict(o.coords).get(a) != v) for o in obs)

    ranked = sorted(space.unobserved(), key=lambda c: (_distance(c), c.key))
    out = [{"cell": c.key, "coords": dict(c.coords), "steps_from_observed": _distance(c)}
           for c in ranked]
    if limit is not None and len(out) > limit:
        dropped = len(out) - limit
        out = out[:limit]
        out.append({"cell": f"... and {dropped} more unobserved cells not listed",
                    "coords": {}, "steps_from_observed": -1})
    return out
