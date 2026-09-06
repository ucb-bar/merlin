"""Capsules passing over time, from the agent's own self-check log.

``selfcheck_log.jsonl`` is the only in-run record of progress: one row per self-check the agent ran,
carrying ``wall_offset_s``, the tier it was graded at, ``n_passed`` / ``n_capsules``, and the names
still ``failing``. It is what makes "capsules passed over time" a measurement rather than a
reconstruction from round boundaries.

TWO HAZARDS, ONE OF WHICH IS NOT THE ONE YOU EXPECT.

``n_passed`` is often suspected of being unreliable. It is not: over all four run roots,
``n_passed == n_capsules - len(failing)`` in 40,335 of 40,335 rows that have a denominator. Zero
mismatches. The complement is kept here as an ASSERTION, not as a substitute.

The real hazard is the 873 rows that have **no denominator** -- ``n_capsules: 0``, ``n_passed: 0``,
``failing: []`` -- of which 368 are an outright ``build_failed``. Those are a self-check that could
not run. Plotted as written they put the curve on the floor, which reads as the agent destroying its
own work and is the single most misleading thing this data can be made to say. They are counted and
excluded, never charted.

``wall_offset_s`` RESETS at every round, so the raw column is a sawtooth. It is rebased onto one
monotone clock here, the way the round transcripts already are.

Regressions are NOT smoothed away at extraction. A score that genuinely went down is a finding about
the run; flattening it here would hide it and there would be no way to get it back. The count is
reported so a plot can decide, with the fact in hand, whether to draw a monotone envelope.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from merlin.agentreport.availability import (Availability, MEASURED, Status, derived,
                                             measured, unavailable)

#: Rows scoped to a subset of the corpus answer a different question than the full-suite ones and
#: cannot share an axis with them. The full-suite scope is the driver's own spelling.
SCOPE_ALL = "all"


@dataclass
class PassPoint:
    t_s: float
    n_passed: int
    n_capsules: int
    tier: str = ""
    sim: str = ""


@dataclass
class PassSeries:
    points: list[PassPoint] = field(default_factory=list)
    n_rows: int = 0
    n_no_denominator: int = 0
    n_build_failed: int = 0
    n_scoped_out: int = 0
    n_regressions: int = 0
    n_inconsistent: int = 0
    wall_s: float = 0.0
    availability: Availability = field(default_factory=Availability)

    @property
    def ok(self) -> bool:
        return bool(self.points)

    @property
    def best(self) -> tuple[int, int] | None:
        if not self.points:
            return None
        top = max(self.points, key=lambda p: p.n_passed)
        return top.n_passed, top.n_capsules

    def milestones(self) -> list[PassPoint]:
        """First time each new high-water mark was reached -- the step plot's risers."""
        out: list[PassPoint] = []
        best = -1
        for p in self.points:
            if p.n_passed > best:
                best = p.n_passed
                out.append(p)
        return out

    def envelope(self) -> list[PassPoint]:
        """Non-decreasing view, for a plot that wants one. The regressions are still counted."""
        out: list[PassPoint] = []
        best = -1
        for p in self.points:
            if p.n_passed >= best:
                best = p.n_passed
                out.append(p)
            else:
                out.append(PassPoint(p.t_s, best, p.n_capsules, p.tier, p.sim))
        return out


def _rows(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _rebase(offsets: list[float]) -> list[float]:
    """One monotone clock from a column that restarts at each round.

    A drop means a new round began, so everything after it is shifted by however far the previous
    round had got. Equal consecutive values are not a reset."""
    out: list[float] = []
    carry = 0.0
    prev: float | None = None
    for raw in offsets:
        if prev is not None and raw < prev:
            carry += prev
        out.append(raw + carry)
        prev = raw
    return out


def read_passes(run_dir: Path, *, scope: str = SCOPE_ALL) -> PassSeries:
    """The pass-over-time series for one run, or a stated reason there is none."""
    series = PassSeries()
    log = run_dir / "selfcheck_log.jsonl"
    if not log.is_file():
        return _from_verdicts(run_dir)

    rows = _rows(log)
    series.n_rows = len(rows)
    kept: list[tuple[float, dict]] = []
    for row in rows:
        if str(row.get("capsules") or SCOPE_ALL) != scope:
            series.n_scoped_out += 1
            continue
        total = row.get("n_capsules")
        if not isinstance(total, int) or total <= 0:
            series.n_no_denominator += 1
            if row.get("build_failed"):
                series.n_build_failed += 1
            continue
        offset = row.get("wall_offset_s")
        if not isinstance(offset, (int, float)):
            continue
        kept.append((float(offset), row))

    if not kept:
        series.availability.set("passes", unavailable(
            f"{series.n_rows} self-check row(s) read, but none carried both the {scope!r} scope and a "
            f"capsule count: {series.n_no_denominator} had no denominator "
            f"({series.n_build_failed} of them a failed build) and {series.n_scoped_out} were scoped "
            f"to a subset of the corpus"))
        return series

    clock = _rebase([off for off, _ in kept])
    prev_passed: int | None = None
    for t_s, (_, row) in zip(clock, kept):
        total = int(row["n_capsules"])
        passed = int(row.get("n_passed") or 0)
        failing = row.get("failing") or []
        # The consistency the corpus has always held. Counted rather than raised: one bad row must
        # not lose a ten-hour run, but a silent disagreement must not pass either.
        if isinstance(failing, list) and passed != total - len(failing):
            series.n_inconsistent += 1
        if prev_passed is not None and passed < prev_passed:
            series.n_regressions += 1
        prev_passed = passed
        series.points.append(PassPoint(t_s, passed, total,
                                       str(row.get("barrier_tier") or ""), str(row.get("sim") or "")))
    series.wall_s = series.points[-1].t_s if series.points else 0.0
    note = f"{len(series.points)} of {series.n_rows} self-check row(s) usable"
    if series.n_no_denominator:
        note += (f"; {series.n_no_denominator} had no capsule count and were excluded "
                 f"({series.n_build_failed} a failed build)")
    if series.n_inconsistent:
        note += f"; {series.n_inconsistent} row(s) disagreed with their own failing list"
    series.availability.set("passes", Status(MEASURED, reason=note, source="selfcheck_log"))
    return series


def _from_verdicts(run_dir: Path) -> PassSeries:
    """Fallback for the continuous schedule, which writes verdicts instead of a self-check log.

    A run graded by the background grader has no ``selfcheck_log.jsonl`` at all -- its progress record
    is the sequence of ``qa_history/verdict_*.json`` files the grader refreshes. Those files carry no
    timestamp field, so the only clock available is the file's own mtime.

    That makes this DERIVED, not measured, and the distinction is not pedantry: an mtime moves if the
    tree is copied, and several of these runs live in a worktree that has been copied. The series is
    still worth having -- it is the only progress record those runs kept -- but a reader has to be
    able to see that its x-axis rests on file metadata rather than on something the run wrote down.
    """
    series = PassSeries()
    history = run_dir / "qa_history"
    if not history.is_dir():
        series.availability.set("passes", unavailable(
            f"{run_dir.name} wrote neither selfcheck_log.jsonl nor a qa_history/ directory, so the "
            f"run kept no in-run progress record"))
        return series

    rows: list[tuple[float, int, int]] = []
    for f in sorted(history.glob("verdict*.json")):
        try:
            v = json.loads(f.read_text(encoding="utf-8", errors="ignore"))
        except (ValueError, OSError):
            continue
        if not isinstance(v, dict):
            continue
        total, passed = v.get("n_capsules"), v.get("n_passed")
        if not isinstance(total, int) or total <= 0 or not isinstance(passed, int):
            series.n_no_denominator += 1
            continue
        rows.append((f.stat().st_mtime, passed, total))
    series.n_rows = len(rows) + series.n_no_denominator
    if not rows:
        series.availability.set("passes", unavailable(
            f"{run_dir.name} has a qa_history/ but none of its verdicts carried a capsule count"))
        return series

    rows.sort()
    t0 = rows[0][0]
    prev: int | None = None
    for mtime, passed, total in rows:
        if prev is not None and passed < prev:
            series.n_regressions += 1
        prev = passed
        series.points.append(PassPoint(mtime - t0, passed, total))
    series.wall_s = series.points[-1].t_s
    series.availability.set("passes", derived(
        f"reconstructed from {len(rows)} qa_history verdict file(s); this run wrote no "
        f"selfcheck_log.jsonl, so the time axis is the verdict files' mtime, not a stamp the run "
        f"recorded. An mtime does not survive a tree copy.", source="qa_history_mtime"))
    return series
