"""What grading one capsule cost, per tier, from the grader's own timing block.

Each ``capsule_result.json`` records, per oracle tier, ``build_s`` / ``sim_active_s`` /
``oracle_wait_s`` / ``adapter_wall_s``. Those four are not interchangeable and three traps live in
the difference.

**A carried tier has no timing at all.** When the same executable was already certified at a tier,
the verdict is reused and ``timing`` is ``null`` -- deliberately, because copying a duration forward
would be a fabricated measurement. Those rows are recorded as carried and excluded from any cost
distribution rather than counted as zero.

**``adapter_wall_s`` is not the simulation cost everywhere.** In the performance lane the measurement
runs ahead of the loop in a prefetch wave, so the adapter call that stamps ``adapter_wall_s`` is only
READING an already-computed result: measured across the corpus, perf-lane L3 rows have
``sim_active_s`` p50 of 52.3 s against ``adapter_wall_s`` p50 of 0.026 s. In the functional lane the
two agree, because the adapter really did wait. So ``sim_active_s`` is the portable number, and
``adapter_wall_s`` is offered with a flag saying whether it is consistent with its own parts.

**A failing capsule is cheap, and mixing it with a passing one destroys the distribution.** A capsule
whose output comes back wrong aborts in hundredths of a second while a passing one simulates for tens
of seconds -- 0.01-0.5 s against 17-35 s on the same tier and engine. The median over both
populations is a number about the pass rate, not about cost. ``status`` is therefore carried on every
row so a caller can split, and :func:`summarize` refuses to pool them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

from merlin.agentreport.availability import (Availability, MEASURED, Status, measured,
                                             unavailable)

#: Fractional disagreement allowed between ``adapter_wall_s`` and the sum of its parts before the
#: wall figure is declared inconsistent (a prefetch wave makes it near-zero, not merely noisy).
_WALL_TOLERANCE = 0.5


@dataclass
class TierTiming:
    """One capsule at one tier: what it cost, whether it ran, and whether it passed."""

    capsule: str
    tier: str
    status: str = ""
    engine: str = ""
    build_s: float | None = None
    sim_active_s: float | None = None
    oracle_wait_s: float | None = None
    adapter_wall_s: float | None = None
    carried: bool = False
    measured_now: bool | None = None
    workers: int | None = None
    reason: str = ""

    @property
    def has_timing(self) -> bool:
        return self.sim_active_s is not None

    @property
    def active_s(self) -> float | None:
        """Work actually done: build plus simulation. Excludes queueing, which is not work."""
        if self.sim_active_s is None:
            return None
        return (self.build_s or 0.0) + self.sim_active_s

    @property
    def wall_is_consistent(self) -> bool | None:
        """Whether ``adapter_wall_s`` agrees with build + sim + wait.

        ``None`` when there is nothing to compare. ``False`` is the prefetch case, and a caller that
        sees it must not read ``adapter_wall_s`` as the cost of this measurement."""
        if self.adapter_wall_s is None or self.sim_active_s is None:
            return None
        parts = (self.build_s or 0.0) + self.sim_active_s + (self.oracle_wait_s or 0.0)
        if parts <= 0:
            return None
        return abs(self.adapter_wall_s - parts) / parts <= _WALL_TOLERANCE


def _num(value) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def read_capsule_timings(grading_dir: Path) -> list[TierTiming]:
    """Every per-tier timing under one grading directory."""
    out: list[TierTiming] = []
    for path in sorted(grading_dir.rglob("capsule_result.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except (ValueError, OSError):
            continue
        if not isinstance(doc, dict):
            continue
        capsule = str(doc.get("capsule") or path.parent.name)
        tiers = doc.get("tiers")
        if not isinstance(tiers, dict):
            continue
        for tier, entry in sorted(tiers.items()):
            if not isinstance(entry, dict):
                continue
            timing = entry.get("timing")
            conc = entry.get("concurrency")
            row = TierTiming(
                capsule=capsule, tier=str(tier), status=str(entry.get("status") or ""),
                engine=str(entry.get("engine") or ""),
                measured_now=entry.get("measured_now") if isinstance(entry.get("measured_now"), bool) else None,
                workers=int(conc["workers"]) if isinstance(conc, dict) and isinstance(conc.get("workers"), int) else None,
                reason=str(entry.get("reason") or "")[:300])
            if isinstance(timing, dict):
                row.build_s = _num(timing.get("build_s"))
                row.sim_active_s = _num(timing.get("sim_active_s"))
                row.oracle_wait_s = _num(timing.get("oracle_wait_s"))
                row.adapter_wall_s = _num(timing.get("adapter_wall_s"))
            else:
                # No timing block is the grader SAYING this tier was not executed here. The reason
                # text distinguishes a carried certificate from a tier that never ran at all.
                row.carried = "carried" in row.reason.lower()
            out.append(row)
    return out


@dataclass
class TierSummary:
    """The cost distribution for one (tier, status) population -- never pooled across statuses."""

    tier: str
    status: str
    n: int = 0
    n_carried: int = 0
    n_no_timing: int = 0
    median_active_s: float | None = None
    p90_active_s: float | None = None
    max_active_s: float | None = None
    total_active_s: float = 0.0
    wall_inconsistent: int = 0
    availability: Availability = field(default_factory=Availability)


def summarize(rows: Sequence[TierTiming], *, tier: str, status: str) -> TierSummary:
    """Cost distribution for one tier and one outcome.

    ``status`` is required, not optional, because pooling a passing capsule's simulation with a
    failing one's early abort produces a median that describes the pass rate rather than the cost."""
    picked = [r for r in rows if r.tier == tier and r.status == status]
    out = TierSummary(tier=tier, status=status, n=len(picked))
    out.n_carried = sum(1 for r in picked if r.carried)
    timed = [r for r in picked if r.has_timing]
    out.n_no_timing = len(picked) - len(timed)
    out.wall_inconsistent = sum(1 for r in timed if r.wall_is_consistent is False)
    if not timed:
        out.availability.set("tier_cost", unavailable(
            f"{len(picked)} capsule(s) at {tier} with status {status!r}, none carrying a timing block"
            + (f" ({out.n_carried} carried a verdict from an earlier grade, which records no duration"
               f" because copying one forward would fabricate a measurement)" if out.n_carried else "")))
        return out
    values = sorted(r.active_s for r in timed if r.active_s is not None)
    out.median_active_s = median(values)
    out.p90_active_s = values[min(int(0.9 * len(values)), len(values) - 1)]
    out.max_active_s = values[-1]
    out.total_active_s = sum(values)
    note = f"{len(timed)} of {len(picked)} capsule(s) timed"
    if out.n_carried:
        note += f"; {out.n_carried} carried"
    if out.wall_inconsistent:
        note += (f"; {out.wall_inconsistent} row(s) have an adapter wall that disagrees with their own "
                 f"parts (a prefetched measurement), so only sim/build time is used")
    out.availability.set("tier_cost", Status(MEASURED, reason=note,
                                             source="capsule_result.timing"))
    return out


def tiers_present(rows: Iterable[TierTiming]) -> list[str]:
    return sorted({r.tier for r in rows})


def statuses_present(rows: Iterable[TierTiming]) -> list[str]:
    return sorted({r.status for r in rows if r.status})
