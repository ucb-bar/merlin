"""What was tried, which instrument found it, whether it helped, and how the metrics moved.

WHY THIS EXISTS. A phase-2 optimization campaign is a search, and a search that does not record its
own history repeats itself and cannot show progress. The previous campaign here recorded
``probe_receipts: []``, ``timing_status: UNMEASURED_FULL_MODEL`` and ``global_speedup_proven: False``
while emitting a candidate byte-identical to its baseline -- and none of that was visible until
someone read the receipts by hand. This module is the ledger that makes a campaign's own history a
first-class artifact: one row per attempt, carrying the mechanism, the SCOPE it acts at, the
INSTRUMENT that surfaced it, the verdict, and the measured deltas that justify the verdict.

FIVE VERDICTS, AND ``unmeasured`` IS NOT ``no_effect``. This tree has a recurring bug class in which
something that could not be measured was recorded as a measured zero. So a candidate whose effect was
never measured is ``unmeasured``, never ``no_effect``, and a ``refuted`` attempt -- one measurement
said it does not work -- is kept, because the most expensive thing a search can do is re-walk a dead
branch. ``blocked`` records an attempt that cannot proceed and WHY, which is how an upstream
dependency stops looking like a compiler deficiency.

EVIDENCE IS REQUIRED FOR A POSITIVE VERDICT. ``helped`` and ``refuted`` both assert a measurement, so
both must name the instrument and carry at least one delta; :meth:`Ledger.problems` reports any row
that claims one without them rather than letting an unevidenced claim sit in a report looking like
the evidenced ones.

WHAT IT DOES NOT DO. It does not rank candidates and it does not decide bound-ness on its own: a
metric is whatever the caller measured, and :func:`arithmetic_intensity` reports MACs per byte and
refuses the compute-bound/memory-bound verdict unless the caller supplies a MEASURED machine balance.
An intensity is derivable from a command buffer; a ridge point is a property of hardware and is not
invented here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["SCOPES", "VERDICTS", "Delta", "Attempt", "Ledger", "arithmetic_intensity"]

#: Where an optimization acts. Recorded because the cheap wins and the structural wins live at
#: different scopes, and a campaign that only ever finds one scope's worth is not done.
SCOPES = (
    "global",          # whole-graph: placement, offload fraction, what reaches the unit at all
    "inter_layer",     # between layers: epilogue fusion, residency across a block, layout handoff
    "local",           # inside one region: loop shape, index strength reduction, tiling
    "encoding",        # instruction selection and packing: which command family, which fields
    "transformation",  # semantics-preserving rewrites: quantization, layout change, reassociation
    "host_lane",       # code quality on the scalar lane
    "build",           # toolchain flags and pipeline order, no compiler change
    "frontend",        # what the compiler can ingest at all
)

#: A verdict asserts what a MEASUREMENT said, or that there wasn't one.
VERDICTS = (
    "helped",      # measured better on at least one workload, with the delta recorded
    "no_effect",   # measured, and the metric did not move
    "refuted",     # measured worse, or rejected outright by hardware/a gate
    "blocked",     # cannot proceed; `blocked_by` says what stops it
    "unmeasured",  # tried or proposed, effect never measured -- NOT the same as no_effect
)

_ASSERTS_MEASUREMENT = frozenset({"helped", "no_effect", "refuted"})


@dataclass(frozen=True)
class Delta:
    """One measured change: a metric on a workload, before and after, by an instrument."""

    workload: str
    metric: str
    before: float | None
    after: float | None
    instrument: str
    unit: str = ""
    note: str = ""
    #: Whether a SMALLER value of this metric is better. True for a cost (cycles, instructions,
    #: bytes); False for a coverage figure (routed MACs, offload fraction). A corrected MEASUREMENT
    #: is neither -- when a metric moved because it used to be wrong, no ratio is meaningful, and
    #: `lower_is_better=None` says so rather than reporting a 0.001x "regression" for a bug fix.
    lower_is_better: bool | None = True

    @property
    def ratio(self) -> float | None:
        """Improvement factor, or None when the metric admits no such reading.

        For a cost metric this is ``before / after``, so >1 means the cost fell. For a coverage
        metric it is inverted, so >1 still means better. For a corrected measurement
        (``lower_is_better is None``) there is no improvement factor at all: the two numbers describe
        different beliefs about the same program, not two programs.
        """
        if self.lower_is_better is None:
            return None
        if not self.before or not self.after or self.before <= 0 or self.after <= 0:
            return None
        return (self.before / self.after) if self.lower_is_better else (self.after / self.before)

    def to_dict(self) -> dict[str, Any]:
        return {"workload": self.workload, "metric": self.metric, "before": self.before,
                "after": self.after, "ratio": self.ratio, "instrument": self.instrument,
                "unit": self.unit, "note": self.note,
                "lower_is_better": self.lower_is_better}


@dataclass(frozen=True)
class Attempt:
    """One thing tried, with the instrument that found it and the evidence for its verdict."""

    mechanism: str
    scope: str
    found_by: str                     # the instrument that surfaced the opportunity
    verdict: str
    hypothesis: str = ""
    deltas: tuple[Delta, ...] = ()
    blocked_by: str = ""
    evidence: str = ""                # where a reader can check it: a path, a commit, a receipt
    iteration: int | None = None

    def problems(self) -> tuple[str, ...]:
        """Why this row may not be reported as it stands. Empty when it is sound."""
        out: list[str] = []
        if self.scope not in SCOPES:
            out.append(f"scope {self.scope!r} is not one of {list(SCOPES)}")
        if self.verdict not in VERDICTS:
            out.append(f"verdict {self.verdict!r} is not one of {list(VERDICTS)}")
        if self.verdict in _ASSERTS_MEASUREMENT:
            if not self.deltas:
                out.append(f"verdict {self.verdict!r} asserts a measurement but carries no delta")
            if not self.found_by:
                out.append(f"verdict {self.verdict!r} asserts a measurement but names no instrument")
        if self.verdict == "blocked" and not self.blocked_by:
            out.append("verdict 'blocked' must say what blocks it")
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {"mechanism": self.mechanism, "scope": self.scope, "found_by": self.found_by,
                "verdict": self.verdict, "hypothesis": self.hypothesis,
                "blocked_by": self.blocked_by, "evidence": self.evidence,
                "iteration": self.iteration, "deltas": [d.to_dict() for d in self.deltas],
                "problems": list(self.problems())}


@dataclass
class Ledger:
    """The campaign's own history, appended to as it runs."""

    target: str
    attempts: list[Attempt] = field(default_factory=list)

    def add(self, attempt: Attempt) -> Attempt:
        self.attempts.append(attempt)
        return attempt

    def problems(self) -> tuple[str, ...]:
        out: list[str] = []
        for index, attempt in enumerate(self.attempts):
            out.extend(f"attempt {index} ({attempt.mechanism}): {why}"
                       for why in attempt.problems())
        return tuple(out)

    def by_verdict(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for attempt in self.attempts:
            counts[attempt.verdict] = counts.get(attempt.verdict, 0) + 1
        return counts

    def by_scope(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for attempt in self.attempts:
            counts[attempt.scope] = counts.get(attempt.scope, 0) + 1
        return counts

    def instruments(self) -> dict[str, int]:
        """Which instrument surfaced how many attempts -- the campaign's own tool audit."""
        counts: dict[str, int] = {}
        for attempt in self.attempts:
            if attempt.found_by:
                counts[attempt.found_by] = counts.get(attempt.found_by, 0) + 1
        return counts

    def series(self, metric: str, workload: str) -> tuple[tuple[int | None, float], ...]:
        """``(iteration, after)`` for one metric on one workload, in recorded order.

        The series is how a campaign shows advance or plateau. Rows whose iteration is unset are
        kept in append order rather than dropped -- an unnumbered measurement is still a measurement.
        """
        out: list[tuple[int | None, float]] = []
        for attempt in self.attempts:
            for delta in attempt.deltas:
                if delta.workload == workload and delta.metric == metric and delta.after is not None:
                    out.append((attempt.iteration, float(delta.after)))
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "merlin_optimization_ledger_v1", "target": self.target,
                "n_attempts": len(self.attempts), "by_verdict": self.by_verdict(),
                "by_scope": self.by_scope(), "instruments": self.instruments(),
                "problems": list(self.problems()),
                "attempts": [a.to_dict() for a in self.attempts]}

    def format_table(self) -> str:
        rows = [f"== optimization ledger: {self.target} ({len(self.attempts)} attempts)",
                f"{'verdict':11} {'scope':14} {'found_by':22} mechanism"]
        order = {v: i for i, v in enumerate(VERDICTS)}
        for attempt in sorted(self.attempts, key=lambda a: order.get(a.verdict, 99)):
            rows.append(f"{attempt.verdict:11} {attempt.scope:14} {attempt.found_by:22} "
                        f"{attempt.mechanism}")
            for delta in attempt.deltas:
                ratio = "" if delta.ratio is None else f"  ({delta.ratio:.3f}x)"
                rows.append(f"{'':11} {'':14} {'':22}   {delta.workload}: {delta.metric} "
                            f"{delta.before} -> {delta.after}{ratio} [{delta.instrument}]")
            if attempt.blocked_by:
                rows.append(f"{'':11} {'':14} {'':22}   BLOCKED BY: {attempt.blocked_by}")
        for why in self.problems():
            rows.append(f"  PROBLEM {why}")
        return "\n".join(rows)


def arithmetic_intensity(routed_macs: int, traffic_bytes: int, *,
                         machine_macs_per_byte: float | None = None) -> dict[str, Any]:
    """MACs per byte for one program, and the bound-ness verdict only if the machine balance is given.

    Intensity is derivable from an emitted program. The RIDGE POINT -- the intensity at which a
    machine stops being memory-bound and starts being compute-bound -- is a property of hardware
    (peak arithmetic rate over achievable bandwidth) and is NOT invented here: without a measured
    ``machine_macs_per_byte`` the verdict is UNKNOWN, because a roofline drawn through a guessed
    ridge would point optimization effort at whichever axis the guess favoured.
    """
    if traffic_bytes <= 0 or routed_macs < 0:
        return {"status": "unavailable", "reason": "a program with no priced traffic has no intensity"}
    intensity = routed_macs / float(traffic_bytes)
    out: dict[str, Any] = {"status": "derived", "macs_per_byte": intensity,
                           "routed_macs": routed_macs, "traffic_bytes": traffic_bytes,
                           "licence": "intensity is derived from the emitted program; the ridge "
                                      "point is a measured machine property and is not assumed"}
    if not machine_macs_per_byte or machine_macs_per_byte <= 0:
        out["bound_by"] = "UNKNOWN"
        out["reason"] = ("no measured machine balance (peak MACs per achievable byte) was supplied, "
                         "and one is not invented here")
        return out
    out["machine_macs_per_byte"] = float(machine_macs_per_byte)
    out["bound_by"] = "compute" if intensity >= machine_macs_per_byte else "memory"
    return out
