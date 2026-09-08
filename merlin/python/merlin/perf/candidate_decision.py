"""Which of two emitted candidates is better, decided by a NAMED instrument or not at all.

WHY THIS EXISTS OUTSIDE THE CAMPAIGN DRIVER. The phase-2 global driver is an authoring and
structural-validation loop: ``global_speedup_proven`` is written 22 times and set ``True`` zero
times, ``global_cost_validated`` 9 times and never ``True`` (asserted ``False`` at 3 further sites),
``full_model_simulation_allowed`` 10 times and never ``True`` (asserted at 7). *"It proved a global
speedup"* is not a state that code has, and ``probe_relevance`` says so in its own words -- *"this
function never promotes a local sample into a global performance verdict."* So a comparison verdict
cannot come from inside it, and forcing one there would mean inverting sixteen assertion sites.

THE TWO WAYS A COMPARISON HAS ALREADY LIED HERE, both of which this module refuses by name:

**Identical emitted bytes.** The last campaign's own receipt recorded
``command_buffer_identical: True`` and ``lowered_identical: True`` beside a performance hypothesis:
the candidate emitted the same program as its baseline. Any metric difference between two identical
programs is noise or instrument drift, and :func:`compare` reports ``IDENTICAL_EMISSION`` rather
than a ratio.

**A metric that cannot see the change.** A 1.642x improvement was measured on host instruction count
across arms whose traffic was byte-identical (169.52 MiB on every one). The number was real and it
answered a question nobody asked, because the axis under optimization was data movement. So a
verdict names the axes that MOVED, and an axis that did not move cannot carry the verdict.

WHAT IT WILL NEVER DO. It does not certify from a band. Two bands that overlap are ``UNKNOWN`` --
not a tie, not a small win -- because ``rate_table.holdout_containment``'s own licence is *"a band
may eliminate a candidate and may never certify one"*. And it does not rank on a proxy whose
degradation correlates with success: Spike prices every accelerator command at one cycle, so it
systematically flatters offloading (measured: a retarget that nearly doubled mesh instructions moved
the Spike metric 0.2%), which is why :class:`Metric` carries ``blind_to``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Axis", "Metric", "CandidateFacts", "Decision", "compare", "AXES"]

#: The axes a compiler change can move, declared so a verdict can say which one it is about. An
#: instrument blind to the axis under change cannot decide, however precise it is on its own axis.
AXES: tuple[str, ...] = (
    "emitted_program",   # the bytes themselves: opcodes, operands, attributes
    "offload",           # how much work reaches the unit at all
    "traffic",           # bytes moved
    "host_work",         # scalar-lane instructions or cycles
    "accelerator_time",  # cycles the unit was busy
    "wall_cycles",       # the whole measured window
)


@dataclass(frozen=True)
class Axis:
    """One axis, and whether the two candidates actually differ along it."""

    name: str
    moved: bool
    detail: str = ""

    def __post_init__(self) -> None:
        if self.name not in AXES:
            raise ValueError(f"axis {self.name!r} is not one of {list(AXES)}")


@dataclass(frozen=True)
class Metric:
    """One instrument's reading of both arms, plus what it is blind to."""

    name: str
    instrument: str
    baseline: float
    candidate: float
    unit: str = ""
    lower_is_better: bool = True
    #: Axes this instrument cannot see. A metric blind to every axis that moved is REPORTED and
    #: excluded from the verdict rather than dropped: "the instrument could not see the change" and
    #: "the change did nothing" are different findings and only this field separates them.
    blind_to: tuple[str, ...] = ()
    #: A band reading is an INTERVAL, not a point. Supplied as (lo, hi) per arm when the instrument
    #: is a band; a verdict is then only reachable when the two intervals are disjoint.
    baseline_interval: tuple[float, float] | None = None
    candidate_interval: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        for axis in self.blind_to:
            if axis not in AXES:
                raise ValueError(f"blind_to axis {axis!r} is not one of {list(AXES)}")

    @property
    def ratio(self) -> float | None:
        if self.baseline <= 0 or self.candidate <= 0:
            return None
        return ((self.baseline / self.candidate) if self.lower_is_better
                else (self.candidate / self.baseline))

    @property
    def is_band(self) -> bool:
        return self.baseline_interval is not None and self.candidate_interval is not None

    def disjoint(self) -> bool | None:
        """For a band metric: are the two intervals disjoint? None when it is not a band."""
        if not self.is_band:
            return None
        b_lo, b_hi = self.baseline_interval          # type: ignore[misc]
        c_lo, c_hi = self.candidate_interval         # type: ignore[misc]
        return c_hi < b_lo or b_hi < c_lo

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "instrument": self.instrument, "baseline": self.baseline,
                "candidate": self.candidate, "unit": self.unit, "ratio": self.ratio,
                "lower_is_better": self.lower_is_better, "blind_to": list(self.blind_to),
                "is_band": self.is_band, "disjoint": self.disjoint()}


@dataclass(frozen=True)
class CandidateFacts:
    """What is known about one arm's emitted program, independent of any comparison."""

    name: str
    #: Digest of the emitted command buffer. Two arms sharing it emitted the SAME program.
    command_buffer_sha256: str = ""
    lowered_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "command_buffer_sha256": self.command_buffer_sha256,
                "lowered_sha256": self.lowered_sha256}


#: The verdicts. Each names a DIFFERENT reason a comparison ended where it did.
BETTER = "BETTER"
WORSE = "WORSE"
NO_EFFECT = "NO_EFFECT"
UNKNOWN = "UNKNOWN"
IDENTICAL_EMISSION = "IDENTICAL_EMISSION"
BLIND = "BLIND"

VERDICTS = (BETTER, WORSE, NO_EFFECT, UNKNOWN, IDENTICAL_EMISSION, BLIND)


@dataclass
class Decision:
    """The verdict, the instrument that reached it, and everything that did not."""

    verdict: str
    why: str
    decided_by: str = ""
    axes: tuple[Axis, ...] = ()
    metrics: tuple[Metric, ...] = ()
    excluded: list[dict[str, Any]] = field(default_factory=list)

    @property
    def moved_axes(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.axes if a.moved)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "merlin_candidate_decision_v1", "verdict": self.verdict, "why": self.why,
                "decided_by": self.decided_by, "moved_axes": list(self.moved_axes),
                "axes": [{"name": a.name, "moved": a.moved, "detail": a.detail} for a in self.axes],
                "metrics": [m.to_dict() for m in self.metrics], "excluded": list(self.excluded),
                "licence": ("a band may eliminate a candidate and may never certify one; an "
                            "instrument blind to every axis that moved cannot decide, however "
                            "precise it is on its own axis")}


def compare(baseline: CandidateFacts, candidate: CandidateFacts, *,
            axes: Sequence[Axis], metrics: Sequence[Metric]) -> Decision:
    """Decide between two arms, or refuse and name what stopped the decision.

    The order of checks is the order in which a comparison goes wrong in practice, and each one
    returns a DIFFERENT verdict rather than collapsing into "no improvement".
    """
    axes = tuple(axes)
    metrics = tuple(metrics)
    seen = [a.name for a in axes]
    if len(set(seen)) != len(seen):
        raise ValueError(f"an axis is declared twice: {sorted({a for a in seen if seen.count(a) > 1})}")

    # 1. THE SAME PROGRAM. Checked first because every later number is then meaningless, and this
    #    is exactly what the last campaign shipped a performance hypothesis on top of.
    same_buffer = (baseline.command_buffer_sha256 and candidate.command_buffer_sha256
                   and baseline.command_buffer_sha256 == candidate.command_buffer_sha256)
    same_lowered = (baseline.lowered_sha256 and candidate.lowered_sha256
                    and baseline.lowered_sha256 == candidate.lowered_sha256)
    if same_buffer or same_lowered:
        which = " and ".join(n for n, y in (("command buffer", same_buffer),
                                            ("lowered module", same_lowered)) if y)
        return Decision(
            verdict=IDENTICAL_EMISSION,
            why=(f"the two arms share their {which} digest, so they are the same emitted program; "
                 f"any metric difference between them is noise or instrument drift, not an effect"),
            axes=axes, metrics=metrics)

    moved = {a.name for a in axes if a.moved}
    if not moved:
        return Decision(
            verdict=NO_EFFECT,
            why=("the arms differ in their emitted bytes but no declared axis moved, so there is "
                 "nothing for a metric to be about"),
            axes=axes, metrics=metrics)

    # 2. AN INSTRUMENT THAT CANNOT SEE THE CHANGE. A metric blind to every moved axis is excluded
    #    with its reason -- this is the 1.642x-on-identical-traffic failure, made unrepeatable.
    usable: list[Metric] = []
    excluded: list[dict[str, Any]] = []
    for metric in metrics:
        blind = moved & set(metric.blind_to)
        if blind and blind == moved:
            excluded.append({
                "metric": metric.name, "instrument": metric.instrument,
                "reason": (f"blind to every axis that moved ({sorted(moved)}), so its reading "
                           f"cannot be about this change however precise it is"),
                "ratio": metric.ratio})
            continue
        if blind:
            excluded.append({
                "metric": metric.name, "instrument": metric.instrument,
                "reason": (f"partially blind: cannot see {sorted(blind)} of the moved axes "
                           f"{sorted(moved)}; kept, but it under-reports this change"),
                "ratio": metric.ratio, "kept": True})
        usable.append(metric)

    if not usable:
        return Decision(
            verdict=BLIND,
            why=(f"axes {sorted(moved)} moved and every supplied instrument is blind to all of "
                 f"them; this is an unmeasured change, NOT an ineffective one"),
            axes=axes, metrics=metrics, excluded=excluded)

    # 3. BANDS ELIMINATE, NEVER CERTIFY. An overlapping pair is UNKNOWN by licence.
    points = [m for m in usable if not m.is_band]
    bands = [m for m in usable if m.is_band]
    for band in bands:
        if not band.disjoint():
            excluded.append({
                "metric": band.name, "instrument": band.instrument,
                "reason": ("the two bands overlap, and an overlapping pair is UNKNOWN rather than a "
                           "tie or a small win: a band may eliminate a candidate and may never "
                           "certify one"),
                "baseline_interval": list(band.baseline_interval or ()),
                "candidate_interval": list(band.candidate_interval or ())})

    decisive_bands = [b for b in bands if b.disjoint()]
    if not points and not decisive_bands:
        return Decision(
            verdict=UNKNOWN,
            why=("no point measurement was supplied and every band overlaps, so nothing separates "
                 "the two arms"),
            axes=axes, metrics=metrics, excluded=excluded)

    # 4. THE VERDICT. Prefer a point measurement; a disjoint band can only ELIMINATE, so it decides
    #    only WORSE (the candidate's whole interval sits above the baseline's).
    for band in decisive_bands:
        b_lo, b_hi = band.baseline_interval          # type: ignore[misc]
        c_lo, c_hi = band.candidate_interval         # type: ignore[misc]
        if c_lo > b_hi:
            return Decision(
                verdict=WORSE, decided_by=band.instrument,
                why=(f"{band.name}: the candidate's entire band [{c_lo:g}, {c_hi:g}] lies above the "
                     f"baseline's [{b_lo:g}, {b_hi:g}], which eliminates it"),
                axes=axes, metrics=metrics, excluded=excluded)

    if not points:
        # A disjoint band favouring the candidate is NOT a win: certification is outside the licence.
        favouring = decisive_bands[0]
        excluded.append({
            "metric": favouring.name, "instrument": favouring.instrument,
            "reason": ("the bands are disjoint in the candidate's favour, which is an elimination "
                       "of the BASELINE and not a certification of the candidate; a band may never "
                       "certify one")})
        return Decision(
            verdict=UNKNOWN,
            why=("the only separating evidence is a band favouring the candidate, and a band may "
                 "not certify a candidate -- only eliminate one"),
            axes=axes, metrics=metrics, excluded=excluded)

    ratios = [(m, m.ratio) for m in points if m.ratio is not None]
    if not ratios:
        return Decision(
            verdict=UNKNOWN,
            why="no point metric yielded a usable ratio (a non-positive reading has none)",
            axes=axes, metrics=metrics, excluded=excluded)
    # The WEAKEST improving ratio decides, so a single flattering instrument cannot carry a verdict
    # that another usable one contradicts.
    worst, worst_ratio = min(ratios, key=lambda pair: pair[1])
    best, best_ratio = max(ratios, key=lambda pair: pair[1])
    if worst_ratio < 1.0 and best_ratio > 1.0:
        return Decision(
            verdict=UNKNOWN, decided_by="",
            why=(f"the instruments disagree: {best.name} reports {best_ratio:.3f}x better while "
                 f"{worst.name} reports {worst_ratio:.3f}x. A verdict from either alone would be "
                 f"the one that happened to be quoted"),
            axes=axes, metrics=metrics, excluded=excluded)
    if worst_ratio > 1.0:
        return Decision(
            verdict=BETTER, decided_by=worst.instrument,
            why=(f"every usable instrument improves; the weakest is {worst.name} at "
                 f"{worst_ratio:.3f}x, and it is quoted rather than the best so a single flattering "
                 f"reading cannot carry the verdict"),
            axes=axes, metrics=metrics, excluded=excluded)
    if best_ratio < 1.0:
        return Decision(
            verdict=WORSE, decided_by=best.instrument,
            why=(f"every usable instrument regresses; the mildest is {best.name} at "
                 f"{best_ratio:.3f}x"),
            axes=axes, metrics=metrics, excluded=excluded)
    return Decision(
        verdict=NO_EFFECT, decided_by=worst.instrument,
        why=("the axes moved and every usable instrument read the same on both arms, so the change "
             "is measured and ineffective -- which is a result, and is not the same as unmeasured"),
        axes=axes, metrics=metrics, excluded=excluded)
