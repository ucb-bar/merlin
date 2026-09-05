"""Does the per-unit cycle rate SURVIVE crossing an operand-residency boundary?

A residency family declares a DIFFERENTIAL: the same work, the same parallel extents, the same
operands and the same drained output, with one thing changed -- how much of the operand store the
reduction's working set occupies. The claim is not that any particular number of cycles is predicted.
It is that the machine's marginal cost per unit of the reduction axis is a DIFFERENT number on the two
sides of a residency boundary. So the comparands are two fitted rates, and the verdict is about
whether they differ, never about how well either one fits an absolute prediction.

⚠️ **A rate fitted inside the machine's fill transient is not a rate.** This is the failure that
refuted the sibling reduction-depth family one level up, and the mechanism transfers directly: while
the engines are still learning to run together the realised overlap keeps growing, the marginal cost
per unit keeps FALLING, and an affine law has nowhere to put that. A residency family is MORE exposed
to it, not less, because its cheapest band deliberately starts at the smallest depth the target can
express -- exactly where the transient lives. So every band is put through
:func:`merlin.perf.fill_transient.transient_verdict` BEFORE any rate is quoted for it, and a band
whose points sit inside the transient is REFUSED. That refusal is a result: it says which side of the
boundary this evidence could price and which it could not.

⚠️ **The guard reads the DEEPEST step, so it is not the only line of defence.**
:func:`~merlin.perf.fill_transient.transient_verdict` answers "had the machine finished filling by the
deepest point in this cohort". A band whose two deepest points are settled but whose SHALLOWEST one is
not therefore reports SATURATED -- and that is the realistic case here, because the cheapest band's
first depth is the smallest the target can express. What catches it is the band's own negative control
below: a contaminated shallow point makes the lower depth range fit a different rate from the upper
one, the control does not fire, and no rate is quoted for the band. The two checks are independent and
both are load-bearing. (:func:`~merlin.perf.fill_transient.settling_depth` would locate the transient's
end directly, but it requires a declared band and a declared confirming-step count -- two constants
this family does not declare, and inventing them here would be the tunable this contract refuses.)

**The noise band is MEASURED, never declared.** "The rates agree within the noise band" needs a band,
and a number written into a contract is a knob somebody can turn until the answer changes. The band
here is read off the evidence instead:

* every member must carry at least TWO replicates, and their cycle counts must be IDENTICAL. Then the
  measured dispersion is zero -- observed, not assumed -- and the band is zero;
* a member with ONE replicate has an UNDETERMINABLE dispersion, not a zero one, and its band is
  refused. This tree has a recurring bug class in which the unmeasured was reported as the
  measured-and-zero, and a deterministic simulator is exactly where that assumption feels safe;
* a member whose replicates DISAGREE is refused rather than averaged: collapsing them invents a point
  the replicate control existed to make unnecessary.

With a measured-zero band, "agree" means the two fitted rates are EXACTLY equal as rationals. Every
number below is a :class:`~fractions.Fraction`; nothing rounds, and there is no tolerance to move.

**The negative control is the same arithmetic applied where the answer is known.** The control is two
disjoint depth ranges INSIDE one band: same residency regime, so the rates must agree, so the
falsifier must FIRE. A control that does not fire means this instrument cannot demonstrate agreement
at all, and a cross-band disagreement it then reports is uninterpretable -- that is
:data:`~merlin.perf.campaign.INERT`, in the same sense and for the same reason.

Nothing here names a target, a store, a band or an axis. Band labels arrive as data on the members and
are ordered by their own measured depths.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from merlin.perf import fill_transient as _ft

__all__ = [
    "BAND_AFFINE_CONTRADICTED", "BAND_CONTROL_DID_NOT_FIRE", "BAND_OVERLAP_UNDETERMINABLE",
    "BAND_REPLICATES_DISAGREE", "BAND_REPLICATE_DISPERSION_UNKNOWN", "BAND_TOO_FEW_DEPTHS",
    "BAND_TRANSIENT", "BAND_USABLE", "ESTABLISHED", "INERT", "Member", "REFUSED", "REFUTED",
    "ResidencyEvidenceError", "residency_verdict",
]

#: The band priced a settled machine, its replicates agreed, and its own two sub-range rates agreed.
BAND_USABLE = "usable"
#: Some member of the band reported more than one distinct cycle count across its replicates.
BAND_REPLICATES_DISAGREE = "replicates_disagree"
#: Some member carries fewer than two replicates, so its dispersion is UNKNOWN rather than zero.
BAND_REPLICATE_DISPERSION_UNKNOWN = "replicate_dispersion_unknown"
#: Fewer depths than a rate and its own negative control need.
BAND_TOO_FEW_DEPTHS = "too_few_depths"
#: Realised overlap was still rising at the band's deepest point: no point priced a settled machine.
BAND_TRANSIENT = "in_fill_transient"
#: The overlap instrument did not resolve every point, so "still filling" is not what the band shows.
BAND_OVERLAP_UNDETERMINABLE = "overlap_undeterminable"
#: The marginal cost falls strictly across the band, which contradicts one constant rate outright.
BAND_AFFINE_CONTRADICTED = "affine_form_contradicted"
#: The band's two disjoint sub-range rates are not equal, so the band has no single rate to quote.
BAND_CONTROL_DID_NOT_FIRE = "negative_control_did_not_fire"

#: The rate differs across every comparable residency boundary.
ESTABLISHED = "ESTABLISHED"
#: The rate agrees across a residency boundary: the falsifier fired on a real comparison.
REFUTED = "REFUTED"
#: Bands survived the evidence checks but no negative control fired, so agreement is undetectable.
INERT = "INERT"
#: The evidence cannot support any of the three above, and the reason is named.
REFUSED = "REFUSED"


class ResidencyEvidenceError(ValueError):
    """Member evidence is malformed in a way no verdict can be read through."""


@dataclass(frozen=True)
class Member:
    """One measured corpus member: where it sits, what band it is in, and what it cost.

    ``realised_overlap`` / ``available_overlap`` are ``None`` TOGETHER when the instrument resolved no
    reading, exactly as in :class:`merlin.perf.fill_transient.Point`; they are never 0.
    """

    label: str
    #: The residency band this member's working set occupies, as the member's own declaration states.
    band: str
    #: The declared reduction depth (the fit's independent variable).
    axis: int
    #: One cycle count per replicate, in the order the run authored them.
    replicate_cycles: tuple[int, ...]
    realised_overlap: int | None = None
    available_overlap: int | None = None
    overlap_detail: str = ""

    def __post_init__(self) -> None:
        for name, value in (("label", self.label), ("band", self.band)):
            if not isinstance(value, str) or not value.strip():
                raise ResidencyEvidenceError(f"a residency member must state a non-empty {name}")
        if isinstance(self.axis, bool) or not isinstance(self.axis, int) or self.axis <= 0:
            raise ResidencyEvidenceError(
                f"member {self.label!r} axis must be a positive integer, got {self.axis!r}")
        cycles = tuple(self.replicate_cycles)
        if not cycles:
            raise ResidencyEvidenceError(f"member {self.label!r} carries no replicate cycle counts")
        for value in cycles:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ResidencyEvidenceError(
                    f"member {self.label!r} has a non-positive-integer cycle count {value!r}")
        if (self.realised_overlap is None) != (self.available_overlap is None):
            raise ResidencyEvidenceError(
                f"member {self.label!r} states one half of an overlap reading; realised and "
                "available are absent together or present together")

    @property
    def cycles(self) -> int | None:
        """The one cycle count the replicates agree on, or ``None`` when they do not agree."""
        distinct = set(self.replicate_cycles)
        return distinct.pop() if len(distinct) == 1 else None

    @property
    def dispersion(self) -> int | None:
        """Measured replicate spread, or ``None`` when fewer than two replicates were run."""
        if len(self.replicate_cycles) < 2:
            return None
        return max(self.replicate_cycles) - min(self.replicate_cycles)

    def to_point(self) -> _ft.Point:
        return _ft.Point(label=self.label, axis=self.axis, cycles=int(self.cycles or 0),
                         realised_overlap=self.realised_overlap,
                         available_overlap=self.available_overlap,
                         overlap_detail=self.overlap_detail)

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "band": self.band, "axis": self.axis,
                "replicate_cycles": list(self.replicate_cycles),
                "replicates": len(self.replicate_cycles),
                "agreed_cycles": self.cycles,
                "replicate_dispersion_cycles": self.dispersion,
                "realised_overlap_cycles": self.realised_overlap,
                "available_overlap_cycles": self.available_overlap,
                "overlap_detail": self.overlap_detail}


def _rational(value: Fraction) -> dict[str, int | float]:
    return {"numerator": value.numerator, "denominator": value.denominator, "value": float(value)}


def _fit(points: Sequence[tuple[int, int]]) -> tuple[Fraction, Fraction]:
    """Exact ordinary least squares of cycles on the axis. Rationals throughout; nothing rounds."""
    n = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    denominator = n * sxx - sx * sx
    if denominator == 0:
        raise ResidencyEvidenceError("the fitted depths are degenerate; no rate is determined")
    slope = Fraction(n * sxy - sx * sy, denominator)
    intercept = Fraction(sy, n) - slope * Fraction(sx, n)
    return slope, intercept


def _agree(left: Fraction, right: Fraction) -> bool:
    """Do two fitted rates AGREE within this family's noise band?

    The band is the measured replicate dispersion, and every band that reaches this comparison has
    been shown to have a dispersion of ZERO -- a member whose replicates disagree, or that carries too
    few replicates to measure the dispersion at all, has already refused its band. A zero band makes
    agreement exact rational equality, which is what this is.

    ⚠️ It is a single function on purpose. The negative control and the cross-boundary comparison must
    be the SAME predicate, or the control stops controlling anything; and a tolerance introduced here
    -- the one repair this contract forbids -- would be visible in one place rather than two.
    """
    return left == right


def _split(ordered: Sequence[Member]) -> tuple[list[Member], list[Member]]:
    """Two depth ranges whose INTERIORS are disjoint -- the declared negative control.

    With an even count the two halves share no member at all. With an odd count the median depth is
    the boundary between the ranges and is the upper endpoint of one and the lower endpoint of the
    other; the ranges are still disjoint as intervals, which is what "two disjoint depth ranges"
    asks. Splitting any other way would either discard a measured point or overlap the ranges.
    """
    n = len(ordered)
    mid = n // 2
    if n % 2 == 0:
        return list(ordered[:mid]), list(ordered[mid:])
    return list(ordered[:mid + 1]), list(ordered[mid:])


def _band_record(band: str, members: Sequence[Member]) -> dict[str, Any]:
    ordered = sorted(members, key=lambda member: member.axis)
    record: dict[str, Any] = {
        "band": band,
        "members": [member.to_dict() for member in ordered],
        "depths": [member.axis for member in ordered],
        "status": BAND_USABLE,
        "reason": "",
        "transient": None,
        "rate": None,
        "intercept": None,
        "negative_control": None,
    }

    if len({member.axis for member in ordered}) != len(ordered):
        record["status"] = BAND_TOO_FEW_DEPTHS
        record["reason"] = (f"band {band!r} repeats a depth; two members at one depth are one point, "
                            "and a rate is not determined by one point")
        return record

    unknown = [m.label for m in ordered if m.dispersion is None]
    if unknown:
        record["status"] = BAND_REPLICATE_DISPERSION_UNKNOWN
        record["reason"] = (
            f"member(s) {unknown} carry fewer than two replicates, so the replicate dispersion is "
            "UNDETERMINABLE rather than zero; the noise band this family compares rates within is "
            "measured from that dispersion and cannot be assumed")
        return record

    disagreeing = [m.label for m in ordered if m.cycles is None]
    if disagreeing:
        record["status"] = BAND_REPLICATES_DISAGREE
        record["reason"] = (
            f"member(s) {disagreeing} report more than one distinct cycle count across replicates; "
            "averaging them would invent a point the replicate control existed to make unnecessary")
        return record

    record["measured_noise_band_cycles"] = max(int(m.dispersion or 0) for m in ordered)

    # Three depths, not two: two separate a rate from an intercept, and the third is what makes this
    # band's OWN negative control checkable rather than exact by construction.
    if len(ordered) < 3:
        record["status"] = BAND_TOO_FEW_DEPTHS
        record["reason"] = (
            f"band {band!r} has {len(ordered)} depth(s); a rate needs two and its declared negative "
            "control needs two disjoint depth ranges inside the band, so three is the minimum")
        return record

    transient = _ft.transient_verdict([member.to_point() for member in ordered])
    record["transient"] = transient
    if transient["state"] == _ft.UNDETERMINABLE:
        record["status"] = BAND_OVERLAP_UNDETERMINABLE
        record["reason"] = (f"band {band!r} has no usable overlap trend: {transient['why']}")
        return record
    if transient["state"] == _ft.IN_FILL_TRANSIENT:
        record["status"] = BAND_TRANSIENT
        record["reason"] = (
            f"band {band!r} lies inside the machine's overlap fill transient: {transient['why']}. "
            "A rate fitted here prices how far the engines had filled, not the residency regime, so "
            "no rate is quoted for this band")
        return record
    if transient["affine_form_contradicted"]:
        record["status"] = BAND_AFFINE_CONTRADICTED
        record["reason"] = (
            f"band {band!r} has a strictly falling marginal cost, which contradicts one constant "
            "rate outright: " + str(transient["affine_contradiction_detail"]))
        return record

    pairs = [(member.axis, int(member.cycles or 0)) for member in ordered]
    slope, intercept = _fit(pairs)
    record["rate"] = _rational(slope)
    record["intercept"] = _rational(intercept)

    lower, upper = _split(ordered)
    lower_rate, _ = _fit([(m.axis, int(m.cycles or 0)) for m in lower])
    upper_rate, _ = _fit([(m.axis, int(m.cycles or 0)) for m in upper])
    fired = _agree(lower_rate, upper_rate)
    record["negative_control"] = {
        "control": "two_disjoint_depth_ranges_inside_one_regime",
        "lower_range": {"members": [m.label for m in lower],
                        "depths": [m.axis for m in lower], "rate": _rational(lower_rate)},
        "upper_range": {"members": [m.label for m in upper],
                        "depths": [m.axis for m in upper], "rate": _rational(upper_rate)},
        "rate_difference": _rational(upper_rate - lower_rate),
        "noise_band_cycles_per_axis_unit": 0.0,
        "agree": fired,
        "fired": fired,
        "reason": (
            "the two disjoint depth ranges inside one residency regime fit the same rate exactly, so "
            "this instrument can demonstrate agreement and a cross-band disagreement means something"
            if fired else
            "the two disjoint depth ranges inside one residency regime fit DIFFERENT rates, so this "
            "band has no single rate to compare across a boundary and the instrument has not shown "
            "it can report agreement"),
    }
    if not fired:
        record["status"] = BAND_CONTROL_DID_NOT_FIRE
        record["reason"] = str(record["negative_control"]["reason"])
    return record


def residency_verdict(members: Sequence[Member]) -> dict[str, Any]:
    """The residency differential's verdict, with every band's usability stated.

    Returns one of :data:`ESTABLISHED`, :data:`REFUTED`, :data:`INERT` or :data:`REFUSED`. There is no
    default and no fourth silent state: a run that cannot decide says which band stopped it and why.
    """
    rows = list(members)
    for row in rows:
        if not isinstance(row, Member):
            raise ResidencyEvidenceError("every residency observation must be a Member")
    labels = [row.label for row in rows]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ResidencyEvidenceError(f"residency evidence repeats member(s) {duplicates}")

    grouped: dict[str, list[Member]] = {}
    for row in rows:
        grouped.setdefault(row.band, []).append(row)
    # Ordered by the bands' OWN measured depths. Nothing here holds a list of band names, so a target
    # whose store yields a band this tree has never seen is ordered correctly anyway.
    order = sorted(grouped, key=lambda band: min(member.axis for member in grouped[band]))
    bands = [_band_record(band, grouped[band]) for band in order]

    usable = [band for band in bands if band["status"] == BAND_USABLE]
    refused_for_transient = [band["band"] for band in bands if band["status"] == BAND_TRANSIENT]
    control_eligible = [band for band in bands
                        if band["status"] in (BAND_USABLE, BAND_CONTROL_DID_NOT_FIRE)]

    result: dict[str, Any] = {
        "observation": "per_regime_fitted_rate_and_intercept",
        "negative_control": "two_disjoint_depth_ranges_inside_one_regime",
        "noise_band": {
            "kind": "measured_replicate_dispersion",
            "declared_constant": None,
            "cycles": max([int(band.get("measured_noise_band_cycles") or 0) for band in bands],
                          default=None),
            "how": ("read off the evidence: every member's replicates must be identical, which makes "
                    "the dispersion zero BY MEASUREMENT, so two rates agree only when they are "
                    "exactly equal as rationals. No constant is declared and none can be moved"),
        },
        "bands": bands,
        "usable_bands": [band["band"] for band in usable],
        "refused_bands": [{"band": band["band"], "status": band["status"], "reason": band["reason"]}
                          for band in bands if band["status"] != BAND_USABLE],
        "bands_refused_for_transient_reasons": refused_for_transient,
        "boundaries": [],
        "status": REFUSED,
        "reason": "",
    }

    if len(usable) < 2:
        if len(control_eligible) >= 2 and not usable:
            result["status"] = INERT
            result["reason"] = (
                "every band that priced a settled machine failed its own negative control: two "
                "disjoint depth ranges inside ONE residency regime already fit different rates, so "
                "this instrument has not shown it can report agreement and a cross-band difference "
                "would be uninterpretable")
            return result
        result["reason"] = (
            f"{len(usable)} band(s) can carry a rate; a residency differential needs two, one on "
            "each side of a boundary. Refused band(s): "
            + "; ".join(f"{band['band']}={band['status']}" for band in bands
                        if band["status"] != BAND_USABLE))
        return result

    boundaries: list[dict[str, Any]] = []
    for index, lower in enumerate(usable):
        for upper in usable[index + 1:]:
            lower_rate = Fraction(int(lower["rate"]["numerator"]), int(lower["rate"]["denominator"]))
            upper_rate = Fraction(int(upper["rate"]["numerator"]), int(upper["rate"]["denominator"]))
            agree = _agree(lower_rate, upper_rate)
            boundaries.append({
                "lower_band": lower["band"],
                "upper_band": upper["band"],
                "lower_rate": dict(lower["rate"]),
                "upper_rate": dict(upper["rate"]),
                "rate_difference": _rational(upper_rate - lower_rate),
                "noise_band_cycles_per_axis_unit": 0.0,
                "agree": agree,
                "falsifier_fired": agree,
            })
    result["boundaries"] = boundaries

    fired = [row for row in boundaries if row["falsifier_fired"]]
    if fired:
        result["status"] = REFUTED
        result["reason"] = (
            "the falsifier fired on a real comparison: the rates fitted on either side of "
            + ", ".join(f"{row['lower_band']}|{row['upper_band']}" for row in fired)
            + " agree within the measured noise band, so crossing that residency boundary does not "
              "change the per-unit cost this evidence can see")
        return result
    result["status"] = ESTABLISHED
    result["reason"] = (
        "every comparable residency boundary changes the fitted rate by more than the measured noise "
        "band, and the negative control fired inside each band, so the instrument was capable of "
        "reporting the agreement it did not report")
    return result
