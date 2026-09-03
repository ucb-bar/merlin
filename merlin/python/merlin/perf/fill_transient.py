"""Is a cohort of design points still inside the machine's overlap FILL TRANSIENT?

A law that says "cycles are affine in the design variable" is a statement about a machine in STEADY
STATE: it asserts that one more unit of the variable always costs the same. A machine with independent
engines does not reach that state immediately. While the engines are still learning to run together the
realised overlap keeps growing, each additional unit hides more of its own cost under the unit before
it, and the marginal cost per unit FALLS. An affine model has nowhere to put that, and no choice of
threshold repairs it -- the contradiction is between "constant marginal cost" and "falling marginal
cost", which is arithmetic, not statistics.

⚠️ **The refutation this produces is THRESHOLD-FREE, and that is the point.** A cohort whose marginals
fall monotonically contradicts the affine form directly. Reporting only "r squared missed its bound"
invites the repair that is forbidden -- move the bound -- because it presents a model error as a
tolerance error. Reporting the falling marginal alongside it names which of the two it is.

**What licenses calling it a FILL transient rather than just curvature.** Curvature alone says the
model is wrong; it does not say why. The overlap reading does: when realised overlap is still RISING at
the deepest point measured, the machine had not finished filling anywhere in the cohort, so every point
priced a different degree of engine cooperation. Without an overlap reading this module returns
UNDETERMINABLE and says so -- a falling marginal with no overlap evidence is a fact about the fit, not
an explanation of it, and this tree has a recurring bug class in which the unmeasured was reported as
the measured-and-zero.

Nothing here names a target, an engine, a unit or a design variable. The overlap quantity is the one
:func:`merlin.perf.hw_counters.eta_from_counters` and :mod:`merlin.perf.falsifier` already use --
deliberately the same number, so this verdict's eta and theirs are one quantity rather than two that
share a name.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

__all__ = [
    "IN_FILL_TRANSIENT", "Point", "SATURATED", "UNDETERMINABLE", "marginal_costs",
    "overlap_trend", "point_from_counter_values", "transient_verdict",
]

#: The cohort's points are all inside the transient: marginal cost still falling, overlap still rising.
IN_FILL_TRANSIENT = "in_fill_transient"
#: The overlap reading stopped rising inside the cohort, so the deepest points priced a settled machine.
SATURATED = "saturated"
#: Spelled the same as everywhere else in this package, and meaning the same thing: not decided.
UNDETERMINABLE = "undeterminable"


@dataclass(frozen=True)
class Point:
    """One measured design point: where it sits on the axis, what it cost, and what overlapped.

    ``realised_overlap`` and ``available_overlap`` are ``None`` TOGETHER whenever the instrument could
    not resolve a reading. They are never 0 -- a zero from an unobservable vector is indistinguishable
    from a machine that genuinely serialises, and only one of those is a measurement.
    """

    label: str
    #: The declared design variable's value at this point (the fit's independent variable).
    axis: int
    #: Total measured cycles for the point.
    cycles: int
    realised_overlap: int | None = None
    available_overlap: int | None = None
    #: Why there is no overlap reading, when there is none. Carried, never inferred from the ``None``.
    overlap_detail: str = ""

    @property
    def eta(self) -> Fraction | None:
        """Realised as a fraction of available overlap, or ``None`` when it is not measurable."""
        if self.realised_overlap is None or not self.available_overlap:
            return None
        return Fraction(int(self.realised_overlap), int(self.available_overlap))

    def to_dict(self) -> dict:
        eta = self.eta
        return {"label": self.label, "axis": self.axis, "cycles": self.cycles,
                "realised_overlap_cycles": self.realised_overlap,
                "available_overlap_cycles": self.available_overlap,
                "eta": None if eta is None else float(eta),
                "overlap_detail": self.overlap_detail}


def point_from_counter_values(label: str, axis: int, cycles: int, values: Mapping[str, int],
                              counters) -> Point:
    """A :class:`Point` whose overlap comes from one bracketed run's combination counters.

    Delegates to :func:`merlin.perf.hw_counters.eta_from_counters` rather than re-deriving the ratio,
    so realised/available here are the SAME quantities the falsifier and the perf ledger hold, and a
    later change to how available overlap is bounded moves all three together.
    """
    from merlin.perf.hw_counters import eta_from_counters

    reading = eta_from_counters(dict(values), counters)
    if reading.get("state") != "measured":
        return Point(label=label, axis=int(axis), cycles=int(cycles),
                     overlap_detail=str(reading.get("why") or "the counter reading is not measured"))
    return Point(label=label, axis=int(axis), cycles=int(cycles),
                 realised_overlap=int(reading["realised_cycles"]),
                 available_overlap=int(reading["available_cycles"]),
                 overlap_detail=str(reading.get("note") or ""))


def _ordered(points: Sequence[Point]) -> list[Point]:
    return sorted(points, key=lambda p: int(p.axis))


def marginal_costs(points: Sequence[Point]) -> list[dict]:
    """Cycles per additional unit of the axis, between each consecutive pair of points.

    Exact rationals, not floats: the whole verdict turns on whether one marginal is strictly below the
    one before it, and a rounded ratio can invert that comparison at the boundary.
    """
    ordered = _ordered(points)
    out: list[dict] = []
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        span = int(later.axis) - int(earlier.axis)
        if span <= 0:
            raise ValueError(f"points {earlier.label!r} and {later.label!r} do not advance the axis")
        rate = Fraction(int(later.cycles) - int(earlier.cycles), span)
        out.append({"from": earlier.label, "to": later.label,
                    "axis_from": int(earlier.axis), "axis_to": int(later.axis),
                    "delta_cycles": int(later.cycles) - int(earlier.cycles),
                    "axis_span": span,
                    "marginal_cycles_per_axis_unit": float(rate),
                    "exact_numerator": rate.numerator, "exact_denominator": rate.denominator})
    return out


def overlap_trend(points: Sequence[Point]) -> dict:
    """Per-point eta, and whether it is still rising at the deepest point in the cohort.

    ``state`` is ``UNDETERMINABLE`` whenever ANY point lacks a reading. A trend computed over the
    subset that happened to read is a trend of a different cohort, and the points that dropped out are
    exactly the ones an instrument fails on for a reason.
    """
    ordered = _ordered(points)
    unread = [p.label for p in ordered if p.eta is None]
    if unread:
        return {"state": UNDETERMINABLE, "unread": unread,
                "why": (f"{len(unread)} point(s) carry no overlap reading ({unread}); an unread point "
                        "is UNKNOWN, never zero overlap, and a trend over the rest is a trend of a "
                        "different cohort"),
                "eta_by_point": [p.to_dict() for p in ordered]}
    if len(ordered) < 3:
        return {"state": UNDETERMINABLE, "unread": [],
                "why": (f"{len(ordered)} point(s): a trend needs at least three, because two points "
                        "define one step and one step cannot be rising or flattening"),
                "eta_by_point": [p.to_dict() for p in ordered]}
    etas = [p.eta for p in ordered]
    # NON-DECREASING, not strictly increasing. Saturation is precisely the case where the last step is
    # zero, so a strict test would report the settled cohort -- the one this verdict must be able to
    # return -- as "not ordered by fill" and collapse SATURATED into UNDETERMINABLE. Only a step that
    # goes DOWN says the points are not ordered by how far the machine had filled.
    rising = all(later >= earlier for earlier, later in zip(etas, etas[1:], strict=False))
    last_step = etas[-1] - etas[-2]
    return {"state": "measured", "unread": [],
            "monotonically_rising": rising,
            "still_rising_at_deepest_point": last_step > 0,
            "final_step": float(last_step),
            "eta_by_point": [p.to_dict() for p in ordered]}


def transient_verdict(points: Sequence[Point]) -> dict:
    """Do these points lie inside the fill transient, and is an affine law testable on them?

    ``affine_form_contradicted`` is reported separately from the transient state and does NOT depend
    on the overlap reading: a strictly falling marginal contradicts "constant marginal cost" by
    itself. The overlap reading supplies the MECHANISM for that contradiction, and its absence
    downgrades the explanation, never the arithmetic.
    """
    ordered = _ordered(points)
    marginals = marginal_costs(ordered)
    rates = [Fraction(m["exact_numerator"], m["exact_denominator"]) for m in marginals]
    falling = len(rates) >= 2 and all(
        later < earlier for earlier, later in zip(rates, rates[1:], strict=False))
    trend = overlap_trend(ordered)

    if trend["state"] != "measured":
        state, why = UNDETERMINABLE, trend["why"]
    elif not trend["monotonically_rising"]:
        state = UNDETERMINABLE
        why = ("realised overlap FALLS somewhere across the cohort, so the points are not ordered by "
               "how far the machine had filled and 'still filling' is not what they show")
    elif trend["still_rising_at_deepest_point"]:
        state = IN_FILL_TRANSIENT
        why = ("realised overlap is still rising at the deepest point measured, so no point in the "
               "cohort priced a settled machine: every one of them charged a different degree of "
               "engine cooperation")
    else:
        state = SATURATED
        why = ("realised overlap stopped rising inside the cohort, so its deepest points priced a "
               "settled machine and a steady-state law is testable on them")

    return {
        "state": state,
        "why": why,
        "affine_form_contradicted": falling,
        "affine_contradiction_detail": (
            "the marginal cost per axis unit falls strictly across every consecutive interval; an "
            "affine law asserts one constant marginal cost, so this cohort contradicts the FORM "
            "independently of any fit statistic or tolerance"
            if falling else
            "the marginal cost per axis unit does not fall strictly across every interval, so the "
            "affine form is not contradicted by the marginals alone"),
        "n_points": len(ordered),
        "marginals": marginals,
        "overlap": trend,
    }
