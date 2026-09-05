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
    "IN_FILL_TRANSIENT", "NEVER_SETTLES_IN_RANGE", "PARTITION_FIELDS", "PartitionEvidenceError",
    "Point", "SATURATED", "SETTLED_AT", "UNDETERMINABLE", "eta_resolution", "marginal_costs",
    "marginal_settling_depth", "overlap_trend", "partition_kwargs", "point_from_counter_values",
    "settling_depth", "transient_verdict",
]

#: The cohort's points are all inside the transient: marginal cost still falling, overlap still rising.
IN_FILL_TRANSIENT = "in_fill_transient"
#: The overlap reading stopped rising inside the cohort, so the deepest points priced a settled machine.
SATURATED = "saturated"
#: Spelled the same as everywhere else in this package, and meaning the same thing: not decided.
UNDETERMINABLE = "undeterminable"

#: A depth was found inside the ladder from which the overlap reading no longer rises, and it stayed
#: that way for every deeper step measured.
SETTLED_AT = "settled_at"
#: Every step in the ladder rises by more than the declared band. This is an ANSWER, not a failure:
#: it says the machine had not finished filling anywhere in the range that was affordable to measure.
NEVER_SETTLES_IN_RANGE = "never_settles_in_range"


#: What :func:`merlin.perf.hw_counters.eta_from_counters` needs in order to PROVE, from the target's
#: own elaborated CIRCT, that its combination counters partition busy time -- without which an overlap
#: reading is a header's naming convention rather than a measurement.
#:
#: ⚠️ **This module cannot derive any of them.** They are facts about one target's RTL: the elaborated
#: HW text, the counter header's event codes, and the two module identities that select the counted
#: structures inside it. A caller reads them off the target boundary the same way
#: :mod:`merlin.targetgen.contract.compile` does -- the backend's ``counter_partition_inputs()`` for
#: ``hw_text`` / ``module`` / ``counter_module`` / ``source``, and
#: :func:`merlin.perf.hw_counters.event_codes` over the shipped header for ``codes`` -- and passes them
#: in. Nothing here substitutes a default for one that is absent.
PARTITION_FIELDS = ("hw_text", "codes", "module", "counter_module")


class PartitionEvidenceError(ValueError):
    """The CIRCT counter-partition evidence is absent or malformed, and the message says WHICH input.

    ⚠️ This type exists so that a missing target artifact cannot arrive at a caller looking like an
    analyzer verdict. The call it guards used to be made with the wrong arity, and the resulting
    ``TypeError`` was caught by a broad ``except (KeyError, TypeError, ValueError)`` and reported as a
    considered REFUSED carrying the type error's text. A named absence is a result; a programming
    error dressed as one is not, so this is raised deliberately and caught by name.
    """


def partition_kwargs(partition) -> dict:
    """Validate caller-supplied partition evidence into :func:`eta_from_counters` keywords.

    Structural validation only -- membership and type checks over a mapping the caller owns. Every
    rejection names the field that is missing or wrong, because the whole point of this gate is that
    the reason reaches the report instead of a stack frame.
    """
    if partition is None:
        raise PartitionEvidenceError(
            "no CIRCT counter-partition evidence was supplied, so realised overlap cannot be called "
            f"measured; the target boundary must supply {list(PARTITION_FIELDS)} -- its elaborated "
            "CIRCT HW text, the shipped counter header's event codes, and the two module identities "
            "that select the counted structures")
    if not isinstance(partition, Mapping):
        raise PartitionEvidenceError(
            f"the CIRCT counter-partition evidence must be a mapping of {list(PARTITION_FIELDS)}, "
            f"not a {type(partition).__name__}")
    # The target boundary reports its own three states. Only "available" carries usable evidence, and
    # an unavailable one is passed through with the target's OWN reason rather than reworded here.
    status = partition.get("status")
    if status is not None and status != "available":
        raise PartitionEvidenceError(
            f"the target reports its CIRCT counter-partition evidence as {str(status)!r}: "
            + str(partition.get("why") or "no reason was given"))
    for field in ("hw_text", "module", "counter_module"):
        value = partition.get(field)
        if not isinstance(value, str) or not value.strip():
            raise PartitionEvidenceError(
                f"the CIRCT counter-partition evidence carries no non-empty {field!r}; without it "
                "the counter exclusivity proof cannot be attempted, and an unproved partition is "
                "UNKNOWN overlap rather than zero overlap")
    codes = partition.get("codes")
    if not isinstance(codes, Mapping) or not codes:
        raise PartitionEvidenceError(
            "the CIRCT counter-partition evidence carries no 'codes' mapping of counter name to the "
            "event code its own header declares; the proof follows those numeric ports into the HW")
    for name, code in codes.items():
        if not isinstance(name, str) or not name:
            raise PartitionEvidenceError("an event code is keyed by something other than a name")
        if isinstance(code, bool) or not isinstance(code, int) or code < 0:
            raise PartitionEvidenceError(
                f"event code for {name!r} is not a non-negative integer, so no port selects it")
    source = partition.get("source")
    if source is not None and not isinstance(source, str):
        raise PartitionEvidenceError("the partition evidence 'source' must be a string when present")
    return {"hw_text": str(partition["hw_text"]), "codes": dict(codes),
            "module": str(partition["module"]), "counter_module": str(partition["counter_module"]),
            "source": source}


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
                              counters, *, partition) -> Point:
    """A :class:`Point` whose overlap comes from one bracketed run's combination counters.

    Delegates to :func:`merlin.perf.hw_counters.eta_from_counters` rather than re-deriving the ratio,
    so realised/available here are the SAME quantities the falsifier and the perf ledger hold, and a
    later change to how available overlap is bounded moves all three together.

    ``partition`` is the target's CIRCT counter-partition evidence (see :data:`PARTITION_FIELDS`) and
    is REQUIRED, because that delegate refuses to call overlap measured until the counters are proved
    exclusive and exhaustive from the elaborated RTL. It cannot be derived here and is never defaulted:
    evidence that is absent produces a Point carrying the reason and NO overlap reading, exactly as a
    counter that did not read does.

    ``cycles`` doubles as the counter window: the readings and the cycle count come from one bracketed
    run, so a partition totalling more than the window it was read in is mixed, corrupt or wrapped, and
    the delegate says so rather than dividing anyway.
    """
    from merlin.perf.hw_counters import eta_from_counters

    try:
        proof = partition_kwargs(partition)
    except PartitionEvidenceError as exc:
        return Point(label=label, axis=int(axis), cycles=int(cycles), overlap_detail=str(exc))
    reading = eta_from_counters(dict(values), counters, measurement_cycles=int(cycles), **proof)
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


def eta_resolution(points: Sequence[Point]) -> list[dict]:
    """The smallest change in eta each consecutive pair of points could have RESOLVED.

    Eta is ``realised / available``, both integer cycle counts, so the finest difference a point can
    express is ``1 / available`` -- and a step between two points is only as fine as the coarser of the
    two. A settling band narrower than that is asking the instrument for a distinction it cannot make,
    which is why a caller's band is compared against this rather than trusted.
    """
    ordered = _ordered(points)
    out: list[dict] = []
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        avail = [p.available_overlap for p in (earlier, later)]
        if any(a is None or not a for a in avail):
            out.append({"from": earlier.label, "to": later.label, "resolution": None,
                        "why": "one of the pair carries no available-overlap reading"})
            continue
        coarsest = min(int(a) for a in avail)
        out.append({"from": earlier.label, "to": later.label,
                    "resolution": float(Fraction(1, coarsest)),
                    "exact_numerator": 1, "exact_denominator": coarsest,
                    "why": f"eta moves in steps of 1/{coarsest} at the coarser of the two points"})
    return out


def settling_depth(points: Sequence[Point], *, band: Fraction, confirming_steps: int) -> dict:
    """The shallowest axis value from which the overlap reading no longer rises -- or a refusal.

    :func:`transient_verdict` answers WHETHER a cohort sits inside the fill transient. This answers
    WHERE the transient ends, which is the question a successor cohort has to be built on: a law
    asserting one constant marginal cost is only testable on points past that depth.

    Both parameters are REQUIRED and are echoed into the result. They are claim decisions, not tuning
    knobs, and a settling depth quoted without the band it was decided under is not reproducible:

    ``band``
        the largest per-step rise in eta that counts as "no longer rising". Compared against
        :func:`eta_resolution` per step, and a step whose band is finer than what the instrument can
        resolve is FLAGGED rather than silently counted as settled.
    ``confirming_steps``
        how many consecutive steps at or past the candidate must stay inside the band. One is not
        enough to trust: a ladder that stopped one point too early has exactly one flat-looking step
        at its end, and that is indistinguishable from a machine that settled there.

    The settling depth is the axis value of the EARLIER point of the first qualifying step. If eta at
    A is already within the band of eta at B, the machine had finished filling by A.

    ``NEVER_SETTLES_IN_RANGE`` is a real answer and the caller must be able to report it as one. It
    says the ladder ended inside the transient, which bounds the successor cohort from below and is
    strictly more useful than a plateau extrapolated from points that are themselves still filling.
    """
    if int(confirming_steps) < 1:
        raise ValueError("confirming_steps must be at least 1; zero steps confirm nothing")
    band = Fraction(band)
    if band < 0:
        raise ValueError("band must not be negative; a negative band accepts a RISING step as settled")

    ordered = _ordered(points)
    trend = overlap_trend(ordered)
    declared = {"band": float(band), "band_numerator": band.numerator,
                "band_denominator": band.denominator,
                "confirming_steps": int(confirming_steps)}
    if trend["state"] != "measured":
        return {"state": UNDETERMINABLE, "why": trend["why"], "declared": declared,
                "settling_axis": None, "steps": [], "overlap": trend,
                "marginals": marginal_costs(ordered) if len(ordered) > 1 else []}

    resolutions = eta_resolution(ordered)
    etas = [p.eta for p in ordered]
    steps: list[dict] = []
    for i, (earlier, later) in enumerate(zip(ordered, ordered[1:], strict=False)):
        delta = etas[i + 1] - etas[i]
        res = resolutions[i].get("resolution")
        steps.append({
            "from": earlier.label, "to": later.label,
            "axis_from": int(earlier.axis), "axis_to": int(later.axis),
            "eta_from": float(etas[i]), "eta_to": float(etas[i + 1]),
            "eta_step": float(delta),
            "within_band": bool(delta <= band),
            "resolution": res,
            # A band finer than the instrument's own step size cannot separate "settled" from "rose by
            # the smallest amount this reading can express". Recorded per step, because resolution
            # improves with depth and the shallow end is where it bites.
            "band_below_resolution": bool(res is not None and float(band) < res),
        })

    n = len(steps)
    settling_index = None
    for i in range(n):
        if n - i < int(confirming_steps):
            break
        if all(s["within_band"] for s in steps[i:]):
            settling_index = i
            break

    deepest = ordered[-1]
    common = {"declared": declared, "steps": steps, "overlap": trend,
              "marginals": marginal_costs(ordered),
              "deepest_axis_measured": int(deepest.axis),
              "deepest_eta": float(etas[-1]),
              "final_step": float(etas[-1] - etas[-2]),
              "unresolvable_steps": [s["from"] + "->" + s["to"] for s in steps
                                     if s["band_below_resolution"]]}
    if settling_index is None:
        return {
            "state": NEVER_SETTLES_IN_RANGE,
            "settling_axis": None,
            "why": (f"no axis value in this ladder is followed by {int(confirming_steps)} consecutive "
                    f"step(s) whose eta rise is within {float(band)}; the deepest point measured is "
                    f"axis {int(deepest.axis)} at eta {float(etas[-1]):.6g}, still rising by "
                    f"{float(etas[-1] - etas[-2]):.6g}. The overlap does not settle anywhere in the "
                    "measured range, which bounds a successor cohort from below rather than supplying "
                    "it a depth"),
            **common}
    settled = ordered[settling_index]
    return {
        "state": SETTLED_AT,
        "settling_axis": int(settled.axis),
        "settling_label": settled.label,
        "why": (f"from axis {int(settled.axis)} onward every one of the {n - settling_index} remaining "
                f"step(s) rises by at most {float(band)}, so the machine had finished filling by that "
                f"depth and a steady-state law is testable at or past it"),
        "confirmed_by_steps": n - settling_index,
        **common}


def marginal_settling_depth(points: Sequence[Point], *, relative_band: Fraction,
                            confirming_steps: int) -> dict:
    """The shallowest axis value from which the MARGINAL cost per axis unit stops changing.

    The companion to :func:`settling_depth`, and the one a successor cohort is actually built on.
    Realised overlap is the MECHANISM -- it explains why the marginal cost falls -- but the property
    an affine law asserts is that the marginal cost is CONSTANT. The two need not end together: a
    machine can still be creeping towards its overlap ceiling while the cycles it charges per extra
    unit have already stopped moving, and it is the second of those that decides where a steady-state
    law becomes testable.

    ``relative_band`` is the largest fractional change between consecutive marginals that counts as
    "unchanged", compared against the LARGER of the pair so the comparison is symmetric. It is
    REQUIRED and echoed, for the same reason the eta band is: a depth quoted without its band is not
    reproducible. A caller that wants this to answer for a particular claim should pass that claim's
    OWN declared tolerance rather than inventing one.

    Exact rationals throughout. The whole verdict turns on whether one ratio sits inside a band, and
    a rounded ratio can cross it at the boundary.
    """
    if int(confirming_steps) < 1:
        raise ValueError("confirming_steps must be at least 1; zero steps confirm nothing")
    relative_band = Fraction(relative_band)
    if relative_band < 0:
        raise ValueError("relative_band must not be negative")

    ordered = _ordered(points)
    marginals = marginal_costs(ordered)
    declared = {"relative_band": float(relative_band),
                "relative_band_numerator": relative_band.numerator,
                "relative_band_denominator": relative_band.denominator,
                "confirming_steps": int(confirming_steps)}
    if len(marginals) < 2:
        return {"state": UNDETERMINABLE, "settling_axis": None, "declared": declared, "steps": [],
                "marginals": marginals,
                "why": (f"{len(marginals)} marginal(s): comparing one marginal to the next needs at "
                        "least two, and one marginal cannot be changing or unchanged")}

    rates = [Fraction(m["exact_numerator"], m["exact_denominator"]) for m in marginals]
    steps: list[dict] = []
    for i, (earlier, later) in enumerate(zip(rates, rates[1:], strict=False)):
        scale = max(abs(earlier), abs(later))
        # A pair of zero marginals is unchanged by construction; guarding it here keeps the ratio
        # from being a division rather than a judgement.
        change = Fraction(0) if scale == 0 else abs(later - earlier) / scale
        steps.append({
            "from": marginals[i]["from"], "through": marginals[i]["to"],
            "to": marginals[i + 1]["to"],
            "axis_from": marginals[i]["axis_from"], "axis_to": marginals[i + 1]["axis_to"],
            "marginal_before": float(earlier), "marginal_after": float(later),
            "relative_change": float(change),
            "within_band": bool(change <= relative_band),
        })

    n = len(steps)
    index = None
    for i in range(n):
        if n - i < int(confirming_steps):
            break
        if all(s["within_band"] for s in steps[i:]):
            index = i
            break

    # The longest contiguous stretch of the ladder over which the marginal DOES hold, whether or not
    # it holds all the way to the end. A cost that goes constant and then changes again has a WINDOW,
    # and a window is still a place a steady-state law can be tested -- bounded from both sides rather
    # than only from below. Reporting only "never settles" would throw that away, and reporting the
    # window as a settling depth would hide that it closes.
    best_start = best_len = run_start = run_len = 0
    for i, step in enumerate(steps):
        if step["within_band"]:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len > best_len:
                best_start, best_len = run_start, run_len
        else:
            run_len = 0
    window = None
    if best_len:
        window = {"axis_from": steps[best_start]["axis_from"],
                  "axis_to": steps[best_start + best_len - 1]["axis_to"],
                  "steps": best_len,
                  "closes_before_the_deepest_point": bool(best_start + best_len < n),
                  "why": ("the marginal cost holds inside the band across this stretch and changes "
                          "again outside it" if best_start + best_len < n else
                          "the marginal cost holds inside the band from here to the deepest point")}

    deepest = ordered[-1]
    common = {"declared": declared, "steps": steps, "marginals": marginals,
              "deepest_axis_measured": int(deepest.axis),
              "longest_within_band_window": window,
              "final_relative_change": steps[-1]["relative_change"]}
    if index is None:
        return {
            "state": NEVER_SETTLES_IN_RANGE, "settling_axis": None,
            "why": (f"no axis value in this ladder is followed by {int(confirming_steps)} consecutive "
                    f"marginal comparison(s) agreeing within {float(relative_band)}; at the deepest "
                    f"point measured (axis {int(deepest.axis)}) consecutive marginals still differ by "
                    f"{steps[-1]['relative_change']:.6g}. An affine law asserts ONE constant marginal "
                    "cost, so it is not testable anywhere in this range"),
            **common}
    # The marginal that first holds spans two intervals, and the depth a law becomes testable FROM is
    # the shallow end of the first of them: that is the first point whose cost is already charged at
    # the settled rate.
    settled_axis = steps[index]["axis_from"]
    return {
        "state": SETTLED_AT, "settling_axis": int(settled_axis),
        "why": (f"from axis {int(settled_axis)} onward every one of the {n - index} remaining "
                f"marginal comparison(s) agrees within {float(relative_band)}, so the cost per extra "
                "axis unit has stopped changing and an affine law is testable at or past that depth"),
        "confirmed_by_steps": n - index,
        **common}
