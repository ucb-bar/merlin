"""WHERE the overlap fill transient ends -- not merely whether a cohort sits inside one.

:mod:`merlin.perf.fill_transient` could already say that a cohort was still filling. That is the
finding that refuted the frozen affine reduction-depth contract, and it is where the argument stopped:
"the depths must reach past saturation" names no depth, so a successor cohort could not be declared.
:func:`~merlin.perf.fill_transient.settling_depth` answers the missing half, and this file pins its
two failure modes rather than only its success.

⚠️ **The verdict must be able to come out BOTH ways, and both ways must be reachable from measured
data.** A settling detector that can only say "settled" would hand a successor cohort a depth on any
evidence at all, and one that can only say "still filling" could never license one. So there is a
positive case that fires on a real ladder, a negative case that fires on the frozen refuted cohort,
and a vacuity check that the two are decided by the data and not by the shape of the code.

The band and the confirming-step count are REQUIRED arguments with no defaults. That is deliberate:
a settling depth quoted without the band it was decided under is not reproducible, and a default band
is a threshold nobody declared. The tests below hold both fixed and move only the measurements.
"""
from __future__ import annotations

from fractions import Fraction

import pytest

from merlin.perf import fill_transient as FT
from merlin.perf import hw_counters as HC

#: The counter block the interlocked target's own shipped header declares, reduced to the ``#define``
#: lines the derivation reads. Stated here rather than read from an external checkout so the test says
#: what it measured on; the ENGINE set is still derived from this text, never typed.
COUNTER_HEADER = """
#define MAIN_LD_CYCLES 1
#define MAIN_ST_CYCLES 2
#define MAIN_EX_CYCLES 3
#define MAIN_LD_ST_CYCLES 4
#define MAIN_LD_EX_CYCLES 5
#define MAIN_ST_EX_CYCLES 6
#define MAIN_LD_ST_EX_CYCLES 7
"""

#: The frozen PK cohort, as recorded in its refutation artifact: citable Verilator L3 cycles and the
#: counter readings the same bracketed runs printed. Reduction depth doubling from one tile to eight
#: at fixed single-tile parallel extents.
PK_COHORT = {
    "PK00_k16": (16, 301, {"MAIN_EX_CYCLES": 70, "MAIN_LD_CYCLES": 39, "MAIN_ST_CYCLES": 43,
                           "MAIN_LD_EX_CYCLES": 28, "MAIN_ST_EX_CYCLES": 0,
                           "MAIN_LD_ST_CYCLES": 0, "MAIN_LD_ST_EX_CYCLES": 0}),
    "PK01_k32": (32, 373, {"MAIN_EX_CYCLES": 83, "MAIN_LD_CYCLES": 46, "MAIN_ST_CYCLES": 43,
                           "MAIN_LD_EX_CYCLES": 51, "MAIN_ST_EX_CYCLES": 0,
                           "MAIN_LD_ST_CYCLES": 0, "MAIN_LD_ST_EX_CYCLES": 0}),
    "PK02_k64": (64, 471, {"MAIN_EX_CYCLES": 83, "MAIN_LD_CYCLES": 43, "MAIN_ST_CYCLES": 43,
                           "MAIN_LD_EX_CYCLES": 128, "MAIN_ST_EX_CYCLES": 0,
                           "MAIN_LD_ST_CYCLES": 0, "MAIN_LD_ST_EX_CYCLES": 0}),
    "PK03_k128": (128, 604, {"MAIN_EX_CYCLES": 83, "MAIN_LD_CYCLES": 42, "MAIN_ST_CYCLES": 43,
                             "MAIN_LD_EX_CYCLES": 281, "MAIN_ST_EX_CYCLES": 0,
                             "MAIN_LD_ST_CYCLES": 0, "MAIN_LD_ST_EX_CYCLES": 0}),
}

#: The claim decisions this file holds fixed. One percentage point of the overlap ceiling per DOUBLING
#: of the reduction depth, confirmed by two consecutive steps. Two and not one: a ladder that stopped
#: one point too early has exactly one flat-looking step at its end, and that is indistinguishable
#: from a machine that settled there.
BAND = Fraction(1, 100)
CONFIRMING_STEPS = 2


def counters():
    return HC.derive_occupancy_counters(COUNTER_HEADER)


def _point(label, axis, cycles, realised, available):
    return FT.Point(label=label, axis=axis, cycles=cycles,
                    realised_overlap=realised, available_overlap=available)


def pk_points() -> list[FT.Point]:
    derived = counters()
    return [FT.point_from_counter_values(name, axis, cycles, values, derived)
            for name, (axis, cycles, values) in PK_COHORT.items()]


def settled_points() -> list[FT.Point]:
    """A ladder whose overlap rises, flattens, and then stays flat for two more steps.

    Not measured: constructed to make the POSITIVE verdict reachable, which is the property under
    test. Every eta here is an exact ratio of integers the way a counter reading is, so the flat
    steps are genuinely flat rather than flat after rounding.
    """
    return [
        _point("a", 16, 300, 28, 104),            # eta 0.2692...
        _point("b", 32, 373, 51, 137),            # eta 0.3722...
        _point("c", 64, 471, 128, 212),           # eta 0.6037...
        _point("d", 128, 604, 292, 400),          # eta 0.73
        _point("e", 256, 900, 584, 800),          # eta 0.73     -- step 0
        _point("f", 512, 1500, 1168, 1600),       # eta 0.73     -- step 0
        _point("g", 1024, 2700, 2336, 3200),      # eta 0.73     -- step 0
    ]


# ---------------------------------------------------------------------------------------------------
# the positive case: a ladder that settles is reported as settling, at the right depth
# ---------------------------------------------------------------------------------------------------


def test_a_ladder_that_flattens_reports_the_depth_it_flattened_at():
    got = FT.settling_depth(settled_points(), band=BAND, confirming_steps=CONFIRMING_STEPS)

    assert got["state"] == FT.SETTLED_AT
    # eta at d already equals eta at e, f and g, so the machine had finished filling BY d.
    assert got["settling_axis"] == 128
    assert got["settling_label"] == "d"
    assert got["confirmed_by_steps"] == 3
    assert [s["within_band"] for s in got["steps"]] == [False, False, False, True, True, True]


def test_the_declared_band_and_confirmation_count_are_echoed_into_the_result():
    """A settling depth quoted without them is not reproducible, so they travel with the number."""
    got = FT.settling_depth(settled_points(), band=BAND, confirming_steps=CONFIRMING_STEPS)

    assert got["declared"] == {"band": 0.01, "band_numerator": 1, "band_denominator": 100,
                               "confirming_steps": 2}


def test_one_flat_step_at_the_end_of_a_ladder_does_not_count_as_settled():
    """The failure mode a single confirmation cannot tell apart from a ladder that stopped too early."""
    truncated = settled_points()[:5]                      # ...rises, then exactly one flat step
    assert FT.settling_depth(truncated, band=BAND, confirming_steps=1)["state"] == FT.SETTLED_AT
    assert FT.settling_depth(truncated, band=BAND,
                             confirming_steps=CONFIRMING_STEPS)["state"] == FT.NEVER_SETTLES_IN_RANGE


# ---------------------------------------------------------------------------------------------------
# the negative case: the frozen cohort, on its own measured numbers
# ---------------------------------------------------------------------------------------------------


def test_the_frozen_refuted_cohort_does_not_settle_anywhere_inside_itself():
    """Which is exactly why its affine law was never testable on it -- and why this is a real answer."""
    got = FT.settling_depth(pk_points(), band=BAND, confirming_steps=CONFIRMING_STEPS)

    assert got["state"] == FT.NEVER_SETTLES_IN_RANGE
    assert got["settling_axis"] is None
    assert got["deepest_axis_measured"] == 128
    assert not any(s["within_band"] for s in got["steps"])
    # The same cohort the existing transient verdict calls IN_FILL_TRANSIENT. The two must agree:
    # a cohort that settles nowhere inside itself is a cohort that is still filling at its deepest
    # point, and a disagreement between them would mean one of the two is measuring something else.
    assert FT.transient_verdict(pk_points())["state"] == FT.IN_FILL_TRANSIENT


def test_the_refusal_names_the_deepest_point_it_reached_rather_than_only_refusing():
    """A bound from below is the useful part of a non-settling answer; it must be in the record."""
    got = FT.settling_depth(pk_points(), band=BAND, confirming_steps=CONFIRMING_STEPS)

    assert "128" in got["why"]
    assert got["final_step"] == pytest.approx(0.16608942879296976)
    assert got["deepest_eta"] == pytest.approx(0.7698630136986301)


# ---------------------------------------------------------------------------------------------------
# vacuity: the verdict is decided by the data, and an unmeasured point never decides it
# ---------------------------------------------------------------------------------------------------


def test_the_two_verdicts_are_reachable_from_data_that_differs_only_in_the_measurements():
    """The vacuity check. Same call, same band, same confirmation count -- opposite verdicts.

    Without this, a detector that hardcoded either answer would pass every other test in this file
    that happened to expect that answer.
    """
    kwargs = {"band": BAND, "confirming_steps": CONFIRMING_STEPS}
    assert FT.settling_depth(settled_points(), **kwargs)["state"] == FT.SETTLED_AT
    assert FT.settling_depth(pk_points(), **kwargs)["state"] == FT.NEVER_SETTLES_IN_RANGE


def test_a_point_with_no_overlap_reading_makes_the_depth_undeterminable_not_settled():
    """The recurring bug class in this package: the unmeasured reported as the measured-and-zero.

    A settling depth is the single number a successor cohort would be built on, so inferring one over
    the points that happened to read is the most expensive possible place to make that mistake.
    """
    partial = settled_points()
    partial[4] = FT.Point(label=partial[4].label, axis=partial[4].axis, cycles=partial[4].cycles,
                          overlap_detail="the bracket did not fire for this point")

    got = FT.settling_depth(partial, band=BAND, confirming_steps=CONFIRMING_STEPS)

    assert got["state"] == FT.UNDETERMINABLE
    assert got["settling_axis"] is None
    assert partial[4].label in got["overlap"]["unread"]


def test_a_band_finer_than_the_instrument_can_resolve_is_flagged_on_the_step_it_bites():
    """Eta moves in steps of 1/available. A band below that asks for a distinction nothing can make."""
    points = settled_points()
    resolutions = FT.eta_resolution(points)
    assert resolutions[0]["resolution"] == pytest.approx(1 / 104)     # the coarser of the first pair

    coarse = FT.settling_depth(points, band=BAND, confirming_steps=CONFIRMING_STEPS)
    assert coarse["unresolvable_steps"] == []

    fine = FT.settling_depth(points, band=Fraction(1, 10000), confirming_steps=CONFIRMING_STEPS)
    assert fine["unresolvable_steps"], "a band of 1e-4 is finer than 1/available at every step here"
    # Flagged, never silently promoted: the flat steps are exactly zero, so they are still inside a
    # band of zero width and the verdict stands -- what changes is that the record says the band was
    # asking for more than the reading could express.
    assert fine["state"] == FT.SETTLED_AT


# ---------------------------------------------------------------------------------------------------
# the arguments are decisions, and a nonsensical one is refused rather than defaulted
# ---------------------------------------------------------------------------------------------------


def test_a_confirmation_count_below_one_is_refused():
    with pytest.raises(ValueError, match="confirm"):
        FT.settling_depth(settled_points(), band=BAND, confirming_steps=0)


def test_a_negative_band_is_refused_because_it_would_accept_a_rising_step():
    with pytest.raises(ValueError, match="band"):
        FT.settling_depth(settled_points(), band=Fraction(-1, 100), confirming_steps=1)


# ---------------------------------------------------------------------------------------------------
# the marginal cost is the property a successor cohort is built on; eta only explains it
# ---------------------------------------------------------------------------------------------------

#: The frozen contract's own proportional allowance, restated so a test that quotes it is greppable
#: from either side. It is never modified here; it is passed IN, which is the point -- a claim's
#: settling depth is decided under that claim's own declared tolerance, not under a new one.
FROZEN_RESIDUAL_FRACTION = Fraction(3, 100)


def test_the_frozen_cohorts_marginal_cost_never_stops_changing_inside_itself():
    """Which is the threshold-free half of its refutation, restated as a depth question."""
    got = FT.marginal_settling_depth(pk_points(), relative_band=FROZEN_RESIDUAL_FRACTION,
                                     confirming_steps=CONFIRMING_STEPS)

    assert got["state"] == FT.NEVER_SETTLES_IN_RANGE
    assert got["settling_axis"] is None
    # Each marginal is about a third below the one before it -- an order of magnitude outside the
    # 3% the family's own contract allows its residuals.
    assert all(s["relative_change"] > 0.3 for s in got["steps"])


def test_a_ladder_whose_marginal_cost_goes_constant_reports_the_depth_it_did_so():
    """The positive case. Cycles become exactly affine from the third point on."""
    settled = [
        FT.Point("a", 16, 300), FT.Point("b", 32, 380),          # marginal 5.0
        FT.Point("c", 64, 500), FT.Point("d", 128, 740),         # marginal 3.75, then 3.75
        FT.Point("e", 256, 1220),                                # marginal 3.75
    ]
    got = FT.marginal_settling_depth(settled, relative_band=FROZEN_RESIDUAL_FRACTION,
                                     confirming_steps=CONFIRMING_STEPS)

    assert got["state"] == FT.SETTLED_AT
    assert got["settling_axis"] == 32
    assert got["confirmed_by_steps"] == 2


def test_the_marginal_verdict_needs_no_overlap_reading_at_all():
    """It is arithmetic about the cycles. The overlap reading supplies the mechanism, not the fact.

    So a cohort whose bracket never fired still yields a marginal verdict, while its eta verdict is
    correctly UNDETERMINABLE. Collapsing the two would make an unmeasured instrument look like a
    machine whose marginal cost could not be decided.
    """
    unread = [FT.Point(p.label, p.axis, p.cycles) for p in pk_points()]

    assert FT.settling_depth(unread, band=BAND,
                             confirming_steps=CONFIRMING_STEPS)["state"] == FT.UNDETERMINABLE
    assert FT.marginal_settling_depth(unread, relative_band=FROZEN_RESIDUAL_FRACTION,
                                      confirming_steps=CONFIRMING_STEPS)["state"] == \
        FT.NEVER_SETTLES_IN_RANGE


def test_two_points_cannot_decide_whether_a_marginal_is_changing():
    got = FT.marginal_settling_depth(pk_points()[:2], relative_band=FROZEN_RESIDUAL_FRACTION,
                                     confirming_steps=1)

    assert got["state"] == FT.UNDETERMINABLE
    assert "at least two" in got["why"]


def test_a_marginal_that_goes_constant_and_then_changes_again_reports_its_WINDOW():
    """The case that cost this derivation its first answer, and the reason the window is reported.

    A ladder truncated before the marginal moves again reports SETTLED_AT and hands a successor cohort
    a depth with no upper edge. Extending it two rungs turns the same evidence into NEVER_SETTLES --
    correctly, because the cost per extra unit is not constant over the whole range -- and the useful
    residue is the stretch over which it DID hold. A window is still a place a steady-state law can be
    tested; it is simply bounded from both sides rather than only from below.
    """
    truncated = [FT.Point("a", 16, 220), FT.Point("b", 32, 319), FT.Point("c", 64, 454),
                 FT.Point("d", 128, 740), FT.Point("e", 256, 1321), FT.Point("f", 512, 2503),
                 FT.Point("g", 1024, 4828), FT.Point("h", 2048, 9532)]
    extended = truncated + [FT.Point("i", 4096, 19493), FT.Point("j", 8192, 42669)]
    kwargs = {"relative_band": FROZEN_RESIDUAL_FRACTION, "confirming_steps": CONFIRMING_STEPS}

    stopped_early = FT.marginal_settling_depth(truncated, **kwargs)
    assert stopped_early["state"] == FT.SETTLED_AT
    assert stopped_early["settling_axis"] == 64

    full = FT.marginal_settling_depth(extended, **kwargs)
    assert full["state"] == FT.NEVER_SETTLES_IN_RANGE
    assert full["settling_axis"] is None

    window = full["longest_within_band_window"]
    assert window["axis_from"] == 64 and window["axis_to"] == 2048
    assert window["closes_before_the_deepest_point"] is True


def test_a_window_that_runs_to_the_end_is_not_reported_as_closing():
    full = FT.marginal_settling_depth(
        [FT.Point("a", 16, 300), FT.Point("b", 32, 380), FT.Point("c", 64, 500),
         FT.Point("d", 128, 740), FT.Point("e", 256, 1220)],
        relative_band=FROZEN_RESIDUAL_FRACTION, confirming_steps=CONFIRMING_STEPS)

    assert full["state"] == FT.SETTLED_AT
    assert full["longest_within_band_window"]["closes_before_the_deepest_point"] is False
