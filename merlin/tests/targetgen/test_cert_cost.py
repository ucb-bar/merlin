"""Sizing a capsule against what its certification actually costs.

A capsule derived at an application's real shape is worthless if nobody can afford to certify it,
and the sweet spot between "too small to generalize" and "too big to simulate" is not something
anyone can pick by eye. These tests pin the two properties that make the cost model usable: it is
fitted from runs already paid for, and it REFUSES rather than guesses everywhere the evidence runs
out -- a target with no history, a budget below the fixed floor, a size past anything measured.

The measured shape of it, on gemmini, is why this module exists at all: a fixed floor of ~115 s that
a capsule pays for existing, and ~0.06 s per operand element on top, so the floor dominates below
~1900 elements while today's capsules are 256-512. The corpus is paying nearly the whole cost of a
certification to exercise a 16x16 tile.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import cert_cost as CC


def _fit_or_skip(target: str = "gemmini"):
    fit = CC.fit_for(target)
    if fit is None:
        pytest.skip(f"{target} has no measured certification history in this checkout")
    return fit


def test_a_target_with_no_certification_history_has_no_cost_model():
    """The refusal that matters most. A default here would be a number nobody measured driving a
    size somebody certifies and then quotes."""
    assert CC.fit_for("definitely_not_a_target") is None


def test_the_fit_rests_on_enough_distinct_sizes_to_be_a_line():
    """Two points define a line through anything, and one x-value defines nothing at all."""
    fit = _fit_or_skip()
    assert fit.n_samples >= 5
    assert fit.elements_max > fit.elements_min
    assert fit.sources, "a fit must name the runs it was built from"


def test_the_fit_predicts_the_runs_it_was_built_from():
    """A fit nobody checked against its own inputs is a straight line with a plausible slope. This
    is deliberately a loose band -- simulator time is noisy -- but it fails on a fit that is simply
    wrong, e.g. one whose intercept and slope have swapped roles."""
    fit = _fit_or_skip()
    mid = (fit.elements_min + fit.elements_max) // 2
    predicted = CC.predict_seconds(fit, mid)
    assert predicted is not None
    assert fit.intercept_s <= predicted <= fit.intercept_s + fit.per_element_s * fit.elements_max * 1.5


def test_a_fixed_floor_is_reported_so_headroom_is_visible():
    """`floor_dominates_below` is the number that says how much larger a capsule can get before it
    is paying for its size rather than for existing -- i.e. where representativeness is nearly free.
    It is the whole reason a derived capsule can be bigger than a tile without costing more."""
    fit = _fit_or_skip()
    assert fit.intercept_s > 0
    if fit.per_element_s > 0:
        assert fit.floor_dominates_below > 0


def test_a_prediction_past_the_measured_range_is_absent_not_large():
    """A fit built on hundreds of elements says nothing about hundreds of thousands. Returning a
    number anyway is how a capsule nobody could afford gets scheduled on the strength of
    arithmetic."""
    fit = _fit_or_skip()
    assert CC.predict_seconds(fit, fit.elements_max * 100) is None
    assert CC.predict_seconds(fit, 0) is None
    assert CC.predict_seconds(None, 1024) is None


def test_a_budget_below_the_fixed_floor_admits_no_capsule_at_all():
    """Not "a very small capsule" -- none. The floor is paid before any work happens, so a budget
    under it is a statement about the budget rather than about the shape."""
    fit = _fit_or_skip()
    assert CC.max_elements_within(fit, fit.intercept_s * 0.5) is None
    assert CC.max_elements_within(None, 600.0) is None


def test_a_generous_budget_is_clamped_to_what_the_evidence_supports():
    """The inverse of the refusal above: a budget large enough to imply a size far past anything
    measured yields the largest size the evidence actually supports, not the arithmetic answer."""
    fit = _fit_or_skip()
    huge = CC.max_elements_within(fit, fit.intercept_s + fit.per_element_s * fit.elements_max * 1000)
    assert huge is not None
    assert huge <= fit.elements_max * 2


def test_a_larger_budget_never_admits_a_smaller_capsule():
    fit = _fit_or_skip()
    small = CC.max_elements_within(fit, fit.intercept_s + 50)
    large = CC.max_elements_within(fit, fit.intercept_s + 500)
    if small is not None and large is not None:
        assert large >= small


def test_the_size_metric_is_the_largest_operand_and_tolerates_a_symbolic_dim():
    """Chosen by measurement, not argument: across the gemmini runs the largest single operand
    predicts cost better than total operand elements, and declared OUTPUT elements is degenerate
    because a capsule records its inputs and not its result shape."""
    assert CC.capsule_elements({"inputs": [
        {"name": "A0", "shape": [16, 32]}, {"name": "W", "shape": [32, 64]}]}) == 2048
    assert CC.capsule_elements({"inputs": []}) == 0
    # A symbolic dim makes THAT operand unmeasurable, not the capsule.
    assert CC.capsule_elements({"inputs": [
        {"name": "A0", "shape": ["?", 32]}, {"name": "W", "shape": [8, 8]}]}) == 64


def test_the_fit_predicts_capsules_it_has_never_seen():
    """The falsification that matters: refit without each capsule, then predict it. A line through
    its own inputs proves nothing — this asks whether the model generalizes to a size it was not
    told about, which is exactly what sizing a NEW capsule requires of it.

    The bound is deliberately loose. Measured on gemmini the median absolute error is 17.5% and the
    worst 51%, so this is a sizing instrument rather than a stopwatch; the assertion exists to catch
    a model that has stopped predicting at all, not to pretend to a precision it does not have."""
    import statistics

    from merlin.common.paths import merlin_dir

    timings = CC._timing_records("gemmini")
    sizes = CC._capsule_sizes([merlin_dir() / "contract" / "capsules"])
    points = [(sizes[n], s) for n, (s, _src) in sorted(timings.items()) if sizes.get(n)]
    if len(points) < CC._MIN_SAMPLES + 1:
        pytest.skip("not enough measured capsules to hold one out")

    def _line(pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        den = sum((x - mx) ** 2 for x in xs)
        if den == 0:
            return None
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
        return my - slope * mx, slope

    errors = []
    for i in range(len(points)):
        held = _line(points[:i] + points[i + 1:])
        if held is None:
            continue
        intercept, slope = held
        x, y = points[i]
        errors.append(abs((intercept + slope * x) - y) / y)
    assert errors
    assert statistics.median(errors) < 0.35, (
        f"the cost model no longer predicts held-out capsules (median error "
        f"{statistics.median(errors):.0%}); sizing against it would be guessing")
    assert sum(1 for e in errors if e <= 0.5) >= 0.8 * len(errors), (
        "fewer than four in five held-out capsules land within 50% of prediction")


# --- which ENGINE produced the second ---------------------------------------------------------------
#
# Two elaborated-RTL engines answer the same capsule at the same fidelity and are NOT interchangeable as
# cost samples: measured on gemmini against the identical ELF, GSIM answers in 3.31 s where Verilator
# takes 86.83 s (hardware_pins.yaml, `gsim_compiler`). A fit over a mixture prices a capsule at neither
# engine's cost. The per-capsule record carries `engine`; the reshaping into `by_tier` used to drop it,
# so the mixture was not merely unhandled, it was invisible.

def test_the_engine_survives_the_reshaping_into_a_by_tier_block():
    doc = {"tiers": {"L3": {"cycle_accurate": True, "engine": "gsim",
                            "timing": {"sim_active_s": 3.31}}}}
    assert CC._per_tier_from_result(doc)["L3"]["engine"] == "gsim"


def test_the_engine_rides_in_the_basis_so_a_mixed_fit_is_visible():
    """The basis is the string every caller already keeps beside the number, which makes this readable
    off the fit's own sources rather than requiring a new channel."""
    fast = {"by_tier": {"L3": {"cycle_accurate": True, "engine": "gsim",
                               "sim_active_s": 3.31}}}
    slow = {"by_tier": {"L3": {"cycle_accurate": True, "engine": "verilator",
                               "sim_active_s": 86.83}}}
    _s_fast, basis_fast = CC._cycle_accurate_seconds(fast)
    _s_slow, basis_slow = CC._cycle_accurate_seconds(slow)
    assert basis_fast.endswith("@gsim")
    assert basis_slow.endswith("@verilator")
    assert basis_fast != basis_slow


def test_a_sample_with_no_recorded_engine_still_yields_a_basis():
    """Older records predate the field. They must keep contributing rather than start being dropped --
    the point is to make the mixture visible, not to discard history."""
    old = {"by_tier": {"L3": {"cycle_accurate": True, "sim_active_s": 12.0}}}
    seconds, basis = CC._cycle_accurate_seconds(old)
    assert seconds == 12.0
    assert basis and "@" not in basis
