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
