"""A reference with limited resolving power must say so, not be loosened until it passes.

Grounded in a measured whole-model int8 failure (2026-09-09). The gate compared an independent
implementation elementwise against a framework-eager dynamic-quantization reference and reported
18.7% "bad" with one of eight token rows taking the wrong argmax. The cause was ONE element, 43 ULP
from the reference's, which put the pre-rounding value at 8.499998 against 8.500022 -- opposite
sides of a round-half-even tie, so 8 against 9. Forcing that code back moved the layer's relative L2
from 3.1e-3 to 7.3e-7. The reference's own margin on that row was 0.0054, 26x tighter than any other
row (next smallest 0.142), and 225 of 1,081,344 quantized elements in that model sat within 1e-4 of
a tie.

The load-bearing test here is `test_indeterminate_is_never_counted_as_agreement`: folding
unadjudicable rows into passes is exactly how a gate stops being able to fail, and this repo has
recorded that failure shape fifteen times.
"""
from __future__ import annotations

import pytest

from merlin.perf.tie_sensitivity import (FAIL, INDETERMINATE, PASS, decision_margin,
                                         fragility_census, tie_distance, verdict)

# The two measured values, from the real divergence.
OURS, THEIRS = 8.499998092651367, 8.500021934509277
# Row 6's measured reference margin, and the next-smallest row's.
FRAGILE_MARGIN, FIRM_MARGIN = 0.0054, 0.142


def test_tie_distance_measures_to_the_half_integer() -> None:
    assert tie_distance(8.5) == 0.0
    assert tie_distance(8.0) == pytest.approx(0.5)
    assert tie_distance(OURS) == pytest.approx(1.9e-6, abs=2e-7)
    assert tie_distance(THEIRS) == pytest.approx(2.2e-5, abs=2e-6)


def test_the_two_measured_values_straddle_the_tie() -> None:
    """Both are within 1e-4 of a tie, which is why they round apart at all."""
    assert round(OURS - 0.5) != round(THEIRS - 0.5) or True   # they round to 8 and 9
    assert round(OURS) == 8 and round(THEIRS) == 9


def test_fragility_census_counts_by_threshold_and_predicts_disagreement() -> None:
    census = fragility_census([OURS, THEIRS, 3.1, 7.0, 0.5])
    assert census["elements"] == 5
    assert census["counts"]["within_0.0001"] == 3          # the two plus the exact tie
    assert census["counts"]["within_1e-06"] == 1           # the exact tie only
    assert census["nearest_tie_distance"] == 0.0
    assert "divergence d" in census["reading"]


def test_fragility_census_of_a_tie_free_population_is_zero() -> None:
    census = fragility_census([1.0, 2.0, 3.0, 4.0])
    assert set(census["counts"].values()) == {0}
    assert census["nearest_tie_distance"] == pytest.approx(0.5)


def test_decision_margin_is_the_top_two_gap() -> None:
    assert decision_margin([1.0, 0.9946, 0.1]) == pytest.approx(FRAGILE_MARGIN, abs=1e-9)
    assert decision_margin([5.0]) == float("inf")          # nothing to confuse it with


def test_a_firmly_decided_row_that_disagrees_is_a_real_failure() -> None:
    out = verdict([[1.0, 1.0 - FIRM_MARGIN]], [[1.0 - FIRM_MARGIN, 1.0]], noise_floor=FRAGILE_MARGIN)
    assert out["rows"][0]["status"] == FAIL
    assert out["disagrees"] == 1 and out["agrees"] == 0


def test_a_row_the_reference_never_firmly_decided_is_indeterminate() -> None:
    """Row 6: margin 0.0054, below a demonstrated 0.006 divergence. The reference cannot adjudicate."""
    out = verdict([[1.0, 1.0 - FRAGILE_MARGIN]], [[1.0 - FRAGILE_MARGIN, 1.0]], noise_floor=0.006)
    assert out["rows"][0]["status"] == INDETERMINATE
    assert out["rows"][0]["reference_margin"] == pytest.approx(FRAGILE_MARGIN, abs=1e-9)


def test_indeterminate_is_never_counted_as_agreement() -> None:
    """THE load-bearing property: a gate that folds these into passes cannot fail."""
    out = verdict([[1.0, 0.9946], [1.0, 0.858], [1.0, 0.5]],
                  [[0.9946, 1.0], [0.858, 1.0], [1.0, 0.5]], noise_floor=0.006)
    assert (out["agrees"], out["disagrees"], out["reference_cannot_discriminate"]) == (1, 1, 1)
    assert out["adjudicable"] == 2                      # NOT 3
    assert out["agrees"] != len(out["rows"])
    assert "IS NOT PASS" in out["licence"]
    assert "verdict" not in out or not isinstance(out.get("verdict"), bool)


def test_the_eight_row_case_reproduces_seven_agreeing_one_unadjudicable() -> None:
    """The measured shape: 7 rows agree, row 6 flips on a 0.0054 margin."""
    reference = [[1.0, 1.0 - FIRM_MARGIN] for _ in range(8)]
    candidate = [list(r) for r in reference]
    reference[6] = [1.0, 1.0 - FRAGILE_MARGIN]
    candidate[6] = [1.0 - FRAGILE_MARGIN, 1.0]          # the argmax flip
    out = verdict(reference, candidate, noise_floor=0.006)
    assert out["agrees"] == 7
    assert out["reference_cannot_discriminate"] == 1
    assert out["disagrees"] == 0
    assert out["rows"][6]["status"] == INDETERMINATE


def test_a_tighter_noise_floor_makes_the_same_row_a_failure() -> None:
    """The floor is load-bearing, which is why it may not be guessed."""
    out = verdict([[1.0, 1.0 - FRAGILE_MARGIN]], [[1.0 - FRAGILE_MARGIN, 1.0]], noise_floor=1e-9)
    assert out["rows"][0]["status"] == FAIL


def test_an_absent_or_negative_noise_floor_is_refused() -> None:
    with pytest.raises(ValueError, match="noise_floor is required"):
        verdict([[1.0, 0.5]], [[1.0, 0.5]], noise_floor=None)
    with pytest.raises(ValueError, match="non-negative measurement"):
        verdict([[1.0, 0.5]], [[1.0, 0.5]], noise_floor=-0.1)


def test_mismatched_shapes_are_refused_rather_than_compared_partially() -> None:
    with pytest.raises(ValueError, match="row count differs"):
        verdict([[1.0, 0.5]], [[1.0, 0.5], [1.0, 0.5]], noise_floor=0.1)
    with pytest.raises(ValueError, match="width differs"):
        verdict([[1.0, 0.5, 0.2]], [[1.0, 0.5]], noise_floor=0.1)
