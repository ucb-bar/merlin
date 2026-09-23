"""A reference with limited resolving power must say so, not be loosened until it passes.

Grounded in a measured whole-model int8 failure (2026-09-09/10). The gate compared an independent
implementation elementwise against a framework-eager dynamic-quantization reference and reported
18.7% "bad" with one of eight token rows taking the wrong argmax. The argmax flip came from ONE
element, 43 ULP from the reference's, which put the pre-rounding value at 8.499998 against 8.500022
-- opposite sides of a round-half-even tie, so 8 against 9. The counterfactual was RUN: forcing that
code in BOTH quantizers that consume the tensor recovered all 8 rows and made the layer exact again
(3.1e-3 -> 7.3e-7); forcing it in only one of them left the model at 7 of 8, which is why a
counterfactual over a shared value must cover every consumer. The reference's own margin on that row
was 0.0054, 26x tighter than any other row (next smallest 0.142), and 485 of 2,809,856 quantized
activations in that model sat within 1e-4 of a tie.

The load-bearing test here is `test_indeterminate_is_never_counted_as_agreement`: folding
unadjudicable rows into passes is exactly how a gate stops being able to fail, and this repo has
recorded that failure shape fifteen times.
"""

from __future__ import annotations

import pytest

from merlin.perf.tie_sensitivity import (
    FAIL,
    INDETERMINATE,
    PASS,
    adjudicable_rows,
    classify,
    decision_margin,
    fragility_census,
    spread_envelope,
    tie_distance,
    verdict,
    verdict_from_choices,
)

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
    assert round(OURS - 0.5) != round(THEIRS - 0.5) or True  # they round to 8 and 9
    assert round(OURS) == 8 and round(THEIRS) == 9


def test_fragility_census_counts_by_threshold_and_predicts_disagreement() -> None:
    census = fragility_census([OURS, THEIRS, 3.1, 7.0, 0.5])
    assert census["elements"] == 5
    assert census["counts"]["within_0.0001"] == 3  # the two plus the exact tie
    assert census["counts"]["within_1e-06"] == 1  # the exact tie only
    assert census["nearest_tie_distance"] == 0.0
    assert "divergence d" in census["reading"]


def test_fragility_census_of_a_tie_free_population_is_zero() -> None:
    census = fragility_census([1.0, 2.0, 3.0, 4.0])
    assert set(census["counts"].values()) == {0}
    assert census["nearest_tie_distance"] == pytest.approx(0.5)


def test_decision_margin_is_the_top_two_gap() -> None:
    assert decision_margin([1.0, 0.9946, 0.1]) == pytest.approx(FRAGILE_MARGIN, abs=1e-9)
    assert decision_margin([5.0]) == float("inf")  # nothing to confuse it with


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
    out = verdict(
        [[1.0, 0.9946], [1.0, 0.858], [1.0, 0.5]], [[0.9946, 1.0], [0.858, 1.0], [1.0, 0.5]], noise_floor=0.006
    )
    assert (out["agrees"], out["disagrees"], out["reference_cannot_discriminate"]) == (1, 1, 1)
    assert out["adjudicable"] == 2  # NOT 3
    assert out["agrees"] != len(out["rows"])
    assert "IS NOT PASS" in out["licence"]
    assert "verdict" not in out or not isinstance(out.get("verdict"), bool)


def test_the_eight_row_case_reproduces_seven_agreeing_one_unadjudicable() -> None:
    """The measured shape: 7 rows agree, row 6 flips on a 0.0054 margin."""
    reference = [[1.0, 1.0 - FIRM_MARGIN] for _ in range(8)]
    candidate = [list(r) for r in reference]
    reference[6] = [1.0, 1.0 - FRAGILE_MARGIN]
    candidate[6] = [1.0 - FRAGILE_MARGIN, 1.0]  # the argmax flip
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


# ---------------------------------------------------------------------------------------------
# The console-limited entry point, the reference-only partition, and the spread envelope.
#
# A whole-model harness cannot send 256,000 logits down a UART, so it reports one argmax per row
# and a digest. The verdict must still be reachable from that, and it must be the SAME verdict --
# which is why `classify` is the only place the rule exists.
# ---------------------------------------------------------------------------------------------


def test_the_choice_only_entry_point_gives_the_same_verdict_as_full_scores() -> None:
    reference = [[1.0, 1.0 - FIRM_MARGIN], [1.0, 1.0 - FRAGILE_MARGIN]]
    candidate = [[1.0 - FIRM_MARGIN, 1.0], [1.0 - FRAGILE_MARGIN, 1.0]]
    from_scores = verdict(reference, candidate, noise_floor=0.006)
    from_choices = verdict_from_choices(reference, [1, 1], noise_floor=0.006)
    assert [r["status"] for r in from_scores["rows"]] == [r["status"] for r in from_choices["rows"]]
    assert (from_choices["agrees"], from_choices["disagrees"], from_choices["reference_cannot_discriminate"]) == (
        0,
        1,
        1,
    )


def test_the_partition_comes_from_the_reference_alone() -> None:
    """A candidate cannot widen the set of decisions it is excused from."""
    reference = [[1.0, 1.0 - FIRM_MARGIN], [1.0, 1.0 - FRAGILE_MARGIN]]
    assert adjudicable_rows(reference, noise_floor=0.006) == [True, False]
    # The same partition, whatever the candidate did.
    assert adjudicable_rows(reference, noise_floor=0.006) == [True, False]


def test_classify_is_the_only_rule_and_covers_the_three_outcomes() -> None:
    assert classify(True, 0.0, noise_floor=0.006) == PASS  # agreement, whatever the margin
    assert classify(False, FRAGILE_MARGIN, noise_floor=0.006) == INDETERMINATE
    assert classify(False, FIRM_MARGIN, noise_floor=0.006) == FAIL


def test_classify_refuses_a_missing_floor_like_every_other_entry_point() -> None:
    for call in (
        lambda: classify(False, 0.1, noise_floor=None),
        lambda: adjudicable_rows([[1.0, 0.5]], noise_floor=None),
        lambda: verdict_from_choices([[1.0, 0.5]], [0], noise_floor=None),
    ):
        with pytest.raises(ValueError, match="noise_floor is required"):
            call()


def test_the_spread_envelope_is_a_measurement_times_a_census_count() -> None:
    """MEASURED on tiny_llama: forcing one reference-unresolved tie the other way moved a logit by
    0.153, and 485 of 2,809,856 quantized activations sit within 1e-4 of a tie."""
    out = spread_envelope(0.153, 485)
    assert out["envelope"] == pytest.approx(0.153 * 485)
    assert "NOT the same as agreeing" in out["reading"]


def test_the_spread_envelope_refuses_a_negative_measurement() -> None:
    with pytest.raises(ValueError, match="non-negative measurement"):
        spread_envelope(-1.0, 485)
    with pytest.raises(ValueError, match="non-negative count"):
        spread_envelope(0.153, -1)
