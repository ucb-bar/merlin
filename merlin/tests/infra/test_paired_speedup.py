"""The performance promotion bound is paired, deterministic, and fail-closed."""

import pytest

from merlin.perf.paired import paired_speedup_confidence, shared_accuracy_verdict


def test_uniform_twenty_percent_win_clears_five_percent_bound():
    result = paired_speedup_confidence(
        ours_ns=[100, 102, 98, 101, 99, 100, 103],
        reference_ns=[120, 122.4, 117.6, 121.2, 118.8, 120, 123.6],
        margin=1.05,
    )
    assert result["geometric_speedup"] == pytest.approx(1.2)
    assert result["lower_confidence_bound"] == pytest.approx(1.2)
    assert result["passes"] is True


def test_noisy_point_win_does_not_become_a_certified_win():
    result = paired_speedup_confidence(
        ours_ns=[100, 100, 100, 100, 100, 100, 100],
        reference_ns=[90, 95, 101, 102, 103, 110, 120],
        margin=1.05,
    )
    assert result["geometric_speedup"] > 1.0
    assert result["lower_confidence_bound"] < 1.05
    assert result["passes"] is False


@pytest.mark.parametrize(
    "ours,reference,reason",
    [
        ([1], [2], "at least two"),
        ([1, 2], [2], "same number"),
        ([1, 0], [2, 2], "positive"),
        ([1, 2], [2, -1], "positive"),
    ],
)
def test_incomplete_or_invalid_pairs_are_refused(ours, reference, reason):
    with pytest.raises(ValueError, match=reason):
        paired_speedup_confidence(ours, reference)


def test_bootstrap_is_reproducible():
    args = dict(
        ours_ns=[101, 99, 103, 97, 100, 102, 98],
        reference_ns=[110, 108, 109, 111, 112, 107, 113],
        resamples=2_000,
        seed=73,
    )
    assert paired_speedup_confidence(**args) == paired_speedup_confidence(**args)


def test_shared_accuracy_requires_the_complete_fp32_comparison():
    bar = {"cos_threshold": 0.99, "rel_threshold": 0.05, "basis": "fixture"}
    good = shared_accuracy_verdict(
        {"fp32_cos": 0.995, "fp32_rel": 0.04, "comparison_complete": True}, bar)
    partial = shared_accuracy_verdict(
        {"fp32_cos": 0.999, "fp32_rel": 0.001, "comparison_complete": False}, bar)
    wrong_tier = shared_accuracy_verdict(
        {"w8a8_cos": 1.0, "w8a8_rel": 0.0, "comparison_complete": True}, bar)
    assert good["passes"] is True
    assert partial["passes"] is False and "complete" in partial["reason"]
    assert wrong_tier["passes"] is False and "fp32" in wrong_tier["reason"]


def test_shared_accuracy_names_the_failed_term():
    bar = {"cos_threshold": 0.99, "rel_threshold": 0.05, "basis": "fixture"}
    result = shared_accuracy_verdict(
        {"fp32_cos": 0.999, "fp32_rel": 0.06, "comparison_complete": True}, bar)
    assert result["passes"] is False
    assert result["cos_passes"] is True and result["rel_passes"] is False
    assert "relative error" in result["reason"]
