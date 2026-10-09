"""A fast cycle signal must survive held-out schedule ordering, not just a fit."""

from dataclasses import replace

import pytest

from merlin.common.jsonio import canonical_sha256 as sha
from merlin.perf import fast_estimate_validation as fast
from merlin.perf import rank_validation as rank
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

POINTER = "/features/work"


def observations(target="descriptor-a", rate=3):
    return [
        fast.Observation(
            sha([group, value]),
            sha(group),
            group,
            sha([target, "scope", "regime"]),
            {POINTER: value},
            7 + rate * value,
            (sha([group, value, "measurement"]),),
        )
        for group in ("family-a", "family-b", "family-c")
        for value in (1, 2, 3, 4)
    ]


def fit(rows):
    return fast.fit_linear_screen(rows, pointers=(POINTER,), include_fixed=True, maximum_condition=100)


def validate(rows, fitter=fit):
    return fast.cross_validate(
        rows,
        fitter,
        maximum_relative_error=1e-10,
        minimum_predictions=12,
        minimum_rank_rate=0.9,
        minimum_decided=12,
        minimum_slice_decided=4,
        minimum_slices=2,
    )


@pytest.mark.parametrize("target,rate", [("descriptor-a", 3), ("descriptor-b", 23)])
def test_grouped_fit_works_on_distinct_targets_without_leaking_labels(target, rate):
    rows = observations(target, rate)
    training = []

    def checked_fit(train):
        training.append({row.group for row in train})
        return fit(train)

    receipt = validate(rows, checked_fit)
    assert receipt["exposable"]
    assert len(training) == 3 and all(len(groups) == 2 for groups in training)
    for row in receipt["predictions"]:
        assert row["program"] not in row["training_programs"]
        assert row["relative_error"] < 1e-12
    model = fit(rows)
    assert model.fixed_cycles == pytest.approx(7)
    assert model.coefficients == pytest.approx((rate,))


def test_absolute_accuracy_does_not_approve_wrong_schedule_ranking():
    rows = observations()

    class Constant:
        def predict(self, features, *, domain_sha256):
            return CycleInterval.point(15)

    result = validate(rows, lambda train: Constant())
    assert not result["exposable"]
    assert result["ranking"]["overall"]["decided"] == 0


def test_overlapping_intervals_never_use_midpoints_to_rank():
    pairs = [(rank.Program("w", "a", 100), rank.Program("w", "b", 200))]
    result = rank.interval_agreement(pairs, {"a": (90, 220), "b": (110, 210)})
    assert result.decided == 0 and result.undecided == 1
    assert rank.interval_agreement(pairs, {"a": (90, 100), "b": (110, 210)}).agreed == 1
    assert rank.interval_agreement(pairs, {"a": (90, 110), "b": (110, 210)}).decided == 0
    with pytest.raises(ValueError):
        rank.interval_agreement(pairs, {"a": (0, float("inf"))})


def test_same_feature_different_time_reports_unavoidable_point_error():
    a = observations()[0]
    b = replace(a, program=sha("different executable"), cycles=20)
    collisions = fast.feature_collisions([a, b], (POINTER,))
    assert collisions[0]["minimum_point_relative_error"] == pytest.approx(1 / 3)
    assert collisions[0]["cycles"] == [10, 20]


def test_unknown_and_extrapolated_features_are_not_zero_cycles():
    model = fit(observations())
    for value in (None, 0, 5, float("nan")):
        estimate = model.predict({POINTER: value}, domain_sha256=observations()[0].domain)
        assert not estimate.resolved and estimate.lo is None and estimate.missing
    assert not model.predict({}, domain_sha256=observations()[0].domain).resolved
    assert not model.predict({POINTER: 2}, domain_sha256=sha("different target")).resolved


def test_held_out_domain_is_checked_against_training_not_whole_corpus():
    rows = observations()
    rows[0] = replace(rows[0], features={POINTER: 100}, cycles=307)
    result = validate(rows)
    refused = next(row for row in result["predictions"] if row["program"] == rows[0].program)
    assert not refused["prediction"]["resolved"]
    assert "training domain" in refused["prediction"]["missing"][0]
    assert not result["exposable"]


def test_finite_features_and_rates_that_overflow_leave_prediction_unknown():
    model = replace(fit(observations()), coefficients=(1e308,), fixed_cycles=1e308)
    estimate = model.predict({POINTER: 4}, domain_sha256=model.domain_sha256)
    assert not estimate.resolved and estimate.lo is None and estimate.hi is None
    assert estimate.missing == ("screening prediction has no finite nonnegative cost",)


def test_workload_variants_cannot_leak_between_folds():
    rows = observations()
    rows[1] = replace(rows[1], group="leaky split")
    with pytest.raises(ValueError, match="one held-out group"):
        validate(rows)


def test_mixed_target_scope_or_regime_refused():
    rows = observations()
    rows[0] = replace(rows[0], domain=sha("different load regime"))
    with pytest.raises(ValueError, match="cannot mix"):
        validate(rows)
    with pytest.raises(ValueError, match="one exact"):
        fit(rows)


def test_underdetermined_collinear_and_negative_rates_refused():
    rows = observations()
    with pytest.raises(ValueError, match="two distinct points"):
        fit(rows[:2])
    correlated = [replace(row, features={**row.features, "/features/alias": row.features[POINTER]}) for row in rows]
    with pytest.raises(ValueError, match="ill-conditioned"):
        fast.fit_linear_screen(
            correlated, pointers=(POINTER, "/features/alias"), include_fixed=False, maximum_condition=100
        )
    decreasing = [replace(row, cycles=20 - row.features[POINTER]) for row in rows]
    with pytest.raises(ValueError, match="nonnegative"):
        fit(decreasing)


def test_model_receipt_binds_feature_selection_and_fixed_policy():
    rows = observations()
    fixed = fit(rows)
    proportional = fast.fit_linear_screen(rows, pointers=(POINTER,), include_fixed=False, maximum_condition=100)
    assert fixed.provenance_sha256 != proportional.provenance_sha256


def test_replica_deduplication_preserves_different_workload_observations():
    rows = observations()
    with pytest.raises(ValueError, match="replicates"):
        validate([*rows, rows[0]])
    rows[4] = replace(rows[4], program=rows[0].program)
    assert validate(rows)["exposable"]
