"""Versioned output policies retain each result's original numerical boundary."""

from copy import deepcopy
from decimal import Decimal

import pytest

from merlin.targetgen import capsule_golden as G
from merlin.targetgen import capsule_runner as R


def policy():
    return {
        "schema": "merlin.numeric_policy.per_output.v1",
        "compare": "per_output",
        "outputs": {
            "count": {"compare": "exact_int", "dtype": "i32"},
            "value": {"compare": "tolerance_float", "dtype": "f32", "atol": 0.02, "rtol": 0.0},
        },
    }


def test_each_output_uses_its_declared_policy():
    expected = {"count": [[16777217]], "value": [[[1.0, 2.0]]]}
    actual = {"count": [[16777217]], "value": [[1.01, 2.01]]}
    selected = policy()
    before = deepcopy(selected)
    assert R._match_by_policy(actual, expected, selected)
    assert selected == before
    actual["count"][0][0] += 1
    assert not R._match_by_policy(actual, expected, selected)


def test_exact_leaf_cannot_use_global_float_default_or_integer_truncation():
    expected = {"count": [[1]], "value": [[2.0]]}
    actual = {"count": [[1.0001]], "value": [[2.0]]}
    assert not R._match_by_policy(actual, expected, policy())


def test_flat_force_cannot_replace_per_output_thresholds():
    selected = policy()
    with pytest.raises(ValueError, match="per-output"):
        R._merge_match_policy({"compare": "float", "atol": 100.0}, selected)


def test_no_force_keeps_original_policy_and_identical_policy_is_not_an_override():
    selected = policy()
    assert R._merge_match_policy(None, selected) is selected
    assert R._merge_match_policy(deepcopy(selected), selected) == selected


@pytest.mark.parametrize("mutation", ["missing", "extra", "empty", "length", "order"])
def test_complete_output_roster_and_scalar_order_are_required(mutation):
    expected = {"count": [[1, 2]], "value": [[3.0, 4.0]]}
    actual = deepcopy(expected)
    if mutation == "missing":
        actual.pop("count")
    elif mutation == "extra":
        actual["other"] = [[0]]
    elif mutation == "empty":
        actual = {}
        expected = {}
    elif mutation == "length":
        actual["value"][0].pop()
    else:
        actual["count"][0].reverse()
    assert not R._match_by_policy(actual, expected, policy())


@pytest.mark.parametrize("mutation", ["version", "mode", "nested", "missing", "extra"])
def test_malformed_or_incomplete_policy_refuses(mutation):
    selected = policy()
    if mutation == "version":
        selected["schema"] = "unknown"
    elif mutation == "mode":
        selected["outputs"]["count"]["compare"] = "bounded_int"
    elif mutation == "nested":
        selected["outputs"]["count"] = policy()
    elif mutation == "missing":
        selected["outputs"].pop("count")
    else:
        selected["outputs"]["extra"] = {"compare": "exact_int", "dtype": "i8"}
    expected = {"count": [[1]], "value": [[2.0]]}
    with pytest.raises(ValueError, match="per-output"):
        R._match_by_policy(expected, expected, selected)


def test_float_leaf_defaults_are_exact_and_relative_bound_uses_reference():
    selected = policy()
    selected["outputs"]["value"].pop("atol")
    expected = {"count": [[1]], "value": [[1.0]]}
    actual = {"count": [[1]], "value": [[1.0001]]}
    assert not R._match_by_policy(actual, expected, selected)
    selected["outputs"]["value"]["rtol"] = 0.01
    assert R._match_by_policy(actual, expected, selected)
    actual["value"][0][0] = 1.02
    assert not R._match_by_policy(actual, expected, selected)


@pytest.mark.parametrize("threshold", [-1.0, float("inf"), float("nan")])
def test_invalid_float_threshold_does_not_create_a_pass(threshold):
    selected = policy()
    selected["outputs"]["value"]["atol"] = threshold
    expected = {"count": [[1]], "value": [[2.0]]}
    with pytest.raises(ValueError, match="per-output"):
        R._match_by_policy(expected, expected, selected)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_values_require_a_separate_explicit_contract(value):
    expected = {"count": [[1]], "value": [[value]]}
    assert not R._match_by_policy(expected, expected, policy())


def test_finite_float_match_agrees_with_existing_versioned_comparator():
    expected = {"count": [[-7]], "value": [[-2.0, 0.0]]}
    for delta in (0.0, 0.01, 0.1):
        actual = {"count": [[-7]], "value": [[-2.0 + delta, delta]]}
        assert R._match_by_policy(actual, expected, policy()) == (
            G.compare(expected, actual, policy())["status"] == "pass"
        )


@pytest.mark.parametrize("value", [1.0001, float("nan"), float("inf"), "1"])
def test_primary_per_output_comparator_cannot_truncate_invalid_integer_values(value):
    expected = {"count": [[1]], "value": [[2.0]]}
    actual = {"count": [[value]], "value": [[2.0]]}
    with pytest.raises(ValueError, match="per-output"):
        G.compare(expected, actual, policy())


@pytest.mark.parametrize("threshold", [-1.0, float("inf"), float("nan")])
def test_primary_per_output_comparator_rejects_invalid_thresholds(threshold):
    selected = policy()
    selected["outputs"]["value"]["atol"] = threshold
    expected = {"count": [[1]], "value": [[2.0]]}
    with pytest.raises(ValueError, match="per-output"):
        G.compare(expected, expected, selected)


@pytest.mark.parametrize("mutation", ["empty", "extra"])
def test_primary_per_output_comparator_requires_closed_nonempty_values(mutation):
    selected = policy()
    expected = {"count": [[1]], "value": [[2.0]]}
    actual = deepcopy(expected)
    if mutation == "empty":
        actual = expected = {}
        selected["outputs"] = {}
    else:
        actual["extra"] = [[0]]
    with pytest.raises(ValueError, match="per-output"):
        G.compare(expected, actual, selected)


def test_legacy_flat_comparison_and_force_merge_keep_their_existing_contract():
    flat = {"compare": "exact_int", "dtype": "i32"}
    expected = {"Y": [[16777217]]}
    assert G.compare(expected, expected, flat)["status"] == "pass"
    assert G.compare(expected, {"Y": [[16777216]]}, flat)["status"] == "fail"
    assert R._merge_match_policy({"compare": "float", "atol": 0.001}, {"atol": 0.01}) == {
        "compare": "float",
        "atol": 0.01,
    }


def test_finite_inputs_cannot_pass_after_difference_and_bound_overflow():
    expected = {"count": [[1]], "value": [[1e308]]}
    actual = {"count": [[1]], "value": [[-1e308]]}
    selected = policy()
    selected["outputs"]["value"].update(dtype="f64", atol=0.0, rtol=1.9)
    a, b, relative = map(Decimal.from_float, (1e308, -1e308, 1.9))
    assert abs(a - b) > relative * abs(a)
    for compare in (
        lambda: G.compare(expected, actual, selected),
        lambda: R._match_by_policy(actual, expected, selected),
    ):
        with pytest.raises(ValueError, match="per-output.*finite"):
            compare()
