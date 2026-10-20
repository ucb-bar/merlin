"""Complete exact policy expansion preserves each original result type."""

import copy

import pytest
from merlin_experiments.phase0 import connected_integer_qualification as Q


def exact(dtype):
    return dict(compare="exact_int", dtype=dtype)


def test_shared_policy_covers_every_homogeneous_original_output_without_numeric_change():
    policy = exact("i32")
    result = Q._exact_output_policies(["first", "second"], [dict(dtype="i32"), dict(dtype="i32")], policy)
    assert result == dict(first=exact("i32"), second=exact("i32"))
    assert result["first"] is policy and result["second"] is policy


def test_legacy_single_and_complete_heterogeneous_policies_keep_exact_contracts():
    assert Q._exact_output_policies(["only"], [dict(dtype="i8")], exact("i8")) == dict(only=exact("i8"))
    policy = dict(compare="per_output", outputs=dict(wide=exact("i32"), state=exact("i8")))
    assert (
        Q._exact_output_policies(["wide", "state"], [dict(dtype="i32"), dict(dtype="i8")], policy) == policy["outputs"]
    )


@pytest.mark.parametrize(
    "policy",
    [
        exact("i8"),
        exact("int32"),
        dict(compare="bounded_int", dtype="i32"),
        dict(compare="exact_int", dtype="i32", atol=0),
        dict(compare="exact_int", dtype="i32", rtol=0.001),
        dict(compare="exact_int", dtype="i32", schema="extra"),
    ],
)
def test_shared_policy_cannot_reinterpret_any_original_result_type_or_tolerance(policy):
    with pytest.raises(ValueError):
        Q._exact_output_policies(["first", "second"], [dict(dtype="i32"), dict(dtype="i32")], policy)


def test_shared_policy_cannot_describe_heterogeneous_outputs():
    with pytest.raises(ValueError):
        Q._exact_output_policies(["wide", "state"], [dict(dtype="i32"), dict(dtype="i8")], exact("i32"))


@pytest.mark.parametrize("mutation", ["missing", "extra", "dtype", "comparison", "tolerance"])
def test_per_output_policy_requires_all_exact_abi_members(mutation):
    policy = dict(compare="per_output", outputs=dict(wide=exact("i32"), state=exact("i8")))
    policy = copy.deepcopy(policy)
    if mutation == "missing":
        policy["outputs"].pop("state")
    elif mutation == "extra":
        policy["outputs"]["other"] = exact("i8")
    elif mutation == "dtype":
        policy["outputs"]["state"]["dtype"] = "i32"
    elif mutation == "comparison":
        policy["outputs"]["wide"]["compare"] = "bounded_int"
    else:
        policy["outputs"]["wide"]["atol"] = 1
    with pytest.raises(ValueError):
        Q._exact_output_policies(["wide", "state"], [dict(dtype="i32"), dict(dtype="i8")], policy)


@pytest.mark.parametrize(
    "names,abis", [([], []), (["x", "x"], [dict(dtype="i32")] * 2), (["x", "y"], [dict(dtype="i32")])]
)
def test_incomplete_or_aliased_output_roster_refuses(names, abis):
    with pytest.raises(ValueError):
        Q._exact_output_policies(names, abis, exact("i32"))
