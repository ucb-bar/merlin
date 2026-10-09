"""Cheap complete finite row comparison controls before original native replay."""

import copy
import json

import pytest
from merlin_experiments.phase0 import original_reference_products as P

from merlin.common.jsonio import canonical_json
from merlin.targetgen.original_pointwise_reference import OriginalPointwiseReferencePolicy


@pytest.fixture
def finite_row():
    policy = OriginalPointwiseReferencePolicy(
        "aten.clamp.default",
        ("float32",),
        ("float32",),
        "float32",
        "finite_f32",
        "not_applicable",
        "elementwise",
        "not_applicable",
        "rne",
        True,
        False,
        "not_applicable",
        0.0,
        0.0,
        "preserve",
    ).record()
    return {
        "original_member_id": "independent",
        "graph_path": "/ordinary/original.json",
        "node": "clamp",
        "target": "aten.clamp.default",
        "call": {"result_arity": 1, "arguments": [{"value": -0.0}, {"value": 1.5}]},
        "cohort": "withheld_transfer",
        "extent": 1,
        "state": "comparison_refuted",
        "required_unknowns": ["original_numerical_domain"],
        "form": {"parameters": {"min": -0.0, "max": 1.5}, "source_numerical_semantics": policy},
        "policy": policy,
        "contract_sha256": "a" * 64,
        "cost": {"scalar_bits": 512},
        "source_cost": {"tensor_elements": 2, "scalar_products": 0, "source_bytes": 100},
        "reason": "complete bounded native/source-reference comparison; domain remains unqualified",
        "comparison": {
            "passed": False,
            "checked_elements": 1,
            "mismatches": [{"slot": "Y", "index": 0, "expected": -0.0, "actual": 0.0}],
        },
    }


def test_complete_finite_policy_bound_and_mismatch_rows_replay(finite_row):
    decoded = json.loads(canonical_json(finite_row))
    assert P._pointwise_equal(finite_row, decoded)


@pytest.mark.parametrize("mutation", ["bool", "int_float", "float_int", "bound_sign", "mismatch_sign", "policy"])
def test_scalar_aliases_and_changed_finite_semantics_do_not_replay(finite_row, mutation):
    actual = copy.deepcopy(finite_row)
    if mutation == "bool":
        actual["extent"] = True
    elif mutation == "int_float":
        actual["source_cost"]["tensor_elements"] = 2.0
    elif mutation == "float_int":
        actual["policy"]["atol"] = 0
    elif mutation == "bound_sign":
        actual["form"]["parameters"]["min"] = 0.0
    elif mutation == "mismatch_sign":
        actual["comparison"]["mismatches"][0]["expected"] = 0.0
    else:
        actual["policy"]["zero_sign"] = "ignore"
    assert not P._pointwise_equal(finite_row, actual)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_never_becomes_a_replay_equality(finite_row, value):
    actual = copy.deepcopy(finite_row)
    actual["policy"]["atol"] = value
    with pytest.raises(ValueError):
        P._pointwise_equal(finite_row, actual)
