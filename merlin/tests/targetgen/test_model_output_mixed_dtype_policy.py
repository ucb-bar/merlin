"""Mixed-dtype model outputs: a complete per-output roster, a 0-d input, and non-finite goldens."""

import math
from types import SimpleNamespace

import pytest

from merlin.targetgen import capsule_source as S
from merlin.targetgen.capture_refusals import M2MUnavailable

BINDING = SimpleNamespace(atol=None, rtol=None)
ABI = [{"dtype": "f32", "shape": [2]}, {"dtype": "i8", "shape": [2]}, {"dtype": "i64", "shape": []}]


def test_legacy_call_still_refuses_mixed_dtypes():
    with pytest.raises(M2MUnavailable, match="different output dtypes"):
        S._model_output_numeric_policy(ABI, [[0.5, 1.0], [1, -2], 3], BINDING)


def test_complete_roster_gets_one_existing_policy_per_result():
    policy = S._model_output_numeric_policy(ABI, [[0.5, 1.0], [1, -2], 3.0], BINDING, output_names=["Y0", "Y1", "Y2"])
    assert policy["compare"] == "per_output"
    assert policy["outputs"]["Y0"]["compare"] == "tolerance_float"
    assert policy["outputs"]["Y1"] == {"compare": "exact_int", "dtype": "i8"}
    assert policy["outputs"]["Y2"] == {"compare": "exact_int", "dtype": "i64"}


def test_single_dtype_roster_is_unchanged():
    abi = [{"dtype": "i32", "shape": [1]}] * 2
    assert S._model_output_numeric_policy(abi, [[1], [2]], BINDING, output_names=["Y0", "Y1"]) == {
        "compare": "exact_int",
        "dtype": "i32",
    }


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_golden_is_refused_not_published(bad):
    with pytest.raises(M2MUnavailable, match="non-finite golden"):
        S._model_output_numeric_policy(ABI, [[0.5, bad], [1, -2], 3.0], BINDING, output_names=["Y0", "Y1", "Y2"])


def test_out_of_range_integer_still_refused_per_output():
    with pytest.raises(M2MUnavailable, match="inexact/out-of-range"):
        S._model_output_numeric_policy(ABI, [[0.5, 1.0], [1, 300], 3.0], BINDING, output_names=["Y0", "Y1", "Y2"])


def test_incomplete_roster_refused():
    with pytest.raises(M2MUnavailable, match="every unique original result"):
        S._model_output_numeric_policy(ABI, [[0.5, 1.0], [1, -2], 3.0], BINDING, output_names=["Y0", "Y0", "Y2"])


def test_flatten_reads_a_zero_d_input_as_one_value():
    assert S._flatten(2.5) == [2.5]
    assert S._flatten([1, 2]) == [1, 2]
    assert S._flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]
