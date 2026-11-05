"""Realized original partitions, independent of any hardware/source shape."""

import struct

import pytest
from test_original_pointwise_reference import contract, policy, tensor

from merlin.targetgen import original_pointwise_stress as S


@pytest.mark.parametrize("dtype", ["float32", "int8", "int16", "int32", "int64"])
@pytest.mark.parametrize("operation", ["aten.relu.default", "aten.round.default", "aten.clamp.default"])
def test_realized_original_storage_partitions_match_complete_reference(dtype, operation):
    parameters = {"min": -1, "max": 1} if operation == "aten.clamp.default" else {}
    selected = policy(operation, dtype)
    values = S.samples(selected, parameters, S.PROFILE)
    original = contract(operation, dtype=dtype, count=len(values), bounds=(-1, 1))
    inputs = (tensor(values, dtype=dtype),)
    stress = S.observe(original, inputs)
    assert all(S.realized(stress)[name] for name in S.required(selected, parameters))
    assert stress["schema"] == S.SCHEMA
    assert stress["output_sha256"] == original.observe_stress(inputs)["output_sha256"]
    assert stress["counts"]["pointwise_values"] == len(values)
    assert stress["counts"]["products"] == stress["counts"]["additions"] == 0
    assert original.observe_stress(inputs)["schema"] == "merlin.original_reference_stress.v2"


@pytest.mark.parametrize(
    "bounds",
    [
        (-0.0, 1.0),
        (None, 0.0),
        (-0.0, None),
        (7, -3),
        (1e-50, -0.0),
        (-(2.0**128 - 2.0**104), None),
        (None, 2.0**128 - 2.0**104),
        ((1 << 60) + (1 << 36) + 1, None),
    ],
)
def test_clamp_required_paths_follow_original_order_bounds_and_defined_storage(bounds):
    selected = policy("aten.clamp.default")
    parameters = dict(zip(("min", "max"), bounds, strict=True))
    values = S.samples(selected, parameters, S.PROFILE)
    original = contract(selected.operation, count=len(values), bounds=bounds)
    trace = S.observe(original, (tensor(values),))
    required = S.required(selected, parameters)
    assert all(S.realized(trace)[name] for name in required)
    if bounds == (7, -3):
        assert "clamp_upper_kept" not in required and "clamp_upper_tie" not in required
        assert trace["counts"]["inverted_clamp"] == len(values)
    if bounds[0] == -(2.0**128 - 2.0**104):
        assert "clamp_lower_applied" not in required
    if bounds[1] == 2.0**128 - 2.0**104:
        assert "clamp_upper_applied" not in required


def test_missing_actual_values_do_not_realize_selected_pointwise_domain():
    original = contract("aten.round.default", count=1)
    observed = S.realized(S.observe(original, (tensor([1.0]),)))
    assert observed["pointwise_values"]
    assert not observed["signed_inputs"] and not observed["rne_positive_even_tie"]
    assert not observed["negative_zero_input"] and not observed["subnormal_input"]


def test_probe_scalar_values_keep_both_zero_signs_and_exact_signed64_endpoints():
    floating = S.samples(policy("aten.relu.default"), {}, S.PROFILE)
    assert struct.pack("<f", -0.0) in {struct.pack("<f", v) for v in floating}
    assert struct.pack("<f", 0.0) in {struct.pack("<f", v) for v in floating}
    integers = S.samples(policy("aten.relu.default", "int64"), {}, S.PROFILE)
    assert integers[0] == -(1 << 63) and integers[-1] == (1 << 63) - 1
    assert all(type(value) is int for value in integers)


def test_unsupported_profile_refuses_without_storage_fallback():
    with pytest.raises(ValueError):
        S.samples(policy("aten.round.default"), {}, "saved_fixture_exists")


def test_scalar_union_requires_actual_positive_and_negative_visits():
    selected = contract("aten.relu.default", count=1, rank=0)
    traces = [S.observe(selected, (tensor([v], shape=[]),)) for v in (-1.0, 1.0)]
    assert not any(S.realized(trace)["signed_inputs"] for trace in traces)
    assert S.combined_realized(traces)["signed_inputs"]
    assert not S.combined_realized(traces[:1])["signed_inputs"]
    with pytest.raises(ValueError):
        S.combined_realized([])
    with pytest.raises(ValueError):
        S.combined_realized([{**traces[0], "schema": "merlin.original_reference_stress.v2"}])
