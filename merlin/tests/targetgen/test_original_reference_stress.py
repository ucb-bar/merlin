"""Realized bounded arithmetic stress; no source or target admission fixtures."""

import importlib.util
import math
import struct
from dataclasses import replace
from pathlib import Path

import pytest


def _fixtures():
    path = Path(__file__).with_name("test_original_operator_reference.py")
    spec = importlib.util.spec_from_file_location("original_stress_unit_contracts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = _fixtures()


def test_real_f32_partial_sum_rounding_and_exact_zero_cancellation():
    contract = F.contract()
    inputs = F.inputs_for(contract, ([2**24, 1, -(2**24)] * 2, [1] * 12))
    actual = contract.observe_stress(inputs)
    assert actual["counts"]["products"] == 24
    assert actual["counts"]["additions"] == 24
    assert actual["counts"]["rounded_additions"] == 8
    assert actual["counts"]["exact_zero_cancellations"] == 8
    assert actual["counts"]["negative_inputs"] == 2
    assert actual["counts"]["zero_outputs"] == 8
    assert contract.evaluate(inputs)[0].values() == (0.0,) * 8
    assert actual["output_sha256"] == contract.compare(inputs, contract.evaluate(inputs))["reference_sha256"]
    assert "no whole-domain" in actual["scope"]


def test_real_signed_readout_wrap_is_distinct_from_accumulator_wrap():
    contract = F.contract(F.policy(dtype="int8", arithmetic="modular_wrap"))
    inputs = F.inputs_for(contract, ([127] * 6, [1] * 12))
    actual = contract.observe_stress(inputs)
    assert actual["counts"]["wrapped_outputs"] == 8
    assert actual["counts"]["wrapped_additions"] == 0
    assert actual["counts"]["wrapped_products"] == 0
    assert contract.evaluate(inputs)[0].values() == (125,) * 8


def test_bounded_exact_readout_failure_cannot_become_completed_stress():
    contract = F.contract(F.policy(dtype="int8"))
    inputs = F.inputs_for(contract, ([127] * 6, [1] * 12))
    with pytest.raises(ValueError, match="readout overflow"):
        contract.observe_stress(inputs)


def test_same_values_without_signed_or_cancelled_stress_are_not_relabelled():
    contract = F.contract()
    inputs = F.inputs_for(contract, ([1] * 6, [1] * 12))
    actual = contract.observe_stress(inputs)
    assert actual["counts"]["negative_inputs"] == 0
    assert actual["counts"]["cancellation_additions"] == 0
    assert actual["counts"]["rounded_additions"] == 0
    changed = F.inputs_for(contract, ([1] * 5 + [-1], [1] * 12))
    assert contract.observe_stress(changed)["output_sha256"] != actual["output_sha256"]


def test_actual_product_rounding_is_retained_before_reduction():
    contract = F.contract()
    value = 1 + 2**-23
    inputs = F.inputs_for(contract, ([value] * 6, [value] * 12))
    actual = contract.observe_stress(inputs)
    assert actual["counts"]["rounded_products"] == 24
    assert actual["counts"]["products"] == 24


def test_original_complete_input_roster_and_finite_domain_remain_required():
    contract = F.contract()
    inputs = F.inputs_for(contract, ([1] * 6, [1] * 12))
    with pytest.raises(ValueError, match="complete ordered tensor"):
        contract.observe_stress(inputs[:-1])
    bad = replace(inputs[0], data=struct.pack("<f", math.inf) + inputs[0].data[4:])
    with pytest.raises(ValueError, match="NaN/Inf"):
        contract.observe_stress((bad, inputs[1]))


def test_original_convolution_bias_and_grouped_reduction_stress_are_observed():
    contract = F.contract(F.policy("aten.conv2d.default", bias=True), extent=1)
    inputs = F.inputs_for(contract, ([-3, 2, 5, -7], [2, 4, -1, 3], [1, -2]))
    actual = contract.observe_stress(inputs)
    assert actual["counts"]["products"] == 4
    assert actual["counts"]["additions"] == 6
    assert actual["counts"]["negative_inputs"] == 4
    assert actual["counts"]["cancellation_additions"] > 0
    assert contract.evaluate(inputs)[0].values() == (3.0, -28.0)
