"""Tests for the mixed-precision policy (merlin.frontends.mixed_precision)."""
from __future__ import annotations

import pytest

from merlin.frontends import mixed_precision as mp


def test_policy_validates_formats():
    with pytest.raises(ValueError):
        mp.MixedPrecisionPolicy(default="not_a_format").validate()
    with pytest.raises(ValueError):
        mp.MixedPrecisionPolicy(default="fp16", rules=(mp.PrecisionRule("mlp", "nope"),)).validate()
    ok = mp.MixedPrecisionPolicy(default="fp16", rules=(mp.PrecisionRule("mlp", "mxfp4"),)).validate()
    assert ok.formats() == {"fp16", "mxfp4"}


def test_to_m2m_per_module_maps_formats_to_schemes():
    policy = mp.MixedPrecisionPolicy(
        default="fp16",
        rules=(mp.PrecisionRule("self_attn", "fp16"), mp.PrecisionRule("mlp", "mxfp4")),
    ).validate()
    per_module = policy.to_m2m_per_module()
    # base float -> None (leave unquantized); quantized -> its torchao scheme.
    assert per_module["*"] is None
    assert per_module["self_attn"] is None
    assert per_module["mlp"] == "mx_dyn_act_mx_weight_mx4"


def test_format_without_torchao_scheme_is_rejected_for_torch_path():
    # bare fp4_e2m1 has no torchao scheme -> clear error pointing at mxfp4/nvfp4 / GGUF.
    policy = mp.MixedPrecisionPolicy(default="fp16", rules=(mp.PrecisionRule("mlp", "fp4_e2m1"),)).validate()
    with pytest.raises(ValueError):
        policy.to_m2m_per_module()


def test_worked_example():
    policy = mp.attention_fp16_mlp_fp4()
    per_module = policy.to_m2m_per_module()
    assert per_module["self_attn"] is None
    assert per_module["mlp"] == "mx_dyn_act_mx_weight_mx4"
