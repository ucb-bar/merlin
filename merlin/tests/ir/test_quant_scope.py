"""Captured quantization reach is derived from quant IR, not framework/model names."""

from merlin.llvmlower import impr_features as F
from merlin.llvmlower.quant_scope import (FEATURE, captured_quantized_contraction,
                                          ensure_registered, traces_to_prequantized)


class _Op:
    def __init__(self, name, operands=(), attributes=None):
        self.name = name
        self.operands = list(operands)
        self.attributes = attributes or {}


class _Value:
    def __init__(self, owner=None):
        self.owner = owner


def _value(name, *operands):
    return _Value(_Op(name, operands))


def test_direct_and_layout_viewed_dequantized_values_are_evidence():
    deq = _value("quant_ext.dequantize_per_channel", _Value(), _Value(), _Value())
    viewed = _value("tensor.expand_shape", _value("linalg.transpose", deq))
    assert traces_to_prequantized(deq)
    assert traces_to_prequantized(viewed)


def test_arithmetic_or_unknown_producer_chain_is_refused():
    deq = _value("quant_ext.dequantize_per_channel", _Value())
    assert not traces_to_prequantized(_value("linalg.generic", deq))
    assert not traces_to_prequantized(_Value())


def test_unregistered_xdsl_spelling_is_recovered_from_its_attribute():
    attr = type("StringAttr", (), {"data": "quant_ext.dequantize_per_channel"})()
    value = _Value(_Op("builtin.unregistered", attributes={"op_name__": attr}))
    assert traces_to_prequantized(value)


def test_only_input_operands_can_admit_a_contraction():
    plain_a, plain_b = _value("tensor.expand_shape", _Value()), _Value()
    quant_weight = _value("quant_ext.dequantize_per_channel", _Value())
    init_that_looks_quantized = quant_weight
    assert captured_quantized_contraction(_Op("linalg.matmul", [plain_a, quant_weight, _Value()]))
    assert not captured_quantized_contraction(
        _Op("linalg.matmul", [plain_a, plain_b, init_that_looks_quantized]))


def test_feature_is_registered_as_a_default_off_policy_request():
    ensure_registered()
    feature = F.get(FEATURE)
    assert feature.edit_pipeline is None
    assert "model" not in FEATURE
