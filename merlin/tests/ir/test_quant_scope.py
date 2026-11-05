"""Captured quantization reach is derived from quant IR, not framework/model names."""

from merlin.llvmlower import impr_features as F
from merlin.llvmlower.quant_scope import (
    FEATURE,
    captured_quantized_contraction,
    ensure_registered,
    traces_to_prequantized,
)


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
    assert not captured_quantized_contraction(_Op("linalg.matmul", [plain_a, plain_b, init_that_looks_quantized]))


def test_feature_is_registered_as_a_default_off_policy_request():
    ensure_registered()
    feature = F.get(FEATURE)
    assert feature.edit_pipeline is None
    assert "model" not in FEATURE


_WIDENED = """
func.func @forward(%w: tensor<4x8xi8>, %a: tensor<2x4xf32>) -> tensor<4x8xf32> {
  %s = arith.constant 5.000000e-01 : f32
  %z = arith.constant 0 : i64
  %d = "quant_ext.dequantize_per_tensor"(%w, %s, %z) : (tensor<4x8xi8>, f32, i64) -> tensor<4x8xbf16>
  %e0 = tensor.empty() : tensor<4x8xbf16>
  %copy = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%d : tensor<4x8xbf16>) outs(%e0 : tensor<4x8xbf16>) {
  ^bb0(%x: bf16, %y: bf16):
    linalg.yield %x : bf16
  } -> tensor<4x8xbf16>
  %e1 = tensor.empty() : tensor<4x8xf32>
  %wide = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%copy : tensor<4x8xbf16>) outs(%e1 : tensor<4x8xf32>) {
  ^bb0(%x: bf16, %y: f32):
    %v = arith.extf %x : bf16 to f32
    linalg.yield %v : f32
  } -> tensor<4x8xf32>
  %e2 = tensor.empty() : tensor<4x8xf32>
  %scaled = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%wide : tensor<4x8xf32>) outs(%e2 : tensor<4x8xf32>) {
  ^bb0(%x: f32, %y: f32):
    %v = arith.mulf %x, %x : f32
    linalg.yield %v : f32
  } -> tensor<4x8xf32>
  return %scaled : tensor<4x8xf32>
}
"""


def _results_by_kind():
    from merlin.common import mlir_query as mq
    from merlin.llvmlower.quant_scope import value_preserving_input

    module = mq.parse(_WIDENED)
    generics = [op for op in module.walk() if op.name == "linalg.generic"]
    copy, wide, scaled = generics
    return value_preserving_input, copy, wide, scaled


def test_an_exact_widening_or_copy_of_a_dequantized_value_is_evidence():
    from merlin.llvmlower.quant_scope import prequantized_source

    value_preserving_input, copy, wide, scaled = _results_by_kind()
    assert value_preserving_input(wide) is wide.inputs[0]
    assert value_preserving_input(copy) is copy.inputs[0]
    assert traces_to_prequantized(wide.results[0])
    # Peeling stops at the dequantized value, so integer lowering sees the captured producer.
    source = prequantized_source(wide.results[0])
    assert source is copy.inputs[0]
    assert source.owner.attributes["op_name__"].data == "quant_ext.dequantize_per_tensor"


def test_arithmetic_on_a_widened_value_is_not_value_preserving():
    from merlin.llvmlower.quant_scope import prequantized_source

    value_preserving_input, _copy, _wide, scaled = _results_by_kind()
    assert value_preserving_input(scaled) is None
    assert not traces_to_prequantized(scaled.results[0])
    assert prequantized_source(scaled.results[0]) is scaled.results[0]
