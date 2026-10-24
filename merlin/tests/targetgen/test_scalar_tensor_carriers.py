# ruff: noqa: E501 -- MLIR fixtures retain one operation per line
"""A private scalar extraction/pack must retain the closed quantized island."""

import pytest
from mlir_fixtures import assert_unchanged

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.structural_folds import fold_scalar_tensor_roundtrips
from merlin.targetgen.core_aten_spec_int8 import quantized_islands

_SOURCE = """
builtin.module {
  func.func @forward(%x: tensor<i8>) -> tensor<1x1xi8> {
    %s0 = arith.constant 0.25 : f32
    %s = tensor.splat %s0 : tensor<f32>
    %z0 = arith.constant 0 : i64
    %z = tensor.splat %z0 : tensor<i64>
    %d = "quant_ext.dequantize_per_tensor"(%x, %s, %z) <{quant_min = -128 : i64, quant_max = 127 : i64}> : (tensor<i8>, tensor<f32>, tensor<i64>) -> tensor<f32>
    %e = tensor.empty() : tensor<f32>
    %n = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%d : tensor<f32>) outs(%e : tensor<f32>) {
    ^bb0(%a: f32, %b: f32):
      %v = arith.negf %a : f32
      linalg.yield %v : f32
    } -> tensor<f32>
    %v = tensor.extract %n[] : tensor<f32>
    %p = tensor.from_elements %v : tensor<1x1xf32>
    %q = "quant_ext.quantize_per_tensor"(%p, %s, %z) <{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}> : (tensor<1x1xf32>, tensor<f32>, tensor<i64>) -> tensor<1x1xi8>
    func.return %q : tensor<1x1xi8>
  }
}
"""


def test_private_pack_remains_a_quantized_tensor_island():
    module = parse_mlir_text(_SOURCE)
    assert not quantized_islands(module)
    assert fold_scalar_tensor_roundtrips(module) == 1
    assert len(quantized_islands(module)) == 2
    assert "tensor.extract " not in str(module)
    assert "tensor.reshape " in str(module)
    assert fold_scalar_tensor_roundtrips(module) == 0


@pytest.mark.parametrize("extra_use", ["%keep = tensor.extract %n[] : tensor<f32>", "%keep = arith.negf %v : f32"])
def test_shared_carriers_are_not_rewritten(extra_use):
    source = _SOURCE.replace("    %p =", "    " + extra_use + "\n    %p =")
    assert_unchanged(parse_mlir_text(source), fold_scalar_tensor_roundtrips)
