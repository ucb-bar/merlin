"""Exact quantization at every byte transition and arbitrary IEEE binary32 inputs."""

import random
from bisect import bisect_right
from fractions import Fraction

import pytest

from merlin.llvmlower.exact_quantize import ordered_float, thresholds
from merlin.targetgen.exact_arithmetic import quantize


@pytest.mark.parametrize(
    "scale", [Fraction(1, 2), Fraction(0.03), Fraction(0.009796262718737125), Fraction(2) ** -149, Fraction(2) ** 120]
)
@pytest.mark.parametrize("zp,low,high", [(0, -128, 127), (-13, -127, 127), (131, 0, 255), (7, 0, 15), (-2, -8, 7)])
def test_all_transitions_and_random_float_codes_against_exact_oracle(scale, zp, low, high):
    table = thresholds(scale, zp, low, high)
    rng = random.Random(9241)
    keys = {key for threshold in table[:-1] for key in (threshold - 1, threshold, threshold + 1)}
    keys.update(rng.randrange(0x00800000, 0xFF800000) for _ in range(1000))
    keys.update((0x7FFFFFFF, 0x80000000))  # negative and positive zero
    for key in sorted(keys):
        if not 0x007FFFFF < key < 0xFF800000:
            continue
        value = ordered_float(key)
        actual = low + bisect_right(table[:-1], key)
        expected = int(quantize(value, scale, zp, low, high))
        assert actual == expected, (scale, key, value, actual, expected)


def test_exact_lowering_contains_no_floating_arithmetic():
    from fractions import Fraction

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.passes_xdsl import lower_quant_ext

    scale = Fraction(0.03)
    source = f'''builtin.module {{
 func.func @forward(%x: tensor<4xf32>) -> tensor<4xi8> {{
  %s = arith.constant 0.03 : f64
  %st = tensor.splat %s : tensor<f64>
  %z = arith.constant 0 : i64
  %zt = tensor.splat %z : tensor<i64>
  %q = "quant_ext.quantize_per_tensor"(%x,%st,%zt)
    <{{quant_min = -128 : i64, quant_max = 127 : i64}}> {{prov.aten = "m2m_q.quantize.default",
    prov.quant.scale_numerator = ["{scale.numerator}"],
    prov.quant.scale_denominator = ["{scale.denominator}"],
    prov.quant.zero_point = 0 : i64,
    prov.quant.quant_min = -128 : i64,
    prov.quant.quant_max = 127 : i64,
    prov.quant.storage_dtype = "int8"}}
    : (tensor<4xf32>, tensor<f64>, tensor<i64>) -> tensor<4xi8>
  func.return %q : tensor<4xi8>
 }}
}}'''
    module = parse_mlir_text(source)
    assert lower_quant_ext(module) == 1
    module.verify()
    forbidden = {"arith.mulf", "arith.divf", "arith.addf", "math.roundeven", "arith.fptosi"}
    assert not any(op.name in forbidden for op in module.walk())
    assert sum(op.name == "tensor.extract" for op in module.walk()) == 8
