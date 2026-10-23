"""Only a complete finite rational zero-threshold identity is exposed."""

import pytest
from mlir_fixtures import assert_unchanged, pointwise

from merlin.llvmlower.quantized_zero_threshold import normalize
from merlin.xdsl_dialects.lowering.contraction_coverage import is_compare_select_relu


def source(alpha=0, scale=1, unknown=False):
    exp = '"fixture.unknown"(%a) : (f32)->f32' if unknown else "math.expm1 %a : f32"
    body = f"""%zero=arith.constant 0.0 : f32
      %alpha=arith.constant {float(alpha)} : f32
      %scale=arith.constant {float(scale)} : f32
      %exp={exp}
      %negative=arith.mulf %alpha,%exp : f32
      %positive=arith.mulf %a,%scale : f32
      %p=arith.cmpf oge,%a,%zero : f32
      %v=arith.select %p,%positive,%negative : f32"""
    return pointwise(["tensor<3xf32>"], "tensor<3xf32>", body)


def test_exact_finite_zero_coefficient_eliminates_the_transcendental_branch():
    module = source()
    assert normalize(module) == 1
    assert sum(is_compare_select_relu(op) for op in module.walk()) == 1
    assert all(op.name != "math.expm1" for op in module.walk())


@pytest.mark.parametrize("options", [dict(alpha=1), dict(scale=2), dict(unknown=True)])
def test_a_different_function_or_unknown_primitive_is_not_a_native_threshold(options):
    assert_unchanged(source(**options), normalize)
