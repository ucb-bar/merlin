"""Scalar promotion is a typed literal boundary, including the rational path."""

from fractions import Fraction

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.quantized_native import simplify_static_controls
from merlin.targetgen.exact_arithmetic import round_binary
from merlin.targetgen.quantized_native_reference import canonical_scalar


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
@pytest.mark.parametrize("scalar", [1.0001, 0.3333333333333333, 0.0001])
def test_reference_promotes_scalars_before_exact_arithmetic(dtype, op, scalar):
    target = dtype if op in {"add", "sub"} or dtype == "f64" else "f32"
    expected = round_binary(
        Fraction(scalar), 16 if target in {"f16", "bf16"} else 32 if target == "f32" else 64, bfloat16=target == "bf16"
    )
    assert canonical_scalar(dtype, "aten." + op + ".Scalar", scalar) == expected


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32"])
def test_lowered_scalar_casts_keep_the_declared_storage_rounding(dtype):
    module = parse_mlir_text(
        """module { func.func @forward() -> f64 {
      %c=arith.constant 1.0001 : f64
      %n=arith.truncf %c : f64 to """
        + dtype
        + """
      %w=arith.extf %n : """
        + dtype
        + """ to f64
      func.return %w : f64
    }}"""
    )
    target = next(op for op in module.walk() if op.name == "arith.extf").result
    values = simplify_static_controls(module)
    assert values[target] == Fraction(canonical_scalar(dtype, "aten.add.Scalar", 1.0001))


@pytest.mark.parametrize("operation", ["add", "sub", "mul", "div"])
def test_real_and_complex_scalar_coefficients_use_the_component_format(operation):
    op = "aten." + operation + ".Scalar"
    expected = canonical_scalar("f32", op, 0.1)
    assert canonical_scalar("complex<f32>", op, 0.1) == expected
    assert canonical_scalar("complex<f32>", op, complex(0.1, -0.1)) == complex(expected, -expected)
