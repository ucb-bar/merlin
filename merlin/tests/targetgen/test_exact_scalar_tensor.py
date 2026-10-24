"""Rank-zero rational arithmetic must retain tensor shape through Q/DQ."""

from fractions import Fraction

import pytest

from merlin.targetgen.exact_arithmetic import ExactTensor, dequantize, materialize, quantize


@pytest.mark.parametrize("numerator", [-3, 0, 3])
def test_scalar_rational_values_keep_rank_zero(numerator):
    value = ExactTensor(Fraction(numerator, 2), None)
    assert value.values.shape == ()
    codes = quantize(value, Fraction(1, 4))
    assert codes.shape == () and codes.item() == 2 * numerator
    decoded = dequantize(codes, Fraction(1, 4))
    assert decoded.values.shape == () and decoded.values.item() == Fraction(numerator, 2)


def test_scalar_materialization_is_not_a_python_scalar():
    torch = pytest.importorskip("torch")
    value = ExactTensor(Fraction(3, 2), torch.float32)
    tensor = materialize(value)
    assert tensor.shape == torch.Size([])
    assert tensor.dtype == torch.float32 and tensor.item() == 1.5
