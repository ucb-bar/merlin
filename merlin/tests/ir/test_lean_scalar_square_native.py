"""Native f32/f64 and exceptional-value evidence, including real-libm counterexamples."""

import pytest
from lean_scalar_square_cases import actual, scalar
from lean_scalar_square_native import differential


@pytest.mark.parametrize("kind", ["f32", "f64"])
@pytest.mark.parametrize("flags", ["none", "contract"])
def test_native_before_after_bits_are_observations(tmp_path, lean_tools, kind, flags):
    before, after, count = actual(scalar(kind, flags))
    assert count == 1
    result = differential(before, after, kind, 2, lean_tools.m2m_python, tmp_path)
    assert result["samples"] > 4000
    assert result["numerical_law_discharged"] is False
    assert any(row["both_nan"] for row in result["observations"])
