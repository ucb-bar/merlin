"""Digit tiles reconstruct integer residuals without implicit shifts or wrapping."""

from fractions import Fraction

import numpy as np
import pytest

from merlin.xdsl_dialects.lowering.k_extension import decompose_coefficient, lower_k_extension


def lower(value, **changes):
    options = dict(
        source_k=5,
        residual_width=3,
        digit_bits=8,
        max_extended_k=100,
        accumulator_bits=32,
        source_bound=80000,
        residual_range=(-128, 127),
    )
    return lower_k_extension(value, **dict(options, **changes))


@pytest.mark.parametrize("coefficient", [0, 1, -1, 127, 128, -255, 301])
def test_radix_digits_and_physical_tiles_reconstruct_exactly(coefficient):
    plan = lower(coefficient)
    assert sum(d * plan["radix"] ** i for i, d in enumerate(plan["digits"])) == coefficient
    assert sum(plan["tile_digits"]) == coefficient
    assert all(abs(d) <= 127 for d in plan["tile_digits"])
    rng = np.random.default_rng(19)
    residual = rng.integers(-128, 128, (2, 3), dtype=np.int64)
    digits = plan["tile_digits"]
    if digits:
        lhs = np.concatenate([residual] * len(digits), axis=1)
        rhs = np.concatenate([np.eye(3, dtype=np.int64) * d for d in digits], axis=0)
        np.testing.assert_array_equal(lhs @ rhs, residual * coefficient)
    assert plan["extended_k"] == 5 + 3 * len(digits)


@pytest.mark.parametrize("coefficient", [Fraction(3, 2), 0.1, float("inf")])
def test_noninteger_must_refuse(coefficient):
    with pytest.raises((ValueError, OverflowError)):
        lower(coefficient)


def test_capacity_and_partial_sum_refusal_twins():
    assert lower(128)["extended_k"] == 11
    assert len(decompose_coefficient(128, digit_bits=8)["digits"]) == 2
    with pytest.raises(ValueError, match="capacity"):
        lower(128, max_extended_k=10)
    with pytest.raises(ValueError, match="overflow"):
        lower(128, accumulator_bits=16)
