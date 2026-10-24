"""Compare emitted integer rounding IR with independent nearest-IEEE search."""

import random
from fractions import Fraction

import pytest
from mlir_fixtures import emitter, run_ops
from xdsl.dialects.builtin import IntegerType, bf16, f16, f32, f64
from xdsl.ir import Block

from merlin.llvmlower.fixed_point_ir import FORMAT, encode_fixed_integer, finite_grid


def reference_bits(numerator, exponent, denominator, token):
    fraction_bits, exponent_bits, bias = FORMAT[token]
    sign = int(numerator < 0) << (fraction_bits + exponent_bits)
    value = abs(Fraction(numerator, denominator) * Fraction(2) ** exponent)
    maximum = (((1 << exponent_bits) - 1) << fraction_bits) - 1

    def decode(raw):
        e, f = raw >> fraction_bits, raw & ((1 << fraction_bits) - 1)
        return Fraction(f if e == 0 else (1 << fraction_bits) + f) * Fraction(2) ** (
            (1 - bias if e == 0 else e - bias) - fraction_bits
        )

    threshold = Fraction(2) ** bias * (2 - Fraction(1, 1 << (fraction_bits + 1)))
    if value >= threshold:
        return sign | (((1 << exponent_bits) - 1) << fraction_bits)
    low, high = 0, maximum
    while low < high:
        mid = (low + high) // 2
        if decode(mid) < value:
            low = mid + 1
        else:
            high = mid
    candidate = min({low, max(0, low - 1)}, key=lambda raw: (abs(decode(raw) - value), raw & 1))
    return sign | candidate


def emitted_bits(numerator, exponent, denominator, dtype, width=128):
    block = Block(arg_types=[IntegerType(width)])
    result = encode_fixed_integer(block.args[0], exponent, dtype, emitter(block), denominator=denominator)
    values = {block.args[0]: numerator & ((1 << width) - 1)}
    run_ops(block.ops, values)
    return values[result]


@pytest.mark.parametrize("dtype", [f16, bf16, f32, f64])
def test_emitted_rne_matches_nearest_ieee_search(dtype):
    rng = random.Random(1767)
    samples = [
        (n, e, d)
        for n in (0, 1, -1, 3, -3, (1 << 119) - 1, -(1 << 119) + 1)
        for e in (-1200, -1074, -300, -149, -24, -1, 0, 128, 1023, 1100)
        for d in (1, 3, 7)
    ]
    samples += [
        (rng.randrange(-(1 << 119), 1 << 119), rng.randrange(-300, 130), rng.choice([1, 3, 17, 65537]))
        for _ in range(200)
    ]
    for numerator, exponent, denominator in samples:
        assert emitted_bits(numerator, exponent, denominator, dtype) == reference_bits(
            numerator, exponent, denominator, str(dtype)
        ), (numerator, exponent, denominator, dtype)


def test_complete_dyadic_grid_refuses_nonfinite_or_unbounded_alignment():
    assert finite_grid([0.0, -0.0, 0.5, 1.25]) == (-2, 5)
    assert finite_grid([256.0, -512.0]) == (8, 2)
    with pytest.raises(ValueError, match="nonfinite"):
        finite_grid([1.0, float("inf")])
    assert finite_grid([1.0, 2.0**-100]) == (-100, 1 << 100)
    with pytest.raises(ValueError, match="512-bit"):
        finite_grid([1.0, 2.0**-600])


@pytest.mark.parametrize("dtype", [f16, bf16, f32, f64])
def test_wide_reconstruction_rounds_without_wide_division_runtime(dtype):
    for n in (0, 1, -1, (1 << 280) + 1, -(1 << 556) + 17, (1 << 1000) - 1):
        for exponent in (-1100, -300, -149, 0):
            for denominator in (1, 3, 65537):
                assert emitted_bits(n, exponent, denominator, dtype, width=1024) == reference_bits(
                    n, exponent, denominator, str(dtype)
                )
