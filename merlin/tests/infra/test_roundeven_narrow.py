"""The cheap round-and-narrow construction must equal round-half-to-even, including at ties.

This is the arithmetic a backend emitter substitutes for ~61 emitted operations, so an error here is
a silent miscompile of every quantized activation: the model still runs and still produces plausible
logits. The reference is `numpy.rint` where available (IEEE roundeven, the same rounding
`math.roundeven` and the Gemmini header's `ROUND_NEAR_EVEN` compute) and Python's own
round-half-to-even otherwise.
"""
from __future__ import annotations

import struct

import pytest

from merlin.llvmlower.roundeven_narrow import (MAX_BOUND, bound_is_usable, quantize_affine,
                                               roundeven_to_int_bounded)


def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _reference(x: float) -> int:
    """Round-half-to-even on an f32 value, without depending on numpy."""
    x = _f32(x)
    floor = int(x // 1)
    frac = x - floor
    if frac > 0.5:
        return floor + 1
    if frac < 0.5:
        return floor
    return floor if floor % 2 == 0 else floor + 1


def test_ties_go_to_even_not_away_from_zero():
    for value, expected in ((0.5, 0), (1.5, 2), (2.5, 2), (3.5, 4),
                            (-0.5, 0), (-1.5, -2), (-2.5, -2), (-3.5, -4),
                            (126.5, 126), (127.5, 128), (-127.5, -128)):
        assert roundeven_to_int_bounded(value) == expected, value


def test_matches_round_half_even_on_a_dense_sweep():
    x = -4096.0
    while x < 4096.0:
        assert roundeven_to_int_bounded(x) == _reference(x), x
        x += 0.0625


def test_matches_round_half_even_on_sixteenths_near_the_int8_range():
    for i in range(-300 * 16, 300 * 16):
        x = i / 16.0
        assert roundeven_to_int_bounded(x) == _reference(x), x


def test_values_past_the_bound_saturate_to_the_bound():
    assert roundeven_to_int_bounded(1e30) == int(MAX_BOUND)
    assert roundeven_to_int_bounded(-1e30) == -int(MAX_BOUND)


def test_a_bound_outside_the_magic_binade_is_refused():
    for bad in (0.0, -1.0, MAX_BOUND * 2, 8388608.0):
        with pytest.raises(ValueError):
            roundeven_to_int_bounded(1.0, bound=bad)


def test_round_then_clamp_is_not_clamp_then_round():
    """The property the pre-clamp width exists to preserve; 127.6 must saturate, not round down."""
    assert quantize_affine(127.6, 1.0, 0, -128, 127) == 127
    assert quantize_affine(126.4, 1.0, 0, -128, 127) == 126
    # and the value that rounds ONTO the limit must reach it
    assert quantize_affine(126.5, 1.0, 0, -128, 127) == 126     # tie to even
    assert quantize_affine(127.4, 1.0, 0, -128, 127) == 127


def test_quantize_affine_matches_the_declared_expression():
    for inv_scale in (1.0, 8.7749, 0.03137, 123.5):
        for zp in (0, -7, 11):
            i = -5000
            while i < 5000:
                x = i / 7.0
                got = quantize_affine(x, inv_scale, zp, -128, 127)
                want = min(max(_reference(_f32(_f32(x) * _f32(inv_scale))) + zp, -128), 127)
                assert got == want, (x, inv_scale, zp, got, want)
                i += 13


def test_bound_must_exceed_the_integer_range():
    assert bound_is_usable(MAX_BOUND, -128, 127)
    assert not bound_is_usable(127.0, -128, 127), "a bound inside the range breaks round-then-clamp"
    assert not bound_is_usable(MAX_BOUND * 2, -128, 127)


def test_a_bound_inside_the_integer_range_is_refused_by_quantize_affine():
    with pytest.raises(ValueError):
        quantize_affine(1.0, 1.0, 0, -128, 127, bound=127.0)
