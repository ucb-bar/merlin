"""Exhaustive finite-BF16 integer pack controls against the rational oracle."""
from pathlib import Path
import sys
import numpy as np
import pytest

pytestmark = pytest.mark.target("atlas")
from merlin.common.paths import repo_root
sys.path.insert(0, str(repo_root() / 'examples/atlas/comparisons'))
from tiny_fp8_pack import pack_words, tensor_scale_bits


def scalar(bits, scale):
    # Independent exact integers encode the BF16 normalized significand, then
    # compare quotient/remainder for nearest-even before the VPU flush/saturate.
    field = (bits >> 7) & 255
    fraction = bits & 127
    if field == 255:
        raise ValueError('nonfinite')
    if not field:
        return 0
    quotient, remainder = divmod(128 + fraction, 16)
    quotient += int(remainder > 8 or remainder == 8 and quotient % 2)
    exponent = field - 127 - scale
    if quotient == 16:
        quotient = 8
        exponent += 1
    if exponent < -6:
        return 0
    sign = 128 if bits & 32768 else 0
    if exponent > 8 or exponent == 8 and quotient >= 15:
        return sign | 0x7e
    return sign | ((exponent + 7) << 3) | (quotient - 8)


@pytest.mark.parametrize('scale', [-126, -64, -16, -8, 0, 8, 64, 127])
def test_all_finite_BF16_signs_ties_underflow_and_saturation(scale):
    values = np.arange(65536, dtype=np.uint16)
    values = values[(values & 0x7f80) != 0x7f80].reshape(1, -1)
    actual = pack_words(values, scale).ravel()
    expected = np.fromiter((scalar(int(x), scale) for x in values.ravel()), dtype=np.uint8, count=values.size)
    assert np.array_equal(actual, expected)
    assert not np.any((actual == 127) | (actual == 255))


@pytest.mark.parametrize('value', [np.array([[0x7f80]], np.uint16), np.array([[0xffc1]], np.uint16),
                                 np.array([], np.uint16), np.array([[1]], np.uint32)])
def test_nonfinite_or_wrong_words_refused(value):
    with pytest.raises(ValueError):
        pack_words(value, 0)


def test_subnormal_tensor_scale_refused_and_signed_zero_canonicalized():
    with pytest.raises(ValueError):
        tensor_scale_bits(np.array([[1]], np.uint16))
    zero = np.array([[0, 0x8000]], np.uint16)
    assert tensor_scale_bits(zero) == 0
    assert not pack_words(zero, 0).any()
