"""Scalar rational convolution preserves every product and final rounding."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from merlin.targetgen.exact_quantized_reference import _convolution


def _convolution_fraction(x, w, bias, stride, padding, dilation, groups):
    dims = x.ndim - 2
    shape = tuple((x.shape[2+i] + 2*padding - dilation*(w.shape[2+i]-1)-1)//stride+1 for i in range(dims))
    padded = np.pad(x, [(0,0),(0,0),*[(padding,padding)]*dims])
    out = np.empty((x.shape[0],w.shape[0],*shape), dtype=object)
    per_group = w.shape[0] // groups
    for n,co,*p in np.ndindex(out.shape):
        window = padded[(n, slice(co//per_group*w.shape[1],(co//per_group+1)*w.shape[1]),
                         *[slice(v*stride,v*stride+dilation*w.shape[2+i],dilation) for i,v in enumerate(p)])]
        out[(n,co,*p)] = sum((Fraction(int(a.numerator), int(a.denominator))*Fraction(int(b.numerator), int(b.denominator)) for a,b in zip(window.flat,w[co].flat)), Fraction(0)) + (bias[co] if bias is not None else 0)
    return out


@pytest.mark.parametrize(
    "dims,groups,stride,padding,dilation",
    [
        (1, 1, 1, 0, 1),
        (1, 2, 2, 1, 2),
        (2, 1, 1, 1, 1),
        (2, 2, 2, 1, 1),
        (2, 3, 1, 2, 2),
        (3, 1, 1, 1, 1),
        (3, 2, 2, 1, 2),
    ],
)
@pytest.mark.parametrize("bias", [False, True])
def test_randomized_grouped_dilated_convolution(dims, groups, stride, padding, dilation, bias):
    rng = np.random.default_rng(166)
    shape = (2, groups * 2, *([5] * dims))
    weight_shape = (groups * 3, 2, *([2] * dims))
    source = rng.integers(-128, 128, shape).astype(object) * Fraction(float(np.float32(0.03)))
    weight = rng.integers(-128, 128, weight_shape).astype(object)
    for co in range(weight.shape[0]):
        weight[co] *= Fraction(float(np.float32(0.02 + co * 0.001)))
    bias_value = np.array([Fraction(co - 3, 37) for co in range(weight.shape[0])], dtype=object) if bias else None
    fast = _convolution(source, weight, bias_value, stride, padding, dilation, groups)
    slow = _convolution_fraction(source, weight, bias_value, stride, padding, dilation, groups)
    np.testing.assert_array_equal(fast, slow)
    # The final quantize must also preserve every asymmetric code and tie.
    for zp in (0, -13, 127):
        scale = Fraction(float(np.float32(0.01)))

        def code(a):
            return np.array([min(127, max(-128, round(v / scale) + zp)) for v in a.flat])

        np.testing.assert_array_equal(code(fast), code(slow))


@pytest.mark.parametrize(
    "source,weight",
    [
        ([-128, 127], [-128, 127]),
        ([-(1 << 31), (1 << 31) - 1], [1, 1]),
        ([(1 << 62), 1], [2, 1]),
        ([1 << 80, 1], [-(1 << 70), 1]),
        ([1 << 80, 1], [0, 0]),
        ([0, 0], [1 << 80, 1]),
        ([Fraction(1, 3), Fraction(1, 7)], [Fraction(1, 11), Fraction(-1, 17)]),
        ([Fraction(2) ** -149, Fraction(2) ** 100], [Fraction(2) ** -149, Fraction(2) ** 100]),
    ],
)
def test_boundaries_and_python_integer_fallback(source, weight):
    x = np.asarray(source, dtype=object).reshape(1, 1, 2)
    w = np.asarray(weight, dtype=object).reshape(1, 1, 2)
    np.testing.assert_array_equal(_convolution(x, w, None, 1, 1, 1, 1), _convolution_fraction(x, w, None, 1, 1, 1, 1))


@pytest.mark.parametrize("first,expected", [(1 << 62, (1 << 63) + 1), ((1 << 61) - 1, (1 << 62) - 1)])
def test_overflow_and_near_limit_bounds_use_python_integer_macs(monkeypatch, first, expected):
    original = np.matmul
    types = []

    def matmul(a, b):
        types.append((a.dtype, b.dtype))
        return original(a, b)

    monkeypatch.setattr(np, "matmul", matmul)
    x = np.array([first, 1], dtype=object).reshape(1, 1, 2)
    w = np.array([2, 1], dtype=object).reshape(1, 1, 2)
    assert _convolution(x, w, None, 1, 0, 1, 1).item() == expected
    assert not types


def test_exhaustive_small_integer_and_half_tie_domain():
    for values in product(range(-2, 3), repeat=4):
        x = np.asarray(values[:2], dtype=object).reshape(1, 1, 2) * Fraction(1, 2)
        w = np.asarray(values[2:], dtype=object).reshape(1, 1, 2) * Fraction(1, 4)
        fast = _convolution(x, w, None, 1, 0, 1, 1)
        slow = _convolution_fraction(x, w, None, 1, 0, 1, 1)
        assert fast.item() == slow.item()
        assert round(fast.item()) == round(slow.item())
