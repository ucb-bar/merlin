"""Integer recipes either match exact arithmetic for every code or refuse."""

import logging
from fractions import Fraction

import numpy as np
import pytest

from merlin.targetgen import recipe_numerics as RN
from merlin.targetgen.exact_quantized_reference import quantize, rational

_X = np.arange(-128, 128)
_A, _B = np.meshgrid(_X, _X)


def check(recipe, values, *codes):
    expected = quantize(values, 1, quant_min=recipe.qmin, quant_max=recipe.qmax)
    assert np.array_equal(recipe.evaluate(*codes), expected)
    eager = np.clip(np.rint(np.asarray(values, dtype=np.float32)), recipe.qmin, recipe.qmax)
    count = int(np.count_nonzero(eager != expected))
    logging.getLogger(__name__).info("eager-float divergence (informational): %d", count)


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("scales", [(0.5, 0.25, 1.0), (0.03, 0.02, 0.01), (0.0137, 0.0193, 0.031)])
def test_two_input_forms_exhaust_every_code_or_refuse(sign, scales):
    sa, sb, so = map(rational, scales)
    try:
        recipe = RN.linear_combination(sa, sb, so, sign_b=sign)
    except RN.RecipeNumericsError as exc:
        assert "exact-arithmetic proof failed" in str(exc)
        return
    check(recipe, (sa * _A + sign * sb * _B) / so, _A, _B)


@pytest.mark.parametrize("trial", range(12))
def test_affine_property_exact_or_refused(trial):
    rng = np.random.default_rng(trial)
    sx, so = map(rational, rng.uniform(0.005, 0.1, size=2))
    offset = rational(rng.uniform(-3, 3))
    gain = rational(rng.choice([1.0, -1.0, 0.37, 2.5]))
    try:
        recipe = RN.affine(sx, so, gain=gain, offset=offset)
    except RN.RecipeNumericsError as exc:
        assert "exact-arithmetic proof failed" in str(exc)
        return
    check(recipe, (sx * gain * _X + offset) / so, _X)


def test_known_exact_forms_are_admitted():
    for gain, offset in [(1, 0), (-1, 0), (2, Fraction(-1, 2))]:
        recipe = RN.affine(0.25, 0.5, gain=gain, offset=offset)
        check(recipe, (Fraction(1, 4) * gain * _X + offset) * 2, _X)
    recipe = RN.affine(0.5, 1, activation="relu", qmin=-127)
    check(recipe, np.maximum(Fraction(1, 2) * _X, 0), _X)


def test_former_near_exact_recipes_are_refused():
    with pytest.raises(RN.RecipeNumericsError, match="exact-arithmetic proof failed"):
        RN.linear_combination(0.0137, 0.0193, 0.031)
    with pytest.raises(RN.RecipeNumericsError, match="tolerance"):
        RN.affine(0.1, 0.1, tolerance=1e-6)


def test_full_int32_readout_is_exact_without_float_cast():
    assert list(RN.readout(np.array([1, 3, 5, -1, -3]), 0.5)) == [0, 2, 2, 0, -2]
    assert list(RN.readout(np.array([-5, 5]), 1.0, relu=True)) == [0, 5]
    assert RN.readout(np.array([2**24 + 1]), 1.5, qmin=-(2**31), qmax=2**31 - 1).tolist() == [25165826]


def test_movement_and_windows_prove_every_reachable_sum():
    copy = RN.requantize(0.1, 0.1)
    assert copy["form"] == "copy" and copy["scale_numerator"] == copy["scale_denominator"] == "1"
    assert RN.constant_code(0.5, 0.25) == 2
    assert RN.constant_code(float("-inf"), 0.1) == -128
    with pytest.raises(RN.RecipeNumericsError, match="NaN"):
        RN.constant_code(float("nan"), 0.1)
    assert RN.window_mean_readout(0.25, 4, 0.5) == 0.125
    assert RN.window_mean_readout(0.03, 2, 0.01) == Fraction(0.03) / (2 * Fraction(0.01))
    assert RN.split_weight(381) == (127, 127, 127)


def test_inexpressible_parameters_refuse():
    with pytest.raises(RN.RecipeNumericsError, match="positive"):
        RN.linear_combination(0.0, 0.1, 0.1)
    with pytest.raises(RN.RecipeNumericsError, match="exact coefficient weights"):
        RN.linear_combination(1.0, 1e-6, 0.1)
    with pytest.raises(RN.RecipeNumericsError, match="divisor"):
        RN.window_mean_readout(0.1, 0, 0.1)
