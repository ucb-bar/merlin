"""Prove integer recipes against exact rational quantized arithmetic.

Device operations are exact integer math with full int32 readout. The host
performs one exact rational RNE quantization. Coefficients and seeds must be
identical before quantization; output-code equivalence cannot license rounding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from merlin.runtime.exact_readout import requantize_int32
from merlin.targetgen.exact_arithmetic import quantize, rational

#: The largest integer weight the forms use, as three signed int8 parts of at most 127 each. A weight
#: is applied as ``sum(part * I)`` contractions into the int32 accumulator, so it never passes an int8
#: load path. Every proposed form requires exact coefficient and seed equality.
MAX_WEIGHT = 381
MAX_PART = 127
INT32 = (-(2**31), 2**31 - 1)


class RecipeNumericsError(ValueError):
    """A form cannot be stated exactly with the given scales (names the obstacle)."""


def _positive(name: str, value: float) -> float:
    value = rational(value)
    if not math.isfinite(value) or value <= 0:
        raise RecipeNumericsError(f"{name} must be a finite positive scale, got {value!r}")
    return value


def split_weight(weight: int, part_max: int = MAX_PART) -> tuple[int, ...]:
    """``weight`` as signed parts of at most ``part_max`` each (the int8 ``c*I`` constants)."""
    if type(weight) is not int:
        raise RecipeNumericsError("an integer weight is required")
    if weight == 0:
        return ()
    sign, remaining, parts = (1 if weight > 0 else -1), abs(weight), []
    while remaining:
        part = min(remaining, part_max)
        parts.append(sign * part)
        remaining -= part
    return tuple(parts)


def best_rational(ratio: float, *, max_weight: int = MAX_WEIGHT) -> tuple[int, int]:
    """Exact ``(p, q)`` with ``q > 0`` and bounded weights; otherwise refuse."""
    ratio = rational(ratio)
    p, q = ratio.numerator, ratio.denominator
    if abs(p) > max_weight or q > max_weight:
        raise RecipeNumericsError("exact-arithmetic proof failed: exact coefficient weights exceed the bound")
    return p, q


@dataclass(frozen=True)
class IntegerRecipe:
    """``y = clamp(RNE(act(sum_i w_i * x_i + constant) * rational_scale))``.

    ``weights`` apply to the int8 operand codes in order; ``constant`` is the int32 accumulator seed;
    ``activation`` is ``none`` or ``relu`` and acts on the accumulator BEFORE the scale, as a readout
    does. ``ratio_error`` records how far the integer form is from the floating coefficients.
    """

    weights: tuple[int, ...]
    constant: int
    readout_scale: object
    activation: str = "none"
    qmin: int = -128
    qmax: int = 127
    ratio_error: float = 0.0
    form: str = ""

    def __post_init__(self) -> None:
        if len(self.weights) not in (1, 2):
            raise RecipeNumericsError("an affine recipe requires at most two tensor operands")
        if not math.isfinite(self.readout_scale):
            raise RecipeNumericsError("readout scale must be finite")
        if self.activation not in {"none", "relu"}:
            raise RecipeNumericsError(f"activation must be none or relu, got {self.activation!r}")
        if self.activation == "relu" and self.readout_scale < 0:
            # The readout applies ReLU before the scale: relu(x)*s with s < 0 is never relu(x*s).
            raise RecipeNumericsError("a readout ReLU cannot pair with a negative readout scale")
        if type(self.constant) is not int or not INT32[0] <= self.constant <= INT32[1]:
            raise RecipeNumericsError("the accumulator constant exceeds int32")
        if any(type(weight) is not int or abs(weight) > MAX_WEIGHT for weight in self.weights):
            raise RecipeNumericsError(f"an integer weight exceeds {MAX_WEIGHT}")

    def parts(self) -> tuple[tuple[int, ...], ...]:
        """Each weight as its signed int8 ``c*I`` parts."""
        return tuple(split_weight(weight) for weight in self.weights)

    def accumulator_bound(self, operand_bound: int = 128) -> int:
        """The largest |accumulator| the form can produce for int8 operands."""
        return sum(abs(weight) for weight in self.weights) * operand_bound + abs(self.constant)

    def evaluate(self, *codes) -> np.ndarray:
        """Execute the form on int8 operand codes (arrays broadcast), exactly as specified."""
        if len(codes) != len(self.weights):
            raise RecipeNumericsError(f"the form takes {len(self.weights)} operand(s), got {len(codes)}")
        accumulator = np.full(np.broadcast_shapes(*(np.shape(c) for c in codes)) if codes else (), self.constant)
        accumulator = accumulator.astype(np.int64)
        for weight, code in zip(self.weights, codes, strict=True):
            accumulator = accumulator + weight * np.asarray(code, dtype=np.int64)
        if np.any(accumulator < INT32[0]) or np.any(accumulator > INT32[1]):
            raise RecipeNumericsError("the int32 accumulator overflowed")
        return readout(accumulator, self.readout_scale, relu=self.activation == "relu", qmin=self.qmin, qmax=self.qmax)

    def to_dict(self) -> dict:
        return {
            "form": self.form,
            "weights": list(self.weights),
            "weight_parts": [list(parts) for parts in self.parts()],
            "constant": self.constant,
            "readout": "full_i32",
            "requantization": "exact_rational_host",
            "scale_numerator": str(rational(self.readout_scale).numerator),
            "scale_denominator": str(rational(self.readout_scale).denominator),
            "zero_point": 0,
            "rounding": "half_to_even",
            "activation": self.activation,
            "quant_min": self.qmin,
            "quant_max": self.qmax,
            "ratio_error": float(self.ratio_error),
        }


def readout(accumulator, scale: float, *, relu: bool = False, qmin: int = -128, qmax: int = 127) -> np.ndarray:
    """Full-width int32 accumulator -> exact host RNE and saturation."""
    values = np.asarray(accumulator)
    if relu:
        values = np.maximum(values, 0)
    coefficient = rational(scale)
    try:
        return requantize_int32(
            values,
            scale_numerator=coefficient.numerator,
            scale_denominator=coefficient.denominator,
            quant_min=qmin,
            quant_max=qmax,
        )
    except ValueError as exc:
        raise RecipeNumericsError(str(exc)) from exc


def retained_affine_readout(coefficients, offset, s_out, *, activation="none", qmin=-128, qmax=127):
    """Describe an exact source host epilogue over full-width integer operands.

    This is not a device schedule proof. No coefficient, bias or partial term is
    rounded. The selected provider must cover the integer producers and retain
    the complete rational expression through its single RNE/clamp boundary.
    """
    scale = rational(_positive("s_out", s_out))
    factors = [rational(value) / scale for value in coefficients]
    bias = rational(offset) / scale
    if len(factors) not in (1, 2) or activation not in {"none", "relu"}:
        raise RecipeNumericsError("unsupported retained affine readout")
    return dict(
        form="retained_affine",
        readout="full_i32",
        requantization="retained_source_exact_integer",
        coefficients_numerator=[str(v.numerator) for v in factors],
        coefficients_denominator=[str(v.denominator) for v in factors],
        offset_numerator=str(bias.numerator),
        offset_denominator=str(bias.denominator),
        activation=activation,
        quant_min=qmin,
        quant_max=qmax,
        rounding="half_to_even",
    )


def linear_combination(
    s_a: float,
    s_b: float,
    s_out: float,
    *,
    sign_b: int = 1,
    activation: str = "none",
    qmin: int = -128,
    qmax: int = 127,
    max_weight: int = MAX_WEIGHT,
) -> IntegerRecipe:
    """``y = (s_a*a + sign_b*s_b*b)/s_out`` (add.Tensor, sub.Tensor with ``sign_b=-1``).

    ``Wa/Wb`` is the exact rational representation of ``s_a/s_b`` within ``max_weight``; the minus of a
    subtraction is the sign of ``Wb``, applied in the int32 accumulator and never to an int8 operand
    (negating -128 on a load saturates). The readout scale is ``s_a/(s_out*Wa)``.
    """
    s_a, s_b, s_out = (
        rational(_positive(name, value)) for name, value in (("s_a", s_a), ("s_b", s_b), ("s_out", s_out))
    )
    if sign_b not in (1, -1):
        raise RecipeNumericsError("sign_b must be +1 or -1")
    wb, wa = best_rational(s_b / s_a, max_weight=max_weight)
    if wb == 0:
        raise RecipeNumericsError(f"s_b/s_a = {s_b / s_a!r} rounds to a zero weight within {max_weight}")
    recipe = IntegerRecipe(
        weights=(wa, sign_b * wb),
        constant=0,
        readout_scale=s_a / (s_out * wa),
        activation=activation,
        qmin=qmin,
        qmax=qmax,
        ratio_error=0,
        form="linear_combination",
    )
    return prove_recipe(recipe, (s_a / s_out, sign_b * s_b / s_out))


def affine(
    s_x: float,
    s_out: float,
    *,
    gain: float = 1.0,
    offset: float = 0.0,
    activation: str = "none",
    qmin: int = -128,
    qmax: int = 127,
    max_weight: int = MAX_WEIGHT,
    tolerance: float = 0,
) -> IntegerRecipe:
    """Propose integer weights/seed, then prove the complete code domain exactly.

    A small coefficient error is insufficient: every rounded/clamped output
    must equal rational evaluation of the original gain and offset.
    """
    s_x, s_out = rational(_positive("s_x", s_x)), rational(_positive("s_out", s_out))
    gain, offset = rational(gain), rational(offset)
    if tolerance != 0:
        raise RecipeNumericsError("exact recipes do not permit an approximation tolerance")
    if not math.isfinite(gain) or gain == 0 or not math.isfinite(offset):
        raise RecipeNumericsError("gain must be finite and nonzero, and offset finite")
    shift = offset / (gain * s_x)  # the offset in input codes
    if shift == 0:
        recipe = IntegerRecipe(
            weights=(1,),
            constant=0,
            readout_scale=gain * s_x / s_out,
            activation=activation,
            qmin=qmin,
            qmax=qmax,
            form="affine",
        )
        return prove_recipe(recipe, (gain * s_x / s_out,), offset / s_out)
    g, seed = shift.denominator, shift.numerator
    if g > max_weight:
        raise RecipeNumericsError("exact-arithmetic proof failed: affine seed requires unbounded integer weights")
    error = 0
    recipe = IntegerRecipe(
        weights=(g,),
        constant=int(seed),
        readout_scale=gain * s_x / (s_out * g),
        activation=activation,
        qmin=qmin,
        qmax=qmax,
        ratio_error=error,
        form="affine",
    )
    return prove_recipe(recipe, (gain * s_x / s_out,), offset / s_out)


def prove_recipe(recipe, coefficients, offset=0):
    """Require identical intermediate coefficients, then exhaust the signed-i8 code domain."""
    factor = rational(recipe.readout_scale)
    if (
        any(rational(c) != w * factor for c, w in zip(coefficients, recipe.weights, strict=True))
        or rational(offset) != recipe.constant * factor
    ):
        raise RecipeNumericsError("exact-arithmetic proof failed: intermediate arithmetic differs")
    codes = np.arange(-128, 128)
    inputs = np.meshgrid(*([codes] * len(coefficients)), indexing="ij")
    values = np.full(inputs[0].shape, rational(offset), dtype=object)
    for coefficient, code in zip(coefficients, inputs, strict=True):
        values += rational(coefficient) * code
    if recipe.activation == "relu":
        values = np.maximum(values, 0)
    expected = quantize(values, 1, quant_min=recipe.qmin, quant_max=recipe.qmax)
    actual = recipe.evaluate(*inputs)
    if not np.array_equal(actual, expected):
        count = int(np.count_nonzero(actual != expected))
        raise RecipeNumericsError(f"exact-arithmetic proof failed for {count} input codes")
    return recipe


def requantize(s_in: float, s_out: float, *, qmin=-128, qmax=127) -> dict:
    """A movement between two scales: an exact copy when equal, else one rounding at ``s_in/s_out``.

    Scaling runs on the host in exact rational arithmetic; it is never an approximate MVIN scale.
    """
    s_in, s_out = _positive("s_in", s_in), _positive("s_out", s_out)
    ratio = rational(s_in) / rational(s_out)
    return dict(
        form="copy" if ratio == 1 else "host_requant",
        readout="full_i32",
        scale_numerator=str(ratio.numerator),
        scale_denominator=str(ratio.denominator),
        zero_point=0,
        quant_min=qmin,
        quant_max=qmax,
        rounding="half_to_even",
    )


def constant_code(value: float, s_out: float, *, qmin: int = -128, qmax: int = 127) -> int:
    """A fill or pad value on the output grid: ``sat(RNE(value/s_out))``, computed at compile time."""
    s_out = _positive("s_out", s_out)
    if math.isnan(value):
        raise RecipeNumericsError("a NaN fill has no exact rational code")
    if not math.isfinite(value):
        # -inf/+inf fills (masks) saturate to the rails, as the reference requantization does.
        return qmin if value < 0 else qmax
    return int(min(qmax, max(qmin, round(rational(value) / rational(s_out)))))


def window_mean_readout(s_in: float, divisor: int, s_out: float, *, qmin=-128, qmax=127) -> float:
    """Readout scale of an average pool / mean over ``divisor`` summed elements: ``s_in/(divisor*s_out)``."""
    if type(divisor) is not int or divisor < 1:
        raise RecipeNumericsError("divisor must be a positive integer count")
    ratio = rational(_positive("s_in", s_in)) / (divisor * rational(_positive("s_out", s_out)))
    return ratio


def narrow_range(qmax: int = 127) -> tuple[int, int]:
    """Symmetric narrow int8 range for the producer of a negation, so ``-(-128)`` never occurs."""
    return -qmax, qmax
