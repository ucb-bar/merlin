"""Explicit finite typed pointwise policies, separate from original reductions.

Known storage values and selected scalar conversion/RNE semantics establish
bounded software reference cases only. NaN/Inf and unsupported formats refuse.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from .original_operator_reference import OriginalReferencePolicy, _Arithmetic, _StressArithmetic
from .original_reference_values import format_record, integer_project, round_f32

POLICY_SCHEMA = "merlin.original_operator_reference_policy.v2"
OPERATIONS = frozenset({"aten.relu.default", "aten.round.default", "aten.clamp.default"})


@dataclass(frozen=True)
class OriginalPointwiseReferencePolicy(OriginalReferencePolicy):
    """No product or reduction is implied by the shared ordered storage fields."""

    def record(self):
        self.verify()
        return {
            "schema": POLICY_SCHEMA,
            **vars(self),
            "operand_dtypes": list(self.operand_dtypes),
            "readout_dtypes": list(self.readout_dtypes),
        }

    def verify(self):
        choices = (
            "operation",
            "accumulator_dtype",
            "arithmetic",
            "product_rounding",
            "reduction_order",
            "reduction_cadence",
            "rounding",
            "bias_position",
            "zero_sign",
        )
        if any(type(getattr(self, key)) is not str for key in choices):
            raise ValueError("pointwise policy needs exact explicit string-valued choices")
        if self.operation not in OPERATIONS:
            raise ValueError("pointwise policy has no implementation of this original operation")
        if (
            type(self.operand_dtypes) is not tuple
            or type(self.readout_dtypes) is not tuple
            or len(self.operand_dtypes) != 1
            or self.readout_dtypes != self.operand_dtypes
            or self.accumulator_dtype != self.operand_dtypes[0]
            or any(type(dtype) is not str for dtype in (*self.operand_dtypes, *self.readout_dtypes))
        ):
            raise ValueError("pointwise reference preserves its single original input/output storage format")
        fmt = format_record(self.accumulator_dtype)
        if any(type(tol) is not float or not math.isfinite(tol) or tol < 0 for tol in (self.atol, self.rtol)):
            raise ValueError("pointwise reference needs explicit finite nonnegative float tolerances")
        if self.finite_only is not True or self.subnormal_operand_flush is not False:
            raise ValueError("pointwise reference implements finite values without operand flushing")
        if (
            self.product_rounding != "not_applicable"
            or self.reduction_order != "elementwise"
            or self.reduction_cadence != "not_applicable"
            or self.bias_position != "not_applicable"
            or self.zero_sign not in {"preserve", "ignore"}
        ):
            raise ValueError("pointwise reference needs explicit no-product/no-reduction semantics")
        if self.arithmetic == "finite_f32":
            if fmt["kind"] != "float_ieee" or self.rounding != "rne":
                raise ValueError("pointwise finite f32 storage requires explicitly selected RNE")
        elif self.arithmetic == "bounded_exact":
            if (
                fmt["kind"] != "int_affine"
                or self.rounding != "exact_integer"
                or self.atol != 0.0
                or self.rtol != 0.0
                or self.zero_sign != "ignore"
            ):
                raise ValueError("pointwise integer storage requires exact bounded values and comparison")
        else:
            raise ValueError("pointwise reference does not implement this arithmetic policy")


def _signed64_f32(value):
    """Direct RNE from an original signed64 Scalar, without a binary64 cast.

    The selected public c10 Scalar integer accessor converts int64 directly to
    float. RNE is the explicit policy; SDK/source correspondence stays separate.
    """
    integer_project(value, "int64", "bounded_exact")
    if value == 0:
        return 0.0
    fmt = format_record("float32")
    fraction_bits = fmt["mant_bits"]
    magnitude = abs(value)
    exponent = magnitude.bit_length() - 1
    if exponent <= fraction_bits:
        significand = magnitude << (fraction_bits - exponent)
    else:
        shift = exponent - fraction_bits
        significand, remainder = divmod(magnitude, 1 << shift)
        midpoint = 1 << (shift - 1)
        if remainder > midpoint or (remainder == midpoint and significand & 1):
            significand += 1
        if significand == 1 << (fraction_bits + 1):
            significand >>= 1
            exponent += 1
    bias = (1 << (fmt["exp_bits"] - 1)) - 1
    encoded = (
        (int(value < 0) << (fmt["element_bits"] - 1))
        | ((exponent + bias) << fraction_bits)
        | (significand - (1 << fraction_bits))
    )
    return struct.unpack("<f", encoded.to_bytes(fmt["element_bits"] // 8, "little"))[0]


def scalar_bound(value, policy):
    """Original finite scalar-to-storage conversion; no clipping or wrap guess."""
    if value is None:
        return None
    if type(value) not in {int, float} or (type(value) is float and not math.isfinite(value)):
        raise ValueError("pointwise scalar bound must be an original finite number or None")
    if policy.arithmetic == "finite_f32":
        return _signed64_f32(value) if type(value) is int else round_f32(value)
    if type(value) is not int:
        raise ValueError("integer clamp needs an original integer bound in the same storage format")
    return integer_project(value, policy.accumulator_dtype, "bounded_exact")


def validate_parameters(parameters, policy):
    expected = {"min", "max"} if policy.operation == "aten.clamp.default" else set()
    if type(parameters) is not dict or set(parameters) != expected:
        raise ValueError("pointwise reference lost its exact original scalar parameter roster")
    if expected:
        bounds = [scalar_bound(parameters[key], policy) for key in ("min", "max")]
        if bounds == [None, None]:
            raise ValueError("clamp reference needs an original lower or upper bound")


def pointwise_value(operation, value, parameters, policy):
    """Evaluate a single defined original value with ordered comparison ties."""
    if operation == "aten.relu.default":
        return (0.0 if policy.arithmetic == "finite_f32" else 0) if value < 0 else value
    if operation == "aten.round.default":
        if policy.arithmetic != "finite_f32":
            return value
        rounded = round(value)
        return math.copysign(0.0, value) if rounded == 0 else round_f32(rounded)
    if operation != "aten.clamp.default" or set(parameters) != {"min", "max"}:
        raise ValueError("pointwise reference lost its exact original operation/scalar roster")
    lower, upper = (scalar_bound(parameters[key], policy) for key in ("min", "max"))
    if lower is None and upper is None:
        raise ValueError("clamp reference needs an original lower or upper bound")
    # Ordered comparisons preserve the original value on equal +/-zero bounds.
    # Upper follows lower, so an inverted interval returns the original upper.
    if lower is not None and value < lower:
        value = lower
    if upper is not None and value > upper:
        value = upper
    return value


def reference_steps(metadata):
    """Bound pointwise comparisons/rounding/readout plus original bound conversion."""
    count = math.prod(metadata["outputs"][0]["shape"])
    # Clamp performs two conversions and two comparisons for each element,
    # followed by readout. Keep that complete work even for omitted bounds.
    return (5 if metadata["target"] == "aten.clamp.default" else 3) * count


def stress_events(operation, value, output):
    """Realized source-value facts only; no reduction or whole-domain inference."""
    floating = type(value) is float
    return {
        "pointwise_values": 1,
        "rounded_pointwise_values": int(operation == "aten.round.default" and output != value),
        "pointwise_rne_ties": int(
            floating and operation == "aten.round.default" and abs(value - math.trunc(value)) == 0.5
        ),
        "negative_zero_inputs": int(floating and value == 0 and math.copysign(1, value) < 0),
        "negative_zero_outputs": int(type(output) is float and output == 0 and math.copysign(1, output) < 0),
        "subnormal_inputs": int(floating and 0 < abs(value) < 2.0**-126),
        "subnormal_outputs": int(type(output) is float and 0 < abs(output) < 2.0**-126),
    }


class _PointwiseArithmetic(_Arithmetic):
    def pointwise(self, operation, value, parameters):
        return pointwise_value(operation, value, parameters, self.policy)


class _PointwiseStressArithmetic(_StressArithmetic):
    def __init__(self, policy):
        super().__init__(policy)
        self.counts.update(dict.fromkeys(stress_events(policy.operation, 0.0, 0.0), 0))

    def pointwise(self, operation, value, parameters):
        output = pointwise_value(operation, value, parameters, self.policy)
        for key, count in stress_events(operation, value, output).items():
            self.counts[key] += count
        return output


def pointwise_arithmetic(policy, *, stress):
    if type(policy) is not OriginalPointwiseReferencePolicy:
        raise ValueError("pointwise reference requires its exact explicit policy type")
    return _PointwiseStressArithmetic(policy) if stress else _PointwiseArithmetic(policy)
