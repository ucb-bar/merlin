"""Explicit finite typed pointwise policies, separate from original reductions.

Known storage values and selected scalar conversion/RNE semantics establish
bounded software reference cases only. NaN/Inf and unsupported formats refuse.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from fractions import Fraction

from merlin.development.targetgen.sources.original_operator_reference import (
    OriginalReferencePolicy,
    _Arithmetic,
    _evaluate,
    _sha,
    _StressArithmetic,
)
from merlin.development.targetgen.sources.original_reference_values import (
    format_record,
    integer_project,
    round_f32,
)

# ---- original_pointwise_reference --------------------------------------------------------------

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


# ---- original_pointwise_stress -----------------------------------------------------------------
# Realized finite pointwise partitions in the original reference traversal.
#
# The explicitly selected probe profile uses storage boundaries and original
# scalar arguments, never source/example geometry. These bounded observations
# prove neither the whole floating domain nor framework/build equivalence.
# Historical original_reference_stress versions are unchanged.

SCHEMA = "merlin.original_pointwise_stress.v1"
PROFILE = "finite_storage_and_original_scalar_boundaries.v1"


def edges(policy):
    policy.verify()
    fmt = format_record(policy.accumulator_dtype)
    if policy.arithmetic == "bounded_exact":
        high = 1 << (fmt["element_bits"] - 1)
        return -high, high - 1, None
    bias = (1 << (fmt["exp_bits"] - 1)) - 1
    maximum = (2.0 - 2.0 ** -fmt["mant_bits"]) * 2.0**bias
    return -maximum, maximum, 2.0 ** (1 - bias - fmt["mant_bits"])


def _neighbor(value, direction, policy):
    if policy.arithmetic == "bounded_exact":
        low, high, _ = edges(policy)
        return max(low, min(high, value + direction))
    if value == 0:
        return direction * edges(policy)[2]
    bits = int.from_bytes(struct.pack("<f", value), "little")
    bits += direction if value > 0 else -direction
    result = struct.unpack("<f", bits.to_bytes(4, "little"))[0]
    return result if math.isfinite(result) else value


def samples(policy, parameters, profile):
    """Small exact scalar roster, before any shaped tensor allocation."""
    if type(policy) is not OriginalPointwiseReferencePolicy or profile != PROFILE:
        raise ValueError("pointwise stress needs its explicit supported original policy/profile")
    low, high, subnormal = edges(policy)
    values = [low, -1, 0, 1, high]
    if subnormal is not None:
        values = [low, -2.5, -1.5, -0.5, -subnormal, -0.0, 0.0, subnormal, 0.5, 1.5, 2.5, high]
    if policy.operation == "aten.clamp.default":
        for name in ("min", "max"):
            bound = scalar_bound(parameters[name], policy)
            if bound is not None:
                values += [_neighbor(bound, -1, policy), bound, _neighbor(bound, 1, policy)]
    # Exact binary encodings distinguish signed zeros and integer values above
    # binary64's precise integer range. No ambient dtype/promotion is selected.
    unique = {}
    for value in values:
        key = struct.pack("<f", value) if subnormal is not None else value
        unique.setdefault(key, float(value) if subnormal is not None else value)
    return list(unique.values())


def required(policy, parameters):
    """Feasible original operation partitions, derived from typed boundaries."""
    low, high, subnormal = edges(policy)
    result = ["signed_inputs", "zero_input", "pointwise_values", "storage_min_input", "storage_max_input"]
    if subnormal is not None:
        result += ["positive_zero_input", "negative_zero_input", "subnormal_input"]
    if policy.operation == "aten.relu.default":
        result += ["relu_clipped", "relu_retained"]
        if subnormal is not None:
            result += ["negative_zero_output", "positive_zero_output", "subnormal_output"]
    elif policy.operation == "aten.round.default":
        if subnormal is None:
            result += ["integer_identity"]
        else:
            result += [
                "rounded_pointwise",
                "rne_positive_even_tie",
                "rne_positive_odd_tie",
                "rne_negative_even_tie",
                "rne_negative_odd_tie",
                "negative_zero_output",
                "positive_zero_output",
            ]
    else:
        lower, upper = (scalar_bound(parameters[key], policy) for key in ("min", "max"))
        result += ["clamp_lower_absent"] if lower is None else ["clamp_lower_kept", "clamp_lower_tie"]
        if lower is not None and lower > low:
            result += ["clamp_lower_applied"]
        if upper is None:
            result += ["clamp_upper_absent"]
        else:
            if lower is None or lower <= upper:
                result += ["clamp_upper_kept", "clamp_upper_tie"]
            if upper < high or (lower is not None and lower > upper):
                result += ["clamp_upper_applied"]
        if lower is not None and upper is not None and lower > upper:
            result += ["inverted_clamp"]
        if subnormal is not None:
            if (lower is None or lower <= 0) and (upper is None or upper >= 0):
                if lower == 0 or upper == 0:
                    result += ["clamp_zero_tie_preserved"]
            if any(
                value is not None and Fraction(value) != Fraction(scalar_bound(value, policy))
                for value in parameters.values()
            ):
                result += ["rounded_original_scalar"]
    return result


class _ObservedArithmetic(_PointwiseStressArithmetic):
    def __init__(self, policy):
        super().__init__(policy)
        self.counts.update(
            dict.fromkeys(
                (
                    "positive_zero_inputs",
                    "positive_zero_outputs",
                    "storage_min_inputs",
                    "storage_max_inputs",
                    "relu_clipped",
                    "relu_retained",
                    "integer_identity",
                    "rne_positive_even_ties",
                    "rne_positive_odd_ties",
                    "rne_negative_even_ties",
                    "rne_negative_odd_ties",
                    "clamp_lower_applied",
                    "clamp_lower_kept",
                    "clamp_lower_ties",
                    "clamp_upper_applied",
                    "clamp_upper_kept",
                    "clamp_upper_ties",
                    "clamp_lower_absent",
                    "clamp_upper_absent",
                    "inverted_clamp",
                    "clamp_zero_ties_preserved",
                    "rounded_original_scalars",
                ),
                0,
            )
        )

    def pointwise(self, operation, value, parameters):
        output = super().pointwise(operation, value, parameters)
        low, high, subnormal = edges(self.policy)
        c = self.counts
        c["storage_min_inputs"] += value == low
        c["storage_max_inputs"] += value == high
        if subnormal is not None:
            c["positive_zero_inputs"] += value == 0 and math.copysign(1, value) > 0
            c["positive_zero_outputs"] += output == 0 and math.copysign(1, output) > 0
        if operation == "aten.relu.default":
            c["relu_clipped"] += value < 0
            c["relu_retained"] += value >= 0
        elif operation == "aten.round.default":
            c["integer_identity"] += subnormal is None and output == value
            if subnormal is not None and abs(value - math.trunc(value)) == 0.5:
                key = "rne_" + ("negative" if value < 0 else "positive")
                key += "_odd_ties" if math.trunc(abs(value)) % 2 else "_even_ties"
                c[key] += 1
        else:
            lower, upper = (scalar_bound(parameters[key], self.policy) for key in ("min", "max"))
            middle = value
            if lower is None:
                c["clamp_lower_absent"] += 1
            else:
                c["clamp_lower_ties"] += value == lower
                c["clamp_lower_applied"] += value < lower
                c["clamp_lower_kept"] += value >= lower
                middle = lower if value < lower else value
            if upper is None:
                c["clamp_upper_absent"] += 1
            else:
                c["clamp_upper_ties"] += middle == upper
                c["clamp_upper_applied"] += middle > upper
                c["clamp_upper_kept"] += middle <= upper
            c["inverted_clamp"] += lower is not None and upper is not None and lower > upper
            if subnormal is not None:
                tied = value == 0 and (lower == 0 or upper == 0)
                c["clamp_zero_ties_preserved"] += tied and struct.pack("<f", value) == struct.pack("<f", output)
                c["rounded_original_scalars"] += sum(
                    raw is not None and Fraction(raw) != Fraction(scalar_bound(raw, self.policy))
                    for raw in parameters.values()
                )
        return output


def observe(contract, inputs):
    if type(contract.policy) is not OriginalPointwiseReferencePolicy:
        raise ValueError("pointwise stress cannot reinterpret an original reduction policy")
    metadata, values = contract._inputs(inputs)
    arithmetic = _ObservedArithmetic(contract.policy)
    for row in values:
        for value in row:
            arithmetic.counts["positive_inputs"] += value > 0
            arithmetic.counts["negative_inputs"] += value < 0
            arithmetic.counts["zero_inputs"] += value == 0
    outputs = _evaluate(metadata, values, arithmetic, contract.output_byteorder)
    scope = "Realized selected finite original reference partitions; no whole-domain/build/compiled/hardware authority."
    return {
        "schema": SCHEMA,
        "contract_sha256": contract.sha256,
        "input_sha256": [_sha(t.data) for t in inputs],
        "output_sha256": [_sha(t.data) for t in outputs],
        "counts": arithmetic.counts,
        "logical_input_shapes": [row["shape"] for row in metadata["inputs"]],
        "logical_output_shapes": [row["shape"] for row in metadata["outputs"]],
        "scope": scope,
    }


def realized(trace):
    if trace["schema"] != SCHEMA:
        raise ValueError("pointwise realized predicates need their exact original traversal version")
    c = trace["counts"]
    result = {
        "signed_inputs": c["positive_inputs"] > 0 and c["negative_inputs"] > 0,
        "zero_input": c["zero_inputs"] > 0,
    }
    keys = {
        "pointwise_values": "pointwise_values",
        "storage_min_input": "storage_min_inputs",
        "storage_max_input": "storage_max_inputs",
        "rounded_pointwise": "rounded_pointwise_values",
        "positive_zero_input": "positive_zero_inputs",
        "negative_zero_input": "negative_zero_inputs",
        "positive_zero_output": "positive_zero_outputs",
        "negative_zero_output": "negative_zero_outputs",
        "subnormal_input": "subnormal_inputs",
        "subnormal_output": "subnormal_outputs",
        "clamp_zero_tie_preserved": "clamp_zero_ties_preserved",
        "rounded_original_scalar": "rounded_original_scalars",
    }
    for key in (
        "relu_clipped",
        "relu_retained",
        "integer_identity",
        "clamp_lower_applied",
        "clamp_lower_kept",
        "clamp_upper_applied",
        "clamp_upper_kept",
        "clamp_lower_absent",
        "clamp_upper_absent",
        "inverted_clamp",
    ):
        keys[key] = key
    for key in (
        "rne_positive_even_tie",
        "rne_positive_odd_tie",
        "rne_negative_even_tie",
        "rne_negative_odd_tie",
        "clamp_lower_tie",
        "clamp_upper_tie",
    ):
        keys[key] = key + "s"
    result.update({key: c[counter] > 0 for key, counter in keys.items()})
    return result


def combined_realized(traces):
    """Require actual visits across the complete selected finite case union."""
    if not traces or any(trace["schema"] != SCHEMA for trace in traces):
        raise ValueError("pointwise finite union needs actual same-version traversal traces")
    keys = set(traces[0]["counts"])
    if any(set(trace["counts"]) != keys for trace in traces):
        raise ValueError("pointwise finite union changed its exact observed counter roster")
    counts = {key: sum(trace["counts"][key] for trace in traces) for key in keys}
    return realized({"schema": SCHEMA, "counts": counts})
