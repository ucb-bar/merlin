"""Realized finite pointwise partitions in the original reference traversal.

The explicitly selected probe profile uses storage boundaries and original
scalar arguments, never source/example geometry. These bounded observations
prove neither the whole floating domain nor framework/build equivalence.
Historical original_reference_stress versions are unchanged.
"""

from __future__ import annotations

import math
import struct
from fractions import Fraction

from .original_operator_reference import _evaluate, _sha
from .original_pointwise_reference import OriginalPointwiseReferencePolicy, _PointwiseStressArithmetic, scalar_bound
from .original_reference_values import format_record

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
