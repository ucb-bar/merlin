"""Framework-independent records for quantized accumulator readout.

These records describe source meaning, implementation candidates and target limits.
Accuracy policy, reference evaluation and admission remain independent consumers.
Serialization and schema identities are preserved from the original contract owner.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

CONTRACT_SCHEMA = "target_neutral_quantized_epilogue_contract_v1"
_GRANULARITIES = frozenset({"per_tensor", "per_axis"})
_BIAS_DOMAINS = frozenset({"none", "real", "accumulator"})
_ROUNDINGS = frozenset({"round_to_nearest_even"})
_EXECUTABLE_STAGES = frozenset({
    "bias_i32", "scale_f32", "round_to_nearest_even", "output_zero_point", "clamp", "relu",
})

__all__ = [
    "CONTRACT_SCHEMA",
    "QuantParameter",
    "QuantizedEpilogueCandidate",
    "QuantizedEpilogueCapability",
    "QuantizedEpilogueContract",
    "canonical_sha256",
]


def canonical_sha256(value: Mapping[str, Any]) -> str:
    """Hash one receipt using the framework's canonical JSON representation."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rows(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


@dataclass(frozen=True)
class QuantParameter:
    """Placement and binding of one exact source quantization parameter."""

    granularity: str
    axis: int | None
    dtype: str
    binding: str

    def __post_init__(self) -> None:
        if self.granularity not in _GRANULARITIES:
            raise ValueError(f"unsupported quantization granularity {self.granularity!r}")
        if (self.granularity == "per_axis") != (self.axis is not None):
            raise ValueError("per_axis parameters require an axis; per_tensor parameters forbid it")
        if self.axis is not None and self.axis < 0:
            raise ValueError("a quantization axis must be nonnegative")
        if not self.dtype.strip() or not self.binding.strip():
            raise ValueError("quantization parameter dtype and binding are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "granularity": self.granularity,
            "axis": self.axis,
            "dtype": self.dtype,
            "binding": self.binding,
        }


@dataclass(frozen=True)
class QuantizedEpilogueContract:
    """Exact source semantics at one accumulator-to-quantized boundary."""

    accumulator_dtype: str
    output_dtype: str
    channel_axis: int
    channel_count: int
    activation_scale: QuantParameter
    weight_scale: QuantParameter
    output_scale_reciprocal: QuantParameter
    input_zero_point: QuantParameter
    weight_zero_point: QuantParameter
    output_zero_point: QuantParameter
    bias_domain: str
    bias: QuantParameter | None
    ordered_stages: tuple[str, ...]
    rounding: str
    saturation: tuple[int, int]
    activation: str = "none"

    def __post_init__(self) -> None:
        if not self.accumulator_dtype.strip() or not self.output_dtype.strip():
            raise ValueError("accumulator and output dtypes are required")
        if self.channel_axis < 0 or self.channel_count < 1:
            raise ValueError("channel axis/count must be nonnegative/positive")
        if self.bias_domain not in _BIAS_DOMAINS:
            raise ValueError(f"unsupported bias domain {self.bias_domain!r}")
        if (self.bias_domain == "none") != (self.bias is None):
            raise ValueError("bias domain and bias parameter disagree")
        if self.rounding not in _ROUNDINGS:
            raise ValueError(f"unsupported rounding {self.rounding!r}")
        if len(self.saturation) != 2 or self.saturation[0] >= self.saturation[1]:
            raise ValueError("saturation bounds must be two ordered integers")
        if not self.ordered_stages or any(not stage.strip() for stage in self.ordered_stages):
            raise ValueError("the exact ordered source stages are required")
        if not self.activation.strip():
            raise ValueError("activation semantics must be named explicitly")
        parameters = (
            self.activation_scale,
            self.weight_scale,
            self.output_scale_reciprocal,
            self.input_zero_point,
            self.weight_zero_point,
            self.output_zero_point,
            *(() if self.bias is None else (self.bias,)),
        )
        wrong_axes = [
            parameter.binding
            for parameter in parameters
            if parameter.granularity == "per_axis" and parameter.axis != self.channel_axis
        ]
        if wrong_axes:
            raise ValueError(f"per-axis parameters disagree with channel axis: {wrong_axes}")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema": CONTRACT_SCHEMA,
            "accumulator_dtype": self.accumulator_dtype,
            "output_dtype": self.output_dtype,
            "channel_axis": self.channel_axis,
            "channel_count": self.channel_count,
            "scales": {
                "activation": self.activation_scale.to_dict(),
                "weight": self.weight_scale.to_dict(),
                "output_reciprocal": self.output_scale_reciprocal.to_dict(),
            },
            "zero_points": {
                "input": self.input_zero_point.to_dict(),
                "weight": self.weight_zero_point.to_dict(),
                "output": self.output_zero_point.to_dict(),
            },
            "bias_domain": self.bias_domain,
            "bias": None if self.bias is None else self.bias.to_dict(),
            "ordered_stages": list(self.ordered_stages),
            "rounding": self.rounding,
            "saturation": list(self.saturation),
            "activation": self.activation,
        }
        result["semantic_contract_sha256"] = canonical_sha256(result)
        return result


@dataclass(frozen=True)
class QuantizedEpilogueCandidate:
    """One concrete accumulator readout proposed by a target adapter or tuner."""

    capability: str
    scale_granularity: str
    scale_axis: int | None
    scales: tuple[float, ...]
    bias_domain: str
    bias: tuple[int, ...]
    output_zero_point: int
    ordered_stages: tuple[str, ...]
    rounding: str
    saturation: tuple[int, int]
    activation: str = "none"

    def __post_init__(self) -> None:
        if not self.capability.strip():
            raise ValueError("candidate capability is required")
        if self.scale_granularity not in _GRANULARITIES:
            raise ValueError("candidate scale granularity must be per_tensor or per_axis")
        if (self.scale_granularity == "per_axis") != (self.scale_axis is not None):
            raise ValueError("candidate per-axis scale requires an axis")
        if self.scale_axis is not None and self.scale_axis < 0:
            raise ValueError("candidate scale axis must be nonnegative")
        if not self.scales or any(not math.isfinite(value) or value <= 0 for value in self.scales):
            raise ValueError("candidate scales must be finite and positive")
        if self.scale_granularity == "per_tensor" and len(self.scales) != 1:
            raise ValueError("per_tensor candidate scale must have one value")
        if self.bias_domain not in {"none", "accumulator"}:
            raise ValueError("candidate bias must be absent or in the accumulator domain")
        if (self.bias_domain == "none") != (not self.bias):
            raise ValueError("candidate bias domain and values disagree")
        if self.rounding not in _ROUNDINGS:
            raise ValueError("candidate rounding is unsupported by the reference evaluator")
        if len(self.saturation) != 2 or self.saturation[0] >= self.saturation[1]:
            raise ValueError("candidate saturation requires two ordered bounds")
        if not self.activation.strip():
            raise ValueError("candidate activation semantics must be explicit")
        if not self.ordered_stages or len(set(self.ordered_stages)) != len(self.ordered_stages):
            raise ValueError("candidate stage order must be nonempty and duplicate-free")
        unknown = set(self.ordered_stages) - _EXECUTABLE_STAGES
        if unknown:
            raise ValueError(f"candidate has unsupported semantic stages {sorted(unknown)}")
        required = {"scale_f32", "round_to_nearest_even", "clamp"}
        if self.bias:
            required.add("bias_i32")
        if self.output_zero_point:
            required.add("output_zero_point")
        if self.activation == "relu":
            required.add("relu")
        if set(self.ordered_stages) != required:
            raise ValueError(
                f"candidate stage set {list(self.ordered_stages)!r} != {sorted(required)!r}"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> QuantizedEpilogueCandidate:
        scale = value.get("scale") if isinstance(value.get("scale"), Mapping) else {}
        bias = value.get("bias") if isinstance(value.get("bias"), Mapping) else {}
        return cls(
            capability=str(value.get("capability") or ""),
            scale_granularity=str(scale.get("granularity") or ""),
            scale_axis=None if scale.get("axis") is None else int(scale["axis"]),
            scales=tuple(float(item) for item in _rows(scale.get("values"))),
            bias_domain=str(bias.get("domain") or "none"),
            bias=tuple(int(item) for item in _rows(bias.get("values"))),
            output_zero_point=int(value.get("output_zero_point", 0)),
            ordered_stages=tuple(str(item) for item in _rows(value.get("ordered_stages"))),
            rounding=str(value.get("rounding") or ""),
            saturation=tuple(int(item) for item in _rows(value.get("saturation"))),
            activation=str(value.get("activation") or "none"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "scale": {
                "granularity": self.scale_granularity,
                "axis": self.scale_axis,
                "values": list(self.scales),
            },
            "bias": {"domain": self.bias_domain, "values": list(self.bias)},
            "output_zero_point": self.output_zero_point,
            "ordered_stages": list(self.ordered_stages),
            "rounding": self.rounding,
            "saturation": list(self.saturation),
            "activation": self.activation,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class QuantizedEpilogueCapability:
    """Target-supplied limits.  Accuracy evidence cannot override any of them."""

    name: str
    accumulator_dtypes: tuple[str, ...]
    output_dtypes: tuple[str, ...]
    scale_granularities: tuple[str, ...]
    bias_domains: tuple[str, ...]
    output_zero_points: tuple[int, ...]
    roundings: tuple[str, ...]
    saturations: tuple[tuple[int, int], ...]
    ordered_stage_templates: tuple[tuple[str, ...], ...]
    activations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("capability name is required")
        rows = (
            self.accumulator_dtypes,
            self.output_dtypes,
            self.scale_granularities,
            self.bias_domains,
            self.roundings,
            self.saturations,
            self.ordered_stage_templates,
            self.activations,
        )
        if any(not row for row in rows):
            raise ValueError("capability sets must be explicit and nonempty")
        if set(self.scale_granularities) - _GRANULARITIES:
            raise ValueError("capability names an unknown scale granularity")
        if set(self.bias_domains) - {"none", "accumulator"}:
            raise ValueError("capability names an unsupported bias domain")
        if set(self.roundings) - _ROUNDINGS:
            raise ValueError("capability names an unsupported reference rounding")

    def refusal_reasons(
        self, candidate: QuantizedEpilogueCandidate, source: QuantizedEpilogueContract
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if candidate.capability != self.name:
            reasons.append("candidate_capability_not_supported")
        if source.accumulator_dtype not in self.accumulator_dtypes:
            reasons.append("accumulator_dtype_not_supported")
        if source.output_dtype not in self.output_dtypes:
            reasons.append("output_dtype_not_supported")
        if candidate.scale_granularity not in self.scale_granularities:
            reasons.append("candidate_scale_granularity_not_supported")
        if candidate.scale_granularity == "per_axis" and (
            candidate.scale_axis != source.channel_axis
            or len(candidate.scales) != source.channel_count
        ):
            reasons.append("candidate_scale_axis_or_extent_not_supported")
        if candidate.bias_domain not in self.bias_domains:
            reasons.append("candidate_bias_domain_not_supported")
        if candidate.bias and len(candidate.bias) not in (1, source.channel_count):
            reasons.append("candidate_bias_extent_not_supported")
        if candidate.output_zero_point not in self.output_zero_points:
            reasons.append("candidate_output_zero_point_not_supported")
        if candidate.rounding not in self.roundings:
            reasons.append("candidate_rounding_not_supported")
        if candidate.saturation not in self.saturations:
            reasons.append("candidate_saturation_not_supported")
        if candidate.ordered_stages not in self.ordered_stage_templates:
            reasons.append("candidate_stage_order_not_supported")
        if candidate.activation not in self.activations:
            reasons.append("candidate_activation_not_supported")
        if candidate.activation != source.activation:
            reasons.append("candidate_activation_does_not_match_source")
        return tuple(dict.fromkeys(reasons))

    def to_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "accumulator_dtypes": list(self.accumulator_dtypes),
            "output_dtypes": list(self.output_dtypes),
            "scale_granularities": list(self.scale_granularities),
            "bias_domains": list(self.bias_domains),
            "output_zero_points": list(self.output_zero_points),
            "roundings": list(self.roundings),
            "saturations": [list(item) for item in self.saturations],
            "ordered_stage_templates": [list(item) for item in self.ordered_stage_templates],
            "activations": list(self.activations),
        }
        result["capability_sha256"] = canonical_sha256(result)
        return result
