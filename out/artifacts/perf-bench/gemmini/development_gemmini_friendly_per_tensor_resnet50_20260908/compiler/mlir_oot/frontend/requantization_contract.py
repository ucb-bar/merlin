"""Target-neutral contracts and an opt-in accuracy gate for narrow requantization.

The source contract records *what the program says*.  A target adapter separately records the
narrow readout it can actually execute.  Calibration evidence may select a non-identical target
readout only when an explicit, source-bound budget passes.  Absence, ambiguity, stale evidence,
or unsupported target semantics always means refusal; exact lowering remains the default.

This module intentionally contains no model names, shapes, target names, or compiler IR parsing.
It is reusable by convolution, matmul, and other accumulator-producing backends.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence


_GRANULARITIES = {"per_tensor", "per_axis"}
_BIAS_DOMAINS = {"none", "real_f32", "accumulator_i32"}
_ROUNDINGS = {"round_to_nearest_even"}


def _canonical_sha(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: Sequence[int]) -> str:
    payload = json.dumps(list(value), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


@dataclass(frozen=True)
class QuantParameter:
    """Quantization parameter placement, independent of its compile/runtime binding."""

    granularity: str
    axis: int | None
    dtype: str
    binding: str

    def __post_init__(self) -> None:
        if self.granularity not in _GRANULARITIES:
            raise ValueError(f"unsupported quantization granularity {self.granularity!r}")
        if (self.granularity == "per_axis") != (self.axis is not None):
            raise ValueError("per_axis parameters require an axis; per_tensor parameters forbid it")
        if not self.dtype or not self.binding:
            raise ValueError("quantization parameter dtype and binding are required")

    def receipt(self) -> dict[str, Any]:
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
    relu: bool

    def __post_init__(self) -> None:
        if self.bias_domain not in _BIAS_DOMAINS:
            raise ValueError(f"unsupported bias domain {self.bias_domain!r}")
        if (self.bias_domain == "none") != (self.bias is None):
            raise ValueError("bias domain and bias parameter disagree")
        if self.rounding not in _ROUNDINGS:
            raise ValueError(f"unsupported rounding {self.rounding!r}")
        if self.channel_count < 1 or self.channel_axis < 0:
            raise ValueError("channel axis/count must be non-negative/positive")
        if self.saturation[0] >= self.saturation[1]:
            raise ValueError("saturation bounds must be ordered")
        if len(set(self.ordered_stages)) != len(self.ordered_stages):
            raise ValueError("ordered stages must be unique")

    def receipt(self) -> dict[str, Any]:
        result = {
            "schema": "target_neutral_quantized_epilogue_contract_v1",
            "accumulator_dtype": self.accumulator_dtype,
            "output_dtype": self.output_dtype,
            "channel_axis": self.channel_axis,
            "channel_count": self.channel_count,
            "scales": {
                "activation": self.activation_scale.receipt(),
                "weight": self.weight_scale.receipt(),
                "output_reciprocal": self.output_scale_reciprocal.receipt(),
            },
            "zero_points": {
                "input": self.input_zero_point.receipt(),
                "weight": self.weight_zero_point.receipt(),
                "output": self.output_zero_point.receipt(),
            },
            "bias_domain": self.bias_domain,
            "bias": None if self.bias is None else self.bias.receipt(),
            "ordered_stages": list(self.ordered_stages),
            "rounding": self.rounding,
            "saturation": list(self.saturation),
            "relu": self.relu,
        }
        result["semantic_contract_sha256"] = _canonical_sha(result)
        return result


@dataclass(frozen=True)
class NativeNarrowCandidate:
    """A concrete accumulator readout proposed by a target adapter or tuner."""

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
    relu: bool

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "NativeNarrowCandidate":
        scale = row.get("scale") or {}
        bias = row.get("bias") or {}
        result = cls(
            capability=str(row.get("capability", "")),
            scale_granularity=str(scale.get("granularity", "")),
            scale_axis=(None if scale.get("axis") is None else int(scale["axis"])),
            scales=tuple(float(value) for value in scale.get("values", ())),
            bias_domain=str(bias.get("domain", "none")),
            bias=tuple(int(value) for value in bias.get("values", ())),
            output_zero_point=int(row.get("output_zero_point", 0)),
            ordered_stages=tuple(str(stage) for stage in row.get("ordered_stages", ())),
            rounding=str(row.get("rounding", "")),
            saturation=tuple(int(value) for value in row.get("saturation", ())),
            relu=bool(row.get("relu", False)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.capability:
            raise ValueError("candidate capability is required")
        if self.scale_granularity not in _GRANULARITIES:
            raise ValueError("candidate scale granularity must be per_tensor or per_axis")
        if (self.scale_granularity == "per_axis") != (self.scale_axis is not None):
            raise ValueError("candidate per-axis scale requires an axis")
        if len(self.scales) < 1 or any(not math.isfinite(x) or x <= 0 for x in self.scales):
            raise ValueError("candidate scales must be finite and positive")
        if self.scale_granularity == "per_tensor" and len(self.scales) != 1:
            raise ValueError("per_tensor candidate scale must have one value")
        if self.bias_domain not in {"none", "accumulator_i32"}:
            raise ValueError("native candidate bias must be absent or accumulator_i32")
        if self.bias_domain == "none" and self.bias:
            raise ValueError("bias values supplied for a no-bias candidate")
        if self.bias_domain == "accumulator_i32" and not self.bias:
            raise ValueError("accumulator_i32 candidate needs bias values")
        if self.rounding not in _ROUNDINGS:
            raise ValueError("candidate rounding is unsupported")
        if len(self.saturation) != 2 or self.saturation[0] >= self.saturation[1]:
            raise ValueError("candidate saturation requires two ordered bounds")
        expected = ({"bias_i32"} if self.bias else set()) | {
            "acc_scale", "round_to_nearest_even", "saturate_i8"}
        if self.output_zero_point:
            expected.add("zero_point")
        if self.relu:
            expected.add("relu")
        if len(set(self.ordered_stages)) != len(self.ordered_stages):
            raise ValueError("candidate stage order contains duplicates")
        if set(self.ordered_stages) != expected:
            raise ValueError(
                f"candidate stage set {list(self.ordered_stages)!r} != {sorted(expected)!r}")

    def receipt(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "scale": {"granularity": self.scale_granularity, "axis": self.scale_axis,
                      "values": list(self.scales)},
            "bias": {"domain": self.bias_domain, "values": list(self.bias)},
            "output_zero_point": self.output_zero_point,
            "ordered_stages": list(self.ordered_stages),
            "rounding": self.rounding,
            "saturation": list(self.saturation),
            "relu": self.relu,
        }


@dataclass(frozen=True)
class NativeNarrowCapability:
    """Target-adapter limits; the accuracy gate cannot override them."""

    name: str
    scale_granularities: tuple[str, ...]
    scale_axis: int | None
    bias_domains: tuple[str, ...]
    output_zero_points: tuple[int, ...]
    rounding: tuple[str, ...]
    saturation: tuple[int, int]
    ordered_stage_templates: tuple[tuple[str, ...], ...]

    def refusal(self, candidate: NativeNarrowCandidate, source: QuantizedEpilogueContract) -> str | None:
        if candidate.capability != self.name:
            return "candidate_capability_not_supported"
        if candidate.scale_granularity not in self.scale_granularities:
            return "candidate_scale_granularity_not_supported"
        if candidate.scale_granularity == "per_axis":
            if candidate.scale_axis != self.scale_axis or len(candidate.scales) != source.channel_count:
                return "candidate_scale_axis_or_extent_not_supported"
        if candidate.bias_domain not in self.bias_domains:
            return "candidate_bias_domain_not_supported"
        if candidate.bias and len(candidate.bias) not in (1, source.channel_count):
            return "candidate_bias_extent_not_supported"
        if candidate.output_zero_point not in self.output_zero_points:
            return "candidate_output_zero_point_not_supported"
        if candidate.rounding not in self.rounding:
            return "candidate_rounding_not_supported"
        if candidate.saturation != self.saturation:
            return "candidate_saturation_not_supported"
        if candidate.ordered_stages not in self.ordered_stage_templates:
            return "candidate_stage_order_not_supported"
        if candidate.relu != source.relu:
            return "candidate_activation_does_not_match_source"
        return None


@dataclass(frozen=True)
class AccuracyBudget:
    min_samples: int
    min_channel_coverage_fraction: float
    max_abs_error_lsb: int
    max_mean_abs_error_lsb: float
    max_mismatch_fraction: float
    max_saturation_mismatch_fraction: float

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "AccuracyBudget":
        result = cls(
            min_samples=int(row.get("min_samples", 0)),
            min_channel_coverage_fraction=float(row.get("min_channel_coverage_fraction", 0.0)),
            max_abs_error_lsb=int(row.get("max_abs_error_lsb", -1)),
            max_mean_abs_error_lsb=float(row.get("max_mean_abs_error_lsb", -1.0)),
            max_mismatch_fraction=float(row.get("max_mismatch_fraction", -1.0)),
            max_saturation_mismatch_fraction=float(
                row.get("max_saturation_mismatch_fraction", -1.0)),
        )
        if result.min_samples < 1 or result.max_abs_error_lsb < 0:
            raise ValueError("accuracy budget requires positive samples and nonnegative error")
        for value in (result.min_channel_coverage_fraction, result.max_mismatch_fraction,
                      result.max_saturation_mismatch_fraction):
            if not 0.0 <= value <= 1.0:
                raise ValueError("accuracy budget fractions must be in [0,1]")
        if result.max_mean_abs_error_lsb < 0.0:
            raise ValueError("mean absolute error budget must be nonnegative")
        return result

    def receipt(self) -> dict[str, Any]:
        return {
            "min_samples": self.min_samples,
            "min_channel_coverage_fraction": self.min_channel_coverage_fraction,
            "max_abs_error_lsb": self.max_abs_error_lsb,
            "max_mean_abs_error_lsb": self.max_mean_abs_error_lsb,
            "max_mismatch_fraction": self.max_mismatch_fraction,
            "max_saturation_mismatch_fraction": self.max_saturation_mismatch_fraction,
        }


@dataclass(frozen=True)
class AccuracyPolicy:
    """Explicit deployment policy. Merely providing evidence never enables approximation."""

    opt_in: bool
    mode: str
    budget: AccuracyBudget
    sites: tuple[Mapping[str, Any], ...]
    policy_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AccuracyPolicy":
        if value.get("schema") != "target_neutral_requant_accuracy_policy_v1":
            raise ValueError("unknown requantization accuracy policy schema")
        mode = str(value.get("mode", ""))
        if mode != "heldout_calibration_bounded":
            raise ValueError("accuracy policy mode must be heldout_calibration_bounded")
        budget = AccuracyBudget.from_mapping(value.get("budget") or {})
        sites = tuple(value.get("sites") or ())
        if not all(isinstance(site, Mapping) for site in sites):
            raise ValueError("accuracy policy sites must be objects")
        canonical = dict(value)
        canonical.pop("policy_sha256", None)
        return cls(bool(value.get("opt_in", False)), mode, budget, sites,
                   _canonical_sha(canonical))

    @classmethod
    def load(cls, path: str | Path) -> "AccuracyPolicy":
        return cls.from_mapping(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class RequantizationDecision:
    selected: bool
    exact: bool
    reason: str
    candidate: NativeNarrowCandidate | None
    receipt: Mapping[str, Any]


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _parameter(values: tuple[float, ...] | tuple[int, ...], granularity: str,
               channel: int) -> float | int:
    return values[0] if granularity == "per_tensor" or len(values) == 1 else values[channel]


def evaluate_candidate(candidate: NativeNarrowCandidate, accumulators: Sequence[int],
                       channels: Sequence[int]) -> list[int]:
    """Execute the declared native readout with explicit binary32/RNE boundaries."""
    if len(accumulators) != len(channels):
        raise ValueError("accumulator samples and channel indices differ in length")
    result = []
    lo, hi = candidate.saturation
    for accumulator, channel in zip(accumulators, channels):
        if not -(1 << 31) <= int(accumulator) < (1 << 31) or int(channel) < 0:
            raise ValueError("calibration sample exceeds i32 or has a negative channel")
        bias = (0 if not candidate.bias else int(_parameter(
            candidate.bias, "per_tensor" if len(candidate.bias) == 1 else "per_axis", channel)))
        scale = float(_parameter(candidate.scales, candidate.scale_granularity, channel))
        current: float | int = int(accumulator)
        for stage in candidate.ordered_stages:
            if stage == "bias_i32":
                current = int(current) + bias
            elif stage == "acc_scale":
                current = _f32(_f32(current) * _f32(scale))
            elif stage == "round_to_nearest_even":
                current = int(round(float(current)))
            elif stage == "zero_point":
                current = int(current) + candidate.output_zero_point
            elif stage == "saturate_i8":
                current = max(lo, min(hi, int(current)))
            elif stage == "relu":
                current = max(0, current)
            else:  # NativeNarrowCandidate.validate makes this unreachable.
                raise ValueError(f"unsupported native stage {stage!r}")
        result.append(int(current))
    return result


def _refusal(reason: str, *, details: Mapping[str, Any] | None = None) -> RequantizationDecision:
    return RequantizationDecision(False, False, reason, None, {
        "selection": "refused", "reason": reason, **dict(details or {})})


def evaluate_accuracy_gate(
    policy: AccuracyPolicy | None,
    *,
    normalized_source_sha256: str,
    producer_source_op_index: int,
    source_contract: QuantizedEpilogueContract,
    capability: NativeNarrowCapability,
) -> RequantizationDecision:
    """Admit one empirically bounded readout, or return a machine-readable refusal."""
    if policy is None:
        return _refusal("accuracy_policy_not_provided")
    if not policy.opt_in:
        return _refusal("accuracy_policy_not_opted_in", details={
            "policy_sha256": policy.policy_sha256})
    matches = [site for site in policy.sites
               if int(site.get("producer_source_op_index", -1)) == producer_source_op_index]
    if len(matches) != 1:
        return _refusal("accuracy_policy_site_missing_or_ambiguous", details={
            "policy_sha256": policy.policy_sha256})
    site = matches[0]
    contract_receipt = source_contract.receipt()
    bindings = {
        "normalized_source_sha256": normalized_source_sha256,
        "semantic_contract_sha256": contract_receipt["semantic_contract_sha256"],
    }
    if site.get("normalized_source_sha256") != normalized_source_sha256:
        return _refusal("accuracy_evidence_source_hash_mismatch", details=bindings)
    if site.get("semantic_contract_sha256") != bindings["semantic_contract_sha256"]:
        return _refusal("accuracy_evidence_semantic_contract_mismatch", details=bindings)
    evidence = site.get("evidence") or {}
    if evidence.get("split") != "holdout":
        return _refusal("accuracy_evidence_is_not_heldout", details=bindings)
    provenance = str(evidence.get("dataset_sha256", ""))
    reference_provenance = str(evidence.get("independent_reference_sha256", ""))
    if not _is_sha256(provenance) or not _is_sha256(reference_provenance):
        return _refusal("accuracy_evidence_provenance_missing", details=bindings)
    try:
        candidate = NativeNarrowCandidate.from_mapping(site.get("candidate") or {})
    except (TypeError, ValueError) as exc:
        return _refusal("accuracy_candidate_invalid", details={**bindings, "detail": str(exc)})
    unsupported = capability.refusal(candidate, source_contract)
    if unsupported:
        return _refusal(unsupported, details=bindings)
    try:
        accumulators = [int(value) for value in evidence.get("accumulators", ())]
        channels = [int(value) for value in evidence.get("channel_indices", ())]
        expected = [int(value) for value in evidence.get("reference_outputs", ())]
        if any(channel >= source_contract.channel_count for channel in channels):
            raise ValueError("calibration channel index exceeds the source contract")
        actual = evaluate_candidate(candidate, accumulators, channels)
    except (IndexError, TypeError, ValueError, OverflowError) as exc:
        return _refusal("accuracy_evidence_invalid", details={**bindings, "detail": str(exc)})
    if not expected or len(expected) != len(actual):
        return _refusal("accuracy_evidence_sample_count_mismatch", details=bindings)
    if any(value < source_contract.saturation[0] or value > source_contract.saturation[1]
           for value in expected):
        return _refusal("accuracy_reference_output_out_of_range", details=bindings)
    declared_reference_sha = str(evidence.get("reference_output_sha256", ""))
    if declared_reference_sha != _sha256(expected):
        return _refusal("accuracy_reference_output_hash_mismatch", details=bindings)

    count = len(expected)
    covered_channels = len(set(channels))
    errors = [abs(a - b) for a, b in zip(actual, expected)]
    mismatch_count = sum(error != 0 for error in errors)
    saturation_mismatches = sum(
        (a in candidate.saturation) != (b in source_contract.saturation)
        for a, b in zip(actual, expected))
    metrics = {
        "sample_count": count,
        "channel_count": source_contract.channel_count,
        "covered_channel_count": covered_channels,
        "channel_coverage_fraction": covered_channels / source_contract.channel_count,
        "max_abs_error_lsb": max(errors),
        "mean_abs_error_lsb": sum(errors) / count,
        "mismatch_count": mismatch_count,
        "mismatch_fraction": mismatch_count / count,
        "saturation_mismatch_count": saturation_mismatches,
        "saturation_mismatch_fraction": saturation_mismatches / count,
    }
    budget = policy.budget
    failures = []
    if count < budget.min_samples:
        failures.append("sample_count")
    if metrics["channel_coverage_fraction"] < budget.min_channel_coverage_fraction:
        failures.append("channel_coverage")
    if metrics["max_abs_error_lsb"] > budget.max_abs_error_lsb:
        failures.append("max_abs_error_lsb")
    if metrics["mean_abs_error_lsb"] > budget.max_mean_abs_error_lsb:
        failures.append("mean_abs_error_lsb")
    if metrics["mismatch_fraction"] > budget.max_mismatch_fraction:
        failures.append("mismatch_fraction")
    if metrics["saturation_mismatch_fraction"] > budget.max_saturation_mismatch_fraction:
        failures.append("saturation_mismatch_fraction")
    receipt = {
        "selection": "selected" if not failures else "refused",
        "reason": ("heldout_accuracy_budget_passed" if not failures
                   else "heldout_accuracy_budget_exceeded"),
        "proof_scope": "empirical_heldout_only",
        "exact_for_all_inputs": False,
        "policy_sha256": policy.policy_sha256,
        **bindings,
        "dataset_sha256": provenance,
        "independent_reference_sha256": reference_provenance,
        "reference_output_sha256": declared_reference_sha,
        "candidate_output_sha256": _sha256(actual),
        "budget": budget.receipt(),
        "metrics": metrics,
        "failed_budget_dimensions": failures,
        "candidate": candidate.receipt(),
        "source_contract": contract_receipt,
    }
    if failures:
        return RequantizationDecision(False, False, receipt["reason"], None, receipt)
    return RequantizationDecision(True, False, receipt["reason"], candidate, receipt)


def contract_from_formation(value: Any) -> QuantizedEpilogueContract:
    """Translate the structural frontend formation into the public semantic contract.

    The bindings are semantic roles, not SSA names, so the receipt is deterministic across parser
    allocation details.  The normalized source hash used by the gate binds their actual values.
    Integer contraction operands are already qvalues; their implicit zero points are recorded
    explicitly rather than pretending the epilogue owns an earlier QDQ operation.
    """
    axis = int(value.channel_axis)
    shape = tuple(int(item) for item in value.shape)
    if axis >= len(shape):
        raise ValueError("quantized epilogue channel axis exceeds result rank")
    scalar_f32 = lambda binding: QuantParameter("per_tensor", None, "f32", binding)
    scalar_i32 = lambda binding: QuantParameter("per_tensor", None, "i32", binding)
    channel_f32 = lambda binding: QuantParameter("per_axis", axis, "f32", binding)
    return QuantizedEpilogueContract(
        accumulator_dtype="i32",
        output_dtype="i8",
        channel_axis=axis,
        channel_count=shape[axis],
        activation_scale=scalar_f32("accumulator_scale"),
        weight_scale=channel_f32("per_channel_weight_scale"),
        output_scale_reciprocal=scalar_f32("output_scale_reciprocal"),
        input_zero_point=scalar_i32("implicit_quantized_input_zero_point"),
        weight_zero_point=QuantParameter(
            "per_axis", axis, "i32", "implicit_quantized_weight_zero_point"),
        output_zero_point=scalar_i32("output_zero_point"),
        bias_domain="real_f32",
        bias=channel_f32("per_channel_real_bias"),
        ordered_stages=tuple(value.stages),
        rounding="round_to_nearest_even",
        saturation=(-128, 127),
        relu=bool(value.relu),
    )
