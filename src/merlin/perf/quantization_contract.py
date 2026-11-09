"""Independent quantized-epilogue accuracy policy and admission.

The source contract, target capability, and deployment accuracy policy are deliberately separate.
A target adapter cannot weaken source semantics, and merely finding calibration data cannot enable
an approximation.  In the absence of a valid, explicitly opted-in policy the only selectable path
is one carrying a source/candidate-bound exact proof.

This module interprets a small semantic epilogue vocabulary for held-out checking.  It contains no
target identities, model identities, tensor shapes, instruction encodings, or target constants.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Transitional exports preserve existing callers and class identity. Numerical
# record consumers use the compiler owner; evaluator policy stays in this module.
from merlin.compiler.quantization.contract import (
    CONTRACT_SCHEMA,
    QuantizedEpilogueCandidate,
    QuantizedEpilogueCapability,
    QuantizedEpilogueContract,
    QuantParameter,
    canonical_sha256,
)

POLICY_SCHEMA = "target_neutral_quantized_accuracy_policy_v2"
_HEX = frozenset("0123456789abcdef")

__all__ = [
    "POLICY_SCHEMA",
    "CONTRACT_SCHEMA",
    "AccuracyBudget",
    "AccuracyCorpus",
    "AccuracyPolicy",
    "AccuracyPolicyParseResult",
    "AccuracySiteEvidence",
    "EpilogueAdmission",
    "ExactEpilogueProof",
    "QuantParameter",
    "QuantizedEpilogueCandidate",
    "QuantizedEpilogueCapability",
    "QuantizedEpilogueContract",
    "admit_epilogue",
    "canonical_sha256",
    "evaluate_candidate",
    "parse_accuracy_policy",
]


def _sequence_sha256(values: Sequence[int]) -> str:
    payload = json.dumps(list(values), separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value.lower())
    )


def _rows(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


@dataclass(frozen=True)
class AccuracyBudget:
    """Deployment limits measured in output quantization codes (LSBs)."""

    min_samples: int
    min_channel_coverage_fraction: float
    max_abs_error_lsb: int
    max_accumulated_error_lsb: int
    max_mean_abs_error_lsb: float
    max_mismatch_fraction: float
    max_saturation_mismatch_fraction: float

    def __post_init__(self) -> None:
        if (self.min_samples < 1 or self.max_abs_error_lsb < 0
                or self.max_accumulated_error_lsb < self.max_abs_error_lsb):
            raise ValueError(
                "accuracy budget requires positive samples and an accumulated bound no smaller "
                "than its nonnegative per-site bound"
            )
        if self.max_mean_abs_error_lsb < 0 or not math.isfinite(self.max_mean_abs_error_lsb):
            raise ValueError("mean absolute error budget must be nonnegative and finite")
        for value in (
            self.min_channel_coverage_fraction,
            self.max_mismatch_fraction,
            self.max_saturation_mismatch_fraction,
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("accuracy budget fractions must be finite and in [0,1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AccuracyBudget:
        return cls(
            min_samples=int(value.get("min_samples", 0)),
            min_channel_coverage_fraction=float(
                value.get("min_channel_coverage_fraction", -1)
            ),
            max_abs_error_lsb=int(value.get("max_abs_error_lsb", -1)),
            max_accumulated_error_lsb=int(value.get("max_accumulated_error_lsb", -1)),
            max_mean_abs_error_lsb=float(value.get("max_mean_abs_error_lsb", -1)),
            max_mismatch_fraction=float(value.get("max_mismatch_fraction", -1)),
            max_saturation_mismatch_fraction=float(
                value.get("max_saturation_mismatch_fraction", -1)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_samples": self.min_samples,
            "min_channel_coverage_fraction": self.min_channel_coverage_fraction,
            "max_abs_error_lsb": self.max_abs_error_lsb,
            "max_accumulated_error_lsb": self.max_accumulated_error_lsb,
            "max_mean_abs_error_lsb": self.max_mean_abs_error_lsb,
            "max_mismatch_fraction": self.max_mismatch_fraction,
            "max_saturation_mismatch_fraction": self.max_saturation_mismatch_fraction,
        }


@dataclass(frozen=True)
class AccuracyCorpus:
    """Content identities that make a claimed holdout corpus auditable."""

    corpus_id: str
    role: str
    dataset_sha256: str
    sampling_policy_sha256: str
    selection_corpus_sha256: str
    independent_reference_sha256: str

    def __post_init__(self) -> None:
        if not self.corpus_id.strip() or self.role != "holdout":
            raise ValueError("accuracy corpus requires a nonempty id and role='holdout'")
        hashes = (
            self.dataset_sha256,
            self.sampling_policy_sha256,
            self.selection_corpus_sha256,
            self.independent_reference_sha256,
        )
        if not all(_is_sha256(value) for value in hashes):
            raise ValueError("accuracy corpus identities must be SHA-256 values")
        if self.dataset_sha256 == self.selection_corpus_sha256:
            raise ValueError("holdout and candidate-selection corpora must have different identities")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AccuracyCorpus:
        return cls(
            corpus_id=str(value.get("id") or ""),
            role=str(value.get("role") or ""),
            dataset_sha256=str(value.get("dataset_sha256") or ""),
            sampling_policy_sha256=str(value.get("sampling_policy_sha256") or ""),
            selection_corpus_sha256=str(value.get("selection_corpus_sha256") or ""),
            independent_reference_sha256=str(value.get("independent_reference_sha256") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.corpus_id,
            "role": self.role,
            "dataset_sha256": self.dataset_sha256,
            "sampling_policy_sha256": self.sampling_policy_sha256,
            "selection_corpus_sha256": self.selection_corpus_sha256,
            "independent_reference_sha256": self.independent_reference_sha256,
        }


@dataclass(frozen=True)
class AccuracySiteEvidence:
    """Held-out samples and candidate bound to one exact source epilogue."""

    site_id: str
    normalized_source_sha256: str
    source_operation_index: int
    semantic_contract_sha256: str
    corpus_id: str
    candidate: QuantizedEpilogueCandidate
    accumulators: tuple[int, ...]
    channel_indices: tuple[int, ...]
    reference_outputs: tuple[int, ...]
    reference_output_sha256: str

    def __post_init__(self) -> None:
        if not self.site_id.strip() or not self.corpus_id.strip():
            raise ValueError("accuracy evidence requires site and corpus ids")
        if self.source_operation_index < 0:
            raise ValueError("source operation index must be nonnegative")
        if not _is_sha256(self.normalized_source_sha256) or not _is_sha256(
            self.semantic_contract_sha256
        ):
            raise ValueError("accuracy evidence requires source and semantic SHA-256 identities")
        if not _is_sha256(self.reference_output_sha256):
            raise ValueError("accuracy evidence requires a reference-output SHA-256")
        if not self.accumulators or not (
            len(self.accumulators) == len(self.channel_indices) == len(self.reference_outputs)
        ):
            raise ValueError("accuracy samples must be nonempty equal-length vectors")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AccuracySiteEvidence:
        samples = value.get("samples") if isinstance(value.get("samples"), Mapping) else {}
        candidate = value.get("candidate")
        if not isinstance(candidate, Mapping):
            raise ValueError("accuracy site candidate must be an object")
        return cls(
            site_id=str(value.get("site_id") or ""),
            normalized_source_sha256=str(value.get("normalized_source_sha256") or ""),
            source_operation_index=int(value.get("source_operation_index", -1)),
            semantic_contract_sha256=str(value.get("semantic_contract_sha256") or ""),
            corpus_id=str(value.get("corpus_id") or ""),
            candidate=QuantizedEpilogueCandidate.from_mapping(candidate),
            accumulators=tuple(int(item) for item in _rows(samples.get("accumulators"))),
            channel_indices=tuple(int(item) for item in _rows(samples.get("channel_indices"))),
            reference_outputs=tuple(int(item) for item in _rows(samples.get("reference_outputs"))),
            reference_output_sha256=str(samples.get("reference_output_sha256") or ""),
        )

    def binding_dict(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "normalized_source_sha256": self.normalized_source_sha256,
            "source_operation_index": self.source_operation_index,
            "semantic_contract_sha256": self.semantic_contract_sha256,
            "corpus_id": self.corpus_id,
        }


@dataclass(frozen=True)
class AccuracyPolicy:
    """Explicit host/deployment authorization for bounded approximation."""

    opt_in: bool
    mode: str
    budget: AccuracyBudget
    corpora: tuple[AccuracyCorpus, ...]
    sites: tuple[AccuracySiteEvidence, ...]
    policy_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AccuracyPolicy:
        if value.get("schema") != POLICY_SCHEMA:
            raise ValueError(f"accuracy policy schema must be {POLICY_SCHEMA}")
        if value.get("mode") != "heldout_calibration_bounded":
            raise ValueError("accuracy policy mode must be heldout_calibration_bounded")
        if not isinstance(value.get("opt_in"), bool):
            raise ValueError("accuracy policy opt_in must be an explicit boolean")
        budget = AccuracyBudget.from_mapping(
            value.get("budget") if isinstance(value.get("budget"), Mapping) else {}
        )
        corpora = tuple(
            AccuracyCorpus.from_mapping(row)
            for row in _rows(value.get("corpora"))
            if isinstance(row, Mapping)
        )
        sites = tuple(
            AccuracySiteEvidence.from_mapping(row)
            for row in _rows(value.get("sites"))
            if isinstance(row, Mapping)
        )
        raw_corpora = _rows(value.get("corpora"))
        raw_sites = _rows(value.get("sites"))
        if len(corpora) != len(raw_corpora) or len(sites) != len(raw_sites):
            raise ValueError("accuracy policy corpora and sites must be objects")
        corpus_ids = [corpus.corpus_id for corpus in corpora]
        site_ids = [site.site_id for site in sites]
        if not corpora or len(corpus_ids) != len(set(corpus_ids)):
            raise ValueError("accuracy policy requires unique nonempty corpora")
        if len(site_ids) != len(set(site_ids)):
            raise ValueError("accuracy policy site ids must be unique")
        known_corpora = set(corpus_ids)
        if any(site.corpus_id not in known_corpora for site in sites):
            raise ValueError("accuracy site names an unknown corpus")
        canonical = dict(value)
        canonical.pop("policy_sha256", None)
        return cls(
            opt_in=value["opt_in"],
            mode="heldout_calibration_bounded",
            budget=budget,
            corpora=corpora,
            sites=sites,
            policy_sha256=canonical_sha256(canonical),
        )

    @classmethod
    def load(cls, path: str | Path) -> AccuracyPolicy:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("accuracy policy document must be an object")
        return cls.from_mapping(value)

    def corpus(self, corpus_id: str) -> AccuracyCorpus | None:
        return next((corpus for corpus in self.corpora if corpus.corpus_id == corpus_id), None)

    def site(self, site_id: str) -> AccuracySiteEvidence | None:
        return next((site for site in self.sites if site.site_id == site_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": POLICY_SCHEMA,
            "policy_sha256": self.policy_sha256,
            "opt_in": self.opt_in,
            "mode": self.mode,
            "budget": self.budget.to_dict(),
            "corpora": [corpus.to_dict() for corpus in self.corpora],
            "site_count": len(self.sites),
            "scope": (
                "hash-distinct heldout/selection corpora; semantic sample disjointness must be "
                "established by the sampling-policy artifact"
            ),
        }


@dataclass(frozen=True)
class AccuracyPolicyParseResult:
    policy: AccuracyPolicy | None
    receipt: Mapping[str, Any]


def parse_accuracy_policy(value: Mapping[str, Any] | None) -> AccuracyPolicyParseResult:
    """Parse without turning missing/malformed deployment data into an exception-only outcome."""
    if value is None:
        return AccuracyPolicyParseResult(None, {
            "status": "refused", "reason_codes": ["accuracy_policy_not_provided"]
        })
    try:
        policy = AccuracyPolicy.from_mapping(value)
    except (TypeError, ValueError, OverflowError) as exc:
        return AccuracyPolicyParseResult(None, {
            "status": "refused",
            "reason_codes": ["accuracy_policy_invalid"],
            "detail": str(exc),
        })
    return AccuracyPolicyParseResult(policy, {
        "status": "available",
        "policy_sha256": policy.policy_sha256,
        "opt_in": policy.opt_in,
    })


@dataclass(frozen=True)
class ExactEpilogueProof:
    """Host-verified exact proof, bound to source semantics and one candidate."""

    status: str
    method: str
    normalized_source_sha256: str
    source_operation_index: int
    semantic_contract_sha256: str
    candidate_sha256: str
    evidence_sha256: str

    def refusal_reasons(
        self,
        *,
        normalized_source_sha256: str,
        source_operation_index: int,
        semantic_contract_sha256: str,
        candidate_sha256: str,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.status != "verified":
            reasons.append("exact_proof_not_verified")
        if not self.method.strip() or not _is_sha256(self.evidence_sha256):
            reasons.append("exact_proof_provenance_missing")
        if self.normalized_source_sha256 != normalized_source_sha256:
            reasons.append("exact_proof_source_hash_mismatch")
        if self.source_operation_index != source_operation_index:
            reasons.append("exact_proof_source_operation_mismatch")
        if self.semantic_contract_sha256 != semantic_contract_sha256:
            reasons.append("exact_proof_semantic_contract_mismatch")
        if self.candidate_sha256 != candidate_sha256:
            reasons.append("exact_proof_candidate_mismatch")
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "method": self.method,
            "normalized_source_sha256": self.normalized_source_sha256,
            "source_operation_index": self.source_operation_index,
            "semantic_contract_sha256": self.semantic_contract_sha256,
            "candidate_sha256": self.candidate_sha256,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True)
class EpilogueAdmission:
    selected: bool
    exact: bool
    reason: str
    candidate: QuantizedEpilogueCandidate | None
    max_observed_error_lsb: int | None
    proof_scope: str
    receipt: Mapping[str, Any]


def _f32(value: float | int) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _parameter(values: Sequence[float] | Sequence[int], granularity: str, channel: int) -> float | int:
    return values[0] if granularity == "per_tensor" or len(values) == 1 else values[channel]


def evaluate_candidate(
    candidate: QuantizedEpilogueCandidate,
    accumulators: Sequence[int],
    channels: Sequence[int],
) -> list[int]:
    """Execute declared candidate semantics with explicit binary32 and RNE boundaries."""
    if len(accumulators) != len(channels):
        raise ValueError("accumulator samples and channel indices differ in length")
    result: list[int] = []
    low, high = candidate.saturation
    for accumulator, channel in zip(accumulators, channels, strict=True):
        accumulator = int(accumulator)
        channel = int(channel)
        if not -(1 << 31) <= accumulator < (1 << 31) or channel < 0:
            raise ValueError("calibration sample exceeds i32 or has a negative channel")
        bias = 0 if not candidate.bias else int(
            _parameter(candidate.bias, "per_tensor" if len(candidate.bias) == 1 else "per_axis", channel)
        )
        scale = float(_parameter(candidate.scales, candidate.scale_granularity, channel))
        current: int | float = accumulator
        for stage in candidate.ordered_stages:
            if stage == "bias_i32":
                current = int(current) + bias
            elif stage == "scale_f32":
                current = _f32(_f32(current) * _f32(scale))
            elif stage == "round_to_nearest_even":
                current = int(round(float(current)))
            elif stage == "output_zero_point":
                current = int(current) + candidate.output_zero_point
            elif stage == "clamp":
                current = max(low, min(high, int(current)))
            elif stage == "relu":
                current = max(0, int(current))
            else:  # Dataclass validation makes this unreachable.
                raise ValueError(f"unsupported candidate stage {stage!r}")
        result.append(int(current))
    return result


def _refusal(reason_codes: Sequence[str], **details: Any) -> EpilogueAdmission:
    reasons = tuple(dict.fromkeys(str(reason) for reason in reason_codes if reason))
    primary = reasons[0] if reasons else "unspecified_refusal"
    return EpilogueAdmission(False, False, primary, None, None, "none", {
        "status": "refused", "reason": primary, "reason_codes": list(reasons), **details
    })


def _accuracy_admission(
    policy: AccuracyPolicy | None,
    *,
    site_id: str,
    normalized_source_sha256: str,
    source_operation_index: int,
    source_contract: QuantizedEpilogueContract,
    capability: QuantizedEpilogueCapability,
) -> EpilogueAdmission:
    if policy is None:
        return _refusal(("accuracy_policy_not_provided",))
    if not policy.opt_in:
        return _refusal(
            ("accuracy_policy_not_opted_in",), policy_sha256=policy.policy_sha256
        )
    evidence = policy.site(site_id)
    if evidence is None:
        return _refusal(
            ("accuracy_policy_site_missing",), policy_sha256=policy.policy_sha256
        )
    contract = source_contract.to_dict()
    semantic_sha = contract["semantic_contract_sha256"]
    binding_reasons: list[str] = []
    if evidence.normalized_source_sha256 != normalized_source_sha256:
        binding_reasons.append("accuracy_evidence_source_hash_mismatch")
    if evidence.source_operation_index != source_operation_index:
        binding_reasons.append("accuracy_evidence_source_operation_mismatch")
    if evidence.semantic_contract_sha256 != semantic_sha:
        binding_reasons.append("accuracy_evidence_semantic_contract_mismatch")
    corpus = policy.corpus(evidence.corpus_id)
    if corpus is None:
        binding_reasons.append("accuracy_evidence_corpus_missing")
    if binding_reasons:
        return _refusal(
            binding_reasons,
            policy_sha256=policy.policy_sha256,
            expected_bindings={
                "site_id": site_id,
                "normalized_source_sha256": normalized_source_sha256,
                "source_operation_index": source_operation_index,
                "semantic_contract_sha256": semantic_sha,
            },
        )
    candidate = evidence.candidate
    capability_reasons = capability.refusal_reasons(candidate, source_contract)
    if capability_reasons:
        return _refusal(
            capability_reasons,
            policy_sha256=policy.policy_sha256,
            capability=capability.to_dict(),
        )
    try:
        if any(channel >= source_contract.channel_count for channel in evidence.channel_indices):
            raise ValueError("sample channel exceeds source channel count")
        actual = evaluate_candidate(candidate, evidence.accumulators, evidence.channel_indices)
    except (IndexError, TypeError, ValueError, OverflowError) as exc:
        return _refusal(("accuracy_evidence_invalid",), detail=str(exc))
    if evidence.reference_output_sha256 != _sequence_sha256(evidence.reference_outputs):
        return _refusal(("accuracy_reference_output_hash_mismatch",))
    low, high = source_contract.saturation
    if any(value < low or value > high for value in evidence.reference_outputs):
        return _refusal(("accuracy_reference_output_out_of_range",))

    errors = [
        abs(actual_value - reference_value)
        for actual_value, reference_value in zip(actual, evidence.reference_outputs, strict=True)
    ]
    count = len(errors)
    covered_channels = len(set(evidence.channel_indices))
    mismatch_count = sum(error != 0 for error in errors)
    candidate_low, candidate_high = candidate.saturation
    saturation_mismatch_count = sum(
        (actual_value in (candidate_low, candidate_high))
        != (reference_value in (low, high))
        for actual_value, reference_value in zip(actual, evidence.reference_outputs, strict=True)
    )
    metrics = {
        "sample_count": count,
        "channel_count": source_contract.channel_count,
        "covered_channel_count": covered_channels,
        "channel_coverage_fraction": covered_channels / source_contract.channel_count,
        "max_abs_error_lsb": max(errors),
        "mean_abs_error_lsb": sum(errors) / count,
        "mismatch_count": mismatch_count,
        "mismatch_fraction": mismatch_count / count,
        "saturation_mismatch_count": saturation_mismatch_count,
        "saturation_mismatch_fraction": saturation_mismatch_count / count,
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
        "status": "refused" if failures else "selected",
        "reason": "heldout_accuracy_budget_exceeded" if failures else "heldout_accuracy_budget_passed",
        "reason_codes": ["heldout_accuracy_budget_exceeded"] if failures else [],
        "proof_scope": "empirical_heldout_only",
        "exact_for_all_inputs": False,
        "policy_sha256": policy.policy_sha256,
        "corpus": corpus.to_dict() if corpus is not None else None,
        "bindings": evidence.binding_dict(),
        "candidate": candidate.to_dict(),
        "candidate_sha256": candidate.sha256,
        "candidate_output_sha256": _sequence_sha256(actual),
        "reference_output_sha256": evidence.reference_output_sha256,
        "source_contract": contract,
        "capability": capability.to_dict(),
        "budget": budget.to_dict(),
        "metrics": metrics,
        "failed_budget_dimensions": failures,
        "claim_boundary": (
            "heldout local output-code evidence; not all-input equivalence or end-to-end accuracy"
        ),
    }
    if failures:
        return EpilogueAdmission(
            False, False, receipt["reason"], None, None, "empirical_heldout_only", receipt
        )
    return EpilogueAdmission(
        True,
        False,
        receipt["reason"],
        candidate,
        int(metrics["max_abs_error_lsb"]),
        "empirical_heldout_only",
        receipt,
    )


def admit_epilogue(
    *,
    site_id: str,
    normalized_source_sha256: str,
    source_operation_index: int,
    source_contract: QuantizedEpilogueContract,
    capability: QuantizedEpilogueCapability,
    policy: AccuracyPolicy | None = None,
    exact_candidate: QuantizedEpilogueCandidate | None = None,
    exact_proof: ExactEpilogueProof | None = None,
) -> EpilogueAdmission:
    """Prefer bound exact proof, then try explicit held-out authorization, else fail closed."""
    if not site_id.strip() or source_operation_index < 0 or not _is_sha256(
        normalized_source_sha256
    ):
        raise ValueError("epilogue site requires a stable id, source index, and source SHA-256")
    exact_reasons: list[str] = []
    contract_receipt = source_contract.to_dict()
    semantic_sha = contract_receipt["semantic_contract_sha256"]
    if exact_candidate is None:
        exact_reasons.append("exact_candidate_not_provided")
    else:
        exact_reasons.extend(capability.refusal_reasons(exact_candidate, source_contract))
        if exact_proof is None:
            exact_reasons.append("exact_proof_not_provided")
        else:
            exact_reasons.extend(exact_proof.refusal_reasons(
                normalized_source_sha256=normalized_source_sha256,
                source_operation_index=source_operation_index,
                semantic_contract_sha256=semantic_sha,
                candidate_sha256=exact_candidate.sha256,
            ))
    if not exact_reasons and exact_candidate is not None and exact_proof is not None:
        receipt = {
            "status": "selected",
            "reason": "source_exact_proof_verified",
            "reason_codes": [],
            "proof_scope": "exact_all_inputs_as_verified_by_bound_proof",
            "exact_for_all_inputs": True,
            "source_contract": contract_receipt,
            "candidate": exact_candidate.to_dict(),
            "candidate_sha256": exact_candidate.sha256,
            "capability": capability.to_dict(),
            "exact_proof": exact_proof.to_dict(),
            "accuracy_policy_consulted": False,
        }
        return EpilogueAdmission(
            True, True, receipt["reason"], exact_candidate, 0, receipt["proof_scope"], receipt
        )

    accuracy = _accuracy_admission(
        policy,
        site_id=site_id,
        normalized_source_sha256=normalized_source_sha256,
        source_operation_index=source_operation_index,
        source_contract=source_contract,
        capability=capability,
    )
    receipt = dict(accuracy.receipt)
    receipt["exact_gate"] = {
        "status": "refused",
        "reason_codes": list(dict.fromkeys(exact_reasons)),
        "proof": None if exact_proof is None else exact_proof.to_dict(),
    }
    if not accuracy.selected:
        all_reasons = list(dict.fromkeys([*exact_reasons, *receipt.get("reason_codes", ())]))
        receipt["reason_codes"] = all_reasons
        receipt["reason"] = all_reasons[0]
        return EpilogueAdmission(
            False, False, receipt["reason"], None, None, accuracy.proof_scope, receipt
        )
    return EpilogueAdmission(
        accuracy.selected,
        accuracy.exact,
        accuracy.reason,
        accuracy.candidate,
        accuracy.max_observed_error_lsb,
        accuracy.proof_scope,
        receipt,
    )
