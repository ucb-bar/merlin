"""Validate the explicit W8A8 numerical construction and its unchanged checker evidence."""

from __future__ import annotations

from typing import Any

from merlin.verify.lean import numerical_evidence, rotary_evidence, rotary_policy

PREFIX = "MerlinVerify.Check.PolicyW8A8IEEEPrefix.entry_bound"
ROOTS = {
    PREFIX: "MerlinVerify.Check.PolicyW8A8IEEEPrefix",
    "MerlinVerify.Check.PolicyW8A8IEEEQualified.check_sound": "MerlinVerify.Check.PolicyW8A8IEEEQualified",
    "MerlinVerify.Check.PolicyW8A8IEEEQualified.check_eq_numerical": "MerlinVerify.Check.PolicyW8A8IEEEQualified",
    "MerlinVerify.Check.PolicyW8A8IEEEQualified.domain_inhabited": "MerlinVerify.Check.PolicyW8A8IEEEQualified",
    "MerlinVerify.Check.PolicyW8A8IEEEContext.inhabited": "MerlinVerify.Check.PolicyW8A8IEEEContext",
    "MerlinVerify.Rules.W8A8IEEEPrimitives.agreement": "MerlinVerify.Rules.W8A8IEEEPrimitives",
    "MerlinVerify.Rules.W8A8IEEELaws.finite_laws": "MerlinVerify.Rules.W8A8IEEELaws",
    "MerlinVerify.Rules.W8A8IEEELaws.dynamic_laws": "MerlinVerify.Rules.W8A8IEEELaws",
}


def validate(
    value: Any,
    *,
    entries: dict[str, dict[str, Any]],
    bounds: dict[str, list[dict[str, int] | None]],
    before: dict[str, Any],
    generic: dict[str, Any],
    index_bits: int,
    scope: str,
) -> None:
    fixed = {
        "schema": "merlin.w8a8.ieee-context-candidate.v1",
        "checker": "MerlinVerify.Check.PolicyW8A8IEEEQualified.check",
        "theorem": "MerlinVerify.Check.PolicyW8A8IEEEQualified.check_sound",
        "bound_theorem": PREFIX,
        "joint_domain_theorem": "MerlinVerify.Check.PolicyW8A8IEEEQualified.domain_inhabited",
        "context_witness_theorem": "MerlinVerify.Check.PolicyW8A8IEEEContext.inhabited",
        "logical_model": "W8A8ByteModel.extend (W8A8IEEEPrimitives.model fallback)",
        "fallback_scope": (
            "arbitrary shared operations outside the constrained slice; witness uses NaN/poison sentinels"
        ),
        "rsqrt_recipe": "binary32 division of one by the binary32 rounded square root",
        "source_laws_proved": 7,
        "dynamic_laws_proved": 3,
        "arithmetic_agreement_fields_proved": 9,
        "additional_numerical_hypotheses": 0,
        "logical_context_inhabitation_proved": True,
        "native_primitive_fidelity_proved": False,
        "frontend_composite_identified": False,
        "whole_pass_verified": False,
        "policy_reviewed": False,
        "primitive_model_laws_verified": False,
        "concrete_context_inhabited": False,
        "index_bits": index_bits,
        "policy_scope": scope,
        "verdict": "abstracted",
    }
    fields = {"generic_rotary", "generic_numerical", "joint_input_domain_checked", "complete_boundary_checked"}
    if not isinstance(value, dict) or set(value) != set(fixed) | fields:
        raise ValueError("explicit-model evidence has an unexpected schema")
    if any(type(value[key]) is not type(expected) or value[key] != expected for key, expected in fixed.items()):
        raise ValueError("explicit-model evidence changed the construction, theorem or unproved scope")
    if any(type(value[key]) is not bool for key in ("joint_input_domain_checked", "complete_boundary_checked")):
        raise ValueError("explicit-model qualification results must be JSON booleans")
    rotary = value["generic_rotary"]
    rotary_evidence.validate(rotary, entries=entries, before=before, index_bits=index_bits, scope=scope)
    if not rotary_policy._same_json(rotary, generic):
        raise ValueError("explicit-model prefix evidence differs from its actual generic check")
    numerical = value["generic_numerical"]
    numerical_evidence.validate(
        numerical, bounds=bounds, before=before, index_bits=index_bits, scope=scope, suffix_count=0
    )
    inhabited = bool(rotary["functions"]) and all(function["domain_inhabited"] for function in rotary["functions"])
    if value["joint_input_domain_checked"] is not inhabited:
        raise ValueError("explicit-model joint domain does not match the actual input contracts")
    if value["complete_boundary_checked"] is not numerical["conditional_initialized_policy_refinement"]:
        raise ValueError("explicit and generic complete-boundary checkers disagree")
