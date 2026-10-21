"""Validate conditional W8A8 diagnostics without promoting their explicit numerical premises."""

from __future__ import annotations

from typing import Any

QUALIFIED_MODULE = "MerlinVerify.Check.PolicyW8A8DynamicQualified"
CHAIN_MODULE = "MerlinVerify.Check.PolicyW8A8DynamicChain"


def validate(
    result: dict[str, Any], *, chain: bool, scope: str, index_bits: int, suffix_count: int
) -> tuple[dict[str, Any], str]:
    """Check the report identity, domain and coverage before binding its theorem's actual axioms."""
    evidence = result.get("conditional_numerical")
    if not isinstance(evidence, dict):
        raise ValueError("missing conditional W8A8 numerical evidence")
    module = CHAIN_MODULE if chain else QUALIFIED_MODULE
    expected = {
        "checker": f"{module}.check",
        "theorem": f"{module}.check_sound",
        "index_bits": index_bits,
        "policy_scope": scope,
        "input_domain": "typed_initialized_inputs_with_finite_f32_values",
        "floating_domain": "byte_extension_with_explicit_source_and_dynamic_primitive_laws",
        "source_primitive_laws": 7,
        "additional_dynamic_primitive_laws": 3,
    }
    if any(type(evidence.get(key)) is not type(value) or evidence[key] != value for key, value in expected.items()):
        raise ValueError("conditional W8A8 checker returned a different theorem or numerical domain")
    for key in ("policy_reviewed", "primitive_model_laws_verified", "whole_pass_verified"):
        if evidence.get(key) is not False:
            raise ValueError("conditional W8A8 checker attempted to approve unproved premises")
    for key in ("boundary_qualification_passed", "conditional_initialized_policy_refinement", "suffix_checked"):
        if type(evidence.get(key)) is not bool:
            raise ValueError("conditional W8A8 qualification status must be Boolean")
    functions = evidence.get("functions")
    if not isinstance(functions, list) or any(not isinstance(fn, dict) for fn in functions):
        raise ValueError("conditional W8A8 function obligations are missing")
    if [fn.get("function") for fn in functions] != result["functions"]:
        raise ValueError("conditional W8A8 evidence changed the selected function roster")
    for fn in functions:
        if any(type(fn.get(key)) is not bool for key in ("source_initialized", "reference_initialized")):
            raise ValueError("conditional W8A8 initialization status must be Boolean")
    boundary = evidence["boundary_qualification_passed"]
    if boundary and (
        not functions or any(not (fn["source_initialized"] and fn["reference_initialized"]) for fn in functions)
    ):
        raise ValueError("conditional W8A8 qualification omitted an initialization obligation")
    _suffix(evidence, boundary=boundary, count=suffix_count)
    return evidence, module


def _suffix(evidence: dict[str, Any], *, boundary: bool, count: int) -> None:
    links = evidence.get("suffix_boundaries")
    if not isinstance(links, list) or len(links) != count:
        raise ValueError("conditional W8A8 evidence did not cover every downstream boundary")
    for index, link in enumerate(links):
        if not isinstance(link, dict) or type(link.get("index")) is not int or link["index"] != index:
            raise ValueError("conditional W8A8 suffix boundary order changed")
        matched = link.get("conditional_match")
        if boundary:
            if type(matched) is not bool or link.get("status") != ("checked" if matched else "refused"):
                raise ValueError("conditional W8A8 suffix result is inconsistent")
        elif matched is not None or link.get("status") != "unmeasured":
            raise ValueError("unqualified conditional W8A8 suffix was reported as checked")
    expected_match = all(link["conditional_match"] for link in links) if boundary else None
    actual_match = evidence.get("suffix_match")
    if (
        evidence["suffix_checked"] is not boundary
        or type(actual_match) is not type(expected_match)
        or actual_match != expected_match
    ):
        raise ValueError("conditional W8A8 suffix summary does not match its boundaries")
    if evidence["conditional_initialized_policy_refinement"] is not (boundary and expected_match is True):
        raise ValueError("conditional W8A8 result does not match its checked obligations")
