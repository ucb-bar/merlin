"""Validate W8A8 numerical-candidate evidence without approving its stronger domain."""

from __future__ import annotations

import json
from typing import Any

MODULE = "MerlinVerify.Check.PolicyW8A8NumericalChain"
INPUT_DOMAIN = "candidate_finite_nonzero_f32_and_declared_integer_intervals_per_function"
PREFIX_MODULE = "MerlinVerify.Check.PolicyW8A8NumericalPrefix"
ROOTS = {
    f"{MODULE}.check_sound": MODULE,
    f"{PREFIX_MODULE}.entry_tracked_defined": PREFIX_MODULE,
    f"{PREFIX_MODULE}.entry_sound": PREFIX_MODULE,
    "MerlinVerify.Check.PolicyW8A8NumericalInhabited.check_sound": "MerlinVerify.Check.PolicyW8A8NumericalInhabited",
}
_CLASSES = {"any", "finite", "non_nan", "nonnegative", "finite_nonzero"}


def _object(value: Any, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} has missing or unexpected fields")


def _natural(value: Any) -> bool:
    return type(value) is int and value >= 0


def _ids(value: Any) -> bool:
    return isinstance(value, list) and all(_natural(item) for item in value)


def _prefix(value: Any, source_ops: list[dict[str, Any]] | None = None) -> None:
    if value is None:
        return
    _object(
        value,
        {"checked", "operation_count", "initialized_source_quantizers", "finite_source_dequantizers", "first_refusal"},
        "prefix",
    )
    if type(value["checked"]) is not bool or not _natural(value["operation_count"]):
        raise ValueError("prefix acceptance/count is malformed")
    for key in ("initialized_source_quantizers", "finite_source_dequantizers"):
        if not _ids(value[key]):
            raise ValueError("prefix source quantization IDs are malformed")
        if not value["checked"] and value[key]:
            raise ValueError("an unchecked prefix cannot certify quantization facts")
    count = value["operation_count"]
    if source_ops is not None:
        for key, name in (
            ("initialized_source_quantizers", "quant_ext.quantize_per_tensor"),
            ("finite_source_dequantizers", "quant_ext.dequantize_per_tensor"),
        ):
            eligible = {result["id"] for op in source_ops[:count] if op["name"] == name for result in op["results"]}
            if count > len(source_ops) or not set(value[key]) <= eligible:
                raise ValueError("prefix count or quantization facts do not belong to the actual source prefix")
    refusal = value["first_refusal"]
    if refusal is None:
        return
    _object(refusal, {"operation", "results", "operands", "input_facts"}, "prefix refusal")
    if not isinstance(refusal["operation"], str) or not _ids(refusal["results"]) or not _ids(refusal["operands"]):
        raise ValueError("prefix refusal operation is malformed")
    if not isinstance(refusal["input_facts"], list) or len(refusal["input_facts"]) != len(refusal["operands"]):
        raise ValueError("prefix refusal omitted operand facts")
    for fact in refusal["input_facts"]:
        if fact is None:
            continue
        _object(fact, {"class", "initialized", "literal_f32"}, "operand fact")
        literal = fact["literal_f32"]
        if (
            not isinstance(fact["class"], str)
            or fact["class"] not in _CLASSES
            or type(fact["initialized"]) is not bool
            or (literal is not None and not (_natural(literal) and literal < 2**32))
        ):
            raise ValueError("prefix refusal contains a malformed numerical fact")
    if source_ops is not None:
        if count >= len(source_ops):
            raise ValueError("prefix refusal is past the actual source")
        expected = source_ops[count]
        if (refusal["operation"], refusal["results"], refusal["operands"]) != (
            expected["name"],
            [result["id"] for result in expected["results"]],
            expected["operands"],
        ):
            raise ValueError("prefix refusal does not name the actual next operation")


def validate(
    evidence: Any,
    *,
    bounds: dict[str, list[dict[str, int] | None]],
    before: dict[str, Any],
    index_bits: int,
    scope: str,
    suffix_count: int,
) -> None:
    expected = {
        "schema": "merlin.w8a8.numerical-candidate.v1",
        "checker": f"{MODULE}.check",
        "theorem": f"{MODULE}.check_sound",
        "prefix_theorem": "MerlinVerify.Check.PolicyW8A8NumericalPrefix.entry_tracked_defined",
        "prefix_facts_theorem": "MerlinVerify.Check.PolicyW8A8NumericalPrefix.entry_sound",
        "nonvacuity_theorem": "MerlinVerify.Check.PolicyW8A8NumericalInhabited.check_sound",
        "verdict": "abstracted",
        "index_bits": index_bits,
        "policy_scope": scope,
        "input_domain": INPUT_DOMAIN,
        "floating_domain": "byte_extension_with_source_dynamic_laws_and_explicit_arithmetic_agreement",
        "source_primitive_laws": 7,
        "additional_dynamic_primitive_laws": 3,
        "arithmetic_agreement_assumed": True,
        "concrete_context_inhabited": False,
        "policy_reviewed": False,
        "primitive_model_laws_verified": False,
        "whole_pass_verified": False,
    }
    statuses = {"structural_match", "boundary_qualification_passed", "conditional_initialized_policy_refinement"}
    _object(evidence, set(expected) | statuses | {"functions", "suffix_boundaries"}, "numerical evidence")
    if any(type(evidence[key]) is not type(value) or evidence[key] != value for key, value in expected.items()):
        raise ValueError("numerical evidence changed its domain, theorem or unproved premises")
    if any(type(evidence[key]) is not bool for key in statuses):
        raise ValueError("numerical qualification statuses must be Boolean")
    source = {}
    for op in before["module"]["ops"]:
        if op["name"] == "func.func" and op["regions"] and op["regions"][0]["blocks"]:
            name = {**op["attrs"], **op.get("props", {})}["sym_name"]["value"]
            source[name] = op["regions"][0]["blocks"]
    details = evidence["functions"]
    if not isinstance(details, list) or len(details) != len(bounds):
        raise ValueError("numerical evidence omitted a source function")
    roster = []
    for function in details:
        _object(
            function,
            {
                "function",
                "input_intervals",
                "domain_inhabited",
                "source_initialized",
                "reference_initialized",
                "source_prefix",
                "reference_prefix",
            },
            "function evidence",
        )
        name = function["function"]
        if (
            not isinstance(name, str)
            or name not in bounds
            or json.dumps(function["input_intervals"], sort_keys=True) != json.dumps(bounds[name], sort_keys=True)
        ):
            raise ValueError("numerical evidence changed a function's input contract")
        if any(
            type(function[key]) is not bool
            for key in ("domain_inhabited", "source_initialized", "reference_initialized")
        ):
            raise ValueError("function numerical statuses must be Boolean")
        blocks = source.get(name, [])
        _prefix(function["source_prefix"], blocks[0]["ops"] if len(blocks) == 1 else None)
        _prefix(function["reference_prefix"])
        roster.append(name)
    if roster != list(source) or set(roster) != set(bounds):
        raise ValueError("numerical evidence changed the actual source function roster")
    boundary = evidence["boundary_qualification_passed"]
    if boundary and (
        not evidence["structural_match"]
        or not details
        or any(
            not all(function[key] for key in ("domain_inhabited", "source_initialized", "reference_initialized"))
            for function in details
        )
    ):
        raise ValueError("complete-boundary qualification omitted an obligation")
    links = evidence["suffix_boundaries"]
    if not isinstance(links, list) or len(links) != suffix_count:
        raise ValueError("numerical evidence omitted a suffix boundary")
    for index, link in enumerate(links):
        _object(link, {"index", "checked", "status"}, "suffix evidence")
        if type(link["index"]) is not int or link["index"] != index or type(link["checked"]) is not bool:
            raise ValueError("suffix index/status is malformed")
        expected_status = "unmeasured" if not boundary else "checked" if link["checked"] else "refused"
        if link["status"] != expected_status or (not boundary and link["checked"]):
            raise ValueError("suffix evidence changed its measured scope")
    if evidence["conditional_initialized_policy_refinement"] != (boundary and all(link["checked"] for link in links)):
        raise ValueError("numerical composition status contradicts its boundary obligations")
