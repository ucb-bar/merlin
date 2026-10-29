"""Validate explicit-model contraction evidence with separate source/reference demands."""

from __future__ import annotations

from typing import Any

from merlin.verify.lean import rotary_evidence, rotary_policy

PREFIX = "MerlinVerify.Check.PolicyW8A8OrderedPrefix.entry_bound"
ROOTS = {
    PREFIX: "MerlinVerify.Check.PolicyW8A8OrderedPrefix",
    "MerlinVerify.Check.PolicyW8A8OrderedRule.rule_sound": "MerlinVerify.Check.PolicyW8A8OrderedRule",
    "MerlinVerify.Check.PolicyW8A8OrderedOp.execute": "MerlinVerify.Check.PolicyW8A8OrderedOp",
    "MerlinVerify.Check.PolicyW8A8OrderedRefined.generic_refined": "MerlinVerify.Check.PolicyW8A8OrderedRefined",
    "MerlinVerify.Check.Definedness.Contextual.inferPrefix_sound": "MerlinVerify.Check.DefinedContextPrefix",
    "MerlinVerify.Rules.W8A8OrderedTensor.projected_bounded": "MerlinVerify.Rules.W8A8OrderedTensor",
    "MerlinVerify.Rules.W8A8OrderedDot.evaluate_bounded": "MerlinVerify.Rules.W8A8OrderedDot",
    "MerlinVerify.Check.PolicyW8A8IEEEQualified.domain_inhabited": "MerlinVerify.Check.PolicyW8A8IEEEQualified",
    "MerlinVerify.Check.PolicyW8A8IEEEQualified.check_sound": "MerlinVerify.Check.PolicyW8A8IEEEQualified",
}

SIDES = {"source_prefix": "source_required_bounds", "reference_prefix": "reference_required_bounds"}


def contracts(value: Any, roster: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Bind input caps to actual arguments and preserve each program's requested SSA bounds."""
    if not isinstance(value, list):
        raise ValueError("ordered exponent contracts must be an array")
    entries = {}
    for entry in value:
        rotary_evidence._object(entry, {"function", "exponent_caps", *SIDES.values()}, "ordered contract")
        name = entry["function"]
        if not isinstance(name, str) or name in entries or name not in roster:
            raise ValueError("ordered contracts must name each source function exactly once")
        caps = entry["exponent_caps"]
        if not isinstance(caps, list) or len(caps) != len(roster[name]):
            raise ValueError("ordered input caps do not match the actual arguments")
        for cap, argument in zip(caps, roster[name], strict=True):
            if cap is None:
                continue
            ty = argument["type"]
            element = ty["element"] if ty["kind"] == "tensor" else ty
            if not rotary_evidence._natural(cap) or cap > 254 or element != {"kind": "float", "format": "f32"}:
                raise ValueError("ordered input caps require f32 arguments and natural exponents in [0,254]")
        for side in SIDES.values():
            required = entry[side]
            if not isinstance(required, list):
                raise ValueError("ordered numerical demands must be arrays")
            seen = set()
            for demand in required:
                rotary_evidence._object(demand, {"value", "upper"}, "ordered numerical demand")
                if (
                    not rotary_evidence._natural(demand["value"])
                    or demand["value"] in seen
                    or not rotary_evidence._natural(demand["upper"])
                    or demand["upper"] > 254
                ):
                    raise ValueError("ordered demands require unique SSA IDs and natural exponents in [0,254]")
                seen.add(demand["value"])
        entries[name] = entry
    if set(entries) != set(roster) or not any(entry[side] for entry in entries.values() for side in SIDES.values()):
        raise ValueError("ordered contracts must cover every function and request at least one bound")
    return entries


def validate(value: Any, *, entries: dict[str, dict[str, Any]], before: dict[str, Any], index_bits: int, scope: str):
    """Reject changed theorem labels, weak JSON types and demands moved between programs."""
    fixed = {
        "schema": "merlin.w8a8.ordered-contractions.v1",
        "verdict": "abstracted",
        "index_bits": index_bits,
        "policy_scope": scope,
        "input_domain": "candidate_finite_nonzero_f32_with_per_argument_exponent_caps",
        "logical_model": "W8A8ByteModel.extend (W8A8IEEEPrimitives.model fallback)",
        "prefix_theorem": "MerlinVerify.Check.PolicyW8A8OrderedPrefix.entry_sound",
        "bound_theorem": PREFIX,
        "rule_theorem": "MerlinVerify.Check.PolicyW8A8OrderedRule.rule_sound",
        "nonvacuity_theorem": "MerlinVerify.Check.PolicyW8A8IEEEQualified.domain_inhabited",
        "additional_numerical_hypotheses": 0,
        "transcendental_bounds_assumed": False,
        "native_primitive_fidelity_proved": False,
        "whole_pass_verified": False,
        "policy_reviewed": False,
        "primitive_model_laws_verified": False,
        "concrete_context_inhabited": False,
    }
    rotary_evidence._object(
        value, {*fixed, "structural_match", "complete_boundary_checked", "functions"}, "ordered evidence"
    )
    if any(type(value[key]) is not type(expected) or value[key] != expected for key, expected in fixed.items()):
        raise ValueError("ordered evidence changed its theorem, model or unresolved scope")
    if any(type(value[key]) is not bool for key in ("structural_match", "complete_boundary_checked")):
        raise ValueError("ordered checker results must be JSON booleans")
    if not isinstance(value["functions"], list):
        raise ValueError("ordered function evidence must be an array")
    functions = {
        op["attrs"]["sym_name"]["value"]: op
        for op in before["module"]["ops"]
        if op["name"] == "func.func" and op["regions"] and op["regions"][0]["blocks"]
    }
    seen = set()
    for item in value["functions"]:
        rotary_evidence._object(item, {"function", "domain_inhabited", *SIDES}, "ordered function")
        name = item["function"]
        if not isinstance(name, str) or name not in entries or name in seen:
            raise ValueError("ordered evidence changed the function roster")
        seen.add(name)
        if type(item["domain_inhabited"]) is not bool:
            raise ValueError("ordered input-domain result must be a JSON boolean")
        block = functions[name]["regions"][0]["blocks"][0]
        for prefix, demands in SIDES.items():
            rotary_evidence._prefix(item[prefix], entries[name][demands], block if prefix == "source_prefix" else None)
            if not item["domain_inhabited"] and item[prefix] is not None and item[prefix]["checked"]:
                raise ValueError("a certified ordered prefix requires an inhabited input domain")
    if seen != set(entries):
        raise ValueError("ordered evidence omitted a source function")


def requested_passed(value: dict[str, Any]) -> bool:
    return (
        value["structural_match"]
        and bool(value["functions"])
        and all(
            function["domain_inhabited"]
            and all(function[side] is not None and function[side]["all_required_bounds_checked"] for side in SIDES)
            for function in value["functions"]
        )
    )


def same(left: Any, right: Any) -> bool:
    """Keep the shared strict JSON comparison at every replay boundary."""
    return rotary_policy._same_json(left, right)
