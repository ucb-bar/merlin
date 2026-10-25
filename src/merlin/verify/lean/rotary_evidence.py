"""Validate scoped, conditional W8A8 prefix-bound evidence against actual exports."""

from __future__ import annotations

from typing import Any

ROOTS = {
    "MerlinVerify.Check.PolicyW8A8RotaryPrefix.entry_sound": "MerlinVerify.Check.PolicyW8A8RotaryPrefix",
    "MerlinVerify.Check.PolicyW8A8RotaryPrefix.entry_bound": "MerlinVerify.Check.PolicyW8A8RotaryPrefix",
    "MerlinVerify.Check.PolicyW8A8RotaryDomain.check_inhabited": "MerlinVerify.Check.PolicyW8A8RotaryDomain",
    "MerlinVerify.Rules.W8A8IEEEBounds.mul_bounded": "MerlinVerify.Rules.W8A8IEEEBounds",
    "MerlinVerify.Rules.W8A8ByteBounds.check_sound": "MerlinVerify.Rules.W8A8ByteBounds",
}


def _object(value: Any, keys: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} has missing or unexpected fields")


def _natural(value: Any) -> bool:
    return type(value) is int and value >= 0


def contracts(value: Any, roster: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Reject weakly typed JSON and bind each bound to an actual f32 argument."""
    if not isinstance(value, list):
        raise ValueError("exponent contracts must be an array")
    result = {}
    for entry in value:
        _object(entry, {"function", "exponent_caps", "required_bounds"}, "exponent contract")
        name = entry["function"]
        if not isinstance(name, str) or name in result or name not in roster:
            raise ValueError("exponent contracts must name each source function exactly once")
        caps, required = entry["exponent_caps"], entry["required_bounds"]
        if not isinstance(caps, list) or len(caps) != len(roster[name]):
            raise ValueError("exponent caps must match the actual argument count")
        for cap, argument in zip(caps, roster[name], strict=True):
            if cap is None:
                continue
            ty = argument["type"]
            elt = ty["element"] if ty["kind"] == "tensor" else ty
            if not _natural(cap) or cap > 254 or elt != {"kind": "float", "format": "f32"}:
                raise ValueError("exponent caps require f32 arguments and an integer in [0,254]")
        if not isinstance(required, list):
            raise ValueError("required bounds must be an array")
        ids = set()
        for demand in required:
            _object(demand, {"value", "upper"}, "required bound")
            if (
                not _natural(demand["value"])
                or demand["value"] in ids
                or not _natural(demand["upper"])
                or demand["upper"] > 254
            ):
                raise ValueError("required bounds need unique SSA IDs and exponent caps in [0,254]")
            ids.add(demand["value"])
        result[name] = entry
    if set(result) != set(roster) or not any(entry["required_bounds"] for entry in result.values()):
        raise ValueError("contracts must cover every function and request at least one numerical bound")
    return result


def _prefix(value: Any, required: list[dict[str, int]], block: dict[str, Any] | None) -> None:
    if value is None:
        return
    _object(
        value,
        {
            "checked",
            "operation_count",
            "tracked_facts",
            "first_refusal",
            "required_bounds",
            "all_required_bounds_checked",
        },
        "prefix",
    )
    if type(value["checked"]) is not bool or not _natural(value["operation_count"]):
        raise ValueError("prefix acceptance/count is malformed")
    count = value["operation_count"]
    eligible = None
    if block is not None:
        if count > len(block["ops"]):
            raise ValueError("prefix count is outside the source")
        eligible = {item["id"] for item in block["args"]}
        eligible.update(item["id"] for op in block["ops"][:count] for item in op["results"])
    tracked = value["tracked_facts"]
    if not isinstance(tracked, list) or (not value["checked"] and tracked):
        raise ValueError("only checked prefixes may publish numerical facts")
    facts = {}
    for fact in tracked:
        _object(fact, {"id", "initialized", "class", "exponent_upper", "integer_interval"}, "tracked fact")
        cap = fact["exponent_upper"]
        if (
            not _natural(fact["id"])
            or fact["id"] in facts
            or (eligible is not None and fact["id"] not in eligible)
            or type(fact["initialized"]) is not bool
            or not isinstance(fact["class"], str)
            or fact["class"] not in {"any", "finite", "non_nan", "nonnegative", "finite_nonzero"}
            or (cap is not None and (not _natural(cap) or cap > 254))
        ):
            raise ValueError("tracked numerical fact does not belong to the checked source prefix")
        interval = fact["integer_interval"]
        if interval is not None:
            _object(interval, {"lower", "upper"}, "integer interval")
            if not all(_natural(item) for item in interval.values()):
                raise ValueError("integer intervals must contain natural bounds")
        facts[fact["id"]] = fact
    demands = value["required_bounds"]
    if not isinstance(demands, list) or len(demands) != len(required):
        raise ValueError("prefix changed the requested bounds")
    for actual, expected in zip(demands, required, strict=True):
        _object(actual, {"value", "upper", "checked"}, "checked demand")
        if (
            type(actual["checked"]) is not bool
            or not _natural(actual["value"])
            or not _natural(actual["upper"])
            or {key: actual[key] for key in expected} != expected
        ):
            raise ValueError("prefix changed the requested bound or its acceptance type")
        if actual["checked"]:
            fact = facts.get(actual["value"])
            if (
                not value["checked"]
                or fact is None
                or not fact["initialized"]
                or fact["exponent_upper"] is None
                or fact["exponent_upper"] > actual["upper"]
            ):
                raise ValueError("accepted bound lacks a checked initialized numerical fact")
    if type(value["all_required_bounds_checked"]) is not bool or value["all_required_bounds_checked"] != (
        value["checked"] and all(demand["checked"] for demand in demands)
    ):
        raise ValueError("aggregate numerical result disagrees with its individual demands")
    refusal = value["first_refusal"]
    if refusal is not None:
        _object(refusal, {"operation", "results", "operands", "input_facts"}, "refusal")
        if block is not None:
            if count >= len(block["ops"]):
                raise ValueError("refusal is outside the source")
            op = block["ops"][count]
            if (refusal["operation"], refusal["results"], refusal["operands"]) != (
                op["name"],
                [result["id"] for result in op["results"]],
                op["operands"],
            ):
                raise ValueError("refusal does not identify the actual next source operation")


def validate(
    evidence: Any, *, entries: dict[str, dict[str, Any]], before: dict[str, Any], index_bits: int, scope: str
) -> None:
    fixed = {
        "schema": "merlin.w8a8.rotary-candidate.v1",
        "verdict": "abstracted",
        "index_bits": index_bits,
        "policy_scope": scope,
        "input_domain": "candidate_finite_nonzero_f32_with_per_argument_exponent_caps",
        "prefix_theorem": "MerlinVerify.Check.PolicyW8A8RotaryPrefix.entry_sound",
        "bound_theorem": "MerlinVerify.Check.PolicyW8A8RotaryPrefix.entry_bound",
        "nonvacuity_theorem": "MerlinVerify.Check.PolicyW8A8RotaryDomain.check_inhabited",
        "source_primitive_laws": 7,
        "additional_dynamic_primitive_laws": 3,
        "arithmetic_agreement_assumed": True,
        "transcendental_bounds_assumed": False,
        "concrete_context_inhabited": False,
        "policy_reviewed": False,
        "primitive_model_laws_verified": False,
        "whole_pass_verified": False,
    }
    _object(evidence, {*fixed, "structural_match", "functions"}, "rotary evidence")
    if any(type(evidence[key]) is not type(expected) or evidence[key] != expected for key, expected in fixed.items()):
        raise ValueError("rotary evidence changed its theorem, domain or unresolved premises")
    if type(evidence["structural_match"]) is not bool or not isinstance(evidence["functions"], list):
        raise ValueError("structural acceptance/functions are malformed")
    functions = {
        op["attrs"]["sym_name"]["value"]: op
        for op in before["module"]["ops"]
        if op["name"] == "func.func" and op["regions"] and op["regions"][0]["blocks"]
    }
    seen = set()
    for item in evidence["functions"]:
        _object(
            item,
            {"function", "domain_inhabited", "exponent_caps", "source_prefix", "reference_prefix"},
            "function evidence",
        )
        name = item["function"]
        if not isinstance(name, str) or name not in entries or name in seen:
            raise ValueError("rotary evidence changed the function roster")
        seen.add(name)
        caps = item["exponent_caps"]
        if (
            type(item["domain_inhabited"]) is not bool
            or not isinstance(caps, list)
            or any(cap is not None and not _natural(cap) for cap in caps)
            or caps != entries[name]["exponent_caps"]
        ):
            raise ValueError("rotary evidence changed its input domain")
        block = functions[name]["regions"][0]["blocks"][0]
        _prefix(item["source_prefix"], entries[name]["required_bounds"], block)
        _prefix(item["reference_prefix"], entries[name]["required_bounds"], None)
        if not item["domain_inhabited"] and any(
            prefix is not None and prefix["checked"] for prefix in (item["source_prefix"], item["reference_prefix"])
        ):
            raise ValueError("a certified prefix requires an inhabited input domain")
    if seen != set(entries):
        raise ValueError("rotary evidence omitted a source function")


def requested_passed(evidence: dict[str, Any]) -> bool:
    return evidence["structural_match"] and all(
        function["domain_inhabited"]
        and all(
            function[name] is not None and function[name]["all_required_bounds_checked"]
            for name in ("source_prefix", "reference_prefix")
        )
        for function in evidence["functions"]
    )
