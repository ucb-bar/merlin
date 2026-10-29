"""Decode declared integer domains and validate bounded-policy evidence without approving it."""

from __future__ import annotations

import json
from typing import Any

from merlin.verify.lean.policy_numerics import _suffix

MODULE = "MerlinVerify.Check.PolicyW8A8BoundedChain"
INPUT_DOMAIN = "finite_f32_and_declared_integer_intervals_per_function"


def decode(text: str) -> dict[str, list[dict[str, int] | None]]:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate input contract object key: {key}")
            result[key] = value
        return result

    return contracts(json.loads(text, object_pairs_hook=unique))


def contracts(value: Any) -> dict[str, list[dict[str, int] | None]]:
    if not isinstance(value, list):
        raise ValueError("input contracts must be an array")
    result = {}
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"function", "intervals"}:
            raise ValueError("input contract requires exactly function and intervals")
        name, intervals = entry["function"], entry["intervals"]
        if not isinstance(name, str) or name in result or not isinstance(intervals, list):
            raise ValueError("input contract names must be unique strings with interval arrays")
        for interval in intervals:
            if interval is not None and (
                not isinstance(interval, dict)
                or set(interval) != {"lower", "upper"}
                or any(type(bound) is not int or bound < 0 for bound in interval.values())
            ):
                raise ValueError("integer intervals require exactly nonnegative integer lower and upper bounds")
        result[name] = intervals
    return result


def functions(document: dict[str, Any]) -> list[str]:
    result = []
    for op in document["module"]["ops"]:
        if op["name"] != "func.func" or not op["regions"] or not op["regions"][0]["blocks"]:
            continue
        attributes = {**op["attrs"], **op.get("props", {})}
        name = attributes["sym_name"]["value"]
        if not isinstance(name, str) or name in result:
            raise ValueError("exported function names must be unique strings")
        result.append(name)
    return result


def validate(
    evidence: dict[str, Any],
    *,
    bounds: dict[str, list[dict[str, int] | None]],
    roster: list[str],
    index_bits: int,
    scope: str,
    suffix_count: int,
) -> None:
    expected = {
        "checker": f"{MODULE}.check",
        "theorem": f"{MODULE}.check_sound",
        "verdict": "abstracted",
        "index_bits": index_bits,
        "policy_scope": scope,
        "input_domain": INPUT_DOMAIN,
        "floating_domain": "byte_extension_with_explicit_source_and_dynamic_primitive_laws",
        "source_primitive_laws": 7,
        "additional_dynamic_primitive_laws": 3,
        "policy_reviewed": False,
        "primitive_model_laws_verified": False,
        "whole_pass_verified": False,
    }
    status = {"boundary_qualification_passed", "conditional_initialized_policy_refinement", "suffix_checked"}
    if not isinstance(evidence, dict) or set(evidence) != set(expected) | status | {
        "functions",
        "suffix_match",
        "suffix_boundaries",
    }:
        raise ValueError("bounded W8A8 evidence has missing or unexpected fields")
    if any(type(evidence[key]) is not type(value) or evidence[key] != value for key, value in expected.items()):
        raise ValueError("bounded W8A8 evidence changed the checker, domain or unproved premises")
    if any(type(evidence[key]) is not bool for key in status):
        raise ValueError("bounded W8A8 qualification status must be Boolean")
    details = evidence["functions"]
    if not isinstance(details, list) or any(
        not isinstance(fn, dict)
        or set(fn) != {"function", "input_intervals", "source_initialized", "reference_initialized"}
        for fn in details
    ):
        raise ValueError("bounded W8A8 function obligations are missing or malformed")
    # No expansion means the driver could not measure any function obligations.
    measured = [fn["function"] for fn in details]
    if set(bounds) != set(roster) or (measured != roster and measured != []):
        raise ValueError("bounded W8A8 evidence changed the actual source function roster")
    for fn in details:
        if json.dumps(fn["input_intervals"], sort_keys=True) != json.dumps(bounds[fn["function"]], sort_keys=True):
            raise ValueError("bounded W8A8 evidence changed a function's declared input intervals")
        if any(type(fn[key]) is not bool for key in ("source_initialized", "reference_initialized")):
            raise ValueError("bounded W8A8 initialization status must be Boolean")
    boundary = evidence["boundary_qualification_passed"]
    if boundary and (
        not details or any(not (fn["source_initialized"] and fn["reference_initialized"]) for fn in details)
    ):
        raise ValueError("bounded W8A8 qualification omitted an initialization obligation")
    links = evidence["suffix_boundaries"]
    if not isinstance(links, list) or any(
        not isinstance(link, dict) or set(link) != {"index", "conditional_match", "status"} for link in links
    ):
        raise ValueError("bounded W8A8 suffix evidence is malformed")
    _suffix(evidence, boundary=boundary, count=suffix_count)
