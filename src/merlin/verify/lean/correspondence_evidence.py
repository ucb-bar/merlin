"""Validate the exact claim boundary of modular W8A8 correspondence evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODULE = "MerlinVerify.Check.PolicyW8A8Correspondence"
ROOTS = {f"{MODULE}.{name}": MODULE for name in ("check_sound", "successful_reference", "defined_reference")}
THEOREM = f"{MODULE}.check_sound"


def read(path: Path) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key in correspondence evidence")
            result[key] = value
        return result

    return json.loads(path.read_text(), object_pairs_hook=unique)


def same(left: Any, right: Any) -> bool:
    """JSON boolean, integer and float representations remain distinct."""
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def functions(program: dict[str, Any]) -> list[str]:
    return [
        op["attrs"]["sym_name"]["value"]
        for op in program["module"]["ops"]
        if op["name"] == "func.func" and op["regions"] and op["regions"][0]["blocks"]
    ]


def validate(
    evidence: Any,
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    suffix_count: int,
    index_bits: int,
    scope: str,
) -> None:
    fixed = {
        "schema": "merlin.w8a8.correspondence-candidate.v1",
        "checker": f"{MODULE}.check",
        "theorem": THEOREM,
        "successful_reference_theorem": f"{MODULE}.successful_reference",
        "defined_reference_theorem": f"{MODULE}.defined_reference",
        "index_bits": index_bits,
        "policy_scope": scope,
        "semantic_interpretation": "MerlinVerify.Rules.W8A8SourceSemantics.opSem",
        "correspondence_domain": "all_contexts_functions_and_inputs_in_this_operational_semantics",
        "source_functions": functions(before),
        "target_functions": functions(after),
        "numerical_premises_for_correspondence": 0,
        "original_source_definedness_required_for_correspondence": False,
        "successful_reference_required_for_success_transfer": True,
        "whole_model_initialized_execution_established": False,
        "original_source_equivalence_claimed": False,
        "whole_pass_verified": False,
        "policy_reviewed": False,
        "primitive_model_laws_verified": False,
        "concrete_context_inhabited": False,
        "verdict": "abstracted",
    }
    booleans = {
        "reference_expanded",
        "actual_boundary_changed",
        "complete_boundary_matched",
        "complete_correspondence_passed",
    }
    if not isinstance(evidence, dict) or set(evidence) != set(fixed) | booleans | {
        "reference_functions",
        "suffix_boundaries",
    }:
        raise ValueError("correspondence evidence has an unexpected schema")
    if any(not same(evidence[key], value) for key, value in fixed.items()):
        raise ValueError("correspondence evidence changed its theorem, domain, premise or source roster")
    if any(type(evidence[key]) is not bool for key in booleans):
        raise ValueError("correspondence results must be JSON booleans")
    expected = fixed["source_functions"] if evidence["reference_expanded"] else []
    if not same(evidence["reference_functions"], expected):
        raise ValueError("correspondence reference roster differs from its actual source")
    links = evidence["suffix_boundaries"]
    if not isinstance(links, list) or len(links) != suffix_count:
        raise ValueError("correspondence evidence omitted an actual suffix boundary")
    for index, link in enumerate(links):
        if (
            not isinstance(link, dict)
            or set(link) != {"index", "matched"}
            or type(link["index"]) is not int
            or link["index"] != index
            or type(link["matched"]) is not bool
        ):
            raise ValueError("malformed correspondence suffix evidence")
    if evidence["complete_boundary_matched"] and not evidence["reference_expanded"]:
        raise ValueError("correspondence accepted a missing policy reference")
    passed = evidence["complete_boundary_matched"] and all(link["matched"] for link in links)
    if evidence["complete_correspondence_passed"] is not passed:
        raise ValueError("correspondence result disagrees with its complete boundary checks")
