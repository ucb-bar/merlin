"""Exact-source and linked-build accounting for opt-in host pointwise bodies.

This proves a typed source form, not host numerical equivalence or placement by itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from merlin.common.digest import is_sha256
from merlin.frontends.linalg_patterns import (
    STATIC_POINTWISE_SOURCE_BODY_SCHEMA,
    InvalidLinalgPattern,
    recognize_static_pointwise,
    static_pointwise_ordered_types,
)

PENDING = "source_pointwise_host_support_pending_build"
LINKED = "source_pointwise_host_support_linked"
SCOPE = "exact normalized source pointwise bodies and linked whole-program build; no numerical equivalence"


def begin(raw_sha256: str, normalized_sha256: str, n_operations: int) -> dict[str, Any]:
    """Create the mandatory current-result witness before examining host admissions."""
    if (
        not is_sha256(raw_sha256)
        or not is_sha256(normalized_sha256)
        or type(n_operations) is not int
        or n_operations < 1
    ):
        raise ValueError("pointwise source identity is malformed")
    return {
        "status": PENDING,
        "scope": SCOPE,
        "raw_source_sha256": raw_sha256,
        "normalized_source_sha256": normalized_sha256,
        "n_source_operations": n_operations,
        "count": 0,
        "occurrences": [],
    }


def record(
    witness: dict[str, Any],
    row: Mapping[str, Any],
    admission: Mapping[str, Any],
    parsed: tuple[Any, ...],
    source_rows: Mapping[int, Mapping[str, Any]],
) -> None:
    """Record every actual ordinal of one source-body-admitted grouped row."""
    body = admission.get("source_body_proof")
    if body is None:
        return
    ordinals, patterns = row.get("ordinals"), body.get("patterns") if isinstance(body, Mapping) else None
    if (
        admission.get("status") != "admitted"
        or admission.get("reviewed") is not True
        or not isinstance(body, Mapping)
        or not is_sha256(body.get("capability_spec_sha256"))
        or not isinstance(body.get("profile"), str)
        or not body["profile"]
        or not isinstance(body.get("declaration"), str)
        or not body["declaration"]
        or not isinstance(ordinals, list)
        or not isinstance(patterns, list)
        or type(row.get("count")) is not int
        or row["count"] != len(ordinals)
        or len(patterns) != len(ordinals)
    ):
        raise ValueError("pointwise host admission has no exact source occurrence roster")
    seen = {item["ordinal"] for item in witness["occurrences"]}
    for ordinal, pattern in zip(ordinals, patterns, strict=True):
        try:
            actual = (
                asdict(recognize_static_pointwise(parsed[ordinal]))
                if type(ordinal) is int and 0 <= ordinal < len(parsed)
                else None
            )
        except InvalidLinalgPattern as exc:
            raise ValueError("pointwise host proof differs from an exact parsed source occurrence") from exc
        if (
            type(ordinal) is not int
            or ordinal < 0
            or ordinal >= len(parsed)
            or ordinal in seen
            or source_rows.get(ordinal) is not row
            or actual != pattern
            or pattern["operation"] != body.get("operation")
            or pattern["predicate"] != body.get("predicate")
        ):
            raise ValueError("pointwise host proof differs from an exact parsed source occurrence")
        witness["occurrences"].append(
            {
                "ordinal": ordinal,
                "profile": body["profile"],
                "capability_spec_sha256": body["capability_spec_sha256"],
                "declaration": body["declaration"],
                **pattern,
            }
        )
        seen.add(ordinal)
    witness["occurrences"].sort(key=lambda item: item["ordinal"])
    witness["count"] = len(witness["occurrences"])


def link(source: Mapping[str, Any], linked_build: Mapping[str, str]) -> None:
    witness = source.get("pointwise_host_support")
    if not isinstance(witness, dict) or witness.get("status") != PENDING:
        raise ValueError("pointwise source proof is not pending the selected build")
    witness["status"] = LINKED
    witness["linked_build"] = dict(linked_build)


def linked_complete(source: Mapping[str, Any], entry: Mapping[str, Any], candidate_sha256: str) -> bool:
    """Require the source-only witness to cite the same linked whole-program bytes."""
    proof = source.get("pointwise_host_support")
    if (
        not isinstance(proof, Mapping)
        or set(proof)
        != {
            "status",
            "scope",
            "raw_source_sha256",
            "normalized_source_sha256",
            "n_source_operations",
            "count",
            "occurrences",
            "linked_build",
        }
        or proof.get("status") != LINKED
        or proof.get("scope") != SCOPE
    ):
        return False
    occurrences = proof.get("occurrences")
    if (
        proof.get("raw_source_sha256") != source.get("source_sha256")
        or proof.get("normalized_source_sha256") != source.get("normalized_source_sha256")
        or not is_sha256(proof.get("raw_source_sha256"))
        or not is_sha256(proof.get("normalized_source_sha256"))
        or entry.get("source_sha256") != source.get("source_sha256")
        or type(proof.get("n_source_operations")) is not int
        or proof["n_source_operations"] < 1
        or proof["n_source_operations"] != source.get("n_source_operations")
        or not isinstance(occurrences, list)
        or type(proof.get("count")) is not int
        or proof["count"] != len(occurrences)
    ):
        return False
    ordinals = []
    for item in occurrences:
        if not isinstance(item, Mapping):
            return False
        ordinal, shape, types = item.get("ordinal"), item.get("shape"), item.get("ordered_types")
        if (
            set(item)
            != {
                "ordinal",
                "profile",
                "capability_spec_sha256",
                "declaration",
                "operation",
                "shape",
                "ordered_types",
                "predicate",
            }
            or type(ordinal) is not int
            or not 0 <= ordinal < proof["n_source_operations"]
            or not isinstance(shape, (list, tuple))
            or not shape
            or any(type(dim) is not int or dim < 0 for dim in shape)
            or not isinstance(types, (list, tuple))
            or any(not isinstance(dtype, str) or not dtype for dtype in types)
            or not isinstance(item.get("declaration"), str)
            or not item["declaration"]
            or not isinstance(item.get("profile"), str)
            or not item["profile"]
            or not is_sha256(item.get("capability_spec_sha256"))
        ):
            return False
        declaration = {"schema": STATIC_POINTWISE_SOURCE_BODY_SCHEMA, "operation": item.get("operation")}
        if item.get("predicate") is not None:
            declaration["predicate"] = item["predicate"]
        try:
            if tuple(types) != static_pointwise_ordered_types(declaration):
                return False
        except InvalidLinalgPattern:
            return False
        ordinals.append(ordinal)
    if ordinals != sorted(set(ordinals)):
        return False
    linked = proof.get("linked_build")
    return bool(
        isinstance(linked, Mapping)
        and set(linked) == {"candidate_tree_sha256", "capture_tree_sha256", "elf_sha256"}
        and all(is_sha256(linked[key]) for key in linked)
        and linked["candidate_tree_sha256"] == candidate_sha256
        and linked["capture_tree_sha256"] == entry.get("capture_tree_sha256")
        and linked["elf_sha256"] == entry.get("elf_sha256")
    )
