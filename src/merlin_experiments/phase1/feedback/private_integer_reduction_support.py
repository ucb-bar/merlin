"""Mandatory source and linked-build witness for reviewed integer reductions.

This checks prepared sum, prefix-sum and paired minimum source bodies. It does
not create a host admission, prove frontend equivalence or certify numerical
behavior of the compiled image. The selected index width is joined separately.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

from merlin.common import mlir_query as mq
from merlin.common.digest import is_sha256
from merlin.frontends.linalg_extremum_patterns import recognize_static_i64_argmin
from merlin.frontends.linalg_integer_reductions import recognize_static_integer_reduction
from merlin.frontends.linalg_patterns import (
    InvalidLinalgPattern,
    _screen_static_linalg_source,
    recognize_static_pointwise,
)
from merlin.frontends.linalg_reduction_source_body import INTEGER_REDUCTION_TARGETS as _TARGETS
from merlin.frontends.linalg_reduction_source_body import (
    STATIC_INTEGER_REDUCTION_SOURCE_BODY_SCHEMA,
    serialized_reduction_pattern,
)
from merlin_experiments.phase1.feedback import private_control_support as control
from merlin_experiments.phase1.feedback.private_index_source import _same_json_value

FIELD = "integer_reduction_host_support"
PENDING = "source_integer_reduction_host_pending_build"
LINKED = "source_integer_reduction_host_linked"
SCOPE = "reviewed host integer-reduction source ordinals and exact linked build; no numerical equivalence"
_KINDS = frozenset(kind for _, kind in _TARGETS.values())
_PREFIXES = ("aten.sum.", "aten.cumsum.", "aten.min.")
_BUILD_KEYS = {"candidate_tree_sha256", "capture_tree_sha256", "elf_sha256"}
_TOP = {
    "status",
    "scope",
    "raw_source_sha256",
    "normalized_source_sha256",
    "n_source_operations",
    "selected_index_observation",
    "count",
    "occurrences",
    "occurrences_sha256",
    "source_verified",
    "linked_build",
}
_OCCURRENCE = {"ordinal", "kind", "profile", "capability_spec_sha256", "pattern"}


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(f"integer reduction host support: {reason}")


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _matching_digest(value: object, digest: object) -> bool:
    try:
        return is_sha256(digest) and _digest(value) == digest
    except (TypeError, ValueError):
        return False


def _selected_width(selected: object) -> int:
    _need(
        isinstance(selected, Mapping)
        and selected.get("schema") == "merlin.selected-index-lowering.v1"
        and type(selected.get("index_bits")) is int
        and 2 <= selected["index_bits"] <= 128,
        "selected producer-owned index width is absent or malformed",
    )
    return selected["index_bits"]


def begin(
    raw_sha256: str, normalized_sha256: str, n_operations: int, selected: Mapping[str, Any] | None = None
) -> dict:
    """Create the roster; absent width permits source-only empty diagnostics, never a build claim."""
    _need(
        is_sha256(raw_sha256) and is_sha256(normalized_sha256) and type(n_operations) is int and n_operations > 0,
        "source identity or parsed operation count is malformed",
    )
    if selected is not None:
        _selected_width(selected)
    return {
        "status": PENDING,
        "scope": SCOPE,
        "raw_source_sha256": raw_sha256,
        "normalized_source_sha256": normalized_sha256,
        "n_source_operations": n_operations,
        "selected_index_observation": deepcopy(dict(selected)) if selected is not None else None,
        "count": 0,
        "occurrences": [],
        "occurrences_sha256": _digest([]),
        "source_verified": True,  # An empty roster has no reduction source obligation.
    }


def _kind_for_op(op) -> str | None:
    tag = mq.attr_str(op, "prov.aten")
    if tag not in _TARGETS:
        if isinstance(tag, str) and tag.startswith(_PREFIXES):
            raise ValueError("integer reduction host support: unproved frontend reduction form")
        return None
    operation, kind = _TARGETS[tag]
    if mq.op_name(op) == operation:
        return kind
    if tag == "aten.sum.dim_IntList" and mq.op_name(op) == "linalg.generic":
        # The Boolean-to-i64 input cast is a separate host operation. Its own
        # reviewed admission is still required, but it is not the reduction.
        try:
            pattern = recognize_static_pointwise(op)
        except InvalidLinalgPattern as exc:
            raise ValueError("integer reduction host support: sum precursor is not an exact cast") from exc
        if pattern.operation == "arith.extui" and pattern.ordered_types == ("i1", "i64", "i64"):
            return None
    if mq.op_name(op).startswith("linalg."):
        raise ValueError("integer reduction host support: selected source form has an unproved Linalg owner")
    return None  # Nested scalar/support rows keep their separate reviewed obligations.


def _pattern(op, kind: str, width: int) -> dict:
    try:
        if kind == "i64_min_first_index":
            result = recognize_static_i64_argmin(op, index_bits=width)
        else:
            result = recognize_static_integer_reduction(op, index_bits=width)
    except InvalidLinalgPattern as exc:
        raise ValueError("integer reduction host support: parsed source body is not a closed reduction") from exc
    _need(result.operation == kind, "source body differs from selected reduction kind")
    return serialized_reduction_pattern(result)


def record(
    witness: dict,
    row: Mapping[str, Any],
    host_decision: Mapping[str, Any],
    parsed: tuple[Any, ...],
    source_rows: Mapping[int, Mapping[str, Any]],
) -> None:
    """Recompute every host-admitted root ordinal from the actual parsed source."""
    _need(witness.get("status") == PENDING, "source witness is not pending")
    _need(isinstance(row, Mapping) and isinstance(host_decision, Mapping), "host row or admission is malformed")
    ordinals = row.get("ordinals")
    _need(
        isinstance(ordinals, list)
        and bool(ordinals)
        and type(row.get("count")) is int
        and row["count"] == len(ordinals)
        and all(type(value) is int and 0 <= value < len(parsed) for value in ordinals)
        and len(set(ordinals)) == len(ordinals)
        and len(parsed) == witness.get("n_source_operations")
        and all(source_rows.get(value) is row for value in ordinals),
        "host source row has no exact parsed ordinal ownership",
    )
    kinds = {_kind_for_op(parsed[ordinal]) for ordinal in ordinals}
    body = host_decision.get("source_body_proof")
    tag = row.get("frontend_op")
    if isinstance(tag, str) and tag.startswith(_PREFIXES) and tag not in _TARGETS:
        raise ValueError("integer reduction host support: row selects an unproved frontend form")
    if tag in _TARGETS and any(mq.attr_str(parsed[ordinal], "prov.aten") != tag for ordinal in ordinals):
        raise ValueError("integer reduction host support: row frontend differs from parsed source")
    _need(len(kinds) == 1, "source row mixes reductions with unrelated operations")
    kind = kinds.pop()
    if isinstance(body, Mapping) and body.get("schema") == STATIC_INTEGER_REDUCTION_SOURCE_BODY_SCHEMA:
        _need(kind is not None, "integer source-body proof has no reduction root")
    if kind is None:
        return
    _need(tag in _TARGETS, "selected reduction row has no matching frontend identity")
    _need(
        row.get("mlir_operation") == _TARGETS[tag][0],
        "selected reduction row differs from parsed operation",
    )
    _need(
        host_decision.get("status") == "admitted" and host_decision.get("reviewed") is True,
        "selected reduction has no reviewed host admission",
    )
    selected_profiles = host_decision.get("profiles")
    _need(isinstance(selected_profiles, (list, tuple)), "reviewed host profiles are malformed")
    profiles = [
        item
        for item in selected_profiles
        if isinstance(item, Mapping) and item.get("status") == "admitted" and item.get("reviewed") is True
    ]
    _need(
        len(profiles) == 1
        and isinstance(profiles[0].get("profile"), str)
        and bool(profiles[0]["profile"])
        and is_sha256(profiles[0].get("capability_spec_sha256")),
        "selected reduction has no unique reviewed host profile identity",
    )
    selected = witness["selected_index_observation"]
    width = _selected_width(selected)
    profile_decisions = profiles[0].get("decisions")
    if body is None:
        _need(
            not isinstance(profile_decisions, list)
            or not any(
                isinstance(decision, Mapping)
                and decision.get("status") == "admitted"
                and isinstance(decision.get("source_body_proof"), Mapping)
                and decision["source_body_proof"].get("schema") == STATIC_INTEGER_REDUCTION_SOURCE_BODY_SCHEMA
                for decision in profile_decisions
            ),
            "integer source-body admission lost its selected source proof",
        )
    if body is not None:
        _need(
            isinstance(body, Mapping)
            and set(body)
            == {
                "schema",
                "declaration",
                "operation",
                "selected_index_observation",
                "patterns",
                "profile",
                "capability_spec_sha256",
            }
            and body["schema"] == STATIC_INTEGER_REDUCTION_SOURCE_BODY_SCHEMA
            and body["operation"] == kind
            and isinstance(body["declaration"], str)
            and bool(body["declaration"])
            and _same_json_value(body["selected_index_observation"], selected)
            and body["profile"] == profiles[0]["profile"]
            and body["capability_spec_sha256"] == profiles[0]["capability_spec_sha256"]
            and isinstance(body["patterns"], list)
            and len(body["patterns"]) == len(ordinals),
            "integer source-body admission differs from selected source/profile context",
        )
        expected_inner = {key: value for key, value in body.items() if key not in {"profile", "capability_spec_sha256"}}
        _need(
            isinstance(profile_decisions, list)
            and sum(
                isinstance(decision, Mapping)
                and decision.get("status") == "admitted"
                and decision.get("declaration") == body["declaration"]
                and _same_json_value(decision.get("source_body_proof"), expected_inner)
                for decision in profile_decisions
            )
            == 1,
            "integer source-body admission has no unique reviewed declaration",
        )
    seen = {item["ordinal"] for item in witness["occurrences"]}
    _need(not seen.intersection(ordinals), "selected reduction ordinal is repeated")
    new = []
    for position, ordinal in enumerate(ordinals):
        pattern = _pattern(parsed[ordinal], kind, width)
        if body is not None:
            _need(
                _same_json_value(body["patterns"][position], pattern),
                "integer source-body admission differs from parsed ordinal",
            )
        new.append(
            {
                "ordinal": ordinal,
                "kind": kind,
                "profile": profiles[0]["profile"],
                "capability_spec_sha256": profiles[0]["capability_spec_sha256"],
                "pattern": pattern,
            }
        )
    witness["occurrences"].extend(new)
    witness["occurrences"].sort(key=lambda item: item["ordinal"])
    witness["count"] = len(witness["occurrences"])
    witness["occurrences_sha256"] = _digest(witness["occurrences"])
    witness["source_verified"] = False


def verify_source(witness: dict, source: Path) -> None:
    """Independently reparse and rehash the complete source once before build."""
    _need(witness.get("status") == PENDING, "source witness is not pending")
    occurrences = witness.get("occurrences")
    _need(isinstance(occurrences, list) and len(occurrences) == witness.get("count"), "source roster is incomplete")
    if not occurrences:
        return
    width = _selected_width(witness.get("selected_index_observation"))
    expected = {item["ordinal"]: item for item in occurrences}
    _need(len(expected) == len(occurrences), "source ordinal roster contains a duplicate")

    def recognize(op):
        kind = _kind_for_op(op)
        _need(kind is not None, "selected ordinal no longer has a reduction body")
        return _pattern(op, kind, width)

    try:
        raw, normalized, evidence = _screen_static_linalg_source(source, tuple(expected), recognize)
    except InvalidLinalgPattern as exc:
        raise ValueError("integer reduction host support: full source reparse failed") from exc
    _need(
        raw == witness.get("raw_source_sha256")
        and normalized == witness.get("normalized_source_sha256")
        and all(
            expected[ordinal]["kind"] == expected[ordinal]["pattern"].get("operation")
            and _same_json_value(expected[ordinal]["pattern"], pattern)
            for ordinal, pattern in evidence
        ),
        "source bytes or recomputed ordinal bodies changed",
    )
    witness["source_verified"] = True


def link(source: Mapping[str, Any], actual_index: Mapping[str, Any], linked_build: Mapping[str, Any]) -> None:
    """Call after private_control_support.link_selected_build on the same build."""
    witness = source.get(FIELD)
    _need(isinstance(witness, dict) and witness.get("status") == PENDING, "source roster is not pending")
    selected = witness.get("selected_index_observation")
    _need(
        witness.get("source_verified") is True
        and _same_json_value(selected, source.get("selected_index_observation"))
        and control._record_matches_selected(actual_index, selected),  # noqa: PLC2701 -- shared trusted join
        "actual linked compiler index observation differs from source premise",
    )
    _need(
        isinstance(linked_build, Mapping)
        and set(linked_build) == _BUILD_KEYS
        and all(is_sha256(linked_build[key]) for key in _BUILD_KEYS),
        "linked candidate/capture/ELF tuple is incomplete",
    )
    witness["status"] = LINKED
    witness["linked_build"] = dict(linked_build)


def _shape(value: object, *, scalar: bool = False) -> tuple[int, ...] | None:
    if type(value) is not list or (not value and not scalar):
        return None
    if any(type(dim) is not int or dim <= 0 for dim in value):
        return None
    return tuple(value)


def _valid_pattern(kind: str, pattern: object, width: int) -> bool:
    if not isinstance(pattern, Mapping) or pattern.get("operation") != kind:
        return False
    inp = _shape(pattern.get("input_shape"))
    out = _shape(pattern.get("output_shape"), scalar=True)
    if inp is None or out is None:
        return False
    maximum = (1 << (width - 1)) - 1
    if any(extent > maximum for extent in (*inp, *out)):
        return False
    if kind in {"sum", "cumsum"}:
        if set(pattern) != {"operation", "input_shape", "output_shape", "axis", "input_type", "index_bits_premise"}:
            return False
        axis = pattern.get("axis")
        if (
            type(axis) is not list
            or not axis
            or any(type(dim) is not int or not 0 <= dim < len(inp) for dim in axis)
            or tuple(sorted(set(axis))) != tuple(axis)
            or pattern.get("input_type") not in {"i1", "i64"}
            or type(pattern.get("index_bits_premise")) is not int
            or pattern["index_bits_premise"] != width
        ):
            return False
        if kind == "sum":
            return out == tuple(extent for dim, extent in enumerate(inp) if dim not in axis)
        return len(axis) == 1 and out == inp
    if kind == "i64_min_first_index":
        axis = pattern.get("axis")
        return (
            set(pattern) == {"operation", "axis", "input_shape", "output_shape", "ordered_types", "index_bits"}
            and type(axis) is int
            and 0 <= axis < len(inp)
            and out == tuple(extent for dim, extent in enumerate(inp) if dim != axis)
            and type(pattern.get("ordered_types")) is list
            and tuple(pattern["ordered_types"]) == ("i64",) * 5
            and type(pattern.get("index_bits")) is int
            and pattern["index_bits"] == width
            and all(extent <= (1 << 63) - 1 for extent in inp)
        )
    return False


def linked_complete(source: Mapping[str, Any], entry: Mapping[str, Any], candidate_sha256: str) -> bool:
    """Reject missing, old, malformed or differently linked source evidence."""
    if not isinstance(source, Mapping) or not isinstance(entry, Mapping) or not is_sha256(candidate_sha256):
        return False
    witness = source.get(FIELD)
    if (
        not isinstance(witness, Mapping)
        or set(witness) != _TOP
        or witness.get("status") != LINKED
        or witness.get("scope") != SCOPE
        or witness.get("source_verified") is not True
        or not is_sha256(witness.get("raw_source_sha256"))
        or not is_sha256(witness.get("normalized_source_sha256"))
        or witness.get("raw_source_sha256") != source.get("source_sha256")
        or witness.get("normalized_source_sha256") != source.get("normalized_source_sha256")
        or entry.get("source_sha256") != source.get("source_sha256")
        or type(witness.get("n_source_operations")) is not int
        or witness["n_source_operations"] < 1
        or witness["n_source_operations"] != source.get("n_source_operations")
        or not isinstance(witness.get("occurrences"), list)
        or type(witness.get("count")) is not int
        or witness["count"] != len(witness["occurrences"])
        or not _matching_digest(witness["occurrences"], witness.get("occurrences_sha256"))
    ):
        return False
    selected = witness.get("selected_index_observation")
    try:
        width = _selected_width(selected)
    except ValueError:
        return False
    if not _same_json_value(selected, source.get("selected_index_observation")) or not control._record_matches_selected(  # noqa: PLC2701
        entry.get("index_lowering"), selected
    ):
        return False
    ordinals = []
    for item in witness["occurrences"]:
        if not isinstance(item, Mapping) or set(item) != _OCCURRENCE:
            return False
        ordinal = item.get("ordinal")
        if (
            type(ordinal) is not int
            or not 0 <= ordinal < witness["n_source_operations"]
            or item.get("kind") not in _KINDS
            or not isinstance(item.get("profile"), str)
            or not item["profile"]
            or not is_sha256(item.get("capability_spec_sha256"))
            or not _valid_pattern(item["kind"], item.get("pattern"), width)
        ):
            return False
        ordinals.append(ordinal)
    if ordinals != sorted(set(ordinals)):
        return False
    linked = witness.get("linked_build")
    return bool(
        isinstance(linked, Mapping)
        and set(linked) == _BUILD_KEYS
        and all(is_sha256(linked[key]) for key in _BUILD_KEYS)
        and linked["candidate_tree_sha256"] == candidate_sha256
        and linked["capture_tree_sha256"] == entry.get("capture_tree_sha256")
        and linked["elf_sha256"] == entry.get("elf_sha256")
    )
