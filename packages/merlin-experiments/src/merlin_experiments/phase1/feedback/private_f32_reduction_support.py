"""Mandatory source and linked-build witness for reviewed f32 host reductions.

A host-admitted reduction whose parsed source is f32 comes here, never to the
integer-reduction witness. Every occurrence must be the exact closed form the
declaration names (an f32 sum: +0.0 init, one fast-math-free addf of element
into accumulator, f32 accumulation, static axes), admitted by exactly one
reviewed declaration of that frontend operation. It does not create a host
admission, prove frontend equivalence, fix the summation order or certify the
numerical behavior of the compiled image. The selected index width is joined
separately.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

from merlin.common import mlir_query as mq
from merlin.common.digest import is_sha256
from merlin.frontends.linalg_f32_reductions import (
    F32_POSITIVE_ZERO_BITS,
    recognize_static_f32_sum,
    serialized_f32_reduction_pattern,
)
from merlin.frontends.linalg_f32_reductions import F32_REDUCTION_TARGETS as _TARGETS
from merlin.frontends.linalg_patterns import InvalidLinalgPattern, _screen_static_linalg_source
from merlin_experiments.phase1.feedback import private_control_support as control
from merlin_experiments.phase1.feedback.private_index_source import _same_json_value

FIELD = "f32_reduction_host_support"
PENDING = "source_f32_reduction_host_pending_build"
LINKED = "source_f32_reduction_host_linked"
SCOPE = "reviewed host f32-reduction source ordinals and exact linked build; no numerical equivalence"
_KINDS = frozenset(kind for _, kind in _TARGETS.values())
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
_OCCURRENCE = {"ordinal", "kind", "declaration", "profile", "capability_spec_sha256", "pattern"}
_PATTERN = {"operation", "input_shape", "output_shape", "axis", "accumulator_type", "init_bits", "index_bits_premise"}


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(f"f32 reduction host support: {reason}")


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


def _f32_reduce(op) -> bool:
    from xdsl.dialects.builtin import TensorType
    from xdsl.dialects.linalg.ops import ReduceOp

    return type(op) is ReduceOp and isinstance(op.input.type, TensorType) and str(op.input.type.element_type) == "f32"


def selects(row: Mapping[str, Any], parsed: tuple[Any, ...]) -> bool:
    """True only for an exact f32 reduction row; an integer row keeps its own witness."""
    tag = row.get("frontend_op") if isinstance(row, Mapping) else None
    ordinals = row.get("ordinals") if isinstance(row, Mapping) else None
    if tag not in _TARGETS or not isinstance(ordinals, list) or not ordinals:
        return False
    if any(type(value) is not int or not 0 <= value < len(parsed) for value in ordinals):
        return False
    kinds = {_f32_reduce(parsed[value]) for value in ordinals}
    _need(len(kinds) == 1, "source row mixes f32 and non-f32 reductions")
    return kinds.pop()


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
        "selected_index_observation": dict(selected) if selected is not None else None,
        "count": 0,
        "occurrences": [],
        "occurrences_sha256": _digest([]),
        "source_verified": True,  # An empty roster has no reduction source obligation.
    }


def _pattern(op, kind: str, width: int) -> dict:
    _need(kind == "f32_sum", "selected f32 reduction kind has no closed recognizer")
    try:
        result = recognize_static_f32_sum(op, index_bits=width)
    except InvalidLinalgPattern as exc:
        raise ValueError("f32 reduction host support: parsed source body is not a closed f32 sum") from exc
    _need(result.operation == kind, "source body differs from selected reduction kind")
    return serialized_f32_reduction_pattern(result)


def _declaration(profile: Mapping[str, Any], tag: str) -> str:
    decisions = profile.get("decisions")
    admitted = (
        [decision for decision in decisions if isinstance(decision, Mapping) and decision.get("status") == "admitted"]
        if isinstance(decisions, list)
        else []
    )
    _need(
        len(admitted) == 1
        and admitted[0].get("op") == tag
        and isinstance(admitted[0].get("declaration"), str)
        and bool(admitted[0]["declaration"])
        and admitted[0].get("review_status") == "reviewed"
        and "source_body_proof" not in admitted[0],
        "selected f32 reduction has no unique reviewed declaration of its frontend operation",
    )
    return admitted[0]["declaration"]


def record(
    witness: dict,
    row: Mapping[str, Any],
    host_decision: Mapping[str, Any],
    parsed: tuple[Any, ...],
    source_rows: Mapping[int, Mapping[str, Any]],
) -> None:
    """Recompute every host-admitted f32 reduction ordinal from the actual parsed source."""
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
    tag = row.get("frontend_op")
    _need(tag in _TARGETS, "selected f32 reduction row has no matching frontend identity")
    operation, kind = _TARGETS[tag]
    _need(row.get("mlir_operation") == operation, "selected f32 reduction row differs from parsed operation")
    _need(
        all(mq.op_name(parsed[value]) == operation and _f32_reduce(parsed[value]) for value in ordinals)
        and all(mq.attr_str(parsed[value], "prov.aten") == tag for value in ordinals),
        "selected f32 reduction row frontend or operation differs from parsed source",
    )
    _need(
        host_decision.get("status") == "admitted"
        and host_decision.get("reviewed") is True
        and "source_body_proof" not in host_decision,
        "selected f32 reduction has no reviewed host admission",
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
        "selected f32 reduction has no unique reviewed host profile identity",
    )
    declaration = _declaration(profiles[0], tag)
    width = _selected_width(witness["selected_index_observation"])
    seen = {item["ordinal"] for item in witness["occurrences"]}
    _need(not seen.intersection(ordinals), "selected reduction ordinal is repeated")
    new = [
        {
            "ordinal": ordinal,
            "kind": kind,
            "declaration": declaration,
            "profile": profiles[0]["profile"],
            "capability_spec_sha256": profiles[0]["capability_spec_sha256"],
            "pattern": _pattern(parsed[ordinal], kind, width),
        }
        for ordinal in ordinals
    ]
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
        tag = mq.attr_str(op, "prov.aten")
        _need(tag in _TARGETS and _f32_reduce(op), "selected ordinal no longer has an f32 reduction body")
        return _pattern(op, _TARGETS[tag][1], width)

    try:
        raw, normalized, evidence = _screen_static_linalg_source(source, tuple(expected), recognize)
    except InvalidLinalgPattern as exc:
        raise ValueError("f32 reduction host support: full source reparse failed") from exc
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
    if not isinstance(pattern, Mapping) or set(pattern) != _PATTERN or pattern.get("operation") != kind:
        return False
    inp = _shape(pattern.get("input_shape"))
    out = _shape(pattern.get("output_shape"), scalar=True)
    if inp is None or out is None:
        return False
    maximum = (1 << (width - 1)) - 1
    if any(extent > maximum for extent in (*inp, *out)):
        return False
    axis = pattern.get("axis")
    return bool(
        type(axis) is list
        and axis
        and all(type(dim) is int and 0 <= dim < len(inp) for dim in axis)
        and tuple(sorted(set(axis))) == tuple(axis)
        and out == tuple(extent for dim, extent in enumerate(inp) if dim not in axis)
        and pattern.get("accumulator_type") == "f32"
        and pattern.get("init_bits") == F32_POSITIVE_ZERO_BITS
        and type(pattern.get("index_bits_premise")) is int
        and pattern["index_bits_premise"] == width
    )


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
            or not isinstance(item.get("declaration"), str)
            or not item["declaration"]
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
