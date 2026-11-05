"""Mandatory prepared-source and linked-build witness for reviewed f32 maximum.

This does not grant host admission or claim PyTorch numerical equivalence.
In particular, the selected maximumf reduction may choose another signed zero.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from math import prod
from pathlib import Path
from typing import Any

from merlin.common import mlir_query as mq
from merlin.common.digest import is_sha256
from merlin.compile.model_execution_inputs import file_sha256, strict_tree_sha256
from merlin.frontends.capture_normalization import normalize_capture_mlir
from merlin.frontends.linalg_f32_maximum_patterns import (
    STATIC_F32_MAXIMUM_SOURCE_BODY_SCHEMA,
    recognize_static_f32_maximum,
    serialized_f32_maximum_pattern,
)
from merlin.frontends.linalg_patterns import InvalidLinalgPattern
from merlin.llvmlower.target_data_layout import selected_index_bits
from merlin_experiments.phase1.feedback import private_control_support as control
from merlin_experiments.phase1.feedback import private_linked_source_preparation as preparation
from merlin_experiments.phase1.feedback.private_index_source import _same_json_value

FIELD = "f32_maximum_host_support"
PENDING = "source_f32_maximum_host_pending_build"
LINKED = "source_f32_maximum_host_linked"
SCOPE = "reviewed prepared maximumf source and exact linked build; no frontend or numerical equivalence"
_BUILD_KEYS = {"candidate_tree_sha256", "capture_tree_sha256", "elf_sha256"}
_BODY_KEYS = {
    "schema",
    "declaration",
    "operation",
    "selected_index_observation",
    "patterns",
    "profile",
    "capability_spec_sha256",
}
_OCCURRENCE = {"ordinal", "profile", "capability_spec_sha256", "pattern"}
_TOP = {
    "status",
    "scope",
    "raw_source_sha256",
    "normalized_source_sha256",
    "capture_receipt_sha256",
    "n_source_operations",
    "selected_index_observation",
    "expected_ordinals",
    "count",
    "occurrences",
    "occurrences_sha256",
    "source_verified",
    "source_capture_path",
    "linked_build",
}


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(f"f32 maximum host support: {reason}")


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _selected_width(selected: object) -> int:
    width = selected_index_bits(selected)
    _need(
        type(width) is int and 2 <= width <= 128,
        "selected producer-owned index width is absent or malformed",
    )
    return width


def _pattern(op: Any, width: int) -> dict:
    try:
        return serialized_f32_maximum_pattern(recognize_static_f32_maximum(op, index_bits=width))
    except InvalidLinalgPattern as exc:
        raise ValueError("f32 maximum host support: parsed source is not a closed maximumf reduction") from exc


def begin(
    raw_sha256: str,
    normalized_sha256: str,
    receipt_sha256: str,
    parsed: tuple[Any, ...],
    selected: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """An empty source diagnostic may omit width; a recorded root never may."""
    _need(
        all(is_sha256(value) for value in (raw_sha256, normalized_sha256, receipt_sha256))
        and isinstance(parsed, tuple)
        and bool(parsed),
        "source or receipt identity is incomplete",
    )
    expected = sorted(
        ordinal
        for ordinal, op in enumerate(parsed)
        if mq.attr_str(op, "prov.aten") == "aten.amax.default" and mq.op_name(op) == "linalg.reduce"
    )
    if expected:
        _selected_width(selected)
    if selected is not None:
        _selected_width(selected)
    return {
        "status": PENDING,
        "scope": SCOPE,
        "raw_source_sha256": raw_sha256,
        "normalized_source_sha256": normalized_sha256,
        "capture_receipt_sha256": receipt_sha256,
        "n_source_operations": len(parsed),
        "selected_index_observation": deepcopy(dict(selected)) if selected is not None else None,
        "expected_ordinals": expected,
        "count": 0,
        "occurrences": [],
        "occurrences_sha256": _digest([]),
        "source_verified": True,
    }


def record(
    witness: dict[str, Any],
    row: Mapping[str, Any],
    decision: Mapping[str, Any],
    parsed: tuple[Any, ...],
    source_rows: Mapping[int, Mapping[str, Any]],
) -> None:
    """Join every admitted maximum root to one reviewed exact inner declaration."""
    body = decision.get("source_body_proof")
    tag = row.get("frontend_op")
    schema = body.get("schema") if isinstance(body, Mapping) else None
    if tag != "aten.amax.default" and schema != STATIC_F32_MAXIMUM_SOURCE_BODY_SCHEMA:
        return
    _need(
        tag == "aten.amax.default"
        and row.get("mlir_operation") == "linalg.reduce"
        and schema == STATIC_F32_MAXIMUM_SOURCE_BODY_SCHEMA
        and witness.get("status") == PENDING,
        "reviewed maximum root has no exact source-body declaration",
    )
    ordinals = row.get("ordinals")
    _need(
        isinstance(ordinals, list)
        and bool(ordinals)
        and type(row.get("count")) is int
        and row["count"] == len(ordinals)
        and all(type(ordinal) is int and 0 <= ordinal < len(parsed) for ordinal in ordinals)
        and len(set(ordinals)) == len(ordinals)
        and set(ordinals) <= set(witness["expected_ordinals"])
        and len(parsed) == witness["n_source_operations"]
        and all(source_rows.get(ordinal) is row for ordinal in ordinals)
        and all(
            mq.op_name(parsed[ordinal]) == "linalg.reduce" and mq.attr_str(parsed[ordinal], "prov.aten") == tag
            for ordinal in ordinals
        ),
        "host row omits or mixes parsed maximum root ordinals",
    )
    _need(
        decision.get("status") == "admitted" and decision.get("reviewed") is True,
        "maximum root lacks a reviewed host decision",
    )
    profiles = decision.get("profiles")
    _need(isinstance(profiles, (list, tuple)), "host profile roster is malformed")
    approved = [
        item
        for item in profiles
        if isinstance(item, Mapping) and item.get("status") == "admitted" and item.get("reviewed") is True
    ]
    _need(
        len(approved) == 1
        and isinstance(approved[0].get("profile"), str)
        and bool(approved[0]["profile"])
        and is_sha256(approved[0].get("capability_spec_sha256")),
        "maximum root has no unique reviewed profile",
    )
    profile = approved[0]
    selected = witness["selected_index_observation"]
    width = _selected_width(selected)
    _need(
        isinstance(body, Mapping)
        and set(body) == _BODY_KEYS
        and body["schema"] == STATIC_F32_MAXIMUM_SOURCE_BODY_SCHEMA
        and body["operation"] == "arith.maximumf"
        and isinstance(body["declaration"], str)
        and bool(body["declaration"])
        and _same_json_value(body["selected_index_observation"], selected)
        and body["profile"] == profile["profile"]
        and body["capability_spec_sha256"] == profile["capability_spec_sha256"]
        and isinstance(body["patterns"], list)
        and len(body["patterns"]) == len(ordinals),
        "source-body proof differs from reviewed profile or selected compiler",
    )
    inner = {key: value for key, value in body.items() if key not in {"profile", "capability_spec_sha256"}}
    decisions = profile.get("decisions")
    _need(
        isinstance(decisions, list)
        and sum(
            isinstance(item, Mapping)
            and item.get("status") == "admitted"
            and item.get("declaration") == body["declaration"]
            and _same_json_value(item.get("source_body_proof"), inner)
            for item in decisions
        )
        == 1,
        "source-body proof has no unique reviewed inner declaration",
    )
    seen = {item["ordinal"] for item in witness["occurrences"]}
    _need(not seen.intersection(ordinals), "source ordinal is admitted twice")
    added = []
    for ordinal, claimed in zip(ordinals, body["patterns"], strict=True):
        pattern = _pattern(parsed[ordinal], width)
        _need(_same_json_value(claimed, pattern), "source-body proof differs from parsed ordinal")
        added.append(
            {
                "ordinal": ordinal,
                "profile": profile["profile"],
                "capability_spec_sha256": profile["capability_spec_sha256"],
                "pattern": pattern,
            }
        )
    witness["occurrences"].extend(added)
    witness["occurrences"].sort(key=lambda item: item["ordinal"])
    witness["count"] = len(witness["occurrences"])
    witness["occurrences_sha256"] = _digest(witness["occurrences"])
    witness["source_verified"] = False


def _screen(path: Path, witness: Mapping[str, Any]) -> bool:
    """Reparse once and prove the complete root roster, not a requested subset."""
    occurrences = witness.get("occurrences")
    if not isinstance(occurrences, list):
        return False
    try:
        width = _selected_width(witness.get("selected_index_observation"))
        raw = path.read_bytes()
        raw_sha = sha256(raw).hexdigest()
        normalized, receipt = normalize_capture_mlir(raw.decode("utf-8"))
        normalized_sha = sha256(normalized.encode("utf-8")).hexdigest()
        if (
            raw_sha != witness.get("raw_source_sha256")
            or normalized_sha != witness.get("normalized_source_sha256")
            or receipt.get("input_sha256") != raw_sha
            or receipt.get("output_sha256") != normalized_sha
        ):
            return False
        module = preparation.parse(path, normalized, raw=raw)
        module.verify()
        parsed = tuple(mq.walk(module))
        actual_roots = [
            ordinal
            for ordinal, op in enumerate(parsed)
            if mq.attr_str(op, "prov.aten") == "aten.amax.default" and mq.op_name(op) == "linalg.reduce"
        ]
        if (
            len(parsed) != witness.get("n_source_operations")
            or not _same_json_value(actual_roots, witness.get("expected_ordinals"))
            or not _same_json_value(actual_roots, [item["ordinal"] for item in occurrences])
        ):
            return False
        return all(
            _same_json_value(_pattern(parsed[ordinal], width), item["pattern"])
            for ordinal, item in zip(actual_roots, occurrences, strict=True)
        )
    except Exception:  # noqa: BLE001 -- xDSL parsing and verification must fail closed
        return False


def verify_source(witness: dict[str, Any], source: Path) -> None:
    _need(witness.get("status") == PENDING, "source witness is not pending")
    occurrences = witness.get("occurrences")
    _need(
        isinstance(occurrences, list) and len(occurrences) == witness.get("count"),
        "source occurrence roster is incomplete",
    )
    _need(
        _same_json_value([item["ordinal"] for item in occurrences], witness.get("expected_ordinals")),
        "an independently identified maximum root lacks reviewed admission",
    )
    if not occurrences and witness.get("selected_index_observation") is None:
        return
    _need(
        source.is_file() and not source.is_symlink() and _screen(source, witness),
        "source bytes or recomputed maximum ordinals changed",
    )
    witness["source_verified"] = True


def _captured(capture_path: Path, witness: Mapping[str, Any], linked: Mapping[str, Any]) -> bool:
    if not capture_path.is_absolute() or capture_path.is_symlink() or not capture_path.is_dir():
        return False
    model, receipt = capture_path / "model.mlir", capture_path / "capture_receipt.json"
    if any(path.is_symlink() or not path.is_file() for path in (model, receipt)):
        return False
    try:

        def pinned() -> bool:
            return bool(
                strict_tree_sha256(capture_path)["sha256"] == linked["capture_tree_sha256"]
                and file_sha256(model) == witness["raw_source_sha256"]
                and file_sha256(receipt) == witness["capture_receipt_sha256"]
            )

        return pinned() and _screen(model, witness) and pinned()
    except (OSError, KeyError, TypeError, ValueError):
        return False


def link(
    source: Mapping[str, Any],
    actual_index: Mapping[str, Any],
    linked_build: Mapping[str, Any],
    *,
    capture_path: Path | None,
) -> None:
    """Require exact selected producer width, captured tree and linked tuple."""
    witness = source.get(FIELD)
    _need(isinstance(witness, dict) and witness.get("status") == PENDING, "source roster is not pending")
    selected = witness.get("selected_index_observation")
    _need(
        witness.get("source_verified") is True
        and _same_json_value(selected, source.get("selected_index_observation"))
        and control._record_matches_selected(actual_index, selected),  # noqa: PLC2701 -- shared selected-build join
        "actual linked compiler differs from source index premise",
    )
    _need(
        isinstance(linked_build, Mapping)
        and set(linked_build) == _BUILD_KEYS
        and all(is_sha256(linked_build[key]) for key in _BUILD_KEYS),
        "linked candidate/capture/ELF tuple is incomplete",
    )
    if capture_path is not None:
        _need(
            isinstance(capture_path, Path) and _captured(capture_path, witness, linked_build),
            "selected capture tree or maximum source changed after build",
        )
    else:
        _need(not witness["occurrences"], "selected maximum root has no captured source path")
    witness["status"] = LINKED
    witness["linked_build"] = dict(linked_build)
    witness["source_capture_path"] = str(capture_path) if capture_path is not None else None


def _valid_pattern(value: object, width: int) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "axis",
        "input_shape",
        "output_shape",
        "ordered_types",
        "index_bits_premise",
    }:
        return False
    inp, out, axis = value.get("input_shape"), value.get("output_shape"), value.get("axis")
    if (
        type(inp) is not list
        or not inp
        or type(out) is not list
        or type(axis) is not int
        or not 0 <= axis < len(inp)
        or type(value.get("ordered_types")) is not list
        or value["ordered_types"] != ["f32", "f32", "f32"]
        or type(value.get("index_bits_premise")) is not int
        or value["index_bits_premise"] != width
    ):
        return False
    maximum = (1 << (width - 1)) - 1
    if any(type(dim) is not int or not 0 < dim <= maximum for dim in (*inp, *out)):
        return False
    return out == [dim for position, dim in enumerate(inp) if position != axis] and all(
        prod(shape) <= maximum for shape in (inp, out)
    )


def linked_complete(source: Mapping[str, Any], entry: Mapping[str, Any], candidate_sha256: str) -> bool:
    """Reject old/omitted, malformed or differently linked source evidence."""
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
        or not is_sha256(witness.get("capture_receipt_sha256"))
        or witness["raw_source_sha256"] != source.get("source_sha256")
        or witness["normalized_source_sha256"] != source.get("normalized_source_sha256")
        or witness["capture_receipt_sha256"] != source.get("capture_receipt_sha256")
        or source.get("source_sha256") != entry.get("source_sha256")
        or type(witness.get("n_source_operations")) is not int
        or witness["n_source_operations"] < 1
        or witness["n_source_operations"] != source.get("n_source_operations")
        or not isinstance(witness.get("occurrences"), list)
        or not isinstance(witness.get("expected_ordinals"), list)
        or type(witness.get("count")) is not int
        or witness["count"] != len(witness["occurrences"])
    ):
        return False
    try:
        if _digest(witness["occurrences"]) != witness.get("occurrences_sha256"):
            return False
        width = _selected_width(witness.get("selected_index_observation"))
    except (ValueError, TypeError):
        return False
    selected = witness["selected_index_observation"]
    if not _same_json_value(selected, source.get("selected_index_observation")) or not control._record_matches_selected(  # noqa: PLC2701
        entry.get("index_lowering"), selected
    ):
        return False
    ordinals = []
    for item in witness["occurrences"]:
        if (
            not isinstance(item, Mapping)
            or set(item) != _OCCURRENCE
            or type(item.get("ordinal")) is not int
            or not 0 <= item["ordinal"] < witness["n_source_operations"]
            or not isinstance(item.get("profile"), str)
            or not item["profile"]
            or not is_sha256(item.get("capability_spec_sha256"))
            or not _valid_pattern(item.get("pattern"), width)
        ):
            return False
        ordinals.append(item["ordinal"])
    if ordinals != sorted(set(ordinals)):
        return False
    if not _same_json_value(ordinals, witness["expected_ordinals"]):
        return False
    linked = witness.get("linked_build")
    if not (
        isinstance(linked, Mapping)
        and set(linked) == _BUILD_KEYS
        and all(is_sha256(linked[key]) for key in _BUILD_KEYS)
        and linked["candidate_tree_sha256"] == candidate_sha256
        and linked["capture_tree_sha256"] == entry.get("capture_tree_sha256")
        and linked["elf_sha256"] == entry.get("elf_sha256")
    ):
        return False
    capture = witness.get("source_capture_path")
    return type(capture) is str and _captured(Path(capture), witness, linked)
