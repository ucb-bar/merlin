"""Mandatory build-only host witness for independently prepared tensor-index forms.

This consumes the grant-none receipt/trace/source proof, requires separate exact
reviewed host admissions for every compute root, and binds the actual linked
image. It does not prove association between prepared nodes, original frontend
equivalence, compiled bounds, or numerical behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

from merlin.common import mlir_query as mq
from merlin.common.digest import is_sha256
from merlin.common.jsonio import canonical_json
from merlin.development.backend.feedback import private_control_support as control
from merlin.development.backend.feedback import private_index_source as source_api
from merlin.frontends.prepared_index_source_body import (
    SCHEMA as BODY_SCHEMA,
)
from merlin.frontends.prepared_index_source_body import (
    _record_role,
    _same_json,
    _selected_width,
    validate_source_record,
)
from merlin.runtime.modelinputs import file_sha256, strict_tree_sha256

FIELD = "prepared_index_host_support"
PENDING = "source_prepared_index_host_pending_build"
LINKED = "source_prepared_index_host_linked"
SCOPE = "reviewed prepared index compute roots and exact linked build; no original/compiled numerical equivalence"
_BUILD_KEYS = {"candidate_tree_sha256", "capture_tree_sha256", "elf_sha256"}
_SOURCE_KEYS = {
    "status",
    "scope",
    "raw_source_sha256",
    "normalized_source_sha256",
    "capture_receipt_sha256",
    "n_source_operations",
    "selected_index_observation",
    "source_proof",
    "source_proof_sha256",
    "expected_ordinals",
    "admissions",
    "source_verified",
}


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(f"prepared index host support: {reason}")


def _digest(value: object) -> str:
    return sha256(canonical_json(value)).hexdigest()


def _expected(proof: Mapping[str, Any]) -> list[int]:
    return sorted(ordinal for record in proof["records"] for ordinal in record["compute_ordinals"])


def _role(proof: Mapping[str, Any], ordinal: int) -> str | None:
    roles = [_record_role(record, ordinal) for record in proof["records"] if ordinal in record["compute_ordinals"]]
    return roles[0] if len(roles) == 1 else None


def begin(
    capture: Path,
    parsed: tuple[Any, ...],
    raw_sha256: str,
    normalized_sha256: str,
    capture_receipt_sha256: str,
    selected: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Independently prove the complete index compute-root roster before admission."""
    actual = sorted(
        ordinal
        for ordinal, op in enumerate(parsed)
        if mq.attr_str(op, "prov.aten") == "aten.index.Tensor" and mq.op_name(op) in {"linalg.generic", "linalg.reduce"}
    )
    _need(
        len(parsed) > 0 and all(is_sha256(value) for value in (raw_sha256, normalized_sha256, capture_receipt_sha256)),
        "capture source identity is incomplete",
    )
    proof = None
    if actual:
        width = _selected_width(selected)
        _need(width is not None, "index roots need a selected producer-owned compiler width")
        proof = source_api.prove_index_source(capture, index_bits=width)
        validate_source_record(proof, selected=selected)
        _need(
            proof["raw_source_sha256"] == raw_sha256
            and proof["normalized_source_sha256"] == normalized_sha256
            and proof["capture_receipt_sha256"] == capture_receipt_sha256
            and proof["n_source_operations"] == len(parsed)
            and _expected(proof) == actual,
            "independent prepared source proof omits or changes an index root",
        )
    return {
        "status": PENDING,
        "scope": SCOPE,
        "raw_source_sha256": raw_sha256,
        "normalized_source_sha256": normalized_sha256,
        "capture_receipt_sha256": capture_receipt_sha256,
        "n_source_operations": len(parsed),
        "selected_index_observation": deepcopy(dict(selected)) if selected is not None else None,
        "source_proof": proof,
        "source_proof_sha256": _digest(proof) if proof is not None else None,
        "expected_ordinals": actual,
        "admissions": [],
        "source_verified": not actual,
    }


def record(
    witness: dict[str, Any],
    row: Mapping[str, Any],
    host_decision: Mapping[str, Any],
    parsed: tuple[Any, ...],
    source_rows: Mapping[int, Mapping[str, Any]],
) -> None:
    """Join each reviewed source-body admission to the independently proved role."""
    ordinals = row.get("ordinals")
    _need(type(ordinals) is list and all(type(item) is int for item in ordinals), "host row ordinals are malformed")
    expected = set(witness["expected_ordinals"])
    if not expected.intersection(ordinals):
        _need(
            not (
                row.get("frontend_op") == "aten.index.Tensor"
                and row.get("mlir_operation") in {"linalg.generic", "linalg.reduce"}
            ),
            "index compute row has no independent prepared source proof",
        )
        return
    _need(
        witness.get("status") == PENDING
        and row.get("frontend_op") == "aten.index.Tensor"
        and type(row.get("count")) is int
        and row["count"] == len(ordinals)
        and bool(ordinals)
        and len(set(ordinals)) == len(ordinals)
        and set(ordinals) <= expected
        and len(parsed) == witness["n_source_operations"]
        and all(
            mq.op_name(parsed[ordinal]) == row.get("mlir_operation")
            and mq.attr_str(parsed[ordinal], "prov.aten") == "aten.index.Tensor"
            for ordinal in ordinals
        )
        and all(source_rows.get(ordinal) is row for ordinal in ordinals),
        "reviewed host row omits or mixes prepared index ordinals",
    )
    _need(
        host_decision.get("status") == "admitted" and host_decision.get("reviewed") is True,
        "prepared index row has no exact reviewed host admission",
    )
    body = host_decision.get("source_body_proof")
    proof = witness["source_proof"]
    _need(type(body) is dict and proof is not None, "reviewed index admission has no source-body proof")
    roles = [{"ordinal": ordinal, "role": _role(proof, ordinal)} for ordinal in ordinals]
    _need(
        set(body)
        == {
            "schema",
            "declaration",
            "operation",
            "selected_index_observation",
            "source_record_sha256",
            "roles",
            "profile",
            "capability_spec_sha256",
        }
        and body["schema"] == BODY_SCHEMA
        and type(body["declaration"]) is str
        and body["declaration"]
        and body["operation"] == roles[0]["role"]
        and all(item["role"] == body["operation"] for item in roles)
        and _same_json(body["roles"], roles)
        and _same_json(body["selected_index_observation"], witness["selected_index_observation"])
        and body["source_record_sha256"] == witness["source_proof_sha256"]
        and is_sha256(body["capability_spec_sha256"]),
        "reviewed source-body declaration/role/context differs from independent proof",
    )
    profiles = host_decision.get("profiles")
    _need(type(profiles) is list, "reviewed host profile roster is malformed")
    selected_profiles = [
        item
        for item in profiles
        if isinstance(item, Mapping) and item.get("status") == "admitted" and item.get("reviewed") is True
    ]
    _need(len(selected_profiles) == 1, "prepared index row has no unique reviewed host profile")
    profile = selected_profiles[0]
    _need(
        body["profile"] == profile.get("profile")
        and body["capability_spec_sha256"] == profile.get("capability_spec_sha256"),
        "prepared index profile identity differs",
    )
    inner = {key: value for key, value in body.items() if key not in {"profile", "capability_spec_sha256"}}
    decisions = profile.get("decisions")
    _need(
        type(decisions) is list
        and sum(
            isinstance(item, Mapping)
            and item.get("status") == "admitted"
            and item.get("declaration") == body["declaration"]
            and _same_json(item.get("source_body_proof"), inner)
            for item in decisions
        )
        == 1,
        "prepared index source has no unique reviewed declaration",
    )
    seen = {item["ordinal"] for item in witness["admissions"]}
    _need(not seen.intersection(ordinals), "prepared index root was admitted twice")
    witness["admissions"].extend(
        {
            "ordinal": ordinal,
            "role": body["operation"],
            "profile": body["profile"],
            "capability_spec_sha256": body["capability_spec_sha256"],
        }
        for ordinal in ordinals
    )
    witness["admissions"].sort(key=lambda item: item["ordinal"])


def verify_source(witness: dict[str, Any], capture: Path) -> None:
    """Re-read captured source, trace and receipt before accepting a source roster."""
    _need(witness.get("status") == PENDING, "source witness is not pending")
    proof = witness.get("source_proof")
    if proof is not None:
        _need(_digest(proof) == witness.get("source_proof_sha256"), "prepared proof bytes changed")
        source_api.verify_index_source_record(capture, proof)
    _need(
        sorted(item["ordinal"] for item in witness["admissions"]) == witness["expected_ordinals"],
        "reviewed host admissions omit a prepared index root",
    )
    witness["source_verified"] = True


def _capture_matches(
    witness: Mapping[str, Any],
    source: Mapping[str, Any],
    linked_build: Mapping[str, str],
    capture_path: Path | None,
) -> bool:
    """Reread the selected capture before and after independent source reproof."""
    proof = witness.get("source_proof")
    if proof is None:
        return True
    if capture_path is None or not capture_path.is_absolute() or capture_path.is_symlink():
        return False
    model = capture_path / "model.mlir"
    receipt = capture_path / "capture_receipt.json"
    if not model.is_file() or model.is_symlink() or not receipt.is_file() or receipt.is_symlink():
        return False
    if not isinstance(proof, dict):
        return False

    def pinned() -> bool:
        return bool(
            strict_tree_sha256(capture_path)["sha256"] == linked_build.get("capture_tree_sha256")
            and file_sha256(model)
            == witness.get("raw_source_sha256")
            == source.get("source_sha256")
            == proof.get("raw_source_sha256")
            and file_sha256(receipt)
            == witness.get("capture_receipt_sha256")
            == source.get("capture_receipt_sha256")
            == proof.get("capture_receipt_sha256")
            and witness.get("normalized_source_sha256")
            == source.get("normalized_source_sha256")
            == proof.get("normalized_source_sha256")
            and witness.get("n_source_operations")
            == source.get("n_source_operations")
            == proof.get("n_source_operations")
            and _digest(proof) == witness.get("source_proof_sha256")
        )

    try:
        if not pinned():
            return False
        source_api.verify_index_source_record(capture_path, proof)
        return pinned()
    except (OSError, ValueError, TypeError, KeyError):
        return False


def link(
    source: Mapping[str, Any],
    actual_index: Mapping[str, Any],
    linked_build: Mapping[str, str],
    *,
    capture_path: Path | None = None,
) -> None:
    witness = source.get(FIELD)
    _need(
        type(witness) is dict and witness.get("status") == PENDING and witness.get("source_verified") is True,
        "source witness is not independently verified",
    )
    _need(
        witness["raw_source_sha256"] == source.get("source_sha256")
        and witness["normalized_source_sha256"] == source.get("normalized_source_sha256")
        and witness["capture_receipt_sha256"] == source.get("capture_receipt_sha256")
        and _same_json(witness["selected_index_observation"], source.get("selected_index_observation"))
        and control._record_matches_selected(actual_index, witness["selected_index_observation"]),  # noqa: PLC2701
        "source, receipt or actual compiler index observation differs",
    )
    _need(
        type(linked_build) is dict
        and set(linked_build) == _BUILD_KEYS
        and all(is_sha256(linked_build[key]) for key in _BUILD_KEYS),
        "candidate/capture/ELF linked tuple is incomplete",
    )
    _need(
        _capture_matches(witness, source, linked_build, capture_path),
        "selected capture/source/trace/receipt differs from the linked image",
    )
    witness["status"] = LINKED
    witness["linked_build"] = dict(linked_build)
    witness["source_capture_path"] = str(capture_path) if witness["source_proof"] is not None else None


def linked_complete(source: Mapping[str, Any], entry: Mapping[str, Any], candidate_sha256: str) -> bool:
    witness = source.get(FIELD)
    if not isinstance(witness, Mapping) or set(witness) != _SOURCE_KEYS | {"linked_build", "source_capture_path"}:
        return False
    proof = witness.get("source_proof")
    selected = witness.get("selected_index_observation")
    try:
        if proof is not None:
            validate_source_record(proof, selected=selected)
        expected = _expected(proof) if proof is not None else []
    except (ValueError, TypeError, KeyError):
        return False
    admissions = witness.get("admissions")
    linked = witness.get("linked_build")
    if (
        witness.get("status") != LINKED
        or witness.get("scope") != SCOPE
        or witness.get("source_verified") is not True
        or not all(
            is_sha256(witness.get(key))
            for key in ("raw_source_sha256", "normalized_source_sha256", "capture_receipt_sha256")
        )
        or witness["raw_source_sha256"] != source.get("source_sha256")
        or witness["normalized_source_sha256"] != source.get("normalized_source_sha256")
        or witness["capture_receipt_sha256"] != source.get("capture_receipt_sha256")
        or entry.get("source_sha256") != source.get("source_sha256")
        or type(witness.get("n_source_operations")) is not int
        or witness["n_source_operations"] < 1
        or witness["n_source_operations"] != source.get("n_source_operations")
        or not _same_json(selected, source.get("selected_index_observation"))
        or witness.get("source_proof_sha256") != (_digest(proof) if proof is not None else None)
        or proof is not None
        and (
            proof["raw_source_sha256"] != witness["raw_source_sha256"]
            or proof["normalized_source_sha256"] != witness["normalized_source_sha256"]
            or proof["capture_receipt_sha256"] != witness["capture_receipt_sha256"]
            or proof["n_source_operations"] != witness["n_source_operations"]
        )
        or type(witness.get("expected_ordinals")) is not list
        or witness["expected_ordinals"] != expected
        or type(admissions) is not list
        or any(type(item) is not dict for item in admissions)
        or [item.get("ordinal") for item in admissions] != expected
        or not isinstance(linked, Mapping)
        or set(linked) != _BUILD_KEYS
        or not all(is_sha256(linked[key]) for key in _BUILD_KEYS)
        or linked
        != {
            "candidate_tree_sha256": candidate_sha256,
            "capture_tree_sha256": entry.get("capture_tree_sha256"),
            "elf_sha256": entry.get("elf_sha256"),
        }
        or not control._record_matches_selected(entry.get("index_lowering"), selected)  # noqa: PLC2701
    ):
        return False
    captured = witness.get("source_capture_path")
    if proof is not None:
        if type(captured) is not str or not _capture_matches(witness, source, linked, Path(captured)):
            return False
    elif captured is not None:
        return False
    return all(
        type(item) is dict
        and set(item) == {"ordinal", "role", "profile", "capability_spec_sha256"}
        and type(item["ordinal"]) is int
        and item["role"] == _role(proof, item["ordinal"])
        and type(item["profile"]) is str
        and bool(item["profile"])
        and is_sha256(item["capability_spec_sha256"])
        for item in admissions
    )
