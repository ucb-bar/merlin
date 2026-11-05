"""Mandatory linked-build accounting for reviewed host i64 arange admissions.

The source witness is prepared-IR only. A reviewed host declaration remains
necessary, and this record never proves a numerical or frontend equivalence.
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
from merlin_experiments.phase1.feedback import private_literal_arange as source_witness

PENDING = "source_literal_i64_arange_host_pending_build"
LINKED = "source_literal_i64_arange_host_linked"
SCOPE = "reviewed host arange source ordinals and exact linked whole-program bytes; no numerical equivalence"
FIELD = "literal_arange_host_support"
_OP = "aten.arange.start_step"
_OP_PREFIX = "aten.arange."
_BUILD_KEYS = {"candidate_tree_sha256", "capture_tree_sha256", "elf_sha256"}


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(f"literal i64 arange host support: {reason}")


def begin(raw_sha256: str, normalized_sha256: str, n_operations: int, selected: Mapping[str, Any] | None) -> dict:
    """Create a mandatory empty roster; do not admit an absent declaration."""
    _need(
        is_sha256(raw_sha256)
        and is_sha256(normalized_sha256)
        and type(n_operations) is int
        and n_operations > 0
        and (selected is None or isinstance(selected, Mapping)),
        "source identity or selected index observation is malformed",
    )
    return {
        "status": PENDING,
        "scope": SCOPE,
        "raw_source_sha256": raw_sha256,
        "normalized_source_sha256": normalized_sha256,
        "n_source_operations": n_operations,
        "selected_index_observation": deepcopy(dict(selected)) if selected is not None else None,
        "source_proof": None,
        "source_proof_sha256": None,
        "admissions": [],
    }


def record(
    witness: dict,
    capture: Path,
    row: Mapping[str, Any],
    host_decision: Mapping[str, Any],
    parsed: tuple[Any, ...],
    source_rows: Mapping[int, Mapping[str, Any]],
) -> None:
    """Require every actual host-admitted arange occurrence to have a closed source proof."""
    ordinals = row.get("ordinals")
    _need(
        isinstance(ordinals, list)
        and type(row.get("count")) is int
        and row["count"] == len(ordinals)
        and all(type(value) is int and 0 <= value < len(parsed) for value in ordinals),
        "host source row has no exact parsed ordinal roster",
    )
    frontend = row.get("frontend_op")
    marked = (isinstance(frontend, str) and frontend.startswith(_OP_PREFIX)) or any(
        isinstance(tag := mq.attr_str(parsed[value], "prov.aten"), str) and tag.startswith(_OP_PREFIX)
        for value in ordinals
    )
    if not marked:
        return
    _need(
        not isinstance(frontend, str) or not frontend.startswith(_OP_PREFIX) or frontend == _OP,
        "host range uses an unproved frontend form",
    )
    _need(
        host_decision.get("status") == "admitted" and host_decision.get("reviewed") is True,
        "arange has no reviewed host declaration",
    )
    selected = witness["selected_index_observation"]
    _need(
        isinstance(selected, Mapping)
        and selected.get("schema") == "merlin.selected-index-lowering.v1"
        and type(selected.get("index_bits")) is int,
        "arange has no selected producer-owned index width",
    )
    proof = witness["source_proof"]
    if proof is None:
        proof = source_witness.prove_literal_arange_source(capture, index_bits=selected["index_bits"])
        _need(
            proof.get("status") == source_witness.PENDING
            and proof.get("raw_source_sha256") == witness["raw_source_sha256"]
            and proof.get("normalized_source_sha256") == witness["normalized_source_sha256"],
            "captured arange proof differs from the selected source",
        )
        witness["source_proof"] = proof
        witness["source_proof_sha256"] = sha256(
            json.dumps(proof, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    allowed = {value for item in proof["occurrences"] for value in item["lowering_ordinals"]}
    seen = {value for item in witness["admissions"] for value in item["ordinals"]}
    _need(
        bool(ordinals)
        and len(ordinals) == len(set(ordinals))
        and set(ordinals) <= allowed
        and not seen.intersection(ordinals)
        and all(source_rows.get(value) is row for value in ordinals),
        "reviewed arange row has an omitted, repeated or unrelated source ordinal",
    )
    profiles = [
        profile
        for profile in host_decision.get("profiles", ())
        if isinstance(profile, Mapping) and profile.get("status") == "admitted" and profile.get("reviewed") is True
    ]
    _need(bool(profiles), "reviewed host profile is absent")
    profile = profiles[0]
    _need(
        isinstance(profile.get("profile"), str)
        and bool(profile["profile"])
        and is_sha256(profile.get("capability_spec_sha256")),
        "reviewed host profile has no selected capability identity",
    )
    witness["admissions"].append(
        {
            "ordinals": sorted(ordinals),
            "profile": profile["profile"],
            "capability_spec_sha256": profile["capability_spec_sha256"],
        }
    )
    witness["admissions"].sort(key=lambda item: item["ordinals"][0])


def link(source: Mapping[str, Any], actual_index: Mapping[str, Any], linked_build: Mapping[str, Any]) -> None:
    """Call only after the shared control owner joins the actual compiler index observation."""
    witness = source.get(FIELD)
    _need(isinstance(witness, dict) and witness.get("status") == PENDING, "source roster is not pending")
    proof = witness["source_proof"]
    if proof is not None:
        selected = witness["selected_index_observation"]
        _need(
            selected == source.get("selected_index_observation")
            and actual_index.get("index_bits") == selected.get("index_bits") == proof.get("index_bits_premise")
            and proof.get("capture_receipt_sha256") == source.get("capture_receipt_sha256"),
            "actual linked index width or capture receipt differs from the source proof",
        )
    _need(
        set(linked_build) == _BUILD_KEYS and all(is_sha256(linked_build[key]) for key in _BUILD_KEYS),
        "linked program identity is incomplete",
    )
    witness["status"] = LINKED
    witness["linked_build"] = dict(linked_build)


def linked_complete(source: Mapping[str, Any], entry: Mapping[str, Any], candidate_sha256: str) -> bool:
    """Refuse old, omitted or altered source/host/linked-build evidence."""
    witness = source.get(FIELD)
    if (
        not isinstance(witness, Mapping)
        or set(witness)
        != {
            "status",
            "scope",
            "raw_source_sha256",
            "normalized_source_sha256",
            "n_source_operations",
            "selected_index_observation",
            "source_proof",
            "source_proof_sha256",
            "admissions",
            "linked_build",
        }
        or witness.get("status") != LINKED
        or witness.get("scope") != SCOPE
    ):
        return False
    selected, proof, admissions, linked = (
        witness["selected_index_observation"],
        witness["source_proof"],
        witness["admissions"],
        witness["linked_build"],
    )
    if (
        witness.get("raw_source_sha256") != source.get("source_sha256")
        or witness.get("normalized_source_sha256") != source.get("normalized_source_sha256")
        or not is_sha256(witness.get("raw_source_sha256"))
        or not is_sha256(witness.get("normalized_source_sha256"))
        or entry.get("source_sha256") != source.get("source_sha256")
        or type(witness.get("n_source_operations")) is not int
        or witness["n_source_operations"] < 1
        or witness["n_source_operations"] != source.get("n_source_operations")
        or selected != source.get("selected_index_observation")
        or not isinstance(admissions, list)
        or not isinstance(linked, Mapping)
        or set(linked) != _BUILD_KEYS
        or any(not is_sha256(linked[key]) for key in _BUILD_KEYS)
        or linked["candidate_tree_sha256"] != candidate_sha256
        or linked["capture_tree_sha256"] != entry.get("capture_tree_sha256")
        or linked["elf_sha256"] != entry.get("elf_sha256")
    ):
        return False
    if proof is None:
        return not admissions and witness.get("source_proof_sha256") is None
    try:
        observed_proof_sha256 = sha256(json.dumps(proof, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    except (TypeError, ValueError):
        return False
    if not is_sha256(witness.get("source_proof_sha256")) or witness["source_proof_sha256"] != observed_proof_sha256:
        return False
    if (
        not isinstance(proof, Mapping)
        or set(proof)
        != {
            "status",
            "scope",
            "index_bits_premise",
            "raw_source_sha256",
            "normalized_source_sha256",
            "frontend_trace_sha256",
            "frontend_trace_status",
            "source_correspondence_scope",
            "capture_receipt_sha256",
            "count",
            "occurrences",
        }
        or proof.get("status") != source_witness.PENDING
        or proof.get("scope") != source_witness.SCOPE
        or proof.get("raw_source_sha256") != witness["raw_source_sha256"]
        or proof.get("normalized_source_sha256") != witness["normalized_source_sha256"]
        or proof.get("capture_receipt_sha256") != source.get("capture_receipt_sha256")
        or proof.get("frontend_trace_status") not in {"complete", "diagnostic"}
        or proof.get("source_correspondence_scope") != "all captured prepared i64 literal ranges"
        or not is_sha256(proof.get("frontend_trace_sha256"))
        or not isinstance(selected, Mapping)
        or type(selected.get("index_bits")) is not int
        or not 2 <= selected["index_bits"] <= 128
        or proof.get("index_bits_premise") != selected.get("index_bits")
        or (entry.get("index_lowering") or {}).get("index_bits") != selected.get("index_bits")
        or not isinstance(proof.get("occurrences"), list)
        or type(proof.get("count")) is not int
        or proof["count"] != len(proof["occurrences"])
        or not admissions
    ):
        return False
    allowed, generics = set(), set()
    for item in proof["occurrences"]:
        if not isinstance(item, Mapping) or set(item) != {
            "source_ordinal",
            "source_node_id",
            "original_node_id",
            "original_target",
            "original_to_prepared_equivalence",
            "extent",
            "literal_sha256",
            "lowering_ordinals",
        }:
            return False
        ordinals = item["lowering_ordinals"]
        if (
            not isinstance(ordinals, list)
            or len(ordinals) != 9
            or any(type(value) is not int or not 0 <= value < witness["n_source_operations"] for value in ordinals)
            or ordinals != sorted(ordinals)
            or len(set(ordinals)) != len(ordinals)
            or allowed.intersection(ordinals)
            or type(item.get("source_ordinal")) is not int
            or item["source_ordinal"] != ordinals[1]
            or not is_sha256(item.get("literal_sha256"))
            or item.get("original_to_prepared_equivalence") != "not_proved"
            or type(item.get("extent")) is not int
            or not 0 <= item["extent"] <= min((1 << (selected["index_bits"] - 1)) - 1, (1 << 63) - 1)
            or any(
                not isinstance(item.get(key), str) or not item[key]
                for key in ("source_node_id", "original_node_id", "original_target")
            )
        ):
            return False
        allowed.update(ordinals)
        generics.add(ordinals[1])
    admitted = []
    for item in admissions:
        if not isinstance(item, Mapping) or set(item) != {"ordinals", "profile", "capability_spec_sha256"}:
            return False
        ordinals = item["ordinals"]
        if (
            not isinstance(ordinals, list)
            or not ordinals
            or ordinals != sorted(set(ordinals))
            or any(type(value) is not int or value not in allowed for value in ordinals)
            or not isinstance(item.get("profile"), str)
            or not item["profile"]
            or not is_sha256(item.get("capability_spec_sha256"))
        ):
            return False
        admitted.extend(ordinals)
    # The source proof enumerates *all* prepared literal ranges in this program.
    # A partial host roster cannot certify the omitted range merely because one
    # sibling range was admitted. Mixed accelerator/host range routing currently
    # refuses conservatively until an independently bound partition exists.
    return len(admitted) == len(set(admitted)) and generics == set(admitted).intersection(generics)
