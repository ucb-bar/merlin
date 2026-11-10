"""Revalidate completed private full-model build evidence and its exact roster.

This admission remains separate from numerical execution and paper accuracy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.development.backend.feedback import private_compilation_inputs as compilation_support
from merlin.development.backend.feedback import private_source_freeze as source_freeze_api
from merlin.development.backend.feedback.private_source_support_join import (
    linked_source_support_complete as _linked_source_support_complete,
)

RESULT_SCHEMA = "merlin.phase1.private_full_model_build_gate.v15"


BUILD_BOARD_SCOPE = "static_memory_layout_and_host_ISA_only; no board execution"


def _complete_capture_identity(checks: Mapping[str, Any]) -> bool:
    identity = checks.get("capture_identity")
    if not isinstance(identity, Mapping) or set(identity) != {
        "compiler_strict_tree_sha256",
        "issuer_sealed_tree_sha256",
    }:
        return False
    return all(
        isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)
        for value in identity.values()
    )


def complete(
    record: Mapping[str, Any] | None,
    *,
    required_models: Sequence[str],
    required_programs: Mapping[str, Sequence[str]],
    candidate_sha256: str,
) -> bool:
    """Revalidate the claim-bearing roster and exact current submission identity."""
    if (
        not isinstance(record, Mapping)
        or record.get("schema") != RESULT_SCHEMA
        or record.get("passed") is not True
        or record.get("mode") in {"prebuilt_diagnostic", "canonical_model_partition"}
    ):
        return False
    if record.get("full_model_numerical_equivalence") != "not_run" or record.get("paper_accuracy") != "not_claimed":
        return False
    if not source_freeze_api.valid_binding(record.get("authored_source_freeze"), record.get("private_spec_sha256")):
        return False
    rows = record.get("models")
    if not isinstance(rows, list) or len(rows) != len(required_models):
        return False
    if any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("checks"), Mapping)
        or not _complete_capture_identity(row["checks"])
        or not isinstance(row["checks"].get("input_provenance"), Mapping)
        or set(row["checks"]["input_provenance"]) != set(required_programs.get(row.get("model"), ()))
        or any(
            not isinstance(provenance, Mapping)
            or type(provenance.get("paper_ready")) not in {bool, type(None)}
            or type(provenance.get("synthetic_inputs")) not in {bool, type(None)}
            or not isinstance(provenance.get("meta_sha256"), str)
            or len(provenance["meta_sha256"]) != 64
            or provenance.get("scope") != "input provenance only; no paper accuracy or full-model numerical result"
            for provenance in row["checks"]["input_provenance"].values()
        )
        or not isinstance(row["checks"].get("build"), Mapping)
        or not _linked_source_support_complete(
            row["checks"], required_programs.get(row.get("model"), ()), candidate_sha256
        )
        or not isinstance(row["checks"].get("recipe_derivation"), Mapping)
        or row["checks"]["recipe_derivation"].get("scope")
        != "current-spec diagnostic recipe derivation only; no model or hardware admission"
        or any(
            not isinstance(row["checks"]["recipe_derivation"].get(key), str)
            or len(row["checks"]["recipe_derivation"][key]) != 64
            for key in ("evidence_manifest_sha256", "recipe_sha256", "recipe_semantic_sha256")
        )
        for row in rows
    ):
        return False
    return (
        record.get("candidate_tree_sha256") == candidate_sha256
        and record.get("required_models") == list(required_models)
        and record.get("required_programs") == {name: list(required_programs[name]) for name in required_models}
        and {row.get("model") for row in rows if isinstance(row, Mapping) and row.get("status") == "pass"}
        == set(required_models)
        and all(
            (row.get("checks") or {}).get("build", {}).get("candidate_tree_sha256") == candidate_sha256
            and (row.get("checks") or {}).get("build", {}).get("status") == "capture_lower_codegen_link_verified"
            and isinstance((row.get("checks") or {}).get("build", {}).get("static_build_board"), str)
            and all(
                isinstance((row.get("checks") or {}).get("build", {}).get(key), str)
                and Path((row.get("checks") or {}).get("build", {})[key]).is_absolute()
                for key in ("static_build_board_catalog_source", "static_build_board_dts_source")
            )
            and (row.get("checks") or {}).get("build", {}).get("static_build_board_scope") == BUILD_BOARD_SCOPE
            and isinstance((row.get("checks") or {}).get("build", {}).get("accelerator_rtl_config"), str)
            and bool((row.get("checks") or {}).get("build", {}).get("accelerator_rtl_config"))
            and all(
                isinstance((row.get("checks") or {}).get("build", {}).get(key), str)
                and len((row.get("checks") or {}).get("build", {})[key]) == 64
                for key in (
                    "static_build_board_catalog_sha256",
                    "static_build_board_dts_sha256",
                    "accelerator_rtl_facts_sha256",
                )
            )
            and type((row.get("checks") or {}).get("build", {}).get("linked_device_groups")) is int
            and (row.get("checks") or {}).get("build", {}).get("linked_device_groups") > 0
            and isinstance((row.get("checks") or {}).get("build", {}).get("programs"), list)
            and (row.get("checks") or {}).get("build", {}).get("linked_device_groups")
            == sum(
                entry["linked_device_groups"]
                for entry in (row.get("checks") or {}).get("build", {}).get("programs", [])
                if isinstance(entry, Mapping) and type(entry.get("linked_device_groups")) is int
            )
            # Every program states its own linked count: one that omits it is not summed as zero.
            and all(
                isinstance(entry, Mapping)
                and type(entry.get("linked_device_groups")) is int
                and compilation_support.complete(entry)
                for entry in (row.get("checks") or {}).get("build", {}).get("programs", [])
            )
            and [entry.get("program") for entry in (row.get("checks") or {}).get("build", {}).get("programs", [])]
            == list(required_programs[row["model"]])
            and all(
                entry.get("status") == "capture_lower_codegen_link_verified"
                and entry.get("candidate_tree_sha256") == candidate_sha256
                and isinstance(entry.get("elf_sha256"), str)
                and len(entry["elf_sha256"]) == 64
                and type(entry.get("linked_device_groups")) is int
                and isinstance(entry.get("static_host_compute_audit"), list)
                and len(entry["static_host_compute_audit"]) == entry["linked_device_groups"]
                and all(
                    isinstance(audit, Mapping)
                    and audit.get("verdict") == "clean_static_host_compute_audit"
                    and isinstance(audit.get("artifact_sha256"), str)
                    and len(audit["artifact_sha256"]) == 64
                    and isinstance(audit.get("object_sha256"), str)
                    and len(audit["object_sha256"]) == 64
                    and isinstance(audit.get("audit"), Mapping)
                    and isinstance(audit["audit"].get("budget"), Mapping)
                    and isinstance(audit["audit"].get("groups"), list)
                    and bool(audit["audit"]["groups"])
                    and all(group.get("verdict") == "clean" for group in audit["audit"]["groups"])
                    for audit in entry["static_host_compute_audit"]
                )
                for entry in (row.get("checks") or {}).get("build", {}).get("programs", [])
            )
            for row in rows
        )
    )
