"""Stored mechanism evidence admission under the checkpoint's existing policy.

These readers verify retained local evidence without requiring its original host
source files. They do not reconstruct live mechanism authority or impose fresh
work-order authoring rules on historical checkpoint records.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import broker_evidence as BE
from . import contracts as P2_CONTRACTS


def verify_checkpoint_catalog(
    root: Path, binding: Mapping[str, Any] | None, authority: Mapping[str, Any] | None
) -> None:
    """Verify a checkpoint's local frozen catalog without depending on its original host path."""
    if binding is None:
        return
    from merlin.perf.compiler_edit_scope import validate_mechanism_catalog

    if not isinstance(authority, Mapping) or not isinstance(binding, Mapping):
        raise ValueError("checkpoint mechanism catalog has no frozen edit authority")
    body = {key: value for key, value in binding.items() if key != "sha256"}
    frozen = root / "compiler_mechanism_catalog.json"
    receipt = root / "compiler_mechanism_catalog_receipt.json"
    initial = root / "edit_scope_seed"
    catalog = binding.get("catalog")
    frozen_path = binding.get("frozen_path")
    if (
        binding.get("schema") != "host_frozen_compiler_mechanism_catalog_v1"
        or binding.get("sha256") != P2_CONTRACTS.document_sha256(body)
        or binding.get("contract_document_sha256") != authority.get("contract_document_sha256")
        or binding.get("initial_candidate_sha256") != authority.get("initial_candidate_sha256")
        or not isinstance(frozen_path, str)
        or not Path(frozen_path).is_absolute()
        or Path(frozen_path).resolve() != frozen.resolve()
        or frozen.is_symlink()
        or not frozen.is_file()
        or frozen.stat().st_mode & 0o222
        or P2_CONTRACTS.sha256_file(frozen) != binding.get("canonical_bytes_sha256")
        or not isinstance(catalog, Mapping)
        or P2_CONTRACTS.mapping_file(frozen) != catalog
        or P2_CONTRACTS.document_sha256(catalog) != binding.get("catalog_document_sha256")
        or receipt.is_symlink()
        or not receipt.is_file()
        or receipt.stat().st_mode & 0o222
        or P2_CONTRACTS.mapping_file(receipt) != binding
    ):
        raise ValueError("checkpoint compiler mechanism catalog changed")
    validate_mechanism_catalog(catalog, initial, authority["contract"])


def verify_checkpoint_work_order(
    root: Path,
    binding: Mapping[str, Any] | None,
    analysis_binding: Mapping[str, Any] | None,
    catalog_binding: Mapping[str, Any] | None,
    authority: Mapping[str, Any] | None,
    portfolio: Mapping[str, Any] | None,
) -> None:
    """Verify portable local work-order bytes and their exact initial-analysis receipt."""
    if binding is None:
        if analysis_binding is not None:
            raise ValueError("checkpoint has work-order analysis without a work order")
        return
    if not all(isinstance(value, Mapping) for value in (binding, catalog_binding, authority, portfolio)):
        raise ValueError("checkpoint mechanism work order lacks its frozen identities")
    body = {key: value for key, value in binding.items() if key != "sha256"}
    frozen = root / "compiler_mechanism_work_order.json"
    receipt = root / "compiler_mechanism_work_order_receipt.json"
    work_order = binding.get("work_order")
    rows = work_order.get("portfolio_site_bindings") if isinstance(work_order, Mapping) else None
    members = portfolio.get("members")
    catalog = catalog_binding.get("catalog")
    mechanisms = catalog.get("mechanisms") if isinstance(catalog, Mapping) else None
    if (
        binding.get("schema") != "host_frozen_compiler_mechanism_work_order_v1"
        or binding.get("sha256") != P2_CONTRACTS.document_sha256(body)
        or binding.get("mechanism_catalog_binding_sha256") != catalog_binding.get("sha256")
        or binding.get("initial_candidate_sha256") != authority.get("initial_candidate_sha256")
        or binding.get("portfolio_sha256") != P2_CONTRACTS.document_sha256(portfolio)
        or not isinstance(work_order, Mapping)
        or P2_CONTRACTS.document_sha256(work_order) != binding.get("work_order_document_sha256")
        or work_order.get("sha256")
        != P2_CONTRACTS.document_sha256({key: value for key, value in work_order.items() if key != "sha256"})
        or work_order.get("catalog_sha256") != (catalog or {}).get("sha256")
        or work_order.get("contract_sha256") != authority.get("contract", {}).get("sha256")
        or work_order.get("initial_candidate_sha256") != authority.get("initial_candidate_sha256")
        or work_order.get("round_start_candidate_sha256") != authority.get("initial_candidate_sha256")
        or work_order.get("ordered_portfolio") != members
        or work_order.get("portfolio_sha256") != P2_CONTRACTS.document_sha256(portfolio)
        or work_order.get("status") != "ready_for_authoring"
        or work_order.get("source_operation_ids") not in (None, [])
        or not isinstance(mechanisms, list)
        or len(mechanisms) != 1
        or mechanisms[0].get("id") != work_order.get("mechanism_id")
        or not isinstance(rows, list)
        or not isinstance(members, list)
        or len(rows) != len(members)
        or frozen.is_symlink()
        or not frozen.is_file()
        or frozen.stat().st_mode & 0o222
        or P2_CONTRACTS.sha256_file(frozen) != binding.get("canonical_bytes_sha256")
        or P2_CONTRACTS.mapping_file(frozen) != work_order
        or receipt.is_symlink()
        or not receipt.is_file()
        or receipt.stat().st_mode & 0o222
        or P2_CONTRACTS.mapping_file(receipt) != binding
    ):
        raise ValueError("checkpoint compiler mechanism work order changed")
    for expected, row in zip(members, rows, strict=True):
        operation_ids = row.get("source_operation_ids") if isinstance(row, Mapping) else None
        chains = row.get("chains") if isinstance(row, Mapping) else None
        hashes = (
            "compiler_sha256",
            "source_sha256",
            "plan_digest",
            "candidate_command_buffer_sha256",
            "candidate_lowered_sha256",
        )
        if (
            not isinstance(row, Mapping)
            or row.get("capsule") != expected.get("capsule")
            or row.get("capsule_sha256") != expected.get("capsule_sha256")
            or row.get("compiler_sha256") != authority.get("initial_candidate_sha256")
            or any(not BE._is_sha256(row.get(name)) for name in hashes)
            or not isinstance(row.get("inventory"), Mapping)
            or not isinstance(row.get("status"), str)
            or not row.get("status")
            or not isinstance(operation_ids, list)
            or not isinstance(chains, list)
            or any(not isinstance(chain, Mapping) for chain in chains)
        ):
            raise ValueError("checkpoint compiler mechanism work-order site binding changed")
    if analysis_binding is None:
        raise ValueError("checkpoint compiler mechanism work order lacks analyzed site bindings")
    analysis_body = {key: value for key, value in analysis_binding.items() if key != "sha256"}
    analysis_receipt = root / "compiler_mechanism_work_order_analysis.json"
    initial_reference = analysis_binding.get("iteration_record")
    initial_iteration = (
        root / f"iteration_{analysis_binding.get('iteration'):04d}.json"
        if isinstance(analysis_binding.get("iteration"), int)
        and not isinstance(analysis_binding.get("iteration"), bool)
        else None
    )
    if (
        analysis_binding.get("schema") != "compiler_mechanism_work_order_analysis_binding_v1"
        or analysis_binding.get("sha256") != P2_CONTRACTS.document_sha256(analysis_body)
        or analysis_binding.get("work_order_binding_sha256") != binding.get("sha256")
        or analysis_binding.get("portfolio_sha256") != P2_CONTRACTS.document_sha256(portfolio)
        or not isinstance(initial_reference, Mapping)
        or initial_iteration is None
        or Path(initial_reference.get("path", "")).resolve() != initial_iteration.resolve()
        or initial_iteration.is_symlink()
        or not initial_iteration.is_file()
        or P2_CONTRACTS.sha256_file(initial_iteration) != initial_reference.get("sha256")
        or analysis_receipt.is_symlink()
        or not analysis_receipt.is_file()
        or analysis_receipt.stat().st_mode & 0o222
        or P2_CONTRACTS.mapping_file(analysis_receipt) != analysis_binding
    ):
        raise ValueError("checkpoint compiler mechanism work-order analysis changed")
    initial_record = P2_CONTRACTS.mapping_file(initial_iteration)
    bound_members = analysis_binding.get("members")
    if (
        initial_record.get("candidate_sha256") != work_order.get("round_start_candidate_sha256")
        or (initial_record.get("portfolio") or {}).get("portfolio_sha256") != P2_CONTRACTS.document_sha256(portfolio)
        or not isinstance(bound_members, list)
        or len(bound_members) != len(members)
    ):
        raise ValueError("checkpoint compiler mechanism work-order initial analysis changed")
    for index, (identity, site, bound) in enumerate(zip(members, rows, bound_members, strict=True)):
        analysis = (
            initial_record.get("analysis")
            if index == 0
            else ((initial_record.get("portfolio") or {}).get("members") or [])[index].get("analysis")
        )
        plan = (
            (analysis.get("diagnostics") or {}).get("verified_global_plan_emission")
            if isinstance(analysis, Mapping)
            else None
        )
        emission = analysis.get("emission") if isinstance(analysis, Mapping) else None
        expected = {
            "capsule": identity["capsule"],
            "capsule_sha256": identity["capsule_sha256"],
            "compiler_sha256": initial_record.get("candidate_sha256"),
            "source_sha256": (plan or {}).get("source_sha256"),
            "plan_digest": (plan or {}).get("plan_digest"),
            "candidate_command_buffer_sha256": (emission or {}).get("candidate_command_buffer_sha256"),
            "candidate_lowered_sha256": (emission or {}).get("candidate_lowered_sha256"),
            "analysis_sha256": P2_CONTRACTS.document_sha256(analysis),
            "site_binding_sha256": P2_CONTRACTS.document_sha256(site),
        }
        if not isinstance(bound, Mapping) or any(bound.get(key) != value for key, value in expected.items()):
            raise ValueError("checkpoint compiler mechanism work-order artifact binding changed")
