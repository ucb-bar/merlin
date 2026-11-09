"""Restore and verify the original frozen session environment before admission."""

from __future__ import annotations

import yaml


def restore(
    a,
    context,
    *,
    _environment_path,
    _run_config,
    _source_context,
    hidden_capsule_policy,
    _hidden_capsule_policy,
    verify_readback_record,
    _verify_public_build_budget,
    _verify_client_selection,
    _verify_authoring_selection,
    _verify_source_selection,
):
    """Reopen the prior record and apply the caller's original verification hooks."""
    try:
        _environment_record = yaml.safe_load(_environment_path.read_text())
    except Exception as exc:  # noqa: BLE001 — provenance is a formal resume gate
        raise RuntimeError(f"resume refused: environment record unreadable: {exc}") from exc
    if not isinstance(_environment_record, dict):
        raise RuntimeError("resume refused: environment record is not a mapping")
    from .feedback import sandbox_pool

    # A resumed run keeps the sandbox mechanism it started with (none recorded: one per call).
    _run_config["sandbox_reuse"] = sandbox_pool.adopt_recorded(_environment_record.get("run_config"))
    prior_mode = (_environment_record.get("run_config") or {}).get("compiler_execution", "profile_strict_v1")
    if prior_mode != a.compiler_execution:
        raise RuntimeError("resume refused: compiler execution treatment changed; use a new checkpoint continuation")
    if (_environment_record.get("run_config") or {}).get("author_profile", "") != a.author_profile:
        raise RuntimeError("resume refused: versioned author profile changed")
    prior_access = (_environment_record.get("run_config") or {}).get("compiler_access", "selected_library")
    prior_manifest = (_environment_record.get("run_config") or {}).get("full_compiler_manifest_sha256", "")
    if (prior_access, prior_manifest) != (a.compiler_access, a.full_compiler_manifest_sha256):
        raise RuntimeError("resume refused: editable compiler source selection changed")
    prior_reference = (_environment_record.get("run_config") or {}).get("reference_experiments_sha256", "")
    if prior_reference != a.reference_experiments_sha256:
        raise RuntimeError("resume refused: reference experiment feedback selection changed")
    prior_tasks = (_environment_record.get("run_config") or {}).get("reference_experiments_tasks", "on")
    if prior_tasks != a.reference_experiments_tasks:
        raise RuntimeError("resume refused: reference experiment task serving changed")
    from .feedback.capsule_certification import verify_scope

    verify_scope(getattr(a, "validation_scope", "full"), _environment_record)
    verify_readback_record(context, _environment_record.get("readback_policy"))
    hidden_capsule_policy.verify_record(
        _hidden_capsule_policy,
        _environment_record.get("hidden_capsule_validation"),
    )
    _verify_public_build_budget(a, _environment_record)
    _verify_authoring_selection(a, _environment_record)
    _verify_client_selection(a, _environment_record)
    _implementation_sources = _environment_record.get("implementation_sources")
    _verify_source_selection(_implementation_sources, **_source_context)
    return _environment_record, _implementation_sources
