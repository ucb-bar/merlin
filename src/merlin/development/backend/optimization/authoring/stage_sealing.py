"""Seal stage record for a bounded authoring stage."""

from __future__ import annotations

import copy

from merlin.common.digest import sha256_bytes as _sha256
from merlin.development.backend.optimization.author_interface import broker_policy as BP
from merlin.development.backend.optimization.author_interface import broker_service as PB
from merlin.development.backend.optimization.author_interface.broker_service import (
    BROKER_NAME,
)
from merlin.development.backend.optimization.authoring import candidate_claims as RECORD
from merlin.development.backend.optimization.authoring import candidate_record as _merlin_candidate_record_4
from merlin.development.backend.optimization.authoring import candidate_verification as VERIFY
from merlin.development.backend.optimization.authoring import telemetry as _merlin_telemetry_1
from merlin.development.backend.optimization.authoring.authoring_artifacts import candidate_delta
from merlin.development.backend.optimization.authoring.authoring_artifacts import (
    development_feedback_performed as _development_feedback_performed,
)
from merlin.development.backend.optimization.authoring.transcript_audit import audit_codex_transcript
from merlin.development.backend.optimization.campaign import compiler_library_state as LIBRARY
from merlin.development.backend.optimization.contracts import (
    ROUND_DEADLINE_EXIT,
)
from merlin.development.backend.optimization.contracts import (
    canonical_json as _canonical_json,
)
from merlin.development.backend.optimization.contracts import (
    sha256_file as _sha256_file,
)
from merlin.development.backend.optimization.contracts import (
    write_json as _write_json,
)
from merlin.development.backend.optimization.ordinary import capsule_ordinary_scope as CAPSULE_SCOPE
from merlin.development.backend.optimization.ordinary import capsule_source_development as CAPSULE_DEV
from merlin.development.targetgen.sandbox.answer_surfaces import absent_declaration_coverage


def seal_stage_record(
    *,
    after,
    agent_inputs,
    base,
    broker_channel_records,
    broker_transport,
    bwrap_binary,
    bwrap_sha256,
    candidate_seed,
    capsule_scope,
    codex_path,
    codex_sha256,
    compiler_library_selection,
    compiler_pass_preparation,
    compiler_pass_selection,
    declaration_drops,
    descriptor_path,
    descriptor_sha256,
    diagnostic_library_stage,
    effort,
    external_resource_boundary,
    feedback,
    fork_check,
    formal_claim,
    frozen_corpus,
    frozen_functional,
    functional,
    last_inner,
    last_outer,
    max_tool_calls,
    model,
    original_base,
    prepared_action_contract,
    prepared_actions,
    prepared_compilers,
    previous_submission,
    probe_recheck_results,
    probe_results,
    prompt,
    prompt_inputs,
    receipt_records,
    refusal,
    replicates,
    resolved_model,
    round_records,
    round_timeout_seconds,
    rounds,
    sandbox_inputs,
    sandbox_probe,
    sealed,
    sealed_sha,
    search,
    seed_digest,
    smoke_replicates,
    stage_root,
    staged_prompt,
    stopped_by,
    suite,
    target_experiment,
    telemetry_preflight_record,
    tool_timeout_seconds,
    total_calls,
    transcript_paths,
    wall_budget_seconds,
    workflow,
    authoring_source,
):
    from merlin.development.backend.optimization.authoring.authoring import functional_emission_guard

    _admitted_exits = {0, ROUND_DEADLINE_EXIT} if (stopped_by or {}).get("conditions") == ["round_deadline"] else {0}
    exits_clean = all(row["agent_exit_code"] in _admitted_exits for row in round_records)
    if prepared_compilers is not None:
        from merlin.development.backend.optimization.campaign.pass_preparation import verify_pair

        verify_pair(prepared_compilers, compiler_pass_selection, functional_sha256=functional.digest)
    delta = candidate_delta(candidate_seed, previous_submission)
    if not delta["execution_relevant_changed_files"]:
        refusal = "Codex produced no execution-relevant candidate change"
    combined_transcript = stage_root / "rounds" / "combined.transcript.jsonl"
    with combined_transcript.open("wb") as stream:
        for path in transcript_paths:
            payload = path.read_bytes()
            stream.write(payload)
            if payload and not payload.endswith(b"\n"):
                stream.write(b"\n")
    combined_transcript.chmod(0o444)
    telemetry_record = _merlin_telemetry_1.finalize(
        stage_root,
        round_records,
        model=resolved_model,
        target=target_experiment.target,
        suite=suite,
        run_id=stage_root.name,
        preflight_record=telemetry_preflight_record,
    )
    combined_audit = audit_codex_transcript(
        combined_transcript, target_experiment, previous_submission, prepared_actions
    )
    round_audits_clean = all(row["audit"]["clean"] and row["audit"]["commands_seen"] > 0 for row in round_records)
    audits_clean = round_audits_clean and combined_audit["clean"]
    if combined_audit["commands_seen"] <= 0:
        refusal = "combined Codex transcript contains zero command evidence"
    elif not combined_audit["clean"]:
        refusal = "combined Codex transcript failed the answer/tool-access audit"
    # A converged run has FEWER round records than `rounds`, by design. Requiring exact equality
    # would mark the search unconsumable for having finished early, which is the outcome the stop
    # rule exists to produce.
    expected_rounds = len(round_records) if stopped_by is not None else rounds
    receipts_clean = len(receipt_records) == expected_rounds and all(
        row.get("all_required_succeeded") is True and row.get("feedback_successes", 0) >= 1 for row in receipt_records
    )
    if not receipts_clean:
        refusal = refusal or "required host-owned broker receipt evidence is incomplete"
    # THE FUNCTIONAL GUARD. Phase 1 certified the baseline on the functional corpus and this stage
    # Complete unchanged execution dependencies permit reuse; changed dependencies
    # or emission require the ordinary full-output numerical certification.
    functional_guard = CAPSULE_SCOPE.final_guard(
        capsule_scope,
        functional_emission_guard,
        base,
        previous_submission,
        target_experiment,
        external_resource_boundary,
        stage_root,
        frozen_functional,
        sandbox_inputs,
    )
    if capsule_scope is None and functional_guard.get("status") != "clean":
        kinds = sorted({str(row.get("kind")) for row in functional_guard.get("offenders") or ()})
        refusal = refusal or (
            "performance candidate did not clear the certified functional emission guard "
            f"({functional_guard.get('status')}" + (f": {', '.join(kinds)}" if kinds else "") + ")"
        )
    functional_guard_clean = functional_guard.get("status") == "clean"
    library_binding, refusal = LIBRARY.freeze_stage(
        compiler_library_selection, stage_root, diagnostic_library_stage, refusal
    )
    consumable = (
        refusal is None
        and audits_clean
        and exits_clean
        and receipts_clean
        and functional_guard_clean
        and len(round_records) == expected_rounds
        and len(round_records) >= 1
    )
    receipt_manifest = stage_root / "control" / "receipt_manifest.json"
    _write_json(receipt_manifest, {"schema_version": 1, "rounds": receipt_records})
    receipt_manifest.chmod(0o444)
    expected_cells = [
        {"family": cell.family, "capsule": cell.capsule, "simulator": cell.simulator, "replicate": cell.replicate}
        for cell in prompt_inputs.expected_cells
    ]
    family_facts = [
        {
            "family": family.family,
            "claim": family.claim,
            "negative_control": family.negative_control,
            "falsifier_observation": family.falsifier_observation,
            "differential_basis": family.differential_basis,
            "fitted_parameters": list(family.fitted_parameters),
            "acceptance": copy.deepcopy(family.acceptance),
        }
        for family in prompt_inputs.families
    ]
    host_lane = {
        "target": prompt_inputs.host_lane.target,
        "package_id": prompt_inputs.host_lane.package_id,
        "package_path": prompt_inputs.host_lane.package_path,
        "package_sha256": prompt_inputs.host_lane.package_sha256,
        "manifest_path": prompt_inputs.host_lane.manifest_path,
        "integration_seam": prompt_inputs.host_lane.integration_seam,
    }
    model_host_record = dict(functional.model_host_lane_snapshot)
    model_host_record.update(host_lane)
    e2e_sentinel = CAPSULE_SCOPE.sentinel_record(capsule_scope, prompt_inputs.e2e_sentinel)
    frozen_grants = [
        {
            "declared_path": grant.declared_path,
            "destination": str(grant.destination),
            "source": str(grant.source),
            "source_sha256": grant.source_sha256,
        }
        for grant in frozen_functional.grants
    ]
    prompt_facts = {
        "replicates": replicates,
        "formal_replicate_identities": list(prompt_inputs.formal_replicate_identities),
        "formal_claim": copy.deepcopy(formal_claim),
        "smoke_replicates": smoke_replicates,
        "expected_cells": expected_cells,
        "budgets": {
            "wall_budget_seconds": wall_budget_seconds,
            "rounds": rounds,
            "round_timeout_seconds": round_timeout_seconds,
            "max_tool_calls": max_tool_calls,
            "tool_timeout_seconds": tool_timeout_seconds,
        },
        "families": family_facts,
        "host_lane": host_lane,
        "e2e_sentinel": e2e_sentinel,
        "tools": prepared_action_contract,
        "mount_destinations": list(prompt_inputs.allowed_paths),
        **(
            {"compiler_pass_preparation": prompt_inputs.compiler_pass_preparation}
            if prepared_compilers is not None
            else {}
        ),
    }
    record = {
        **LIBRARY.stage_record(compiler_library_selection, library_binding, diagnostic_library_stage),
        **(
            {"schema_version": RECORD.PREPARED_SCHEMA_VERSION, "compiler_pass_preparation": prepared_compilers}
            if prepared_compilers is not None
            else {}
        ),
        "kind": "arm4_performance_candidate",
        "state": "sealed" if consumable else "refused",
        # Per-capsule evidence that the certified functional emission survived this candidate: which
        # capsules are PROVED unchanged (byte-identical emission), which changed, and what the changed
        # ones introduced relative to the certified baseline.
        "functional_guard": functional_guard,
        "target": {
            "name": target_experiment.target,
            "descriptor": str(descriptor_path),
            "descriptor_sha256": descriptor_sha256,
        },
        "base_functional": {
            "run_id": functional.run_id,
            "submission_sha256": functional.digest,
            "snapshot": str(original_base),
            "fork_before": fork_check.to_dict(),
            "fork_after": after.to_dict(),
            "bundle_input_snapshot": {
                "path": str(frozen_functional.root),
                "content_sha256": frozen_functional.content_sha256,
                "manifest": str(frozen_functional.marker),
                "manifest_sha256": frozen_functional.marker_sha256,
                "grants": frozen_grants,
                **(
                    {
                        "host_provenance": frozen_functional.host_provenance,
                        "public_manifest": str(frozen_functional.public_marker),
                        "public_manifest_sha256": frozen_functional.public_marker_sha256,
                        "public_content_sha256": frozen_functional.public_content_sha256,
                    }
                    if frozen_functional.host_provenance is not None
                    else {}
                ),
            },
            "model_host_lane": model_host_record,
            "e2e_sentinel": e2e_sentinel,
        },
        "candidate": {
            "path": str(sealed),
            "initial_sha256": seed_digest,
            "sha256": sealed_sha,
            "rounds_completed": len(round_records),
            "read_only": True,
            "base_submission_overwritten": False,
            "delta": delta,
            **(
                {"comparison_delta": candidate_delta(base, previous_submission)}
                if prepared_compilers is not None
                else {}
            ),
        },
        "prompt": {
            "renderer_path": str(authoring_source.resolve()),
            "renderer_sha256": _sha256_file(authoring_source.resolve()),
            "staged_path": str(staged_prompt),
            "sha256": prompt.sha256,
            "n_bytes": prompt.n_bytes,
            "facts": prompt_facts,
            "facts_sha256": _sha256(_canonical_json(prompt_facts)),
        },
        "performance_corpus": {
            "path": str(frozen_corpus.root),
            "manifest": str(frozen_corpus.manifest_path),
            "manifest_sha256": frozen_corpus.manifest_sha256,
            "capsules_sha256": frozen_corpus.capsules_sha256,
            "agent_input_path": str(agent_inputs.root),
            "agent_input_manifest": str(agent_inputs.manifest_path),
            "agent_input_manifest_sha256": agent_inputs.manifest_sha256,
            "agent_input_sha256": agent_inputs.content_sha256,
            "agent_input_files": agent_inputs.n_files,
            "agent_input_bytes": agent_inputs.n_bytes,
            "replicates": replicates,
            "formal_replicate_identities": list(prompt_inputs.formal_replicate_identities),
            "formal_claim": copy.deepcopy(formal_claim),
            "smoke_replicates": smoke_replicates,
            "expected_cells": expected_cells,
            "families": family_facts,
        },
        "development_feedback": {
            "action": BP.DEVELOPMENT_FEEDBACK_ACTION,
            "required_per_round": True,
            "scope": "frozen_tuning_corpus_only",
            "engine": "gsim",
            **CAPSULE_DEV.feedback_binding(feedback),
            "redaction": "correctness_gsim_cycles_and_paired_deltas_only",
            "round_receipts": [row.get("feedback_receipts", []) for row in receipt_records],
        },
        "sandbox": {
            "preflight": sandbox_probe.as_record(),
            "dropped_declarations": [drop.as_record() for drop in declaration_drops],
            "absent_declaration_coverage": absent_declaration_coverage(target_experiment),
            "outer_codex_control_plane": {
                "engine": "bwrap",
                "network": last_outer.network,
                "clear_environment": last_outer.clear_environment,
                "auth_exception": "isolated_codex_home_explicit_auth_mount",
                "session_history_mounted": False,
                "live_target_toolchain_mounted": False,
                "frozen_functional_grants_mounted": True,
                "frozen_grant_manifest_sha256": (
                    frozen_functional.public_marker_sha256 or frozen_functional.marker_sha256
                ),
                "mount_destinations": list(prompt_inputs.allowed_paths),
                "answer_surface_gap": list(last_outer.answer_surface_gap),
                "bwrap_binary": str(bwrap_binary),
                "bwrap_binary_sha256": bwrap_sha256,
                "policy_sha256": _sha256(_canonical_json(list(last_outer.argv))),
            },
            "inner_execution_plane": {
                "engine": "bwrap",
                "network": last_inner.network,
                "clear_environment": last_inner.clear_environment,
                "credentials": "none",
                "candidate_writable": last_inner.candidate_writable,
                "corpus_read_only": last_inner.corpus_read_only,
                "answer_surface_gap": list(last_inner.answer_surface_gap),
                "required_tools": [probe.label for probe in last_inner.required_tools],
                "tool_probe_results": probe_results,
                "tool_probe_recheck_results": probe_recheck_results,
                "broker_calls": total_calls,
                "frozen_functional_grants_mounted": True,
                "frozen_grant_manifest_sha256": (
                    frozen_functional.public_marker_sha256 or frozen_functional.marker_sha256
                ),
                "policy_sha256": _sha256(_canonical_json(list(last_inner.argv))),
            },
        },
        "broker": {
            **(
                {"transport": broker_transport, "file_channels": broker_channel_records}
                if broker_transport != "tcp"
                else {}
            ),
            "shim_mount": BROKER_NAME,
            "shim_sha256": _sha256(PB._BROKER_SHIM.encode("utf-8")),
            "registry": prepared_action_contract,
            "registry_sha256": _sha256(_canonical_json(prepared_action_contract)),
            "receipt_manifest": str(receipt_manifest),
            "receipt_manifest_sha256": _sha256_file(receipt_manifest),
            "round_receipts": receipt_records,
            "required_actions": sorted(action.name for action in prepared_actions if action.required),
            "all_required_succeeded": receipts_clean,
            "control_owned_by_harness": True,
            "control_writable_by_agent": False,
        },
        "agent": {
            "driver": "codex",
            "model": model,
            "resolved_model": resolved_model,
            "effort": effort,
            "codex_binary": str(codex_path),
            "codex_binary_sha256": codex_sha256,
            "wall_budget_seconds": wall_budget_seconds,
            "round_timeout_seconds": round_timeout_seconds,
            "max_tool_calls": max_tool_calls,
            "tool_timeout_seconds": tool_timeout_seconds,
            "rounds_requested": rounds,
            "rounds": round_records,
            "transcript": str(combined_transcript),
            "transcript_sha256": _sha256(combined_transcript.read_bytes()),
            "audit": combined_audit,
        },
        "telemetry": telemetry_record,
        "admission": {
            "consumable": consumable,
            "refusal": refusal,
            # Distinct from `refusal` on purpose: this names a run that ended because the search
            # reported it was finished, which must not be readable as a failure.
            "stopped_by": stopped_by,
            "development_feedback_performed_by_stage": _development_feedback_performed(receipt_records),
            "evaluation_performed_by_stage": False,
            "success_declared_by_stage": False,
            "consumer": RECORD.MEASUREMENT_CONSUMER,
        },
    }
    if capsule_scope is not None:
        CAPSULE_SCOPE.development_record(
            capsule_scope,
            record,
            refusal is None and audits_clean and exits_clean and receipts_clean,
        )
        consumable = False
    RECORD.verify_sandbox_containment_evidence(record["sandbox"])
    record_path = CAPSULE_SCOPE.record_path(stage_root, capsule_scope)
    _write_json(record_path, record)
    record_path.chmod(0o444)
    if capsule_scope is not None:
        CAPSULE_SCOPE.verify_checkpoint(record)
    else:
        _merlin_candidate_record_4.validate_candidate_record(record, require_consumable=consumable)
    if consumable:
        VERIFY.verify_candidate_record(
            record_path,
            verify_authoring_tools=True,
            target_experiment=target_experiment,
            **LIBRARY.verification_inputs(compiler_library_selection, library_binding),
            **(
                dict(
                    compiler_pass_selection=compiler_pass_selection,
                    compiler_pass_preparation=compiler_pass_preparation,
                    compiler_preparation_verification_root=stage_root / "control" / "final_preparation_replay",
                )
                if prepared_compilers is not None
                else {}
            ),
        )
    if search is not None:
        from merlin.development.backend.optimization.ordinary.ordinary_build_runtime import close_stage

        close_stage(workflow)
    return record_path
