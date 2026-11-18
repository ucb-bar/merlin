"""Candidate record schema, normalized handoff and declared formal-claim admission."""

from __future__ import annotations

import copy
import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.common.digest import sha256_bytes as _sha256
from merlin.development.backend.optimization import contracts as CONTRACTS
from merlin.development.backend.optimization.author_interface import broker_policy as BP
from merlin.development.backend.optimization.author_interface.broker_service import BROKER_NAME
from merlin.development.backend.optimization.authoring import telemetry_accounting as TEL
from merlin.development.backend.optimization.claims import dispatch as CD
from merlin.development.backend.optimization.contracts import ROUND_DEADLINE_EXIT, StageGateError, _is_sha256
from merlin.development.backend.optimization.contracts import canonical_json as _canonical_json
from merlin.development.backend.optimization.measurement import gsim_gate as GATE
from merlin.development.backend.optimization.search import corpus as CORPUS

# ---- candidate_record --------------------------------------------------------------------------


def validate_candidate_record(document: Mapping[str, Any], *, require_consumable: bool = True) -> dict:
    """Pure schema/boundary validator for the paired measurement runner."""
    from merlin.development.backend.optimization.authoring.candidate_claims import (
        LEGACY_MEASUREMENT_CONSUMERS,
        LIBRARY_SCHEMA_VERSION,
        MEASUREMENT_CONSUMER,
        PREPARED_SCHEMA_VERSION,
        SCHEMA_VERSION,
        _validate_audit_requalification,
        _validate_formal_claim_facts,
        verify_sandbox_containment_evidence,
    )

    if not isinstance(document, Mapping) or document.get("schema_version") not in (
        SCHEMA_VERSION,
        LIBRARY_SCHEMA_VERSION,
        PREPARED_SCHEMA_VERSION,
    ):
        raise StageGateError("performance candidate record has an unsupported schema")
    if document.get("kind") != "arm4_performance_candidate":
        raise StageGateError("performance candidate record has a foreign kind")
    additional = document.get("additional_compiler_library")
    if document["schema_version"] in (LIBRARY_SCHEMA_VERSION, PREPARED_SCHEMA_VERSION):
        from merlin.development.backend.optimization.campaign.compiler_library_state import validate_binding

        binding = validate_binding(additional)
        guard = document.get("functional_guard") or {}
        if require_consumable and (
            document.get("diagnostic_library_stage")
            or guard.get("status") != "clean"
            or guard.get("additional_compiler_library") != binding["origin"]
            or (guard.get("library_baseline_guard") or {}).get("status") != "clean"
        ):
            raise StageGateError(
                "additional-library candidate lacks same-release baseline and candidate functional guards"
            )
    elif require_consumable and additional is not None:
        raise StageGateError(
            "additional-library authoring is diagnostic; legacy paired measurement has no "
            "same-release compiler library handoff"
        )
    target = document.get("target")
    base = document.get("base_functional")
    candidate = document.get("candidate")
    prompt = document.get("prompt")
    corpus = document.get("performance_corpus")
    sandbox = document.get("sandbox")
    broker = document.get("broker")
    development_feedback = document.get("development_feedback")
    agent = document.get("agent")
    telemetry = document.get("telemetry")
    admission = document.get("admission")
    if not all(
        isinstance(value, Mapping)
        for value in (
            target,
            base,
            candidate,
            prompt,
            corpus,
            sandbox,
            broker,
            development_feedback,
            agent,
            telemetry,
            admission,
        )
    ):
        raise StageGateError("performance candidate record omits a required evidence mapping")
    for label, value in (
        ("functional submission", base.get("submission_sha256")),
        ("candidate initial", candidate.get("initial_sha256")),
        ("candidate final", candidate.get("sha256")),
        ("prompt", prompt.get("sha256")),
        ("prompt facts", prompt.get("facts_sha256")),
        ("prompt renderer", prompt.get("renderer_sha256")),
        ("performance manifest", corpus.get("manifest_sha256")),
        ("performance corpus", corpus.get("capsules_sha256")),
        ("agent-input manifest", corpus.get("agent_input_manifest_sha256")),
        ("agent-input content", corpus.get("agent_input_sha256")),
        ("target descriptor", target.get("descriptor_sha256")),
        ("Codex binary", agent.get("codex_binary_sha256")),
        ("transcript", agent.get("transcript_sha256")),
        ("broker registry", broker.get("registry_sha256")),
        ("broker receipt manifest", broker.get("receipt_manifest_sha256")),
        ("development GSIM certificate", (development_feedback.get("certificate") or {}).get("sha256")),
    ):
        if not _is_sha256(value):
            raise StageGateError(f"performance candidate record has no valid {label} SHA-256")
    prepared = document.get("compiler_pass_preparation")
    if document["schema_version"] == PREPARED_SCHEMA_VERSION:
        from merlin.development.backend.optimization.campaign.pass_preparation import (
            prompt_projection,
            validate_pair,
        )

        validate_pair(prepared, functional_sha256=base.get("submission_sha256"))
        if (
            prepared["original"]["path"] != base.get("snapshot")
            or candidate.get("initial_sha256") != prepared["candidate_seed"]["sha256"]
            or (prompt.get("facts") or {}).get("compiler_pass_preparation") != prompt_projection(prepared)
        ):
            raise StageGateError("prepared candidate changed its original parent or initial candidate seed")
        guard = document.get("functional_guard") or {}
        if require_consumable and (
            guard.get("compiler_pass_selection") != prepared["selection"] or not guard.get("compiler_pass_realizations")
        ):
            raise StageGateError("prepared candidate lacks original complete final pass-realization checks")
    elif prepared is not None:
        raise StageGateError("legacy candidate cannot silently acquire compiler preparation")
    elif candidate.get("initial_sha256") != base.get("submission_sha256"):
        raise StageGateError("performance candidate was not forked byte-for-byte from functional")
    if (
        not isinstance(target.get("name"), str)
        or not target.get("name")
        or not isinstance(target.get("descriptor"), str)
        or not target.get("descriptor")
    ):
        raise StageGateError("performance candidate record omits its target descriptor identity")
    CONTRACTS.safe_component(str(base.get("run_id") or ""), label="base functional run id")
    bundle_snapshot = base.get("bundle_input_snapshot")
    host_lane = base.get("model_host_lane")
    sentinel = base.get("e2e_sentinel")
    if (
        not isinstance(bundle_snapshot, Mapping)
        or not _is_sha256(bundle_snapshot.get("content_sha256"))
        or not _is_sha256(bundle_snapshot.get("manifest_sha256"))
        or not isinstance(bundle_snapshot.get("grants"), list)
        or not bundle_snapshot["grants"]
    ):
        raise StageGateError("performance candidate omits the frozen functional grant snapshot")
    if (
        not isinstance(host_lane, Mapping)
        or not _is_sha256(host_lane.get("package_sha256"))
        or not isinstance(host_lane.get("integration_seam"), str)
    ):
        raise StageGateError("performance candidate omits its exact functional host lane")
    if (
        not isinstance(sentinel, Mapping)
        or not _is_sha256(sentinel.get("capsule_sha256"))
        or set(sentinel.get("required_lanes") or []) != {"on_mesh", "scalar_rvv_lane"}
        or "L3" not in (sentinel.get("required_tiers") or [])
    ):
        raise StageGateError("performance candidate omits its frozen full-model E2E sentinel")
    for evidence, label in (
        (candidate.get("path"), "sealed candidate path"),
        (prompt.get("staged_path"), "staged prompt path"),
        (corpus.get("path"), "frozen performance corpus path"),
        (agent.get("codex_binary"), "Codex binary path"),
        (agent.get("transcript"), "combined transcript path"),
    ):
        if not isinstance(evidence, str) or not evidence:
            raise StageGateError(f"performance candidate record omits the {label}")
    for attribute in ("agent_input_files", "agent_input_bytes"):
        value = corpus.get(attribute)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise StageGateError(f"performance candidate record has no non-vacuous {attribute}")
    replicates = corpus.get("replicates")
    cells, families = corpus.get("expected_cells"), corpus.get("families")
    formal_claim = corpus.get("formal_claim")
    formal_replicate_identities = corpus.get("formal_replicate_identities")
    smoke_replicates = corpus.get("smoke_replicates")
    prompt_facts = prompt.get("facts")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates <= 0
        or not isinstance(cells, list)
        or not cells
        or not isinstance(families, list)
        or not families
        or not isinstance(prompt_facts, Mapping)
        or prompt_facts.get("expected_cells") != cells
        or prompt_facts.get("families") != families
        or prompt_facts.get("replicates") != replicates
        or prompt_facts.get("formal_claim") != formal_claim
        or prompt_facts.get("formal_replicate_identities") != formal_replicate_identities
        or prompt_facts.get("smoke_replicates") != smoke_replicates
    ):
        raise StageGateError("performance candidate omits exact prompt cells/families/replicates")
    _validate_formal_claim_facts(
        formal_claim, replicates, formal_replicate_identities, smoke_replicates, cells, families
    )
    if candidate.get("read_only") is not True or candidate.get("base_submission_overwritten") is not False:
        raise StageGateError("performance candidate record does not prove a separate read-only snapshot")
    delta = candidate.get("delta")
    if (
        not isinstance(delta, Mapping)
        or not isinstance(delta.get("changed_files"), list)
        or not isinstance(delta.get("execution_relevant_changed_files"), list)
        or delta.get("changed_file_count") != len(delta["changed_files"])
        or delta.get("execution_relevant_changed_file_count") != len(delta["execution_relevant_changed_files"])
    ):
        raise StageGateError("performance candidate record has no structured candidate delta")
    outer = sandbox.get("outer_codex_control_plane")
    inner = sandbox.get("inner_execution_plane")
    if not isinstance(outer, Mapping) or not isinstance(inner, Mapping):
        raise StageGateError("performance candidate record collapses its two sandbox boundaries")
    verify_sandbox_containment_evidence(sandbox, require_present=False)
    network_record = "available_not_an_isolation_claim"
    mounted_manifest_sha256 = bundle_snapshot.get("public_manifest_sha256") or bundle_snapshot.get("manifest_sha256")
    if (
        outer.get("network") != network_record
        or outer.get("clear_environment") is not True
        or outer.get("auth_exception") != "isolated_codex_home_explicit_auth_mount"
        or outer.get("session_history_mounted") is not False
        or outer.get("live_target_toolchain_mounted") is not False
        or outer.get("frozen_functional_grants_mounted") is not True
        or outer.get("frozen_grant_manifest_sha256") != mounted_manifest_sha256
        or outer.get("answer_surface_gap") != []
    ):
        raise StageGateError("outer Codex control-plane evidence is incomplete")
    if not isinstance(outer.get("bwrap_binary"), str) or not outer.get("bwrap_binary"):
        raise StageGateError("outer Codex control-plane evidence omits the bwrap binary path")
    for evidence, label in (
        (outer.get("bwrap_binary_sha256"), "bwrap binary"),
        (outer.get("policy_sha256"), "outer bwrap policy"),
        (inner.get("policy_sha256"), "inner bwrap policy"),
    ):
        if not _is_sha256(evidence):
            raise StageGateError(f"performance candidate record omits the {label} digest")
    if (
        inner.get("network") != network_record
        or inner.get("clear_environment") is not True
        or inner.get("credentials") != "none"
        or inner.get("answer_surface_gap") != []
        or inner.get("candidate_writable") is not True
        or inner.get("corpus_read_only") is not True
        or inner.get("frozen_functional_grants_mounted") is not True
        or inner.get("frozen_grant_manifest_sha256") != mounted_manifest_sha256
    ):
        raise StageGateError("inner execution-plane evidence is not credential-free and answer-masked")
    probes = inner.get("tool_probe_results")
    probe_recheck = inner.get("tool_probe_recheck_results")
    if (
        not isinstance(probes, list)
        or not probes
        or any(
            not isinstance(row, Mapping)
            or row.get("returncode") != 0
            or not isinstance(row.get("label"), str)
            or not row.get("label")
            or not isinstance(row.get("command"), str)
            or not row.get("command")
            for row in probes
        )
    ):
        raise StageGateError("performance candidate record lacks passing inner tool probes")
    if probe_recheck != probes:
        raise StageGateError("performance candidate tool evidence changed between preflight and recheck")
    registry, receipt_rows = broker.get("registry"), broker.get("round_receipts")
    if (
        not isinstance(registry, list)
        or not registry
        or _sha256(_canonical_json(registry)) != broker.get("registry_sha256")
        or not isinstance(receipt_rows, list)
        or any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("path"), str)
            or not _is_sha256(row.get("sha256"))
            or row.get("all_required_succeeded") is not True
            or not _is_sha256(row.get("candidate_sha256"))
            or row.get("final_candidate_feedback_verified") is not True
            or not isinstance(row.get("feedback_successes"), int)
            or row.get("feedback_successes") < 1
            or not isinstance(row.get("feedback_receipts"), list)
            or not row.get("feedback_receipts")
            for row in receipt_rows
        )
        or broker.get("control_owned_by_harness") is not True
        or broker.get("control_writable_by_agent") is not False
        or not isinstance(broker.get("receipt_manifest"), str)
        or not broker.get("receipt_manifest")
    ):
        raise StageGateError("performance candidate lacks immutable broker registry/receipts")
    required_actions = sorted(
        str(row.get("name")) for row in registry if isinstance(row, Mapping) and row.get("required") is True
    )
    if not required_actions or broker.get("required_actions") != required_actions:
        raise StageGateError("performance candidate broker required-action contract is incomplete")
    feedback_actions = [
        row for row in registry if isinstance(row, Mapping) and row.get("name") == BP.DEVELOPMENT_FEEDBACK_ACTION
    ]
    if (
        len(feedback_actions) != 1
        or feedback_actions[0].get("required") is not True
        or feedback_actions[0].get("placeholders") != []
        or feedback_actions[0].get("argv_template") != [BP._HOST_FEEDBACK_SENTINEL]
        or BP.DEVELOPMENT_FEEDBACK_ACTION not in required_actions
    ):
        raise StageGateError("mandatory tuning GSIM feedback action is absent or drifted")
    certificate = development_feedback.get("certificate")
    rtl_identity = development_feedback.get("rtl_identity")
    recorded_feedback_receipts = development_feedback.get("round_receipts")
    if (
        development_feedback.get("action") != BP.DEVELOPMENT_FEEDBACK_ACTION
        or development_feedback.get("required_per_round") is not True
        or development_feedback.get("scope") != "frozen_tuning_corpus_only"
        or development_feedback.get("engine") != "gsim"
        or development_feedback.get("redaction") != "correctness_gsim_cycles_and_paired_deltas_only"
        or not isinstance(certificate, Mapping)
        or certificate.get("target") != target.get("name")
        or certificate.get("fidelity") != GATE.FIDELITY
        or not isinstance(rtl_identity, Mapping)
        or not isinstance(recorded_feedback_receipts, list)
        or recorded_feedback_receipts != [row.get("feedback_receipts") for row in receipt_rows]
    ):
        raise StageGateError("development GSIM feedback evidence is incomplete or drifted")
    budget_facts = prompt_facts.get("budgets")
    expected_budget_facts = {
        "wall_budget_seconds": agent.get("wall_budget_seconds"),
        "rounds": agent.get("rounds_requested"),
        "round_timeout_seconds": agent.get("round_timeout_seconds"),
        "max_tool_calls": agent.get("max_tool_calls"),
        "tool_timeout_seconds": agent.get("tool_timeout_seconds"),
    }
    prompt_host = prompt_facts.get("host_lane")
    host_fact_keys = {"target", "package_id", "package_path", "package_sha256", "manifest_path", "integration_seam"}
    if (
        budget_facts != expected_budget_facts
        or prompt_facts.get("tools") != registry
        or prompt_facts.get("e2e_sentinel") != sentinel
        or not isinstance(prompt_host, Mapping)
        or set(prompt_host) != host_fact_keys
        or any(prompt_host.get(key) != host_lane.get(key) for key in host_fact_keys)
        or prompt_facts.get("mount_destinations") != outer.get("mount_destinations")
    ):
        raise StageGateError("canonical prompt facts drift from recorded launch enforcement")
    round_rows = agent.get("rounds")
    rounds_requested = agent.get("rounds_requested")
    bounded_agent_values = (
        agent.get("wall_budget_seconds"),
        agent.get("rounds_requested"),
        agent.get("round_timeout_seconds"),
        agent.get("max_tool_calls"),
        agent.get("tool_timeout_seconds"),
    )
    if (
        agent.get("driver") != "codex"
        or any(
            not isinstance(agent.get(field), str) or not agent.get(field)
            for field in ("model", "resolved_model", "effort")
        )
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in bounded_agent_values)
        or not isinstance(round_rows, list)
        or not round_rows
        or isinstance(rounds_requested, bool)
        or not isinstance(rounds_requested, int)
        or rounds_requested < len(round_rows)
        or any(
            not isinstance(row, Mapping)
            or isinstance(row.get("agent_exit_code"), bool)
            or not isinstance(row.get("agent_exit_code"), int)
            or not isinstance(row.get("transcript"), str)
            or not row.get("transcript")
            or not isinstance(row.get("audit"), Mapping)
            or not isinstance(row["audit"].get("clean"), bool)
            or not isinstance(row["audit"].get("hits"), list)
            or row["audit"].get("broker_required") != BROKER_NAME
            or not isinstance(row["audit"].get("broker_invocations"), list)
            or not isinstance(row.get("telemetry"), Mapping)
            or row["telemetry"].get("event_count", 0) <= 0
            or (row["telemetry"].get("summary") or {}).get("usage_complete") is not True
            or (row["telemetry"].get("accounting") or {}).get("available") is not True
            or (row["telemetry"].get("accounting") or {}).get("usage_complete") is not True
            or isinstance(row["audit"].get("commands_seen"), bool)
            or not isinstance(row["audit"].get("commands_seen"), int)
            or not _is_sha256(row.get("transcript_sha256"))
            for row in round_rows
        )
    ):
        raise StageGateError("performance candidate was not produced by the bounded Codex driver")
    telemetry_artifacts = telemetry.get("artifacts")
    telemetry_accounting = telemetry.get("accounting")
    telemetry_reconciliation = telemetry.get("aet_reconciliation")
    activity_share = telemetry.get("activity_share")
    activity_seconds = (activity_share or {}).get("seconds_by_category") or {}
    activity_shares = (activity_share or {}).get("share_by_category") or {}
    activity_wall = (activity_share or {}).get("trajectory_wall_seconds")
    activity_occupancy = (activity_share or {}).get("classified_span_occupancy_ratio")
    activity_semantics_valid = (
        isinstance(activity_seconds, Mapping)
        and bool(activity_seconds)
        and isinstance(activity_shares, Mapping)
        and bool(activity_shares)
        and set(activity_seconds) == set(activity_shares)
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
            for value in (*activity_seconds.values(), *activity_shares.values())
        )
        and math.isclose(sum(float(value) for value in activity_shares.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        and isinstance((activity_share or {}).get("classified_seconds"), (int, float))
        and not isinstance((activity_share or {}).get("classified_seconds"), bool)
        and float(activity_share["classified_seconds"]) > 0
        and math.isclose(
            sum(float(value) for value in activity_seconds.values()),
            float(activity_share["classified_seconds"]),
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        and activity_share.get("schema_version") == 2
        and activity_share.get("denominator") == "sum_of_classified_tool_span_seconds_including_overlap"
        and activity_share.get("is_wall_time_partition") is False
        and activity_share.get("overlapping_tool_spans_allowed") is True
        and activity_share.get("occupancy_ratio_may_exceed_one") is True
        and activity_share.get("subagent_tool_calls_tracked") is False
        and isinstance(activity_wall, (int, float))
        and not isinstance(activity_wall, bool)
        and float(activity_wall) > 0
        and isinstance(activity_occupancy, (int, float))
        and not isinstance(activity_occupancy, bool)
        and float(activity_occupancy) > 0
        and math.isclose(
            float(activity_occupancy),
            float(activity_share["classified_seconds"]) / float(activity_wall),
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
    )
    required_telemetry_artifacts = TEL.artifact_names(telemetry)
    if (
        telemetry.get("required") is not True
        or telemetry.get("driver") != "codex"
        or telemetry.get("billing_mode") != "subscription_notional"
        or telemetry.get("rounds_with_complete_usage") != len(round_rows)
        or not isinstance(telemetry.get("raw_event_count"), int)
        or telemetry.get("raw_event_count", 0) <= 0
        or not isinstance(telemetry.get("tool_call_count"), int)
        or telemetry.get("tool_call_count", 0) <= 0
        or telemetry.get("subagent_tool_calls_tracked") is not False
        or not _is_sha256(telemetry.get("preflight_sha256"))
        or not isinstance(telemetry_accounting, Mapping)
        or telemetry_accounting.get("available") is not True
        or telemetry_accounting.get("usage_complete") is not True
        or telemetry_accounting.get("billing_mode") != "subscription_notional"
        or telemetry_accounting.get("estimated_cost_usd") is not None
        or not isinstance(telemetry_accounting.get("subscription_notional_usd"), (int, float))
        or telemetry_accounting.get("subscription_notional_usd", 0) <= 0
        or not isinstance(telemetry_accounting.get("tokens_total"), int)
        or telemetry_accounting.get("tokens_total", 0) <= 0
        or telemetry_accounting.get("tool_calls") != telemetry.get("tool_call_count")
        or not isinstance(telemetry_reconciliation, Mapping)
        or telemetry_reconciliation.get("ok") is not True
        or (telemetry_reconciliation.get("raw_events") or {}).get("reconciled") is not True
        or (telemetry_reconciliation.get("token_ledger") or {}).get("all_match") is not True
        or not isinstance(activity_share, Mapping)
        or activity_share.get("basis") != "aet_native_codex_structured_tool_spans"
        or not activity_semantics_valid
        or not isinstance(telemetry_artifacts, Mapping)
        or set(telemetry_artifacts) != required_telemetry_artifacts
        or any(
            not isinstance(value, Mapping)
            or not isinstance(value.get("path"), str)
            or not value.get("path")
            or not _is_sha256(value.get("sha256"))
            for value in telemetry_artifacts.values()
        )
    ):
        raise StageGateError("performance candidate lacks complete raw/AET/cost/activity telemetry")
    if len(receipt_rows) != len(round_rows) or any(
        receipt.get("candidate_sha256") != round_row.get("candidate_sha256")
        for receipt, round_row in zip(receipt_rows, round_rows)
    ):
        raise StageGateError("per-round tuning GSIM feedback is not bound to the authored candidate bytes")
    audit = agent.get("audit")
    if (
        not isinstance(audit, Mapping)
        or not isinstance(audit.get("clean"), bool)
        or not isinstance(audit.get("hits"), list)
        or audit.get("broker_required") != BROKER_NAME
        or not isinstance(audit.get("broker_invocations"), list)
        or isinstance(audit.get("commands_seen"), bool)
        or not isinstance(audit.get("commands_seen"), int)
    ):
        raise StageGateError("performance candidate lacks structured agent audit evidence")
    if (
        admission.get("evaluation_performed_by_stage") is not False
        or admission.get("development_feedback_performed_by_stage") is not True
        or admission.get("success_declared_by_stage") is not False
        or admission.get("consumer") not in ({MEASUREMENT_CONSUMER} | LEGACY_MEASUREMENT_CONSUMERS)
    ):
        raise StageGateError("performance authoring stage crossed the evaluation boundary")
    _validate_audit_requalification(document)
    if require_consumable:
        if (
            admission.get("consumable") is not True
            or document.get("state") != "sealed"
            or rounds_requested != len(round_rows)
            or audit.get("clean") is not True
            or audit.get("hits") != []
            or audit.get("commands_seen", 0) <= 0
            or any(
                row["audit"].get("commands_seen", 0) <= 0 or row["audit"].get("clean") is not True for row in round_rows
            )
            or delta.get("execution_relevant_changed_file_count", 0) <= 0
            or broker.get("all_required_succeeded") is not True
            or any(row.get("feedback_successes", 0) < 1 for row in receipt_rows)
            or len(receipt_rows) != rounds_requested
            or any(row.get("agent_exit_code") not in (0, ROUND_DEADLINE_EXIT) for row in round_rows)
        ):
            raise StageGateError(f"performance candidate is not consumable: {admission.get('refusal')}")
    return dict(document)
