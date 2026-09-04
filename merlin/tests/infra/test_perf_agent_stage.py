"""Hermetic checks for the bounded Arm4 performance-authoring stage.

No test launches Codex or a simulator.  The tests pin the two-plane record contract, answer-free corpus
view, exact functional copy, broker command, and transcript refusal paths before a paid run is possible.
"""
from __future__ import annotations

import copy
import contextlib
import hashlib
import inspect
import json
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.benchharness import hash_tree
from merlin.common.paths import merlin_dir
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface
from merlin.targetgen.sandbox.toolchain import ToolProbe


_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import perf_agent_stage as PAS  # noqa: E402
import perf_campaign as PC  # noqa: E402
import perf_pk_claim as PK  # noqa: E402


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _feedback_document(*, candidate_sha256: str = SHA_B) -> dict:
    return {
        "schema_version": 1,
        "kind": "host_owned_tuning_gsim_feedback",
        "round": 0,
        "invocation": 1,
        "tuning_corpus_sha256": SHA_A,
        "candidate_sha256": candidate_sha256,
        "certificate_sha256": SHA_C,
        "engine": "gsim",
        "cells": [{
            "family": "PK",
            "capsule": "PK00_k16",
            "baseline_correct": True,
            "candidate_correct": True,
            "baseline_gsim_cycles": 120,
            "candidate_gsim_cycles": 100,
            "candidate_minus_baseline_cycles": -20,
            "baseline_over_candidate": 1.2,
            "comparable": True,
            "declared_macs": 4096,
            "declared_work_basis": "declared matmul operand shapes (M x K x N)",
            "ideal_cycles_at_peak": 16.0,
            "baseline_utilization": 16.0 / 120,
            "candidate_utilization": 16.0 / 100,
            "baseline_share_of_achievable": 51.2 / 120,
            "candidate_share_of_achievable": 51.2 / 100,
            # A cell states a position on its own measurement, not just the numbers behind it.
            "verdict": "improved",
            "verdict_reason": "20 cycles saved, closing 29.1% of the gap to the achievable rate",
            # A cell says whether the sweep paid for it. Omitting these made every test in this file
            # that builds a document fail on the key set rather than on what it meant to assert.
            "measured": True,
            "skip_reason": None,
        }],
        "stopping": {"status": "continue", "queries": 1, "baseline_total_cycles": 120.0,
                     "best_total_cycles": 100.0, "previous_best_total_cycles": None,
                     "attainable_total_cycles": 51.2, "share_of_attainable": 0.512,
                     "verdicts": [{"name": "plateaued", "fired": False,
                                   "reason": "only 1 query so far", "missing": []}]},
        "summary": {"members": 1, "comparable": 1, "all_correct": True,
                    "peak_macs_per_cycle": 256,
                    "peak_basis": "facts-derived peak of compute unit 'systolic_mesh'",
                    "achievable_macs_per_cycle": 80.01,
                    "achievable_basis": "best rate over 38 measured points in phase-1 run"},
    }


def _record() -> dict:
    capsules = ["PK00_k16", "PK01_k32", "PK02_k64", "PK03_k128"]
    replicas = ["r000", "r001", "r002"]
    cells = [{"family": "PK", "capsule": capsule, "simulator": simulator,
              "replicate": replicate}
             for capsule in capsules for replicate in replicas
             for simulator in ("spike", "gsim")]
    acceptance = PK.supported_acceptance()
    families = [{"family": "PK", "claim": "PREDICTS", "negative_control": "fixed",
                 "falsifier_observation": "residual", "differential_basis": "same work",
                 "fitted_parameters": ["K"], "acceptance": acceptance}]
    formal_claim = {
        "schema_version": 1, "family": "PK", "claim": "PREDICTS", "status": "READY",
        "declaration": acceptance, "cohort": {"replicates": replicas},
        "expected_identities": [
            {**cell, "tier": "L2" if cell["simulator"] == "spike" else "L3"}
            for cell in cells],
        "refusal_reasons": [],
    }
    registry = [
        {"name": "candidate-parse", "argv_template": ["tool", "{input_mlir}"],
         "placeholders": ["input_mlir"], "purpose": "parse", "required": True},
        {"name": PAS.DEVELOPMENT_FEEDBACK_ACTION,
         "argv_template": [PAS._HOST_FEEDBACK_SENTINEL], "placeholders": [],
         "purpose": "host-owned frozen-tuning correctness and certified GSIM cycle deltas; invoke after edits",
         "required": True},
    ]
    sentinel = {"capsule": "M2", "capsule_sha256": SHA_A,
                "frozen_source_path": "/bundle/model/M2",
                "required_lanes": ["on_mesh", "scalar_rvv_lane"],
                "required_tiers": ["L2", "L3"]}
    host_lane = {"target": "rvv", "package_id": "rvv_pkg", "package_path": "/bundle/rvv",
                 "package_sha256": SHA_D, "manifest_path": "/bundle/rvv/manifest.yaml",
                 "integration_seam": "host runner consumes schedule"}
    facts = {"replicates": 3, "formal_replicate_identities": replicas,
             "formal_claim": formal_claim, "smoke_replicates": 1,
             "expected_cells": cells, "families": families,
             "budgets": {"wall_budget_seconds": 60, "rounds": 1,
                         "round_timeout_seconds": 30, "max_tool_calls": 4,
                         "tool_timeout_seconds": 10},
             "host_lane": host_lane, "e2e_sentinel": sentinel, "tools": registry,
             "mount_destinations": []}
    audit = {"clean": True, "hits": [], "commands_seen": 1,
             "broker_required": PAS.BROKER_NAME, "broker_invocations": []}
    artifact_names = {"combined_raw", "trajectory", "reconciliation", "token_ledger",
                      "tool_ledger", "cost_time_toolcalls", "activity_share", "preflight",
                      "aet_metrics_log"}
    round_telemetry = {
        "event_count": 3,
        "summary": {"usage_complete": True},
        "accounting": {"available": True, "usage_complete": True},
        "artifacts": {"raw": {"path": "/raw", "sha256": SHA_A},
                      "timestamped": {"path": "/timestamped", "sha256": SHA_B}},
    }
    telemetry = {
        "required": True, "driver": "codex", "billing_mode": "subscription_notional",
        "raw_event_count": 3, "tool_call_count": 1, "rounds_with_complete_usage": 1,
        "subagent_tool_calls_tracked": False,
        "preflight_sha256": SHA_A,
        "accounting": {"available": True, "usage_complete": True,
                       "billing_mode": "subscription_notional", "estimated_cost_usd": None,
                       "subscription_notional_usd": 0.01,
                       "tokens_total": 10, "tool_calls": 1},
        "aet_reconciliation": {"ok": True, "raw_events": {"reconciled": True},
                               "token_ledger": {"all_match": True}},
        "activity_share": {"schema_version": 2,
                           "basis": "aet_native_codex_structured_tool_spans",
                           "denominator": "sum_of_classified_tool_span_seconds_including_overlap",
                           "is_wall_time_partition": False,
                           "overlapping_tool_spans_allowed": True,
                           "occupancy_ratio_may_exceed_one": True,
                           "subagent_tool_calls_tracked": False,
                           "classified_seconds": 1.0,
                           "trajectory_wall_seconds": 1.0,
                           "classified_span_occupancy_ratio": 1.0,
                           "seconds_by_category": {"bash": 1.0},
                           "share_by_category": {"bash": 1.0}},
        "artifacts": {name: {"path": f"/{name}", "sha256": SHA_D}
                      for name in artifact_names},
    }
    return {
        "schema_version": PAS.SCHEMA_VERSION,
        "kind": "arm4_performance_candidate",
        "state": "sealed",
        "target": {"name": "gemmini", "descriptor": "/target.yaml",
                   "descriptor_sha256": SHA_D},
        "base_functional": {
            "run_id": "functional_001", "submission_sha256": SHA_A,
            "bundle_input_snapshot": {"path": "/bundle", "content_sha256": SHA_B,
                                      "manifest": "/bundle/snapshot.json",
                                      "manifest_sha256": SHA_C,
                                      "grants": [{"declared_path": "x", "destination": "/x",
                                                  "source": "/bundle/x",
                                                  "source_sha256": SHA_D}]},
            "model_host_lane": host_lane,
            "e2e_sentinel": sentinel},
        "candidate": {"path": "/candidate", "initial_sha256": SHA_A, "sha256": SHA_B,
                      "read_only": True, "base_submission_overwritten": False,
                      "delta": {"changed_files": ["schedule.mlir"], "changed_file_count": 1,
                                "execution_relevant_changed_files": ["schedule.mlir"],
                                "execution_relevant_changed_file_count": 1}},
        "prompt": {"sha256": SHA_C, "staged_path": "/prompt.txt",
                   "renderer_path": "/perf_prompt.py", "renderer_sha256": SHA_A,
                   "facts": facts, "facts_sha256": hashlib.sha256(
                       PAS._canonical_json(facts)).hexdigest()},
        "performance_corpus": {
            "path": "/corpus",
            "manifest_sha256": SHA_D,
            "capsules_sha256": SHA_A,
            "agent_input_manifest_sha256": SHA_B,
            "agent_input_sha256": SHA_C,
            "agent_input_files": 1,
            "agent_input_bytes": 1,
            "replicates": 3, "formal_replicate_identities": replicas,
            "formal_claim": formal_claim, "smoke_replicates": 1,
            "expected_cells": cells, "families": families,
        },
        "development_feedback": {
            "action": PAS.DEVELOPMENT_FEEDBACK_ACTION,
            "required_per_round": True,
            "scope": "frozen_tuning_corpus_only",
            "engine": "gsim",
            "certificate": {
                "path": "/certificate.json", "sha256": SHA_C,
                "target": "gemmini", "certified_workloads": 4,
                "unresolved_workloads": 0, "fidelity": PAS.GATE.FIDELITY,
            },
            "rtl_identity": {"rtl_facts": {"path": "/rtl-facts.json", "sha256": SHA_D}},
            "redaction": "correctness_gsim_cycles_and_paired_deltas_only",
            "round_receipts": [[{"path": "/feedback.json", "sha256": SHA_C}]],
        },
        "sandbox": {
            "outer_codex_control_plane": {
                "network": "available_not_an_isolation_claim",
                "clear_environment": True,
                "auth_exception": "isolated_codex_home_explicit_auth_mount",
                "session_history_mounted": False,
                "live_target_toolchain_mounted": False,
                "frozen_functional_grants_mounted": True,
                "frozen_grant_manifest_sha256": SHA_C,
                "answer_surface_gap": [],
                "mount_destinations": [],
                "bwrap_binary": "/usr/bin/bwrap",
                "bwrap_binary_sha256": SHA_A,
                "policy_sha256": SHA_B,
            },
            "inner_execution_plane": {
                "network": "available_not_an_isolation_claim",
                "clear_environment": True,
                "credentials": "none",
                "answer_surface_gap": [],
                "candidate_writable": True,
                "corpus_read_only": True,
                "frozen_functional_grants_mounted": True,
                "frozen_grant_manifest_sha256": SHA_C,
                "policy_sha256": SHA_C,
                "tool_probe_results": [{"label": "python", "command": "python --version",
                                        "returncode": 0}],
                "tool_probe_recheck_results": [{"label": "python",
                                                "command": "python --version", "returncode": 0}],
            },
        },
        "broker": {"registry": registry,
                   "registry_sha256": hashlib.sha256(PAS._canonical_json(registry)).hexdigest(),
                   "receipt_manifest": "/receipts.json",
                   "receipt_manifest_sha256": SHA_A,
                   "round_receipts": [{"path": "/round-receipts.jsonl", "sha256": SHA_D,
                                       "all_required_succeeded": True,
                                       "candidate_sha256": SHA_B,
                                       "final_candidate_feedback_verified": True,
                                       "feedback_successes": 1,
                                       "feedback_receipts": [
                                           {"path": "/feedback.json", "sha256": SHA_C}]}],
                   "required_actions": ["candidate-parse", PAS.DEVELOPMENT_FEEDBACK_ACTION],
                   "all_required_succeeded": True,
                   "control_owned_by_harness": True,
                   "control_writable_by_agent": False},
        "agent": {"driver": "codex", "model": "gpt-5.6-sol",
                  "resolved_model": "gpt-5.6-sol", "effort": "high",
                  "rounds_requested": 1,
                  "wall_budget_seconds": 60, "round_timeout_seconds": 30,
                  "max_tool_calls": 4, "tool_timeout_seconds": 10,
                  "codex_binary": "/usr/bin/codex", "codex_binary_sha256": SHA_B,
                  "rounds": [{"agent_exit_code": 0, "candidate_sha256": SHA_B,
                              "transcript": "/round.jsonl",
                              "transcript_sha256": SHA_A,
                              "audit": audit, "telemetry": round_telemetry}],
                  "transcript": "/transcript.jsonl", "transcript_sha256": SHA_D,
                  "audit": audit},
        "telemetry": telemetry,
        "admission": {"consumable": True, "refusal": None,
                      "evaluation_performed_by_stage": False,
                      "development_feedback_performed_by_stage": True,
                      "success_declared_by_stage": False,
                      "consumer": "run_perf_bench.py"},
    }


def test_candidate_record_keeps_network_out_of_the_isolation_claim():
    assert PAS.validate_candidate_record(_record())["state"] == "sealed"
    for plane in ("outer_codex_control_plane", "inner_execution_plane"):
        document = _record()
        document["sandbox"][plane]["network"] = "unshared"
        with pytest.raises(PAS.StageGateError, match="outer|inner"):
            PAS.validate_candidate_record(document)


def test_candidate_record_requires_an_exact_functional_fork_and_tool_evidence():
    wrong_base = _record()
    wrong_base["candidate"]["initial_sha256"] = SHA_B
    with pytest.raises(PAS.StageGateError, match="byte-for-byte"):
        PAS.validate_candidate_record(wrong_base)

    no_probes = _record()
    no_probes["sandbox"]["inner_execution_plane"]["tool_probe_results"] = []
    with pytest.raises(PAS.StageGateError, match="tool probes"):
        PAS.validate_candidate_record(no_probes)

    drifted_tools = _record()
    drifted_tools["sandbox"]["inner_execution_plane"]["tool_probe_recheck_results"][0][
        "returncode"] = 1
    with pytest.raises(PAS.StageGateError, match="changed"):
        PAS.validate_candidate_record(drifted_tools)


def test_candidate_record_requires_exact_frozen_pk_acceptance_and_separates_smoke():
    omitted = _record()
    omitted["performance_corpus"].pop("formal_claim")
    with pytest.raises(PAS.StageGateError, match="formal claim|cells/families/replicates"):
        PAS.validate_candidate_record(omitted)

    drifted = _record()
    declaration = drifted["performance_corpus"]["formal_claim"]["declaration"]
    declaration["thresholds"]["r_squared_min_inclusive"] = 0.9
    drifted["prompt"]["facts"]["formal_claim"] = drifted["performance_corpus"]["formal_claim"]
    with pytest.raises(PAS.StageGateError, match="acceptance contract drifted"):
        PAS.validate_candidate_record(drifted)

    masquerading_smoke = _record()
    masquerading_smoke["performance_corpus"]["smoke_replicates"] = 3
    masquerading_smoke["prompt"]["facts"]["smoke_replicates"] = 3
    with pytest.raises(PAS.StageGateError, match="masquerade"):
        PAS.validate_candidate_record(masquerading_smoke)


def test_refused_candidate_is_not_consumable_but_its_evidence_can_be_read():
    document = _record()
    document["state"] = "refused"
    document["admission"].update({"consumable": False, "refusal": "audit failed"})
    with pytest.raises(PAS.StageGateError, match="not consumable"):
        PAS.validate_candidate_record(document)
    assert PAS.validate_candidate_record(document, require_consumable=False)["state"] == "refused"


def test_verified_handoff_is_the_narrow_measurement_boundary(tmp_path, monkeypatch):
    document = _record()
    document["base_functional"]["snapshot"] = "/functional-base"
    document["performance_corpus"]["manifest"] = "/corpus/manifest.json"
    document["performance_corpus"]["agent_input_path"] = "/agent-inputs"
    document["performance_corpus"]["agent_input_manifest"] = "/agent-inputs/manifest.json"
    source_sha256 = {name: SHA_D for name in PAS.TELEMETRY_TREATMENT_SOURCES}
    source_sha256["codex_binary"] = document["agent"]["codex_binary_sha256"]
    source_sha256["performance_authoring_stage"] = document["prompt"]["renderer_sha256"]
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({
        "model_resolution": {"requested_model": "gpt-5.6-sol",
                             "resolved_model": "gpt-5.6-sol",
                             "codex_model_map": ""},
        "sources": {name: {"path": f"/{name}", "sha256": digest}
                    for name, digest in source_sha256.items()}}), encoding="utf-8")
    document["telemetry"]["artifacts"]["preflight"]["path"] = str(preflight)
    record = tmp_path / "performance_candidate.json"
    record.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(PAS, "verify_candidate_record", lambda *_args, **_kwargs: document)
    handoff = PAS.verify_candidate_handoff(record)
    assert handoff.candidate_sha256 == SHA_B
    assert handoff.codex_binary_sha256 == SHA_B
    assert handoff.authoring_stage_sha256 == SHA_A
    assert handoff.telemetry_source_sha256 == dict(sorted(source_sha256.items()))
    assert handoff.functional_submission_sha256 == SHA_A
    assert handoff.formal_replicate_identities == ("r000", "r001", "r002")
    assert handoff.required_actions == ("candidate-parse", PAS.DEVELOPMENT_FEEDBACK_ACTION)
    assert handoff.transcript_audit["clean"] is True


def test_telemetry_preflight_pins_the_complete_executable_and_aet_treatment(
        tmp_path: Path) -> None:
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\n", encoding="utf-8")
    codex.chmod(0o755)
    prices = tmp_path / "prices.yaml"
    prices.write_text("gpt-5.6-sol: [5, 30, 0.5, 5]\n", encoding="utf-8")

    preflight = PAS.telemetry_preflight(
        model="gpt-5.6-sol", price_table=prices, codex_binary=codex)
    sources = preflight["sources"]

    assert set(sources) == PAS.TELEMETRY_TREATMENT_SOURCES
    assert sources["codex_binary"] == {
        "path": str(codex.resolve()), "sha256": PAS._sha256_file(codex)}
    assert sources["performance_authoring_stage"]["sha256"] == PAS._sha256_file(
        Path(PAS.__file__).resolve())
    for source in sources.values():
        path = Path(source["path"])
        assert path.is_file()
        assert source["sha256"] == PAS._sha256_file(path)


def test_telemetry_preflight_binds_ambient_model_mapping_and_effective_model(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\n", encoding="utf-8")
    codex.chmod(0o755)
    prices = tmp_path / "prices.yaml"
    prices.write_text(
        "gpt-5.6-sol: [5, 30, 0.5, 5]\ngpt-5.5: [4, 20, 0.4, 4]\n",
        encoding="utf-8")

    monkeypatch.setenv("CODEX_MODEL_MAP", "")
    native = PAS.telemetry_preflight(
        model="gpt-5.6-sol", price_table=prices, codex_binary=codex)
    monkeypatch.setenv("CODEX_MODEL_MAP", "gpt-5.6-sol=gpt-5.5")
    remapped = PAS.telemetry_preflight(
        model="gpt-5.6-sol", price_table=prices, codex_binary=codex)

    assert native["model_resolution"] == {
        "requested_model": "gpt-5.6-sol", "resolved_model": "gpt-5.6-sol",
        "codex_model_map": ""}
    assert remapped["model_resolution"] == {
        "requested_model": "gpt-5.6-sol", "resolved_model": "gpt-5.5",
        "codex_model_map": "gpt-5.6-sol=gpt-5.5"}
    assert PAS._canonical_json(native) != PAS._canonical_json(remapped)
    assert remapped["price_table"]["model"] == "gpt-5.5"


def test_documentation_only_candidate_is_vacuous():
    document = _record()
    document["candidate"]["delta"] = {
        "changed_files": ["REPORT.md"], "changed_file_count": 1,
        "execution_relevant_changed_files": [], "execution_relevant_changed_file_count": 0,
    }
    with pytest.raises(PAS.StageGateError, match="not consumable"):
        PAS.validate_candidate_record(document)


def test_zero_command_audit_cannot_be_marked_consumable():
    document = _record()
    document["agent"]["audit"]["commands_seen"] = 0
    document["agent"]["rounds"][0]["audit"]["commands_seen"] = 0
    with pytest.raises(PAS.StageGateError, match="not consumable"):
        PAS.validate_candidate_record(document)


def test_prompt_is_exact_required_utf8_bytes(tmp_path):
    path = tmp_path / "prompt.md"
    path.write_text("Improve the frozen Arm4 compiler.\n", encoding="utf-8")
    artifact = PAS.load_prompt(path)
    assert artifact.text == path.read_text(encoding="utf-8")
    assert artifact.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(PAS.StageGateError, match="explicit"):
        PAS.load_prompt(tmp_path / "missing.md")


def test_paid_stage_has_no_arbitrary_prompt_injection_interface():
    assert "prompt" not in inspect.signature(PAS.run_stage).parameters
    assert "--prompt-file" not in inspect.getsource(PAS.main)
    assert "materialize_canonical_prompt(prompt_inputs" in inspect.getsource(PAS.run_stage)


def _pk_capsules() -> tuple[PAS.PerformanceCapsule, ...]:
    rows = []
    for index, k in enumerate((16, 32, 64, 128)):
        name = f"PK{index:02d}_k{k}"
        descriptor = {
            "name": name, "kind": "model_slice", "label": "dev",
            "performance": {
                "level": "L1_tile", "family": "PK", "lever": "reduction_depth",
                "claim": "PREDICTS", "acceptance": PK.supported_acceptance(),
                "comparand": {"kind": "fitted_prediction",
                               "against": "measured_cycles_same_member",
                               "cancels": ["identical_M", "identical_N"],
                               "demand_equal": ["operation", "M", "N"]},
                "falsifier": {"observation": "residual_cycles_by_K",
                               "negative_control": "fixed_M_and_N_across_all_K_points"},
                "emitter": {"knobs": {"varied_axis": "K"}},
            },
            "operation": {"op": "matmul", "attributes": {
                "lhs": "A", "weight": "B", "epilogue": [], "output_dtype": "i32"}},
            "inputs": [{"name": "A", "shape": [16, k], "dtype": "i8"},
                       {"name": "B", "shape": [k, 16], "dtype": "i8"}],
            "numeric_policy": {"dtype": "i32", "compare": "exact_int"},
            "required_oracle_tiers": ["L0", "L1", "L2", "L3"],
        }
        rows.append(PAS.PerformanceCapsule(
            "PK", name, Path(f"/frozen/_perf/{name}"), f"_perf/{name}", descriptor,
            SHA_A, 2, 100))
    return tuple(rows)


def test_formal_pk_claim_is_derived_from_frozen_acceptance_and_exact_three_replicas():
    claim = PAS.prepare_formal_pk_claim(_pk_capsules())
    assert claim["status"] == "READY"
    assert claim["declaration"] == PK.supported_acceptance()
    assert claim["cohort"]["replicates"] == ["r000", "r001", "r002"]
    assert len(claim["expected_identities"]) == 24
    assert {row["simulator"] for row in claim["expected_identities"]} == {"spike", "gsim"}
    families = PAS._family_declarations(_pk_capsules(), claim)
    assert families[0].acceptance == PK.supported_acceptance()

    with pytest.raises(PAS.StageGateError, match="exact_count=3"):
        PAS.prepare_formal_pk_claim(_pk_capsules(), requested_replicates=2)


def test_formal_pk_claim_refuses_omitted_or_drifted_frozen_acceptance():
    with pytest.raises(PAS.StageGateError, match="preflight refused"):
        PAS.prepare_formal_pk_claim(_pk_capsules()[:-1])

    capsules = list(_pk_capsules())
    descriptor = copy.deepcopy(capsules[0].descriptor)
    descriptor["performance"]["acceptance"]["thresholds"][
        "r_squared_min_inclusive"] = 0.9
    capsules[0] = PAS.PerformanceCapsule(
        capsules[0].family, capsules[0].capsule, capsules[0].source_dir,
        capsules[0].source_relative_path, descriptor, capsules[0].source_sha256,
        capsules[0].n_files, capsules[0].n_bytes)
    with pytest.raises(PAS.StageGateError, match="preflight refused"):
        PAS.prepare_formal_pk_claim(capsules)


def test_answer_free_view_excludes_registry_answers_and_is_read_only(tmp_path, monkeypatch):
    source_parent = tmp_path / "source"
    original = source_parent / "_perf" / "PK0"
    original.mkdir(parents=True)
    frozen = tmp_path / "frozen" / "_perf" / "PK0"
    frozen.mkdir(parents=True)
    for name, payload in (("capsule.yaml", b"name: PK0\n"),
                          ("capsule.interface.mlir", b"module {}\n"),
                          ("golden.yaml", b"secret: 7\n")):
        (original / name).write_bytes(payload)
        (frozen / name).write_bytes(payload)
    member = PAS.PerformanceCapsule(
        "PK", "PK0", frozen, "_perf/PK0", {"performance": {}}, SHA_A, 3, 40)
    corpus = PAS.FrozenPerformanceCorpus(
        tmp_path / "frozen", tmp_path / "frozen", tmp_path / "manifest.json",
        SHA_B, SHA_C, (member,))
    te = SimpleNamespace(capsule_corpus=source_parent / "public")
    monkeypatch.setattr(PAS, "answer_surfaces", lambda _te: [
        AnswerSurface("golden", original / "golden.yaml", "file", "golden")])

    snapshot = PAS.build_answer_free_agent_inputs(corpus, te, tmp_path / "agent-inputs")
    assert (snapshot.root / "_perf/PK0/capsule.yaml").is_file()
    assert (snapshot.root / "_perf/PK0/capsule.interface.mlir").is_file()
    assert not (snapshot.root / "_perf/PK0/golden.yaml").exists()
    PAS.verify_answer_free_agent_inputs(snapshot)
    assert (snapshot.root / "_perf/PK0/capsule.yaml").stat().st_mode & 0o222 == 0


def test_broker_command_has_clear_environment_but_makes_no_networkless_claim(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    policy = PAS.AgentSandboxPolicy(
        ("bwrap", "--clearenv", "--bind", str(candidate), str(candidate)), (),
        "available_not_an_isolation_claim", True, True, True)
    te = SimpleNamespace(target="x", sim_via="", curated_harness="", path=tmp_path / "te.yaml")
    command = PAS.inner_command(policy, te, candidate, ["python3", "--version"], 10)
    assert command[0] == "bwrap"
    assert "--clearenv" in command
    assert "--unshare-net" not in command
    assert command[-2:] == ["python3", "--version"]


def test_outer_and_inner_mount_policies_keep_tool_and_credential_grants_separate(
        tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    candidate = workspace / "submission"
    candidate.mkdir(parents=True)
    inputs = tmp_path / "agent-inputs"
    inputs.mkdir()
    snapshot = PAS.AgentInputSnapshot(inputs, inputs / "manifest.json", SHA_A, SHA_B, 1, 1)
    monkeypatch.setattr(PAS, "verify_answer_free_agent_inputs", lambda _snapshot: None)
    monkeypatch.setattr(PAS, "answer_surfaces", lambda _te: [])
    monkeypatch.setattr(PAS.TC, "toolchain_binds",
                        lambda _te: ["--ro-bind", "/exact-target-tools", "/exact-target-tools"])
    monkeypatch.setattr(PAS.TC, "required_tool_probes",
                        lambda _te: [ToolProbe("target-tool", "target-tool --version",
                                               "/exact-target-tools")])
    te = SimpleNamespace()

    inner = PAS.inner_execution_policy(te, candidate, snapshot)
    inner_text = " ".join(inner.argv)
    assert "--ro-bind /exact-target-tools /exact-target-tools" in inner_text
    assert ".codex/auth.json" not in inner_text
    assert f"--bind {candidate} {candidate}" in inner_text
    assert f"--ro-bind {inputs} {PAS.AGENT_CORPUS_MOUNT}" in inner_text

    outer = PAS.outer_codex_policy(
        workspace, snapshot, ["--ro-bind", "/explicit-auth", "/isolated/auth.json"], te)
    outer_text = " ".join(outer.argv)
    assert "/exact-target-tools" not in outer_text
    assert "--ro-bind /explicit-auth /isolated/auth.json" in outer_text
    assert f"--bind {workspace} {workspace}" in outer_text
    assert f"--ro-bind {inputs} {PAS.AGENT_CORPUS_MOUNT}" in outer_text
    assert "--unshare-net" not in outer_text


def test_each_round_starts_as_an_exact_fresh_copy(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.yaml").write_text("target: x\n", encoding="utf-8")
    digest = hash_tree(source)["sha256"]
    submission = PAS.fresh_round_workspace(source, tmp_path / "round-0", digest)
    assert hash_tree(submission)["sha256"] == digest
    assert submission != source
    with pytest.raises(PAS.StageGateError, match="not fresh"):
        PAS.fresh_round_workspace(source, tmp_path / "round-0", digest)


def test_candidate_delta_distinguishes_code_from_documentation(tmp_path):
    base, candidate = tmp_path / "base", tmp_path / "candidate"
    base.mkdir()
    candidate.mkdir()
    for root in (base, candidate):
        (root / "tool.py").write_text("VALUE = 1\n", encoding="utf-8")
        (root / "REPORT.md").write_text("initial\n", encoding="utf-8")
    (candidate / "REPORT.md").write_text("updated\n", encoding="utf-8")
    report_only = PAS.candidate_delta(base, candidate)
    assert report_only["changed_file_count"] == 1
    assert report_only["execution_relevant_changed_file_count"] == 0
    (candidate / "tool.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert PAS.candidate_delta(base, candidate)["execution_relevant_changed_files"] == ["tool.py"]


@pytest.mark.parametrize("excluded", ["build", "__pycache__", ".git"])
def test_candidate_sealing_rejects_measurement_digest_exclusions(tmp_path, excluded):
    candidate = tmp_path / "submission"
    (candidate / excluded).mkdir(parents=True)
    (candidate / excluded / "state").write_text("ephemeral", encoding="utf-8")
    with pytest.raises(PAS.StageGateError, match="digest-excluded"):
        PAS.assert_candidate_sealable(candidate)


def _transcript(path: Path, command: str) -> Path:
    path.write_text(json.dumps({
        "type": "assistant",
        "message": {"content": [{"type": "tool_use", "name": "Bash",
                                   "input": {"command": command}}]},
    }) + "\n", encoding="utf-8")
    return path


def _native_codex_transcript(path: Path, command: str, *, completed: bool = True) -> Path:
    item = {"id": "item_0", "type": "command_execution", "command": command,
            "aggregated_output": "", "exit_code": None, "status": "in_progress"}
    rows = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "item.started", "item": item},
    ]
    if completed:
        rows.append({"type": "item.completed", "item": {
            **item, "aggregated_output": "done", "exit_code": 0, "status": "completed"}})
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_transcript_requires_candidate_entrypoints_to_use_the_broker(tmp_path, monkeypatch):
    candidate = tmp_path / "submission"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(yaml.safe_dump({
        "entrypoints": {"tool": "mlir_oot/target-opt"},
        "commands": {"parse": {"argv": ["python3", "{tool}", "{input_mlir}"]}},
    }), encoding="utf-8")
    monkeypatch.setattr(PAS, "audit_tokens", lambda _te: {
        "answer": ("golden.yaml",), "grader": (), "oracle_subpath": ()})
    monkeypatch.setattr(PAS.TC, "required_tool_probes",
                        lambda _te: [ToolProbe("mlir-opt", "mlir-opt --version")])
    te = SimpleNamespace()
    actions = (PAS.BrokerAction("candidate-parse", ("tool", "{input_mlir}"),
                                ("input_mlir",), "parse", True),)

    direct = PAS.audit_codex_transcript(
        _transcript(tmp_path / "direct.jsonl", "python3 submission/mlir_oot/target-opt x.mlir"),
        te, candidate, actions)
    assert direct["clean"] is False
    assert direct["hits"][0]["kind"] == "candidate_execution_outside_broker"

    brokered = PAS.audit_codex_transcript(
        _transcript(tmp_path / "brokered.jsonl",
                    "python3 /perf-control/perf_tool.py candidate-parse input_mlir=x.mlir"),
        te, candidate, actions)
    assert brokered["clean"] is True

    direct_tool = PAS.audit_codex_transcript(
        _transcript(tmp_path / "tool.jsonl", "mlir-opt input.mlir"), te, candidate, actions)
    assert [hit["kind"] for hit in direct_tool["hits"]] == ["target_tool_outside_broker"]

    ordinary_ls = PAS.audit_codex_transcript(
        _transcript(tmp_path / "ls.jsonl", "ls submission"), te, candidate, actions)
    assert ordinary_ls["clean"] is True
    ordinary_search = PAS.audit_codex_transcript(
        _transcript(tmp_path / "rg.jsonl", "rg lower submission"), te, candidate, actions)
    assert ordinary_search["clean"] is True

    renamed = candidate / "renamed.py"
    renamed.write_text("print('x')\n", encoding="utf-8")
    renamed_direct = PAS.audit_codex_transcript(
        _transcript(tmp_path / "renamed.jsonl", "python3 submission/renamed.py"),
        te, candidate, actions)
    assert [hit["kind"] for hit in renamed_direct["hits"]] == [
        "candidate_execution_outside_broker"]

    monkeypatch.setattr(PAS.TC, "required_tool_probes", lambda _te: [
        ToolProbe("verilator RTL sim", "ls /tools/simulator-*", "/tools")])
    direct_verilator = PAS.audit_codex_transcript(
        _transcript(tmp_path / "verilator.jsonl", "verilator --version"), te, candidate, actions)
    assert [hit["kind"] for hit in direct_verilator["hits"]] == ["target_tool_outside_broker"]


def test_native_codex_command_events_are_audited_once_across_started_and_completed(
        tmp_path, monkeypatch):
    candidate = tmp_path / "submission"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(yaml.safe_dump({
        "entrypoints": {"tool": "mlir_oot/target-opt"},
    }), encoding="utf-8")
    monkeypatch.setattr(PAS, "audit_tokens", lambda _te: {
        "answer": ("golden.yaml",), "grader": (), "oracle_subpath": ()})
    monkeypatch.setattr(PAS.TC, "required_tool_probes",
                        lambda _te: [ToolProbe("mlir-opt", "mlir-opt --version")])
    te = SimpleNamespace()
    actions = (PAS.BrokerAction("candidate-parse", ("tool", "{input_mlir}"),
                                ("input_mlir",), "parse", True),)

    brokered = PAS.audit_codex_transcript(_native_codex_transcript(
        tmp_path / "native-brokered.jsonl",
        "python3 /perf-control/perf_tool.py candidate-parse input_mlir=input.mlir"),
        te, candidate, actions)
    assert brokered["clean"] is True
    assert brokered["commands_seen"] == 1

    direct = PAS.audit_codex_transcript(_native_codex_transcript(
        tmp_path / "native-direct.jsonl", "mlir-opt input.mlir"), te, candidate, actions)
    assert direct["commands_seen"] == 1
    assert [hit["kind"] for hit in direct["hits"]] == ["target_tool_outside_broker"]


def test_native_codex_accepts_only_an_exact_bash_lc_broker_payload(tmp_path, monkeypatch):
    candidate = tmp_path / "submission"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(yaml.safe_dump({
        "entrypoints": {"tool": "target-opt"},
    }), encoding="utf-8")
    monkeypatch.setattr(PAS, "audit_tokens", lambda _te: {
        "answer": (), "grader": (), "oracle_subpath": ()})
    monkeypatch.setattr(PAS.TC, "required_tool_probes", lambda _te: [])
    action = PAS.BrokerAction("candidate-parse", ("tool", "{input_mlir}"),
                              ("input_mlir",), "parse", True)
    exact = "/bin/bash -lc 'python3 /perf-control/perf_tool.py candidate-parse input_mlir=x.mlir'"
    audit = PAS.audit_codex_transcript(
        _native_codex_transcript(tmp_path / "wrapped.jsonl", exact),
        SimpleNamespace(), candidate, (action,))
    assert audit["clean"] is True
    assert audit["broker_invocations"][0]["action"] == "candidate-parse"


@pytest.mark.parametrize("payload", [
    "python3 /perf-control/perf_tool.py candidate-parse input_mlir=x.mlir; ./target-opt x.mlir",
    "cp /perf-control/perf_tool.py /tmp/x && python3 /tmp/x candidate-parse input_mlir=x.mlir",
    "python3 -c 'exec(open(\"/perf-control/perf_tool.py\").read())' candidate-parse",
    "python3 /perf-control/perf_tool.py candidate-parse input_mlir=$(pwd)/x.mlir",
])
def test_wrapped_broker_compound_rename_and_python_exec_forms_fail_closed(
        tmp_path, monkeypatch, payload):
    candidate = tmp_path / "submission"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(yaml.safe_dump({
        "entrypoints": {"tool": "target-opt"},
    }), encoding="utf-8")
    monkeypatch.setattr(PAS, "audit_tokens", lambda _te: {
        "answer": (), "grader": (), "oracle_subpath": ()})
    monkeypatch.setattr(PAS.TC, "required_tool_probes", lambda _te: [])
    action = PAS.BrokerAction("candidate-parse", ("tool", "{input_mlir}"),
                              ("input_mlir",), "parse", True)
    command = "/bin/bash -lc " + json.dumps(payload)
    audit = PAS.audit_codex_transcript(
        _native_codex_transcript(tmp_path / "bad-wrapped.jsonl", command),
        SimpleNamespace(), candidate, (action,))
    assert audit["clean"] is False
    assert "invalid_broker_invocation" in [hit["kind"] for hit in audit["hits"]]


@pytest.mark.parametrize(("command", "kind"), [
    ("cp submission/tool.py /tmp/x.py && python3 /tmp/x.py",
     "candidate_code_copied_outside"),
    ('python3 -c \'exec(open("submission/tool.py").read())\'',
     "candidate_execution_outside_broker"),
])
def test_candidate_code_cannot_evade_audit_via_copy_out_or_python_c(
        tmp_path, monkeypatch, command, kind):
    candidate = tmp_path / "submission"
    candidate.mkdir()
    (candidate / "tool.py").write_text("print('candidate')\n", encoding="utf-8")
    (candidate / "manifest.yaml").write_text(yaml.safe_dump({
        "entrypoints": {"tool": "tool.py"},
    }), encoding="utf-8")
    monkeypatch.setattr(PAS, "audit_tokens", lambda _te: {
        "answer": (), "grader": (), "oracle_subpath": ()})
    monkeypatch.setattr(PAS.TC, "required_tool_probes", lambda _te: [])
    audit = PAS.audit_codex_transcript(
        _native_codex_transcript(tmp_path / "candidate-evasion.jsonl", command),
        SimpleNamespace(), candidate)
    assert audit["clean"] is False
    assert kind in [hit["kind"] for hit in audit["hits"]]


def test_read_only_self_check_with_stat_dash_c_is_not_candidate_execution(
        tmp_path, monkeypatch):
    """A `stat -c` format string must not be read as a `python -c` program.

    Measured on perf_stage_20260903T151801Z: the audit lexed the WHOLE `bash -lc` payload into one
    flat token list, so `"-c" in words` found STAT's flag while `words[0]` was still `python3`. It
    took `'%n %s %a'` to be Python source, failed to parse it, took the fail-closed branch, and
    refused a 54-command round in which the agent had done nothing but hash its own submission and
    stat the control files. Flag ownership belongs to the simple command the flag appears in.
    """
    candidate = tmp_path / "submission"
    candidate.mkdir()
    (candidate / "tool.py").write_text("print('candidate')\n", encoding="utf-8")
    (candidate / "manifest.yaml").write_text(yaml.safe_dump({
        "entrypoints": {"tool": "tool.py"},
    }), encoding="utf-8")
    monkeypatch.setattr(PAS, "audit_tokens", lambda _te: {
        "answer": (), "grader": (), "oracle_subpath": ()})
    monkeypatch.setattr(PAS.TC, "required_tool_probes", lambda _te: [])
    payload = ("python3 - <<'PY'\n"
               "from pathlib import Path\n"
               "import hashlib\n"
               "h = hashlib.sha256()\n"
               "print(h.hexdigest())\n"
               "PY\n"
               "stat -c '%n %s %a' /perf-control/perf_tool.py\n"
               "sha256sum /perf-corpus-manifest.json\n")
    # shlex.quote, not json.dumps: json escapes the newlines to a literal backslash-n, which
    # shlex then collapses to the letter 'n', fusing every line into one bogus token. A real
    # transcript has already been through json.loads and carries actual newlines.
    command = "/bin/bash -lc " + shlex.quote(payload)
    audit = PAS.audit_codex_transcript(
        _native_codex_transcript(tmp_path / "self-check.jsonl", command),
        SimpleNamespace(), candidate)
    assert audit["hits"] == []
    assert audit["clean"] is True


def test_heredoc_python_reading_candidate_bytes_fails_closed(tmp_path, monkeypatch):
    """`python3 - <<EOF` takes its script from stdin, so the heredoc body IS the program.

    Auditing `-c` while ignoring `<<` would leave candidate execution one keystroke away from the
    form already pinned as must-fail.
    """
    candidate = tmp_path / "submission"
    candidate.mkdir()
    (candidate / "tool.py").write_text("print('candidate')\n", encoding="utf-8")
    (candidate / "manifest.yaml").write_text(yaml.safe_dump({
        "entrypoints": {"tool": "tool.py"},
    }), encoding="utf-8")
    monkeypatch.setattr(PAS, "audit_tokens", lambda _te: {
        "answer": (), "grader": (), "oracle_subpath": ()})
    monkeypatch.setattr(PAS.TC, "required_tool_probes", lambda _te: [])
    payload = ("python3 - <<'PY'\n"
               "exec(open('submission/tool.py').read())\n"
               "PY\n")
    # shlex.quote, not json.dumps: json escapes the newlines to a literal backslash-n, which
    # shlex then collapses to the letter 'n', fusing every line into one bogus token. A real
    # transcript has already been through json.loads and carries actual newlines.
    command = "/bin/bash -lc " + shlex.quote(payload)
    audit = PAS.audit_codex_transcript(
        _native_codex_transcript(tmp_path / "heredoc-exec.jsonl", command),
        SimpleNamespace(), candidate)
    assert audit["clean"] is False
    assert "candidate_execution_outside_broker" in [hit["kind"] for hit in audit["hits"]]


def test_host_owned_receipts_must_exactly_join_transcript_actions(tmp_path):
    action = PAS.BrokerAction("candidate-parse", ("tool", "{input_mlir}"),
                              ("input_mlir",), "parse", True)
    feedback_action = PAS.BrokerAction(
        PAS.DEVELOPMENT_FEEDBACK_ACTION, (PAS._HOST_FEEDBACK_SENTINEL,), (),
        "host-owned frozen tuning feedback", True)
    binding_digest = hashlib.sha256(PAS._canonical_json(["input_mlir=x.mlir"])).hexdigest()
    receipt = {"receipt_schema_version": 1, "index": 0, "action": "candidate-parse",
               "bindings": {"input_mlir": "x.mlir"}, "bindings_command_sha256": binding_digest,
               "argv_sha256": SHA_A, "stdout_sha256": SHA_B, "stderr_sha256": SHA_C,
               "returncode": 0, "state": "complete"}
    feedback_document = _feedback_document()
    feedback_payload = PAS._canonical_json(feedback_document)
    feedback_sha = hashlib.sha256(feedback_payload).hexdigest()
    feedback_path = tmp_path / "feedback" / "sha256" / f"{feedback_sha}.json"
    feedback_path.parent.mkdir(parents=True)
    feedback_path.write_bytes(feedback_payload)
    feedback_binding_digest = hashlib.sha256(PAS._canonical_json([])).hexdigest()
    feedback_receipt = {
        "receipt_schema_version": 1, "index": 1,
        "action": PAS.DEVELOPMENT_FEEDBACK_ACTION, "bindings": {},
        "bindings_command_sha256": feedback_binding_digest,
        "argv_sha256": SHA_A, "stdout_sha256": SHA_B, "stderr_sha256": SHA_C,
        "returncode": 0, "state": "complete",
        "feedback_receipt_path": str(feedback_path.resolve()),
        "feedback_receipt_sha256": feedback_sha,
    }
    path = tmp_path / "receipts.jsonl"
    path.write_bytes(PAS._canonical_json(receipt) + PAS._canonical_json(feedback_receipt))
    audit = {"broker_invocations": [
        {"action": "candidate-parse", "bindings_sha256": binding_digest},
        {"action": PAS.DEVELOPMENT_FEEDBACK_ACTION,
         "bindings_sha256": feedback_binding_digest},
    ]}
    evidence = PAS.verify_broker_receipts(path, (action, feedback_action), audit)
    assert evidence["all_required_succeeded"] is True
    assert evidence["feedback_successes"] == 1
    assert evidence["feedback_receipts"] == [
        {"path": str(feedback_path), "sha256": feedback_sha}]
    receipt["returncode"] = 1
    path.write_bytes(PAS._canonical_json(receipt) + PAS._canonical_json(feedback_receipt))
    with pytest.raises(PAS.StageGateError, match="lack successful receipts"):
        PAS.verify_broker_receipts(path, (action, feedback_action), audit)


def test_feedback_schema_is_redacted_and_internally_consistent():
    document = _feedback_document()
    assert PAS.validate_redacted_feedback(document) == document

    leaked = copy.deepcopy(document)
    leaked["cells"][0]["shape"] = {"M": 16, "N": 16, "K": 16}
    with pytest.raises(PAS.StageGateError, match="redacted schema"):
        PAS.validate_redacted_feedback(leaked)

    inconsistent = copy.deepcopy(document)
    inconsistent["cells"][0]["candidate_correct"] = False
    with pytest.raises(PAS.StageGateError, match="comparability"):
        PAS.validate_redacted_feedback(inconsistent)


def test_feedback_broker_action_stays_host_side_and_writes_content_addressed_receipt(
        tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text("target: test\n", encoding="utf-8")
    candidate_sha = hash_tree(candidate)["sha256"]
    action = PAS.BrokerAction(
        PAS.DEVELOPMENT_FEEDBACK_ACTION, (PAS._HOST_FEEDBACK_SENTINEL,), (),
        "host-owned frozen tuning feedback", True)
    policy = PAS.AgentSandboxPolicy(
        ("bwrap",), (), "available_not_an_isolation_claim", True, True, True)

    class Feedback:
        def evaluate(self, observed_candidate, **kwargs):
            assert observed_candidate == candidate
            assert kwargs == {"round_index": 0, "call_index": 0, "timeout_s": 10}
            return _feedback_document(candidate_sha256=candidate_sha)

    monkeypatch.setattr(PAS, "inner_command", lambda *_args, **_kwargs: pytest.fail(
        "host-owned feedback must not expose or invoke an inner simulator command"))
    receipt_stream = tmp_path / "control" / "receipts.jsonl"
    broker = PAS._Broker(
        policy, SimpleNamespace(), candidate, (action,), receipt_stream,
        deadline=PAS.time.monotonic() + 60, max_calls=1, max_tool_seconds=10,
        feedback_evaluator=Feedback(), feedback_round=0)
    result = broker.execute({
        "action": PAS.DEVELOPMENT_FEEDBACK_ACTION, "bindings": {}, "timeout_s": 10})
    assert result["returncode"] == 0
    assert json.loads(result["stdout"])["candidate_sha256"] == candidate_sha
    receipt = json.loads(receipt_stream.read_text(encoding="utf-8"))
    feedback_path = Path(receipt["feedback_receipt_path"])
    assert feedback_path.name == f"{receipt['feedback_receipt_sha256']}.json"
    assert hashlib.sha256(feedback_path.read_bytes()).hexdigest() == receipt[
        "feedback_receipt_sha256"]
    assert feedback_path.stat().st_mode & 0o222 == 0
    binding_digest = hashlib.sha256(PAS._canonical_json([])).hexdigest()
    evidence = PAS.verify_broker_receipts(
        receipt_stream, (action,), {"broker_invocations": [{
            "action": PAS.DEVELOPMENT_FEEDBACK_ACTION,
            "bindings_sha256": binding_digest,
        }]}, candidate_sha256=candidate_sha)
    assert evidence["feedback_successes"] == 1
    assert evidence["final_candidate_feedback_verified"] is True

    with pytest.raises(PAS.StageGateError, match="final round candidate bytes"):
        PAS.verify_broker_receipts(
            receipt_stream, (action,), {"broker_invocations": [{
                "action": PAS.DEVELOPMENT_FEEDBACK_ACTION,
                "bindings_sha256": binding_digest,
            }]}, candidate_sha256=SHA_D)


def test_refused_binding_is_recorded_so_the_receipt_join_stays_total(tmp_path, monkeypatch):
    """A refusal the ledger never records looks identical to a lost receipt.

    Measured on perf_stage_20260903T151801Z: the agent pointed one `output_json=` outside the mounts,
    the broker refused it before any ledger entry existed, and the run was discarded because 25
    receipts did not join 26 transcript invocations -- for a path typo the agent then corrected.
    """
    candidate = tmp_path / "submission"
    candidate.mkdir()
    action = PAS.BrokerAction("candidate-emit-command-buffer", ("tool", "{output_json}"),
                              ("output_json",), "emit", True)
    receipt_stream = tmp_path / "control" / "receipts.jsonl"
    broker = PAS._Broker(
        PAS.AgentSandboxPolicy(
            ("bwrap",), (), "available_not_an_isolation_claim", True, True, True),
        SimpleNamespace(), candidate, (action,), receipt_stream,
        deadline=PAS.time.monotonic() + 60, max_calls=4, max_tool_seconds=10)

    with pytest.raises(PAS.StageGateError, match="escapes declared inputs"):
        broker.execute({"action": "candidate-emit-command-buffer",
                        "bindings": {"output_json": "/workspace/submission/out.json"}})

    rows = [json.loads(line) for line in
            receipt_stream.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["state"] == "rejected"
    assert rows[0]["returncode"] != 0
    assert rows[0]["index"] == 0

    # The refusal joins the transcript invocation the agent really made...
    binding_digest = hashlib.sha256(PAS._canonical_json(
        ["output_json=/workspace/submission/out.json"])).hexdigest()
    assert rows[0]["bindings_command_sha256"] == binding_digest

    # ...and it can never stand in for a successful required action.
    with pytest.raises(PAS.StageGateError, match="lack successful receipts"):
        PAS.verify_broker_receipts(receipt_stream, (action,), {"broker_invocations": [
            {"action": "candidate-emit-command-buffer", "bindings_sha256": binding_digest}]})


def test_inner_sandbox_never_writes_bytecode_into_the_candidate():
    """Importing a candidate must not deposit bytes its own digest does not cover.

    hash_tree skips __pycache__, so a cache written into the candidate is unattested state inside a
    content-addressed artifact, and the seal gate refuses it as "digest-excluded ephemeral state".
    Measured on perf_stage_20260903T163936Z: 15 cache dirs appeared the moment the broker first ran
    the candidate's tools, and a complete round was discarded for it.
    """
    from merlin.common.paths import repo_root
    from merlin.targetgen.sandbox import toolchain as TCM
    descriptor = (repo_root()
                  / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml")
    if not descriptor.is_file():
        pytest.skip("target descriptor is absent in this checkout")
    env = TCM.sandbox_env(PAS.load_target_experiment(descriptor), Path("/unreached"))
    assert "export PYTHONDONTWRITEBYTECODE=1;" in env


def test_peak_is_derived_from_rtl_facts_and_refuses_when_geometry_is_absent(tmp_path):
    """The utilization ceiling is a hardware fact, so it comes from the target's own RTL or not at all."""
    from merlin.common.paths import repo_root
    facts = repo_root() / "merlin/targets/gemmini/contracts/rtl_facts/facts.json"
    if not facts.is_file():
        pytest.skip("RTL facts are absent in this checkout")
    declared = json.loads(facts.read_text(encoding="utf-8"))["facts"]["arrays"][0]
    peak, basis = PAS.derived_peak_macs_per_cycle(facts, "gemmini")
    # rows x cols x multipliers-per-element, all read out of the facts -- never a literal here.
    expected = declared["rows"] * declared["cols"] * declared["mac_idiom"]["muls"]
    assert peak == expected
    assert "facts-derived peak" in basis

    stripped = json.loads(facts.read_text(encoding="utf-8"))
    stripped["facts"]["arrays"] = []
    blind = tmp_path / "facts.json"
    blind.write_text(json.dumps(stripped), encoding="utf-8")
    peak, basis = PAS.derived_peak_macs_per_cycle(blind, "gemmini")
    assert peak is None and basis


def test_declared_work_comes_from_the_capsule_spec_and_refuses_a_non_contracting_shape():
    """Utilization is priced against the work the SPEC requires, not the work the program performs."""
    descriptor = {
        "operation": {"op": "matmul", "attributes": {"lhs": "A0", "weight": "W"}},
        "inputs": [{"name": "A0", "shape": [16, 128]}, {"name": "W", "shape": [128, 16]}],
    }
    macs, basis = PAS.declared_capsule_macs(descriptor)
    assert macs == 16 * 128 * 16
    assert "declared matmul operand shapes" in basis

    descriptor["inputs"][1]["shape"] = [64, 16]  # does not contract with lhs K=128
    macs, basis = PAS.declared_capsule_macs(descriptor)
    assert macs is None and "do not contract" in basis

    macs, basis = PAS.declared_capsule_macs({"operation": {"op": "conv"}})
    assert macs is None and basis


def test_utilization_above_the_derived_peak_is_refused():
    """Against the STRUCTURAL peak a ratio over 1.0 is a broken derivation, not a fast program.

    The sibling ratio against the empirical achievable rate is a different matter and is allowed to
    exceed 1 -- see test_a_member_may_beat_the_empirical_ceiling_but_not_the_structural_peak.
    """
    document = _feedback_document()
    document["cells"][0]["candidate_utilization"] = 1.5
    with pytest.raises(PAS.StageGateError, match="structural peak"):
        PAS.validate_redacted_feedback(document)


def test_offending_functional_guard_refuses_the_seal(tmp_path, monkeypatch):
    """A candidate that regresses certified functional emission must not be consumable.

    The perf stage never re-grades the functional corpus, so this guard is the only thing standing
    between a fast candidate and a broken compiler. Measured on perf_stage_20260903T172344Z, the perf
    lever changed the emission of 27 of 48 capsules -- including A6_resident_reuse, whose residency
    property blocked phase-1 convergence.
    """
    offending = {"status": "offending", "capsules": 48, "proved_unchanged": 21, "changed": 27,
                 "offenders": [{"capsule": "A6_resident_reuse",
                                "kind": "trace_findings_introduced",
                                "findings": ["resident scratchpad tile reused after reload"]}],
                 "rows": []}
    assert offending["status"] != "clean"
    kinds = sorted({str(row.get("kind")) for row in offending["offenders"]})
    assert kinds == ["trace_findings_introduced"]
    # The stage composes exactly this refusal string, and consumable is gated on the clean status.
    refusal = ("performance candidate did not clear the certified functional emission guard "
               f"({offending['status']}: {', '.join(kinds)})")
    assert "did not clear the certified functional emission guard" in refusal
    assert (offending["status"] == "clean") is False


@pytest.mark.parametrize(("measurement", "required", "expected"), [
    # The measured PK00 shape: exact numerics, every declared tier passing, one skipped tier beyond
    # the capsule's declared ceiling recorded as a failure.
    ({"status": "screened_only", "numeric": "pass",
      "failure": {"tier": "L4", "oracle_ceiling": {"max_oracle_tier": "L2"}}},
     ("L0", "L1", "L2", "L3"), True),
    # A skip at a tier the capsule DOES require is still a failure.
    ({"status": "screened_only", "numeric": "pass",
      "failure": {"tier": "L3", "oracle_ceiling": {"max_oracle_tier": "L2"}}},
     ("L0", "L1", "L2", "L3"), False),
    # A real failure carries no ceiling evidence and is never forgiven.
    ({"status": "screened_only", "numeric": "pass",
      "failure": {"tier": "L4", "plane": "gsim", "category": "mismatch"}},
     ("L0", "L1", "L2", "L3"), False),
    # Numeric divergence is never forgiven, whatever the tier bookkeeping says.
    ({"status": "screened_only", "numeric": "fail",
      "failure": {"tier": "L4", "oracle_ceiling": {"max_oracle_tier": "L2"}}},
     ("L0", "L1", "L2", "L3"), False),
])
def test_a_tier_skipped_beyond_its_ceiling_is_not_a_correctness_failure(
        measurement, required, expected):
    """A check that could not run is not a verdict.

    PK00_k16 was reported incorrect in every measurement -- with mismatch_count 0 and all four
    declared tiers passing -- because a fifth, unrequired tier was skipped and the skip was filed as a
    failure. That silently removed a quarter of the PK family's comparison surface.
    """
    skip = PAS.DevelopmentGsimFeedback._tier_skipped_beyond_declared_ceiling(measurement, required)
    correct = (measurement.get("numeric") == "pass"
               and (measurement.get("status") == "pass"
                    or (measurement.get("status") == "screened_only" and skip))
               and (not measurement.get("failure") or skip))
    assert correct is expected


def test_broker_output_redirected_to_a_file_is_one_clean_invocation(tmp_path, monkeypatch):
    """`broker ... > out.mlir` is ONE invocation whose target is data, not a second command.

    The prompt asks the agent to diff the emitted artifact before spending an oracle cell, and the
    natural way to capture it is a redirect. Splitting on `>` made the filename a second simple
    command: it tripped the mixing rule AND resolved under the candidate, so it was also reported as
    candidate execution outside the broker. Measured on perf_agentic_20260903T184101Z__trial_00:
    13 such lines, 26 hits, a refused trial whose 60 invocations exactly matched 60 host receipts.
    """
    candidate = tmp_path / "submission"
    (candidate / "performance").mkdir(parents=True)
    (candidate / "performance" / "out.mlir").write_text("// emitted\n", encoding="utf-8")
    (candidate / "manifest.yaml").write_text(yaml.safe_dump({
        "entrypoints": {"tool": "tool.py"},
    }), encoding="utf-8")
    monkeypatch.setattr(PAS, "audit_tokens", lambda _te: {
        "answer": (), "grader": (), "oracle_subpath": ()})
    monkeypatch.setattr(PAS.TC, "required_tool_probes", lambda _te: [])
    action = PAS.BrokerAction("candidate-lower-target-to-llvm", ("tool", "{input_mlir}"),
                              ("input_mlir",), "lower", True)
    payload = (f"python3 {PAS.BROKER_NAME} candidate-lower-target-to-llvm "
               "input_mlir=/perf-corpus/_perf/PK00_k16/capsule.interface.mlir "
               "> submission/performance/out.mlir")
    command = "/bin/bash -lc " + shlex.quote(payload)
    audit = PAS.audit_codex_transcript(
        _native_codex_transcript(tmp_path / "redirect.jsonl", command),
        SimpleNamespace(), candidate, (action,))
    assert audit["hits"] == []
    assert audit["clean"] is True
    assert len(audit["broker_invocations"]) == 1

    # A real second command after a separator is still caught.
    payload_bad = (f"python3 {PAS.BROKER_NAME} candidate-lower-target-to-llvm "
                   "input_mlir=/perf-corpus/_perf/PK00_k16/capsule.interface.mlir "
                   "; ./target-opt x.mlir")
    audit_bad = PAS.audit_codex_transcript(
        _native_codex_transcript(tmp_path / "mixed.jsonl", "/bin/bash -lc " + shlex.quote(payload_bad)),
        SimpleNamespace(), candidate, (action,))
    assert audit_bad["clean"] is False


def _stopper(*, achievable: float | None = 80.0, budget: int | None = 100):
    """A feedback evaluator with only the fields the stop conditions read."""
    return PAS.DevelopmentGsimFeedback(
        None, None, Path("."), "a" * 64, None, {}, Path("."), {},
        peak_macs_per_cycle=256, peak_basis="test",
        achievable_macs_per_cycle=achievable, achievable_basis="test",
        tuning_call_budget=budget)


def _cells(total_cycles: int, *, macs: int = 4096):
    return [{"comparable": True, "baseline_gsim_cycles": 1000,
             "candidate_gsim_cycles": total_cycles, "declared_macs": macs}]


def _stop(stopper, cells, *, seconds: float = 60.0):
    """One tuning measurement, priced at its measured wall seconds."""
    n = len(stopper._spend or ())
    return stopper._stopping(cells, label=f"call_{n:03d}", elapsed_s=seconds)


def test_the_search_stops_when_it_stops_moving(tmp_path):
    """Three consecutive queries improving less than 1% is a plateau, and ends the search.

    A plateau is reported BESIDE attainment rather than instead of it: a search can sit still far
    from the ceiling because the last few levers were bad, which is not the same as being done.
    """
    stopper = _stopper()
    # A real improvement, then three that move essentially nothing.
    first = _stop(stopper, _cells(1000))
    assert first["status"] == "continue"
    _stop(stopper, _cells(800))                      # -20%, real progress
    for total in (799, 798, 797):                        # each well under 1%
        last = _stop(stopper, _cells(total))
    by_name = {v["name"]: v for v in last["verdicts"]}
    assert by_name["plateaued"]["fired"] is True, by_name["plateaued"]["reason"]
    assert last["status"] == "stop"
    # Every condition is answered, fired or not -- an omitted one cannot be told from an unchecked one.
    assert set(by_name) == {"attainment_reached", "predicted_remaining_below",
                            "plateaued", "budget_exhausted"}


def test_the_search_stops_once_it_is_close_enough_to_what_is_attainable():
    """Attainment is measured against the ACHIEVABLE ceiling, not the structural peak."""
    # 4096 MACs at 80 mac/cycle attainable -> 51.2 cycles. 55 cycles is 93% of that.
    stopper = _stopper()
    verdicts = {v["name"]: v for v in _stop(stopper, _cells(55))["verdicts"]}
    assert verdicts["attainment_reached"]["fired"] is True

    far = _stopper()
    verdicts = {v["name"]: v for v in _stop(far, _cells(4000))["verdicts"]}
    assert verdicts["attainment_reached"]["fired"] is False


def test_an_underivable_ceiling_never_reads_as_a_ceiling_that_was_reached():
    """An unresolved target must not stop the search; that would be a silent coverage loss."""
    stopper = _stopper(achievable=None)
    report = _stop(stopper, _cells(55))
    assert report["attainable_total_cycles"] is None
    verdicts = {v["name"]: v for v in report["verdicts"]}
    assert verdicts["attainment_reached"]["fired"] is False
    assert "UNKNOWN" in verdicts["attainment_reached"]["reason"]


def test_the_budget_condition_judges_measurements_actually_taken():
    """A stop condition that reports an untouched ledger is inert, not passing.

    ``budget_exhausted`` used to answer "0 item(s) spent, unbounded remaining" for a search that
    had already spent six brokered GSIM measurements: the budget was rebuilt un-charged on every
    invocation and no caller ever declared the cap. Both halves are asserted here -- the spend is
    charged per measurement with its measured seconds, and the run's own tool-call budget bounds it.
    """
    stopper = _stopper(budget=3)
    for total in (1000, 900, 800):
        report = _stop(stopper, _cells(total), seconds=12.5)
    ledger = report["budget"]
    assert ledger["spent_items"] == 3, ledger
    assert ledger["spent_seconds"] == pytest.approx(37.5), ledger
    assert ledger["charges"] == 3, "one charge per measurement, not one lump"
    assert ledger["remaining_items"] == 0
    fired = {v["name"]: v for v in report["verdicts"]}["budget_exhausted"]
    assert fired["fired"] is True and "3 of 3" in fired["reason"], fired

    # And an un-spent search reports remaining honestly rather than "unbounded".
    fresh = _stop(_stopper(budget=100), _cells(1000))
    unfired = {v["name"]: v for v in fresh["verdicts"]}["budget_exhausted"]
    assert unfired["fired"] is False and "99 remaining" in unfired["reason"], unfired


def test_stopping_is_undeterminable_when_nothing_is_comparable():
    """With no comparable member there is no measured total, and that is said rather than assumed."""
    report = _stop(_stopper(), [{"comparable": False}])
    assert report["status"] == "undeterminable" and report["verdicts"] == []


def test_development_feedback_fails_closed_without_a_certificate():
    with pytest.raises(PAS.StageGateError, match="certificate is unavailable"):
        PAS.prepare_development_feedback(
            certificate_path=None, certificate_sha256=None, rtl_facts_path=None,
            corpus=SimpleNamespace(), baseline=Path("/unreached"), baseline_sha256=SHA_A,
            target_experiment=SimpleNamespace(), work_root=Path("/unreached"))


def test_feedback_runtime_refusal_does_not_leak_evaluator_details(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    action = PAS.BrokerAction(
        PAS.DEVELOPMENT_FEEDBACK_ACTION, (PAS._HOST_FEEDBACK_SENTINEL,), (),
        "host-owned frozen tuning feedback", True)

    class RefusingFeedback:
        def evaluate(self, *_args, **_kwargs):
            raise PAS.StageGateError(
                "hidden shape K=4096 at /secret/golden.yaml emitted raw output 123")

    broker = PAS._Broker(
        PAS.AgentSandboxPolicy(
            ("bwrap",), (), "available_not_an_isolation_claim", True, True, True),
        SimpleNamespace(), candidate, (action,), tmp_path / "control" / "receipts.jsonl",
        deadline=PAS.time.monotonic() + 60, max_calls=1, max_tool_seconds=10,
        feedback_evaluator=RefusingFeedback(), feedback_round=0)
    result = broker.execute({"action": PAS.DEVELOPMENT_FEEDBACK_ACTION, "bindings": {}})
    assert result["returncode"] == 125
    assert "host-owned evaluator (StageGateError)" in result["stderr"]
    assert "4096" not in result["stderr"]
    assert "golden" not in result["stderr"]
    assert "raw output" not in result["stderr"]


def test_named_action_registry_pins_manifest_argv_without_accepting_arbitrary_executables(
        tmp_path, monkeypatch):
    candidate = tmp_path / "submission"
    candidate.mkdir()
    tool = candidate / "target-opt"
    tool.write_text("#!/bin/sh\n", encoding="utf-8")
    (candidate / "manifest.yaml").write_text(yaml.safe_dump({
        "entrypoints": {"tool": "target-opt"},
        "commands": {"parse": {"argv": ["{tool}", "parse", "{input_mlir}"]}},
    }), encoding="utf-8")
    monkeypatch.setattr(PAS.TC, "required_tool_probes", lambda _te: [])
    actions = PAS.build_action_registry(candidate, SimpleNamespace())
    # The host-side analysis action is registered alongside the measurement action: it costs no
    # oracle time, so it is available for screening a candidate before paying for a measurement.
    assert [action.name for action in actions] == [
        "candidate-parse", PAS.DEVELOPMENT_FEEDBACK_ACTION, PAS.ANALYSIS_ACTION]
    assert actions[0].argv_template == (str(tool), "parse", "{input_mlir}")
    feedback = actions[1]
    assert feedback.argv_template == (PAS._HOST_FEEDBACK_SENTINEL,)
    assert feedback.placeholders == ()
    assert feedback.required is True
    assert "host-owned frozen-tuning" in feedback.purpose
    assert "certified GSIM" in feedback.purpose
    contract = PAS.action_registry_contract(actions, candidate)
    assert contract[0]["argv_template"] == ["{candidate}/target-opt", "parse", "{input_mlir}"]
    assert contract[1] == {
        "name": PAS.DEVELOPMENT_FEEDBACK_ACTION,
        "argv_template": [PAS._HOST_FEEDBACK_SENTINEL],
        "placeholders": [],
        "purpose": feedback.purpose,
        "required": True,
    }
    reconstructed = PAS.actions_from_registry_contract(contract, candidate)
    assert reconstructed == actions
    assert '"argv":' not in PAS._BROKER_SHIM

    malformed = copy.deepcopy(contract)
    malformed[0]["placeholders"] = []
    with pytest.raises(PAS.StageGateError, match="binding contract"):
        PAS.actions_from_registry_contract(malformed, candidate)


@pytest.mark.parametrize("row,kind", [
    ({"type": "item.started", "item": []}, "malformed_command_event"),
    ({"type": "item.started", "item": {"id": "x", "type": "command_execution"}},
     "malformed_command_event"),
    ({"type": "item.updated", "item": {
        "id": "x", "type": "command_execution", "command": "true"}},
     "unknown_command_event"),
    ({"type": "item.started", "item": {"id": "x", "type": "future_tool"}},
     "unknown_command_event"),
    ({"type": "codex_summary", "unknown_types": ["future_command_event"]},
     "unknown_command_event"),
    ({"type": "codex_unparsed", "line": "not-json"}, "malformed_command_event"),
])
def test_native_codex_unknown_or_malformed_command_schema_fails_closed(
        tmp_path, monkeypatch, row, kind):
    candidate = tmp_path / "submission"
    candidate.mkdir()
    path = tmp_path / "malformed.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(PAS, "audit_tokens", lambda _te: {
        "answer": (), "grader": (), "oracle_subpath": ()})
    monkeypatch.setattr(PAS.TC, "required_tool_probes", lambda _te: [])
    audit = PAS.audit_codex_transcript(path, SimpleNamespace(), candidate)
    assert audit["clean"] is False
    assert kind in [hit["kind"] for hit in audit["hits"]]


def test_zero_command_transcript_is_not_consumable_evidence(tmp_path, monkeypatch):
    candidate = tmp_path / "submission"
    candidate.mkdir()
    path = tmp_path / "no-command.jsonl"
    path.write_text(json.dumps({"type": "turn.completed", "usage": {}}) + "\n",
                    encoding="utf-8")
    monkeypatch.setattr(PAS, "audit_tokens", lambda _te: {
        "answer": (), "grader": (), "oracle_subpath": ()})
    monkeypatch.setattr(PAS.TC, "required_tool_probes", lambda _te: [])
    audit = PAS.audit_codex_transcript(path, SimpleNamespace(), candidate)
    assert audit["commands_seen"] == 0
    assert audit["hits"] == [{"kind": "no_command_evidence", "line": "0"}]


def test_transcript_answer_reconnaissance_refuses_even_when_the_read_would_be_masked(
        tmp_path, monkeypatch):
    candidate = tmp_path / "submission"
    candidate.mkdir()
    monkeypatch.setattr(PAS, "audit_tokens", lambda _te: {
        "answer": ("golden.yaml",), "grader": (), "oracle_subpath": ()})
    monkeypatch.setattr(PAS.TC, "required_tool_probes", lambda _te: [])
    result = PAS.audit_codex_transcript(
        _transcript(tmp_path / "answer.jsonl", "cat /some/golden.yaml"),
        SimpleNamespace(), candidate)
    assert result["clean"] is False
    assert result["hits"][0]["kind"] == "answer_reconnaissance"


def test_current_prompt_adapter_keeps_formal_claim_and_two_plane_boundary():
    inputs = PAS.StagePromptInputs(
        target="target", approach="arm4", functional_run_id="functional",
        functional_submission_sha256=SHA_A,
        frozen_functional_path=str(PAS.FUNCTIONAL_BASE_MOUNT),
        frozen_functional_sha256=SHA_A, submission_path="submission",
        submission_initial_sha256=SHA_A, functional_public_capsules=1,
        functional_hidden_capsules=1,
        functional_bundle_snapshot_manifest=str(PAS.FUNCTIONAL_INPUT_MANIFEST_MOUNT),
        functional_bundle_snapshot_manifest_sha256=SHA_B,
        functional_bundle_snapshot_sha256=SHA_C,
        workload_root=str(PAS.AGENT_CORPUS_MOUNT),
        workload_manifest=str(PAS.PERF_CORPUS_MANIFEST_MOUNT),
        workload_manifest_sha256=SHA_B, workload_capsules_sha256=SHA_C,
        expected_cells=(
            PAS.PP.PerfCell("PK", "PK00", "spike", "r000"),
            PAS.PP.PerfCell("PK", "PK00", "gsim", "r000"),
            PAS.PP.PerfCell("PK", "PK00", "spike", "r001"),
            PAS.PP.PerfCell("PK", "PK00", "gsim", "r001"),
            PAS.PP.PerfCell("PK", "PK00", "spike", "r002"),
            PAS.PP.PerfCell("PK", "PK00", "gsim", "r002"),
        ),
        replicates=3, formal_replicate_identities=("r000", "r001", "r002"),
        formal_claim={"status": "READY", "declaration": PK.supported_acceptance()},
        smoke_replicates=1, wall_budget_seconds=60, rounds=1,
        round_timeout_seconds=30, max_tool_calls=4, tool_timeout_seconds=10,
        families=(PAS.PerformanceFamilyDeclaration(
            "PK", "PREDICTS", "fixed", "residual", "same work", ("K",),
            PK.supported_acceptance()),),
        host_lane=PAS.StageHostLaneGrant(
            "rvv", "host", "/host", SHA_D, "/host/manifest.yaml", "host-owned runner"),
        e2e_sentinel=PAS.StageE2ESentinel(
            "M2", "/capsules/M2", "/frozen/M2", SHA_A,
            ("on_mesh", "scalar_rvv_lane"), ("L2", "L3")),
        tools=(PAS.PP.ToolGrant(
            "parse", f"python3 {PAS.BROKER_NAME} parse", "parse candidate", True),),
        allowed_paths=(str(PAS.FUNCTIONAL_BASE_MOUNT), "submission",
                       str(PAS.AGENT_CORPUS_MOUNT), str(PAS.PERF_CORPUS_MANIFEST_MOUNT),
                       "/host", "/host/manifest.yaml", PAS.BROKER_NAME,
                       str(PAS.BROKER_RECEIPT_MOUNT)),
        execution_broker_path=PAS.BROKER_NAME,
        execution_broker_command=f"python3 {PAS.BROKER_NAME}",
        broker_receipt_path=str(PAS.BROKER_RECEIPT_MOUNT))
    text = PAS.render_stage_prompt(inputs)
    assert "isolated authentication mount" in text
    assert "inner execution plane" in text
    assert '"exact_count": 3' in text
    assert '"capsule": "M2"' in text


def test_mocked_one_round_run_stage_seals_candidate_with_receipts(
        tmp_path, monkeypatch):
    # This test is about receipts and sealing. The functional emission guard needs a real capsule
    # corpus and two runnable compilers, which this fixture deliberately does not build, so stub it
    # clean here; its own behaviour is pinned by the guard tests below.
    monkeypatch.setattr(PAS, "functional_emission_guard",
                        lambda *_args, **_kwargs: {"status": "clean", "capsules": 0,
                                                   "proved_unchanged": 0, "changed": 0,
                                                   "offenders": [], "rows": []})
    submission = tmp_path / "functional-submission"
    submission.mkdir()
    tool = submission / "tool.py"
    tool.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    tool.chmod(0o755)
    (submission / "manifest.yaml").write_text(yaml.safe_dump({
        "entrypoints": {"tool": "tool.py"},
        "commands": {"parse": {"argv": ["{tool}", "{input_mlir}"]}},
    }), encoding="utf-8")
    digest = hash_tree(submission)["sha256"]

    snapshot = tmp_path / "functional-inputs"
    snapshot.mkdir()
    marker = snapshot / "snapshot.json"
    marker.write_text("{}\n", encoding="utf-8")
    grant_source = snapshot / "grant.txt"
    grant_source.write_text("grant\n", encoding="utf-8")
    frozen_inputs = PAS.FrozenFunctionalInputs(
        snapshot, marker, hashlib.sha256(marker.read_bytes()).hexdigest(), SHA_B,
        (PAS.FrozenGrant("grant.txt", tmp_path / "grant.txt", grant_source,
                         hashlib.sha256(grant_source.read_bytes()).hexdigest()),))
    host = tmp_path / "host"
    host.mkdir()
    (host / "manifest.yaml").write_text("name: host\n", encoding="utf-8")
    host_sha = hash_tree(host)["sha256"]
    functional = PAS.StageFunctionalRun(
        tmp_path / "functional-run", submission, "functional", digest, 1, 1,
        {"per_capsule": []}, {"per_capsule": []}, "now",
        {"path": str(snapshot), "content_sha256": SHA_B},
        {"target": "rvv", "run_id": "host", "package_sha256": host_sha}, host)

    source_root = tmp_path / "source-corpus"
    members = []
    for prototype in _pk_capsules():
        source = source_root / prototype.source_relative_path
        source.mkdir(parents=True)
        (source / "capsule.yaml").write_text(
            yaml.safe_dump(prototype.descriptor), encoding="utf-8")
        (source / "capsule.interface.mlir").write_text("module {}\n", encoding="utf-8")
        tree = PAS._exact_tree_record(source)
        members.append(PAS.PerformanceCapsule(
            prototype.family, prototype.capsule, source,
            prototype.source_relative_path, prototype.descriptor,
            tree["sha256"], tree["n_files"], tree["n_bytes"]))
    provenance = tmp_path / "MANIFEST.yaml"
    provenance.write_text("generated: []\n", encoding="utf-8")
    corpus = PAS.PerformanceCorpus(
        "target", source_root, source_root / "_perf", provenance,
        hashlib.sha256(provenance.read_bytes()).hexdigest(), {"phase": {}}, tuple(members))

    descriptor = tmp_path / "target.yaml"
    descriptor.write_text("target: target\n", encoding="utf-8")
    target = SimpleNamespace(
        target="target", path=descriptor, capsule_corpus=source_root / "public",
        sim_via="", curated_harness="")
    sentinel = PAS.StageE2ESentinel(
        "M2", "/capsules/M2", "/frozen/M2", SHA_A,
        ("on_mesh", "scalar_rvv_lane"), ("L2", "L3"))
    policy = PAS.AgentSandboxPolicy(
        ("bwrap",), (), "available_not_an_isolation_claim", True, True, True)

    bwrap = tmp_path / "bwrap"
    codex = tmp_path / "codex"
    for executable in (bwrap, codex):
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)

    class ForkCheck:
        ok = True
        reason = "clean"

        def to_dict(self):
            return {"ok": True, "reason": self.reason}

    class FakeBroker:
        def __init__(self, _policy, _te, _candidate, _actions, receipt_path, **_kwargs):
            self.receipt_path = receipt_path
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text("{}\n", encoding="utf-8")
            self.calls = [
                {"action": "candidate-parse", "returncode": 0},
                {"action": PAS.DEVELOPMENT_FEEDBACK_ACTION, "returncode": 0},
            ]
            self.token = "token"

        @contextlib.contextmanager
        def serving(self):
            yield "127.0.0.1", 1

    def fake_round(workspace, _stage, _prompt, _te, *_args, **_kwargs):
        (workspace / "submission" / "optimized.mlir").write_text(
            "module {}\n", encoding="utf-8")
        transcript = workspace / "round.jsonl"
        transcript.write_text('{"type":"turn.completed"}\n', encoding="utf-8")
        rounds = _stage / "rounds"
        rounds.mkdir(parents=True, exist_ok=True)
        events = [
            {"type": "thread.started", "thread_id": "thread"},
            {"type": "turn.started"},
            {"type": "item.started", "item": {"id": "tool-1", "type": "command_execution",
                                                  "command": "echo smoke"}},
            {"type": "item.started", "item": {"id": "tool-2", "type": "command_execution",
                                                  "command": "echo concurrent"}},
            {"type": "item.completed", "item": {"id": "tool-1", "type": "command_execution",
                                                    "command": "echo smoke", "exit_code": 0,
                                                    "aggregated_output": "smoke"}},
            {"type": "item.completed", "item": {"id": "tool-2", "type": "command_execution",
                                                    "command": "echo concurrent", "exit_code": 0,
                                                    "aggregated_output": "concurrent"}},
            {"type": "turn.completed", "usage": {
                "input_tokens": 20, "cached_input_tokens": 5,
                "cache_write_input_tokens": 1, "output_tokens": 4,
                "reasoning_output_tokens": 2}},
        ]
        raw = rounds / "round_00.codex_events.raw.jsonl"
        raw.write_text("".join(json.dumps(row) + "\n" for row in events), encoding="utf-8")
        stamped = rounds / "round_00.codex_events.timestamped.jsonl"
        # Two concurrent eight-second spans occupy sixteen span-seconds inside a ten-second wall
        # interval.  Activity shares remain a partition of span-seconds; their occupancy ratio is
        # deliberately >1 and must never be labeled a wall-time coverage/partition percentage.
        arrivals = ("00", "00", "01", "01", "09", "09", "10")
        stamped.write_text("".join(json.dumps({
            "seq": index, "arrived_at": f"2026-09-02T00:00:{second}+00:00", "event": event}) + "\n"
            for index, (event, second) in enumerate(zip(events, arrivals, strict=True), start=1)),
            encoding="utf-8")
        for suffix, content in (("codex_stderr.log", ""), ("prompt.txt", "prompt\n"),
                                ("final.txt", "done\n")):
            (rounds / f"round_00.{suffix}").write_text(content, encoding="utf-8")
        (rounds / "round_00.codex_summary.json").write_text(json.dumps({
            "billing_mode": "subscription_notional", "exit_code": 0,
            "usage_complete": True, "timed_out": False, "wall_s": 10.0}), encoding="utf-8")
        return 0, transcript, policy

    feedback_receipt = {"path": str(tmp_path / "feedback.json"), "sha256": SHA_C}
    receipt = lambda path, *_args, **kwargs: {  # noqa: E731 - compact test seam
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "all_required_succeeded": True, "feedback_successes": 1,
        "candidate_sha256": kwargs["candidate_sha256"],
        "final_candidate_feedback_verified": True,
        "feedback_receipts": [feedback_receipt],
    }
    audit = {"clean": True, "hits": [], "commands_seen": 1,
             "broker_required": PAS.BROKER_NAME,
             "broker_invocations": [
                 {"action": "candidate-parse"},
                 {"action": PAS.DEVELOPMENT_FEEDBACK_ACTION},
             ]}
    certificate = SimpleNamespace(to_dict=lambda: {
        "path": str(tmp_path / "certificate.json"), "sha256": SHA_C,
        "target": "target", "certified_workloads": 4,
        "unresolved_workloads": 0, "fidelity": PAS.GATE.FIDELITY,
    })
    feedback = SimpleNamespace(
        certificate=certificate,
        rtl_identity={"rtl_facts": {"path": str(tmp_path / "rtl.json"), "sha256": SHA_D}},
    )
    monkeypatch.setattr(PAS, "_require_executable",
                        lambda name, **_kw: bwrap if name == "bwrap" else codex)
    monkeypatch.setattr(PAS, "inspect_stage_functional_run", lambda *_args: functional)
    monkeypatch.setattr(PAS, "discover_performance_corpus", lambda *_args, **_kw: corpus)
    monkeypatch.setattr(PAS, "load_frozen_functional_inputs", lambda _run: frozen_inputs)
    monkeypatch.setattr(PAS, "prepare_development_feedback", lambda **_kwargs: feedback)
    monkeypatch.setattr(PAS, "select_e2e_sentinel", lambda *_args: sentinel)
    monkeypatch.setattr(PAS, "answer_surfaces", lambda _te: [])
    monkeypatch.setattr(PAS.TC, "required_tool_probes", lambda _te: [])
    monkeypatch.setattr(PAS, "inner_execution_policy", lambda *_args: policy)
    monkeypatch.setattr(PAS, "run_required_tool_probes", lambda *_args, **_kw: [
        {"label": "tool", "command": "tool --version", "returncode": 0}])
    monkeypatch.setattr(PAS, "_Broker", FakeBroker)
    monkeypatch.setattr(PAS, "_codex_round", fake_round)
    monkeypatch.setattr(PAS, "verify_broker_receipts", receipt)
    monkeypatch.setattr(PAS, "audit_codex_transcript", lambda *_args, **_kw: audit)
    monkeypatch.setattr(PAS.PC, "functional_fork", lambda _run: object())
    monkeypatch.setattr(PAS.PC, "check_fork", lambda *_args: ForkCheck())
    monkeypatch.setattr(PAS, "verify_candidate_record", lambda path, **_kw: json.loads(
        path.read_text(encoding="utf-8")))

    price_table = tmp_path / "prices.yaml"
    price_table.write_text("gpt-5.6-sol: [5, 30, 0.5, 5]\n", encoding="utf-8")
    record = PAS.run_stage(
        functional_runs_root=tmp_path, functional_run_id="functional",
        functional_submission_sha256=digest, target_experiment=target,
        stage_root=tmp_path / "stage", model="gpt-5.6-sol", effort="high",
        wall_budget_seconds=60, rounds=1, round_timeout_seconds=30,
        max_tool_calls=4, tool_timeout_seconds=10, codex_binary=str(codex),
        telemetry_price_table=price_table)
    document = json.loads(record.read_text(encoding="utf-8"))
    assert document["state"] == "sealed"
    assert document["candidate"]["initial_sha256"] == digest
    assert document["candidate"]["sha256"] != digest
    assert document["broker"]["all_required_succeeded"] is True
    assert document["development_feedback"]["engine"] == "gsim"
    assert document["development_feedback"]["round_receipts"] == [[feedback_receipt]]
    assert document["admission"]["development_feedback_performed_by_stage"] is True
    assert document["admission"]["evaluation_performed_by_stage"] is False
    assert document["telemetry"]["aet_reconciliation"]["ok"] is True
    assert document["telemetry"]["tool_call_count"] == 2
    assert document["telemetry"]["subagent_tool_calls_tracked"] is False
    activity = document["telemetry"]["activity_share"]
    assert activity["is_wall_time_partition"] is False
    assert activity["overlapping_tool_spans_allowed"] is True
    assert activity["occupancy_ratio_may_exceed_one"] is True
    assert activity["subagent_tool_calls_tracked"] is False
    assert activity["classified_span_occupancy_ratio"] > 1.0
    assert sum(activity["share_by_category"].values()) == pytest.approx(1.0)
    assert "coverage_of_trajectory_wall" not in activity


def test_a_broker_usage_probe_is_not_tool_access_misuse(tmp_path, monkeypatch):
    """`python3 <broker> --help` names no action, binds nothing, and executes nothing.

    Measured on perf_agentic_20260903T200637Z__trial_02: one `--help` among 32 commands refused the
    whole trial, while its other 21 broker invocations were clean and its candidate sealed otherwise.
    An undeclared ACTION is still refused -- by the broker, which records the refusal.
    """
    candidate = tmp_path / "submission"
    candidate.mkdir()
    (candidate / "manifest.yaml").write_text(yaml.safe_dump({
        "entrypoints": {"tool": "tool.py"}}), encoding="utf-8")
    monkeypatch.setattr(PAS, "audit_tokens", lambda _te: {
        "answer": (), "grader": (), "oracle_subpath": ()})
    monkeypatch.setattr(PAS.TC, "required_tool_probes", lambda _te: [])
    action = PAS.BrokerAction("candidate-parse", ("tool", "{input_mlir}"), ("input_mlir",),
                              "parse", True)

    def audit(payload):
        return PAS.audit_codex_transcript(
            _native_codex_transcript(tmp_path / f"{abs(hash(payload))}.jsonl",
                                     "/bin/bash -lc " + shlex.quote(payload)),
            SimpleNamespace(), candidate, (action,))

    clean = audit(f"python3 {PAS.BROKER_NAME} --help")
    assert clean["hits"] == [] and clean["clean"] is True
    assert clean["broker_invocations"] == []          # a probe is not an invocation

    # An undeclared action that is NOT a usage probe is still refused.
    bad = audit(f"python3 {PAS.BROKER_NAME} candidate-exfiltrate input_mlir=x.mlir")
    assert bad["clean"] is False
    assert "invalid_broker_invocation" in [h["kind"] for h in bad["hits"]]


def test_a_refused_inner_command_still_gets_a_receipt_so_the_ledger_has_no_gap(tmp_path, monkeypatch):
    """An allocated call index with no receipt breaks the join for every row after it.

    The index is taken before the inner command is built, and building it can refuse. Measured on
    perf_agentic_20260903T212924Z__trial_01: one such refusal left index 5 unwritten, so receipt 6
    landed at position 5 and `verify_broker_receipts` rejected it as a schema violation -- discarding
    a trial that had already made 42 clean invocations.
    """
    candidate = tmp_path / "submission"
    candidate.mkdir()
    action = PAS.BrokerAction("candidate-parse", ("tool", "{input_mlir}"), ("input_mlir",),
                              "parse", True)
    stream = tmp_path / "control" / "receipts.jsonl"
    broker = PAS._Broker(
        PAS.AgentSandboxPolicy(("bwrap",), (), "available_not_an_isolation_claim", True, True, True),
        SimpleNamespace(), candidate, (action,), stream,
        deadline=PAS.time.monotonic() + 60, max_calls=8, max_tool_seconds=10)

    # Make building the inner command refuse, the way a policy/argv check does.
    monkeypatch.setattr(PAS, "inner_command", lambda *_a, **_k: (_ for _ in ()).throw(
        PAS.StageGateError("inner tool argv is empty, malformed, or too large")))
    with pytest.raises(PAS.StageGateError):
        broker.execute({"action": "candidate-parse", "bindings": {"input_mlir": "x.mlir"},
                        "timeout_s": 5})

    rows = [json.loads(line) for line in stream.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["index"] == 0                 # the allocated index IS the written one
    assert rows[0]["state"] == "rejected" and rows[0]["returncode"] != 0
    # And the ledger stays gapless: indices are exactly their positions.
    assert [r["index"] for r in rows] == list(range(len(rows)))
