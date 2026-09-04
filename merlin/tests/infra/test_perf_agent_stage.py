"""Hermetic checks for the bounded Arm4 performance-authoring stage.

No test launches Codex or a simulator.  The tests pin the two-plane record contract, answer-free corpus
view, exact functional copy, broker command, and transcript refusal paths before a paid run is possible.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
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


def _record() -> dict:
    capsules = ["PK00_k16", "PK01_k32", "PK02_k64", "PK03_k128"]
    replicas = ["r000", "r001", "r002"]
    cells = [{"family": "PK", "capsule": capsule, "simulator": simulator,
              "replicate": replicate}
             for capsule in capsules for replicate in replicas
             for simulator in ("spike", "verilator")]
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
    registry = [{"name": "candidate-parse", "argv_template": ["tool", "{input_mlir}"],
                 "placeholders": ["input_mlir"], "purpose": "parse", "required": True}]
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
                                       "all_required_succeeded": True}],
                   "required_actions": ["candidate-parse"],
                   "all_required_succeeded": True,
                   "control_owned_by_harness": True,
                   "control_writable_by_agent": False},
        "agent": {"driver": "codex", "rounds_requested": 1,
                  "wall_budget_seconds": 60, "round_timeout_seconds": 30,
                  "max_tool_calls": 4, "tool_timeout_seconds": 10,
                  "codex_binary": "/usr/bin/codex", "codex_binary_sha256": SHA_B,
                  "rounds": [{"agent_exit_code": 0, "transcript": "/round.jsonl",
                              "transcript_sha256": SHA_A,
                              "audit": audit}],
                  "transcript": "/transcript.jsonl", "transcript_sha256": SHA_D,
                  "audit": audit},
        "admission": {"consumable": True, "refusal": None,
                      "evaluation_performed_by_stage": False,
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


def _pk_capsules() -> tuple[PC.PerformanceCapsule, ...]:
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
        rows.append(PC.PerformanceCapsule(
            "PK", name, Path(f"/frozen/_perf/{name}"), f"_perf/{name}", descriptor,
            SHA_A, 2, 100))
    return tuple(rows)


def test_formal_pk_claim_is_derived_from_frozen_acceptance_and_exact_three_replicas():
    claim = PAS.prepare_formal_pk_claim(_pk_capsules())
    assert claim["status"] == "READY"
    assert claim["declaration"] == PK.supported_acceptance()
    assert claim["cohort"]["replicates"] == ["r000", "r001", "r002"]
    assert len(claim["expected_identities"]) == 24
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
    capsules[0] = PC.PerformanceCapsule(
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
    member = PC.PerformanceCapsule(
        "PK", "PK0", frozen, "_perf/PK0", {"performance": {}}, SHA_A, 3, 40)
    corpus = PC.FrozenPerformanceCorpus(
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


def test_host_owned_receipts_must_exactly_join_transcript_actions(tmp_path):
    action = PAS.BrokerAction("candidate-parse", ("tool", "{input_mlir}"),
                              ("input_mlir",), "parse", True)
    binding_digest = hashlib.sha256(PAS._canonical_json(["input_mlir=x.mlir"])).hexdigest()
    receipt = {"receipt_schema_version": 1, "index": 0, "action": "candidate-parse",
               "bindings": {"input_mlir": "x.mlir"}, "bindings_command_sha256": binding_digest,
               "argv_sha256": SHA_A, "stdout_sha256": SHA_B, "stderr_sha256": SHA_C,
               "returncode": 0, "state": "complete"}
    path = tmp_path / "receipts.jsonl"
    path.write_bytes(PAS._canonical_json(receipt))
    audit = {"broker_invocations": [{"action": "candidate-parse",
                                     "bindings_sha256": binding_digest}]}
    evidence = PAS.verify_broker_receipts(path, (action,), audit)
    assert evidence["all_required_succeeded"] is True
    receipt["returncode"] = 1
    path.write_bytes(PAS._canonical_json(receipt))
    with pytest.raises(PAS.StageGateError, match="lack successful receipts"):
        PAS.verify_broker_receipts(path, (action,), audit)


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
    assert [action.name for action in actions] == ["candidate-parse"]
    assert actions[0].argv_template == (str(tool), "parse", "{input_mlir}")
    contract = PAS.action_registry_contract(actions, candidate)
    assert contract[0]["argv_template"] == ["{candidate}/target-opt", "parse", "{input_mlir}"]
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
