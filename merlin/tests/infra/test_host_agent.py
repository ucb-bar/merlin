"""Isolation and staging checks for the CPU-host Codex runner."""
from __future__ import annotations

import shutil
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from merlin.benchharness.host_agent import (
    audit_staged_inputs,
    codex_bwrap_argv,
    prepare_isolated_codex_home,
    probe_codex_bwrap_runtime,
    stage_host_workspace,
    verify_trusted_search,
    write_codex_bwrap_wrapper,
)
from merlin.benchharness import host_agent
from merlin.benchharness import capsule_descriptor
from merlin.common.paths import repo_root
from merlin.targetgen.sandbox.bwrap import is_exposed


def _inputs(tmp_path: Path):
    task = tmp_path / "task.md"; task.write_text("build a compiler\n")
    contract = tmp_path / "contract.yaml"; contract.write_text("name: cpu\n")
    plan = tmp_path / "plan.yaml"
    plan.write_text(yaml.safe_dump({"target": "cpu", "dialect_name": "host",
                                    "ops": [{"name": "map"}], "types": []}))
    public = tmp_path / "public"; public.mkdir()
    (public / "train.jsonl").write_text('{"id":"train"}\n')
    (public / "validation.jsonl").write_text('{"id":"validation"}\n')
    submission = tmp_path / "submission.md"; submission.write_text("compiler ABI\n")
    search_space = tmp_path / "space.yaml"
    search_space.write_text("version: 1\nstatus: frozen_definition\nactions: []\n")
    search_runner = tmp_path / "beam.py"; search_runner.write_text("print('search')\n")
    trusted_evaluator = tmp_path / "trusted.py"; trusted_evaluator.write_text("print('trusted')\n")
    return (task, contract, plan, submission, public, search_space, search_runner,
            trusted_evaluator)


def test_workspace_stages_public_inputs_and_arm_specific_generated_scaffold(tmp_path):
    (task, contract, plan, submission, public, search_space, search_runner,
     trusted_evaluator) = _inputs(tmp_path)
    arm1 = stage_host_workspace(
        tmp_path / "arm1", task_path=task, target_contract_path=contract,
        dialect_plan_path=plan, submission_contract_path=submission,
        public_corpus_dir=public, search_space_path=search_space,
        search_runner_path=search_runner, trusted_evaluator_path=trusted_evaluator, arm_id="arm1",
        capabilities={"public_contract"}, treatment="raw")
    arm3 = stage_host_workspace(
        tmp_path / "arm3", task_path=task, target_contract_path=contract,
        dialect_plan_path=plan, submission_contract_path=submission,
        public_corpus_dir=public, search_space_path=search_space,
        search_runner_path=search_runner, trusted_evaluator_path=trusted_evaluator, arm_id="arm3",
        capabilities={"public_contract", "cpp_targetgen_scaffold", "generated_cpu_dialect",
                      "deterministic_candidate_search"}, treatment="generated")

    assert not (arm1.path / "starter").exists()
    assert (arm3.path / "starter" / "CMakeLists.txt").is_file()
    assert (arm3.path / "starter" / "xdsl" / "host_dialect.py").is_file()
    assert (arm3.path / "policy" / "optimization_space.yaml").is_file()
    assert (arm3.path / "policy" / "beam_search.py").is_file()
    assert (arm3.path / "policy" / "trusted_evaluator.py").is_file()
    readme = (arm3.path / "policy" / "README.md").read_text()
    assert "output fixed at scratch/search_work" in readme
    assert "/usr/bin/python3 -B policy/trusted_evaluator.py" in readme
    assert "no other file is permitted" in readme
    assert not (arm1.path / "policy").exists()
    assert (arm3.path / "contracts" / "SUBMISSION_CONTRACT.md").is_file()
    assert (arm3.path / "contracts" / "capsule_descriptor.py").read_bytes() == Path(
        capsule_descriptor.__file__).read_bytes()
    fixtures = sorted((arm3.path / "contracts" / "descriptor_fixtures").glob("*.mlir"))
    assert len(fixtures) == len(capsule_descriptor.FAMILY_CODE) == 6
    assert {path.stem.split("-", 1)[0] for path in fixtures} == set(
        capsule_descriptor.FAMILY_CODE)
    assert all(
        "family_code" in path.read_text() and "semantic_operation_code" in path.read_text()
        for path in fixtures)
    assert not (arm3.path / "corpus" / "sealed").exists()
    assert audit_staged_inputs(arm3)["ok"] is True
    channel = arm3.path / ".trusted_search_channel"; channel.mkdir()
    (channel / "req_1.json").write_text("{}")
    assert audit_staged_inputs(arm3)["ok"] is True
    (arm3.path / "contracts" / "arm.yaml").write_text("changed\n")
    assert audit_staged_inputs(arm3)["ok"] is False


def test_bwrap_exposes_workspace_but_not_checkout_or_sealed_corpus(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"; workspace.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "real-codex"))
    real = tmp_path / "real-codex"; real.mkdir()
    home, _ = prepare_isolated_codex_home(tmp_path / "isolated-codex")
    argv = codex_bwrap_argv(workspace, home)

    assert is_exposed(argv, workspace)
    assert not is_exposed(argv, Path("/scratch/answer/heldout.jsonl"))
    assert not is_exposed(argv, Path("/scratch/agustin/projects/oscar-merlin/.git/config"))


def test_runtime_probe_fails_closed_and_retains_bwrap_diagnostic(monkeypatch):
    observed = {}

    def fail(argv, **kwargs):
        observed.update({"argv": argv, **kwargs})
        return subprocess.CompletedProcess(
            argv, 1, "", "bwrap: setting up uid map: Permission denied\n")

    monkeypatch.setattr(host_agent.subprocess, "run", fail)
    result = probe_codex_bwrap_runtime("/bin/true")

    assert result["ready"] is False
    assert result["returncode"] == 1
    assert result["stderr"] == "bwrap: setting up uid map: Permission denied"
    assert result["token_or_provider_work"] is False
    assert observed["stdin"] is subprocess.DEVNULL
    assert observed["timeout"] == 30
    assert observed["argv"][-2:] == ["/usr/bin/true", "--version"]


@pytest.mark.parametrize("failure", [
    subprocess.TimeoutExpired(["bwrap"], 30),
    OSError("cannot execute bwrap"),
])
def test_runtime_probe_fails_closed_on_controller_exception(monkeypatch, failure):
    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(host_agent.subprocess, "run", fail)
    result = probe_codex_bwrap_runtime("/bin/true")
    assert result["ready"] is False
    assert result["returncode"] is None
    assert result["failure"] in {"timeout", "OSError"}


def test_workspace_audit_allows_only_sanctioned_scratch(tmp_path):
    (task, contract, plan, submission, public, search_space, search_runner,
     trusted_evaluator) = _inputs(tmp_path)
    staged = stage_host_workspace(
        tmp_path / "workspace", task_path=task, target_contract_path=contract,
        dialect_plan_path=plan, submission_contract_path=submission,
        public_corpus_dir=public, search_space_path=search_space,
        search_runner_path=search_runner, trusted_evaluator_path=trusted_evaluator,
        arm_id="arm1", capabilities={"public_contract"}, treatment="raw")
    (staged.path / "scratch" / "build.tmp").write_text("transient\n")
    assert audit_staged_inputs(staged)["ok"] is True
    (staged.path / "unsanctioned.tmp").write_text("leak\n")
    audit = audit_staged_inputs(staged)
    assert audit["ok"] is False
    assert audit["unexpected"] == ["unsanctioned.tmp"]


def test_workspace_audit_rejects_same_byte_symlink_and_pycache(tmp_path):
    (task, contract, plan, submission, public, search_space, search_runner,
     trusted_evaluator) = _inputs(tmp_path)
    staged = stage_host_workspace(
        tmp_path / "workspace", task_path=task, target_contract_path=contract,
        dialect_plan_path=plan, submission_contract_path=submission,
        public_corpus_dir=public, search_space_path=search_space,
        search_runner_path=search_runner, trusted_evaluator_path=trusted_evaluator,
        arm_id="arm1", capabilities={"public_contract"}, treatment="raw")
    arm = staged.path / "contracts" / "arm.yaml"
    copy = staged.path / "scratch" / "same-arm.yaml"
    copy.write_bytes(arm.read_bytes())
    arm.unlink()
    arm.symlink_to(copy)
    audit = audit_staged_inputs(staged)
    assert audit["ok"] is False
    assert "contracts/arm.yaml" in audit["changed_or_missing"]

    arm.unlink()
    arm.write_bytes(copy.read_bytes())
    cache = staged.path / "contracts" / "__pycache__"
    cache.mkdir()
    (cache / "payload.pyc").write_bytes(b"not sanctioned")
    audit = audit_staged_inputs(staged)
    assert audit["ok"] is False
    assert "contracts/__pycache__/payload.pyc" in audit["unexpected"]


def test_trusted_search_replay_seals_policy_and_rejects_tampering(tmp_path):
    runner = repo_root() / "merlin/experiments/cpu_host_compiler_v0/beam_search.py"
    replay = repo_root() / "merlin/experiments/cpu_host_compiler_v0/trusted_search_replay.py"
    module_spec = importlib.util.spec_from_file_location("test_trusted_beam", runner)
    beam = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(beam)
    families = [
        ("contraction", "matmul", "fp32", "row_row", 1),
        ("elementwise_map", "relu", "fp32", "contiguous", 1),
        ("reduction", "sum", "fp32", "contiguous", 1),
        ("movement_layout", "copy", "fp32", "operation_defined", 1),
        ("fusion_epilogue", "matmul_bias", "fp32", "row_row", 1),
        ("runtime_parallel", "static_partition", "fp32", "contiguous", 4),
    ]
    def rows(split):
        return [{"id": f"{split}-{family}", "sha256": f"{index + 1:064x}",
                 "split": split, "family": family, "operation": operation,
                 "dtype": dtype, "shape": {"m": index + 4}, "layout": layout,
                 "state": {}, "core_count": cores}
                for index, (family, operation, dtype, layout, cores) in enumerate(families)]
    train, validation = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    for path, split in ((train, "train"), (validation, "validation")):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows(split)))
    action = {"id": "fast", "group": "schedule", "action_class": "knob",
              "stage": 1, "value": "fast", "affected_families": [
                  "contraction", "elementwise_map", "reduction", "movement_layout",
                  "fusion_epilogue", "runtime_parallel"]}
    space = tmp_path / "space.yaml"
    space.write_text(yaml.safe_dump({
        "version": 1, "status": "frozen_definition", "screen_samples_per_family": 1,
        "confirmation_samples_per_family": 1, "confirmation_width": 1,
        "max_sweeps": 4, "measurement_repeats": 6, "noise_margin": 0.02,
        "minimum_families": 3, "required_empty_sweeps": 1,
        "selection": {"minimum_pairwise_wins": 4}, "actions": [action]}))
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text('''
import argparse,json
p=argparse.ArgumentParser(); p.add_argument("--phase"); p.add_argument("--policy"); p.add_argument("--parent-policy"); p.add_argument("--capsules")
p.add_argument("--split"); p.add_argument("--repeats",type=int); p.add_argument("--output")
a=p.parse_args(); out=open(a.output,"w")
for line in open(a.capsules):
 c=json.loads(line); row={"capsule_id":c["id"],"family":c["family"],"correctness_ok":True,
 "baseline_code_sha256":"a"*64,"candidate_code_sha256":"b"*64}
 if a.phase=="screen": row.update(baseline_cycles=100,candidate_cycles=80)
 else: row.update(baseline_elapsed_ns=[10000]*a.repeats,baseline_calls=[100]*a.repeats,
                  candidate_elapsed_ns=[16000]*a.repeats,candidate_calls=[200]*a.repeats)
 out.write(json.dumps(row)+"\\n")
out.close()
''')
    first = tmp_path / "first"
    beam.run_search(space_path=space, train_path=train, validation_path=validation,
                    evaluator=[sys.executable, str(evaluator)], output=first)
    parent = beam._candidate([])
    candidate = beam._candidate([action])["candidate_sha256"]
    ledger = tmp_path / "ledger"; observations = ledger / "observations"
    observations.mkdir(parents=True)
    entries = {}
    for phase, split, repeats in (("screen", "train", 1), ("confirm", "train", 6),
                                  ("confirm", "validation", 6)):
        work = first / "evaluations" / phase / parent["candidate_sha256"][:16] / candidate[:16] / split
        target = observations / f"{candidate}_{split}_{phase}.jsonl"
        shutil.copy2(work / "observations.jsonl", target)
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        entries[f"{parent['candidate_sha256']}:{candidate}:{split}:{phase}"] = {
            "parent_candidate_sha256": parent["candidate_sha256"],
            "candidate_sha256": candidate,
            "phase": phase, "split": split,
            "measurement_repeats": repeats, "policy_sha256": sha(work / "policy.json"),
            "parent_policy_sha256": sha(work / "parent_policy.json"),
            "capsules_sha256": sha(work / "capsules.jsonl"),
            "observations": f"observations/{target.name}",
            "observations_sha256": sha(target),
        }
    workspace = tmp_path / "workspace"; submission = workspace / "submission"
    search = submission / "search"; search.mkdir(parents=True)
    shutil.copy2(first / "search_record.json", search / "search_record.json")
    shutil.copy2(first / "selected_policy.json", search / "selected_policy.json")
    manifest = {
        "version": 1, "build": {"command": ["python3", "build.py"]},
        "compiler": {"command": ["build/compiler", "{input_mlir}", "{output_dir}",
                                  "{mode}", "{harts}", "{vlen_bits}"]},
        "policy": "policy.yaml",
    }
    (submission / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    shutil.copy2(search / "selected_policy.json", submission / "policy.yaml")
    (submission / "compiler.cc").write_text("// frozen compiler source\n")
    (submission / "build.py").write_text("# reproducible build\n")

    snapshot = ledger / "submission_snapshot"
    shutil.copytree(submission, snapshot)
    shutil.rmtree(snapshot / "search")
    prebuilt = ledger / "prebuilt_search_package"
    shutil.copytree(snapshot, prebuilt)
    (prebuilt / "build").mkdir()
    (prebuilt / "build" / "compiler").write_text("#!/bin/sh\nexit 0\n")
    (prebuilt / "build" / "compiler").chmod(0o755)
    before_tree = host_agent._package_tree_identity(snapshot)
    built_tree = host_agent._package_tree_identity(prebuilt)
    private_manifest = dict(manifest)
    private_manifest["build"] = {"command": ["/bin/true"]}
    (prebuilt / "manifest.yaml").write_text(yaml.safe_dump(private_manifest, sort_keys=False))
    sealed_tree = host_agent._package_tree_identity(prebuilt)

    receipts_dir = ledger / "receipts"; receipts_dir.mkdir()
    requests_dir = ledger / "requests"; requests_dir.mkdir()
    private_dir = ledger / "private_corpus"; private_dir.mkdir()
    private_secret = bytes.fromhex("ab" * 32)
    private_splits = {}
    for phase, split in (("screen", "train"), ("confirm", "train"),
                         ("confirm", "validation")):
        public_private_sample = beam.select_semantic_sample(
            rows(split), per_family=1,
            families=(action["affected_families"] if phase == "confirm" else None))
        private_rows = [host_agent._recompute_private_capsule(
            row, secret=private_secret, phase=phase, split=split)
                        for row in public_private_sample]
        private_path = private_dir / f"{phase}_{split}.jsonl"
        private_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                        for row in private_rows))
        private_splits[f"{phase}:{split}"] = {
            "path": f"private_corpus/{private_path.name}",
            "sha256": hashlib.sha256(private_path.read_bytes()).hexdigest(),
            "count": len(private_rows),
            "aliases": {private["id"]: public["id"]
                        for private, public in zip(
                            private_rows, public_private_sample, strict=True)},
        }
    for evaluation in entries.values():
        private_record = private_splits[f"{evaluation['phase']}:{evaluation['split']}"]
        evaluation["private_capsules_sha256"] = private_record["sha256"]
        evaluation["private_capsule_ids"] = list(private_record["aliases"])
        evaluation["request_multiplicity"] = 1
    terminal_receipts = {}
    for ordinal, (key, evaluation) in enumerate(entries.items()):
        phase, split, repeats = evaluation["phase"], evaluation["split"], evaluation[
            "measurement_repeats"]
        work = (first / "evaluations" / phase / parent["candidate_sha256"][:16] /
                candidate[:16] / split)
        request = requests_dir / f"request_{ordinal}.json"
        request_value = {
            "version": 1, "phase": phase, "split": split, "repeats": repeats,
            "policy": str(work / "policy.json"),
            "parent_policy": str(work / "parent_policy.json"),
            "capsules": str(work / "capsules.jsonl"),
        }
        request.write_text(json.dumps(request_value))
        parsed_request = {
            "version": 1, "phase": phase, "split": split, "repeats": repeats,
            "policy": request_value["policy"],
            "parent_policy": request_value["parent_policy"],
            "capsules": request_value["capsules"],
            "parent_candidate_sha256": parent["candidate_sha256"],
            "candidate_sha256": candidate,
            "parent_policy_sha256": evaluation["parent_policy_sha256"],
            "policy_sha256": evaluation["policy_sha256"],
            "capsules_sha256": evaluation["capsules_sha256"],
        }
        receipt = receipts_dir / f"request_{ordinal}.json"
        receipt.write_text(json.dumps({
            "version": 1, "authority": "driver_trusted_search_broker",
            "request_id": f"request_{ordinal}", "status": "pass",
            "request_artifact": f"requests/{request.name}",
            "request_sha256": hashlib.sha256(request.read_bytes()).hexdigest(),
            "evaluation_key": key, "cache_hit": False, "multiplicity": 1,
            "parsed_request": parsed_request,
            "response_sha256": evaluation["observations_sha256"], "wall_ns": 1,
        }))
        terminal_receipts[f"request_{ordinal}"] = {
            "path": f"receipts/{receipt.name}",
            "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
            "status": "pass", "evaluation_key": key, "cache_hit": False,
            "multiplicity": 1, "response_sha256": evaluation["observations_sha256"],
        }
        if ordinal == 0:
            duplicate_request = requests_dir / "request_cached_0.json"
            duplicate_request.write_bytes(request.read_bytes())
            duplicate = receipts_dir / "request_cached_0.json"
            duplicate.write_text(json.dumps({
                "version": 1, "authority": "driver_trusted_search_broker",
                "request_id": "request_cached_0", "status": "pass",
                "request_artifact": f"requests/{duplicate_request.name}",
                "request_sha256": hashlib.sha256(duplicate_request.read_bytes()).hexdigest(),
                "evaluation_key": key, "cache_hit": True, "multiplicity": 2,
                "parsed_request": parsed_request,
                "response_sha256": evaluation["observations_sha256"], "wall_ns": 1,
            }))
            terminal_receipts["request_cached_0"] = {
                "path": f"receipts/{duplicate.name}",
                "sha256": hashlib.sha256(duplicate.read_bytes()).hexdigest(),
                "status": "pass", "evaluation_key": key, "cache_hit": True,
                "multiplicity": 2, "response_sha256": evaluation["observations_sha256"],
            }
            evaluation["request_multiplicity"] = 2
    index = {"version": 1, "heldout_opened": False, "measurement_repeats": 6,
             "budget": {
                 "screen_evaluations_used": 1, "screen_evaluation_limit": 10,
                 "confirmation_requests_used": 3, "confirmation_request_limit": 10,
                 "package_builds_used": 6, "package_build_limit": 20,
                 "compiler_invocations_used": 12, "compiler_invocation_limit": 40,
                 "spike_checks_used": 12, "spike_check_limit": 40,
                 "k1_programs_used": 120, "k1_program_limit": 2400,
                 "planning_upper_search_seconds": 1000.0, "deadline_exceeded": False},
             "submission_tree_sha256": host_agent._submission_presearch_digest(snapshot),
             "prebuilt_package_sha256": host_agent._submission_presearch_digest(prebuilt),
             "private_shape_corpus": {
                 "authority": "controller_private_after_compiler_snapshot",
                 "secret_hex": private_secret.hex(), "splits": private_splits,
             },
             "private_prebuild": {
                 "authority": "driver_private_prebuild",
                 "private_build_override": ["/bin/true"],
                 "real_build_commands": [["python3", "build.py"]],
                 "real_build_logs": [{"command": ["python3", "build.py"], "returncode": 0,
                                      "wall_seconds": 0.1, "stdout_tail": "", "stderr_tail": ""}],
                 "submitted_manifest_sha256": hashlib.sha256(
                     (snapshot / "manifest.yaml").read_bytes()).hexdigest(),
                 "private_manifest_sha256": hashlib.sha256(
                     (prebuilt / "manifest.yaml").read_bytes()).hexdigest(),
                 "prebuild_tree_sha256": before_tree, "built_tree_sha256": built_tree,
                 "sealed_prebuilt_tree_sha256": sealed_tree,
                 "submitted_entrypoint_identity": None,
                 "built_entrypoint_identity": host_agent._compiler_entrypoint_identity(
                     prebuilt, private_manifest),
             },
             "broker_terminal": {"status": "stopped", "start_monotonic_ns": 1,
                                 "end_monotonic_ns": 2, "wall_ns": 1},
             "terminal_receipts": terminal_receipts,
             "evaluations": entries}
    (ledger / "index.json").write_text(json.dumps(index))
    seal = verify_trusted_search(
        workspace=workspace, ledger=ledger, space_path=space, runner_path=runner,
        replay_path=replay, train_path=train, validation_path=validation)
    assert seal["status"] == "pass", seal
    assert seal["checks"]["exact_final_search_file_set"] is True
    assert seal["checks"]["terminal_receipt_associations"] is True
    receipt = receipts_dir / "request_0.json"
    original_receipt = receipt.read_bytes()
    receipt.write_text('{"status":"fail"}\n')
    failed_receipt = verify_trusted_search(
        workspace=workspace, ledger=ledger, space_path=space, runner_path=runner,
        replay_path=replay, train_path=train, validation_path=validation)
    assert failed_receipt["status"] == "fail"
    assert failed_receipt["checks"]["all_requests_have_passing_terminal_receipts"] is False
    assert failed_receipt["failure_class"] == "harness_invalid"
    receipt.write_bytes(original_receipt)
    mismatched_value = json.loads(receipt.read_text())
    mismatched_value["response_sha256"] = "c" * 64
    receipt.write_text(json.dumps(mismatched_value))
    terminal_receipts["request_0"]["sha256"] = hashlib.sha256(
        receipt.read_bytes()).hexdigest()
    terminal_receipts["request_0"]["response_sha256"] = "c" * 64
    (ledger / "index.json").write_text(json.dumps(index))
    mismatched = verify_trusted_search(
        workspace=workspace, ledger=ledger, space_path=space, runner_path=runner,
        replay_path=replay, train_path=train, validation_path=validation)
    assert mismatched["status"] == "fail"
    assert mismatched["checks"]["terminal_receipt_associations"] is False
    receipt.write_bytes(original_receipt)
    terminal_receipts["request_0"]["sha256"] = hashlib.sha256(
        receipt.read_bytes()).hexdigest()
    terminal_receipts["request_0"]["response_sha256"] = entries[
        json.loads(receipt.read_text())["evaluation_key"]]["observations_sha256"]
    (ledger / "index.json").write_text(json.dumps(index))
    path_mismatch = json.loads(receipt.read_text())
    path_mismatch["parsed_request"]["policy"] = "/different/request/policy.json"
    receipt.write_text(json.dumps(path_mismatch))
    terminal_receipts["request_0"]["sha256"] = hashlib.sha256(
        receipt.read_bytes()).hexdigest()
    (ledger / "index.json").write_text(json.dumps(index))
    mismatched_request = verify_trusted_search(
        workspace=workspace, ledger=ledger, space_path=space, runner_path=runner,
        replay_path=replay, train_path=train, validation_path=validation)
    assert mismatched_request["status"] == "fail"
    assert mismatched_request["checks"]["terminal_receipt_associations"] is False
    receipt.write_bytes(original_receipt)
    terminal_receipts["request_0"]["sha256"] = hashlib.sha256(
        receipt.read_bytes()).hexdigest()
    (ledger / "index.json").write_text(json.dumps(index))
    orphan = entries.pop(next(iter(entries)))
    (ledger / "index.json").write_text(json.dumps(index))
    orphaned = verify_trusted_search(
        workspace=workspace, ledger=ledger, space_path=space, runner_path=runner,
        replay_path=replay, train_path=train, validation_path=validation)
    assert orphaned["status"] == "fail"
    assert orphaned["checks"]["terminal_receipt_associations"] is False
    # Restore by its digest-bound evaluation key; receipts must once again associate exactly.
    orphan_key = (f"{orphan['parent_candidate_sha256']}:{orphan['candidate_sha256']}:"
                  f"{orphan['split']}:{orphan['phase']}")
    entries[orphan_key] = orphan
    (ledger / "index.json").write_text(json.dumps(index))
    private_path = private_dir / "confirm_validation.jsonl"
    original_private = private_path.read_bytes()
    private_path.write_bytes(original_private + b"{}\n")
    failed_private = verify_trusted_search(
        workspace=workspace, ledger=ledger, space_path=space, runner_path=runner,
        replay_path=replay, train_path=train, validation_path=validation)
    assert failed_private["status"] == "fail"
    assert failed_private["checks"]["controller_private_shape_corpus"] is False
    assert failed_private["failure_class"] == "harness_invalid"
    private_path.write_bytes(original_private)
    (search / "extra.log").write_text("untrusted extra\n")
    extra = verify_trusted_search(
        workspace=workspace, ledger=ledger, space_path=space, runner_path=runner,
        replay_path=replay, train_path=train, validation_path=validation)
    assert extra["status"] == "fail"
    assert extra["checks"]["exact_final_search_file_set"] is False
    (search / "extra.log").unlink()
    (submission / "policy.yaml").write_text("{}\n")
    tampered = verify_trusted_search(
        workspace=workspace, ledger=ledger, space_path=space, runner_path=runner,
        replay_path=replay, train_path=train, validation_path=validation)
    assert tampered["status"] == "fail"
    assert tampered["checks"]["submission_policy_byte_match"] is False


def test_failed_broker_receipt_preserves_typed_build_failure_association(tmp_path):
    ledger = tmp_path / "ledger"
    requests = ledger / "requests"; requests.mkdir(parents=True)
    receipts = ledger / "receipts"; receipts.mkdir()
    request = requests / "build_fail.json"
    request.write_text(json.dumps({
        "version": 1, "phase": "screen", "split": "train", "repeats": 1,
        "policy": "/work/policy.json", "parent_policy": "/work/parent.json",
        "capsules": "/work/capsules.jsonl",
    }))
    parsed = {
        "version": 1, "phase": "screen", "split": "train", "repeats": 1,
        "policy": "/work/policy.json", "parent_policy": "/work/parent.json",
        "capsules": "/work/capsules.jsonl",
        "parent_candidate_sha256": "a" * 64, "candidate_sha256": "b" * 64,
        "parent_policy_sha256": "c" * 64, "policy_sha256": "d" * 64,
        "capsules_sha256": "e" * 64,
    }
    key = f"{'a' * 64}:{'b' * 64}:train:screen"
    receipt = receipts / "build_fail.json"
    receipt.write_text(json.dumps({
        "version": 1, "authority": "driver_trusted_search_broker",
        "request_id": "build_fail", "status": "fail",
        "request_artifact": "requests/build_fail.json",
        "request_sha256": hashlib.sha256(request.read_bytes()).hexdigest(),
        "evaluation_key": key, "cache_hit": False, "multiplicity": 1,
        "parsed_request": parsed, "response_sha256": None,
        "failure_class": "treatment_build_fail",
        "error": "submission build failed", "wall_ns": 10,
    }))
    record = {
        "path": "receipts/build_fail.json",
        "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "status": "fail", "evaluation_key": key, "cache_hit": False,
        "multiplicity": 1, "response_sha256": None,
    }
    index = {"terminal_receipts": {"build_fail": record}, "evaluations": {}}
    summary = host_agent._terminal_receipt_summary(ledger, index)
    assert summary == {"integrity": True, "all_pass": False,
                       "failure_classes": ["treatment_build_fail"]}
    record["evaluation_key"] = "orphan"
    assert host_agent._terminal_receipt_summary(ledger, index)["integrity"] is False


def test_compiler_seal_covers_final_search_tree_and_excludes_only_policy(tmp_path):
    workspace = tmp_path / "workspace"
    submission = workspace / "submission"
    search = submission / "search"
    search.mkdir(parents=True)
    (submission / "manifest.yaml").write_text("version: 1\npolicy: policy.yaml\n")
    (submission / "compiler.cc").write_text("// compiler\n")
    (submission / "policy.yaml").write_text("version: 1\n")
    (search / "selected_policy.json").write_text('{"version":1}\n')
    (search / "search_record.json").write_text('{"status":"converged"}\n')

    seal = host_agent.create_compiler_seal(
        workspace=workspace, search_seal={"status": "not_required"})
    source_before = seal["compiler_source_sha256"]
    package_before = seal["compiler_package_sha256"]

    # Policy has its own identity: changing it changes the package, but not the source identity.
    (submission / "policy.yaml").write_text("version: 2\n")
    assert host_agent._submission_source_digest(submission) == source_before
    assert host_agent._submission_package_digest(submission) != package_before
    (submission / "policy.yaml").write_text("version: 1\n")

    # Search records are executable compiler provenance and must never fall through the seal.
    (search / "search_record.json").write_text('{"status":"tampered"}\n')
    assert host_agent._submission_source_digest(submission) != source_before
    assert host_agent._submission_package_digest(submission) != package_before


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap is unavailable")
def test_generated_wrapper_runs_a_fake_codex_inside_bwrap(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"; workspace.mkdir()
    fake = workspace / "fake-codex"
    fake.write_text("#!/bin/sh\nprintf 'done' > \"$3\"\n")
    fake.chmod(0o700)
    real = tmp_path / "real-codex"; real.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(real))
    home, _ = prepare_isolated_codex_home(tmp_path / "isolated-codex")
    wrapper = write_codex_bwrap_wrapper(
        tmp_path / "wrapper", workspace=workspace, codex_home=home, codex_binary=fake)
    output = tmp_path / "output.txt"; output.touch()
    proc = __import__("subprocess").run(
        [str(wrapper), "exec", "--output-last-message", str(output)], capture_output=True, text=True)
    if proc.returncode and "Operation not permitted" in proc.stderr:
        pytest.skip("kernel disallows unprivileged bwrap")
    assert proc.returncode == 0, proc.stderr
    assert output.read_text() == "done"
