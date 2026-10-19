"""Frozen-input drift and incomplete remote observations never qualify a model."""

import json
from pathlib import Path

import pytest
from merlin_experiments.post_freeze import firesim as F

from merlin.common.tree_hash import hash_tree


def save(path, value):
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def case(tmp_path):
    compiler = tmp_path / "compiler"
    compiler.mkdir()
    (compiler / "compiler.py").write_text("# frozen compiler\n")
    freeze = save(tmp_path / "freeze.json", {"submission_sha256": hash_tree(compiler)["sha256"]})
    stage = {"steps": 2, "outputs": [{"dtype": "f32", "shape": [1, 2]}]}
    contract = save(tmp_path / "contract.json", {"models": {"example": {
        "chain": {"kind": "recurrent", "steps": 2}, "selected_chain_steps": 2,
        "declared_arrays_per_chain": 2, "stages": {"step": stage}}}})
    launcher = tmp_path / "fpga-run"
    launcher.write_text("#!/bin/sh\nexit 99\n")
    launcher.chmod(0o700)
    selection = save(tmp_path / "selection.json", {
        "schema": F.SELECTION_SCHEMA, "phase": "phase1", "compiler": str(compiler),
        "freeze_record": {"path": str(freeze), "compiler_sha256_field": "submission_sha256"},
        "workload_contract": str(contract), "deployment": {
            "mode": "workhorse_locked_ssm_on_demand", "launcher": str(launcher),
            "region": "example-region", "manager_instance": "i-example", "ssm_document": "LockedRun",
            "agfi": "agfi-example", "results_prefix": "s3://example/results/", "target_clock_hz": 30000000,
            "observed_host_clock_hz": 110000000, "manager_contract": None},
        "budget": {"max_jobs": 1, "job_wall_s": 120, "retries": 0}, "models": {}})
    return tmp_path, selection, compiler, freeze


def test_preparation_preserves_complete_recurrence_without_launching(case):
    root, selection, _, _ = case
    request = F.prepare(selection)
    assert request["status"] == "blocked"
    assert request["paid_jobs_launched"] == 0
    assert request["scientific_qualification"] is False
    assert request["models"][0]["output_roster"] == [
        {"stage": "step", "step": i, "index": 0, "dtype": "f32", "shape": [1, 2]} for i in range(2)]
    assert any("administrator registration" in text for text in request["blockers"])
    assert not (root / "execution").exists()


@pytest.mark.parametrize("changed", ["compiler", "freeze", "launcher", "contract", "selection"])
def test_input_drift_is_rejected(case, changed):
    _, selection, compiler, freeze = case
    request = F.prepare(selection)
    paths = {"compiler": compiler / "compiler.py", "freeze": freeze,
             "launcher": Path(request["launcher"]["path"]),
             "contract": Path(request["workload_contract"]["path"]), "selection": selection}
    with paths[changed].open("a") as stream:
        stream.write("\n# changed\n")
    with pytest.raises(ValueError, match="changed"):
        F.verify_request(request)


def test_expected_freeze_digest_is_required(case):
    _, selection, _, freeze = case
    save(freeze, {"submission_sha256": "0" * 64})
    with pytest.raises(ValueError, match="disagree"):
        F.prepare(selection)


def test_phase2_reuses_contract_but_binds_its_own_freeze(case):
    _, selection, compiler, freeze = case
    (compiler / "compiler.py").write_text("# independently frozen Phase 2 compiler\n")
    save(freeze, {"submission_sha256": hash_tree(compiler)["sha256"]})
    spec = F.load(selection)
    spec["phase"] = "phase2"
    save(selection, spec)
    request = F.prepare(selection)
    assert request["phase"] == "phase2"
    assert request["compiler"]["hash_tree_sha256"] == hash_tree(compiler)["sha256"]
    assert len(request["models"][0]["output_roster"]) == 2


def test_bare_success_marker_is_only_incomplete_observation(case):
    root, selection, _, _ = case
    request_path = save(root / "request.json", F.prepare(selection))
    result = root / "result"
    result.mkdir()
    (result / "uartlog").write_text("*** PASSED *** after 900 cycles\nHost Frequency: 110 MHz\n")
    (result / "HW_CFG_SUMMARY").write_text("AGFI: agfi-example\n")
    (result / "sim-run.sh").write_text("# original simulation command\n")
    invocation = save(root / "invocation.json", {
        "InstanceId": "i-example", "DocumentName": "LockedRun", "CommandId": "id1",
        "Status": "Success", "ResponseCode": 0})
    report = F.collect(request_path, result, invocation)
    assert report["status"] == "incomplete"
    assert report["hardware_agfi_matches"] is True
    assert report["model_validation_complete"] is False
    assert "missing result_manifest.json" in report["failures"]


def completed_observation(case):
    root, selection, compiler, _ = case
    spec = F.load(selection)
    assets = []
    for role in sorted(F.ASSET_ROLES):
        path = root / (role + ".bin")
        path.write_bytes(role.encode())
        assets.append({"role": role, "path": str(path)})
    manager = save(root / "manager.json", {"schema": "test_manager_build_selection"})
    spec["deployment"]["manager_contract"] = F.pin(manager)
    registration = save(root / "registration.json", {
        "phase": "phase1", "model": "example", "workload": "frozen-example",
        "compiler_sha256": hash_tree(compiler)["sha256"],
        "contract_sha256": F.pin(Path(spec["workload_contract"]))["sha256"],
        "agfi": "agfi-example", "assets": [
            {"role": item["role"], **F.pin(Path(item["path"]))} for item in assets]})
    spec["models"] = {"example": {"registered_workload": "frozen-example",
                                 "registration": F.pin(registration), "assets": assets}}
    save(selection, spec)
    request = F.prepare(selection)
    assert request["status"] == "prepared"
    request_path = save(root / "request.json", request)
    result = root / "result"
    result.mkdir()
    (result / "uartlog").write_text("complete raw UART\n")
    (result / "HW_CFG_SUMMARY").write_text("AGFI: agfi-example\n")
    (result / "sim-run.sh").write_text("# original simulation command\n")
    invocation = save(root / "invocation.json", {
        "InstanceId": "i-example", "DocumentName": "LockedRun", "CommandId": "id1",
        "Status": "Success", "ResponseCode": 0})
    model = request["models"][0]
    outputs = []
    for i, row in enumerate(model["output_roster"]):
        path = result / f"output-{i}.bin"
        path.write_bytes(bytes([i]) * 8)
        outputs.append({**row, "file": path.name, "sha256": F.pin(path)["sha256"]})
    manifest = {"schema": F.MANAGER_SCHEMA, "phase": "phase1", "model": "example", "workload": "frozen-example",
                "compiler_sha256": request["compiler"]["hash_tree_sha256"],
                "contract_sha256": request["workload_contract"]["sha256"],
                "request_sha256": F.pin(request_path)["sha256"], "agfi": "agfi-example",
                "ssm_command_id": "id1", "assets": model["assets"], "completed_chain_steps": 2,
                "output_array_count": 2, "outputs": outputs}
    save(result / "result_manifest.json", manifest)
    return request_path, result, invocation, manifest


def test_intake_keeps_numerical_gate_pending_even_with_complete_raw_files(case):
    request, result, invocation, _ = completed_observation(case)
    report = F.collect(request, result, invocation)
    assert report["status"] == "observation_imported"
    assert report["model_validation_complete"] is False
    assert report["numeric_status"] == "pending_existing_all_output_checker"
    assert len(report["artifacts"]) == 6


@pytest.mark.parametrize("changed", ["source", "command", "array", "shape", "path", "SSM"])
def test_wrong_or_incomplete_remote_binding_never_passes(case, changed):
    request, result, invocation, manifest = completed_observation(case)
    if changed == "source":
        manifest["compiler_sha256"] = "0" * 64
    elif changed == "command":
        manifest["ssm_command_id"] = "another-command"
    elif changed == "array":
        (result / "output-1.bin").write_bytes(b"changed")
    elif changed == "shape":
        manifest["outputs"][1]["shape"] = [1, 1]
    elif changed == "path":
        manifest["outputs"][1]["file"] = "../outside.bin"
    else:
        save(invocation, {"InstanceId": "i-example", "DocumentName": "OtherDocument",
                          "CommandId": "id1", "Status": "Failed", "ResponseCode": 1})
    save(result / "result_manifest.json", manifest)
    if changed in {"array", "shape", "path"}:
        with pytest.raises(ValueError):
            F.collect(request, result, invocation)
    else:
        assert F.collect(request, result, invocation)["status"] == "incomplete"
