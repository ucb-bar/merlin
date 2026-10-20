"""Frozen-input drift and incomplete remote observations never qualify a model."""

import base64
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
    assert request["queue"] == "none; submit one registered workload at a time"
    assert request["scope"].startswith("Registration/preparation only. ")
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


def direct_observation(case):
    request_path, result, invocation, manifest = completed_observation(case)
    root, selection, _, _ = case
    spec = F.load(selection)
    launcher = root / "fpga-run-elf"
    launcher.write_text("#!/bin/sh\nexit 99\n")
    launcher.chmod(0o700)
    spec["deployment"].update(mode=F.DIRECT_MODE, ssm_document="FsimRunElf",
                              launcher=str(launcher), upload_prefix="s3://example/firesim-elfs/")
    selected = spec["models"]["example"]
    selected.pop("registered_workload")
    selected.pop("registration")
    boot_path = next(Path(a["path"]) for a in selected["assets"] if a["role"] == "boot_elf")
    boot_path.write_bytes(b"\x7fELFcomplete recurrent chain")
    boot = {"role": "boot_elf", **F.pin(boot_path)}
    model = F.load(request_path)["models"][0]
    chain_path = next(Path(a["path"]) for a in selected["assets"] if a["role"] == "chain_manifest")
    save(chain_path, {"schema": F.CHAIN_SCHEMA, "phase": "phase1", "model": "example",
                      "execution_scope": "complete_chain", "compiler_sha256": manifest["compiler_sha256"],
                      "contract_sha256": manifest["contract_sha256"], "boot_elf": boot,
                      "chain_steps": 2, "output_roster": model["output_roster"]})
    save(selection, spec)
    request = F.prepare(selection)
    assert request["status"] == "prepared"
    assert not any("registration" in reason for reason in request["blockers"])
    assert request["models"][0]["launch_argv"] == [request["launcher"]["path"], str(boot_path), "example"]
    save(request_path, request)
    save(invocation, {**F.load(invocation), "DocumentName": "FsimRunElf"})
    key = "firesim-elfs/example-20261020000000.elf"
    identity = {"sha256": boot["sha256"], "bytes": boot["bytes"]}
    manifest.update(request_sha256=F.pin(request_path)["sha256"], assets=request["models"][0]["assets"],
                    elf_key=key, uploaded_elf={**identity, "version_id": "observed-version"}, consumed_elf=identity)
    manifest.pop("workload")
    save(result / "result_manifest.json", manifest)
    command = save(root / "command.json", {"Command": {
        "CommandId": "id1", "DocumentName": "FsimRunElf", "InstanceIds": ["i-example"],
        "Parameters": {"elfKey": [key]}}})
    downloaded = result / "uploaded.elf"
    downloaded.write_bytes(boot_path.read_bytes())
    response = save(root / "get-object.json", {"ContentLength": boot["bytes"], "VersionId": "observed-version",
        "ChecksumSHA256":
        base64.b64encode(bytes.fromhex(boot["sha256"])).decode()})
    upload = save(root / "upload.json", {"schema": F.UPLOAD_SCHEMA, "bucket": "example", "key": key,
                  "version_id": "observed-version", "downloaded_elf": F.pin(downloaded),
                  "s3_get_object": F.pin(response)})
    return request_path, result, invocation, command, upload


def test_direct_complete_chain_has_no_registration_and_reopens_actual_upload(case):
    request, result, invocation, command, upload = direct_observation(case)
    report = F.collect(request, result, invocation, ssm_command_path=command, elf_upload_path=upload)
    assert report["status"] == "observation_imported"
    assert report["uploaded_elf_identity_matches"] is True
    assert report["model_validation_complete"] is False


@pytest.mark.parametrize("changed", ["key", "document", "checksum", "downloaded", "consumed", "size", "version",
                                    "missing_upload", "missing_manifest"])
def test_direct_transport_does_not_infer_execution_from_shell_or_uart(case, changed):
    request, result, invocation, command, upload = direct_observation(case)
    if changed in {"key", "document"}:
        document = F.load(command)
        if changed == "key":
            document["Command"]["Parameters"]["elfKey"] = ["firesim-elfs/foreign.elf"]
        else:
            document["Command"]["DocumentName"] = "OtherDocument"
        save(command, document)
    elif changed in {"checksum", "size", "version"}:
        record = F.load(upload)
        path = Path(record["s3_get_object"]["path"])
        head = F.load(path)
        field = {"checksum": "ChecksumSHA256", "size": "ContentLength", "version": "VersionId"}[changed]
        head[field] = "wrong"
        save(path, head)
        record["s3_get_object"] = F.pin(path)
        save(upload, record)
    elif changed == "downloaded":
        record = F.load(upload)
        path = Path(record["downloaded_elf"]["path"])
        path.write_bytes(b"another executable")
        record["downloaded_elf"] = F.pin(path)
        save(upload, record)
    elif changed == "consumed":
        path = result / "result_manifest.json"
        record = F.load(path)
        record["consumed_elf"] = {"sha256": "0" * 64, "bytes": 1}
        save(path, record)
    elif changed == "missing_upload":
        upload = None
    else:
        (result / "result_manifest.json").unlink()
    report = F.collect(request, result, invocation, ssm_command_path=command, elf_upload_path=upload)
    assert report["status"] == "incomplete"
    assert report["uploaded_elf_identity_matches"] is False


def test_actual_versioned_download_does_not_require_checksum_metadata(case):
    request, result, invocation, command, upload = direct_observation(case)
    record = F.load(upload)
    response_path = Path(record["s3_get_object"]["path"])
    response = F.load(response_path)
    response.pop("ChecksumSHA256")
    save(response_path, response)
    record["s3_get_object"] = F.pin(response_path)
    save(upload, record)
    assert F.collect(request, result, invocation, ssm_command_path=command,
                     elf_upload_path=upload)["status"] == "observation_imported"


@pytest.mark.parametrize("changed", ["stage", "duplicate", "not_elf", "wrong_document", "wrong_prefix", "launcher"])
def test_direct_preparation_rejects_stage_images_or_ambiguous_transport(case, changed):
    direct_observation(case)
    _, selection, _, _ = case
    spec = F.load(selection)
    assets = spec["models"]["example"]["assets"]
    if changed == "stage":
        path = next(Path(a["path"]) for a in assets if a["role"] == "chain_manifest")
        save(path, {**F.load(path), "chain_steps": 1})
    elif changed == "duplicate":
        assets.append(next(a for a in assets if a["role"] == "boot_elf"))
    elif changed == "not_elf":
        next(Path(a["path"]) for a in assets if a["role"] == "boot_elf").write_bytes(b"not ELF")
    elif changed == "launcher":
        spec["deployment"]["launcher"] = str(case[0] / "fpga-run")
    else:
        spec["deployment"]["ssm_document" if changed == "wrong_document" else "upload_prefix"] = "wrong"
    save(selection, spec)
    with pytest.raises(ValueError):
        F.prepare(selection)


def version_selected_request(case):
    request, result, invocation, command, upload = direct_observation(case)
    selection = case[1]
    spec = F.load(selection)
    contract = Path(spec["deployment"]["manager_contract"]["path"])
    definition = {
        "schema": "merlin.post_freeze.firesim.manager_contract.v1",
        "boot_argument_prefix": "+prog0=",
        "helpers": {name: {"sha256": "a" * 64, "bytes": 10}
                    for name in ("elf_custody", "chain_observation", "manager_result")},
    }
    save(contract, definition)
    spec["deployment"].update(manager_contract=F.pin(contract), ssm_document_version="2",
                              ssm_document_sha256="b" * 64, manager_entrypoint_sha256="c" * 64)
    save(selection, spec)
    prepared = F.prepare(selection)
    save(request, prepared)
    return prepared, request, result, invocation, command, upload, definition


def test_actual_preparation_binds_reviewed_manager_contract_and_reopens_it(case):
    prepared, *_, definition = version_selected_request(case)
    assert prepared["manager_contract_definition"] == definition
    assert "launch_argv" not in prepared["models"][0]
    assert prepared["models"][0]["staging_module"] == "merlin_experiments.post_freeze.elf_upload"
    assert prepared["workload_contract"]["sha256"] != definition["helpers"]["elf_custody"]["sha256"]
    F.verify_request(prepared)


def test_version_selected_intake_refuses_old_claimed_consumed_identity(case):
    _, request, result, invocation, command, upload, _ = version_selected_request(case)
    manifest = F.load(result / "result_manifest.json")
    manifest["request_sha256"] = F.pin(request)["sha256"]
    save(result / "result_manifest.json", manifest)
    report = F.collect(request, result, invocation, ssm_command_path=command, elf_upload_path=upload)
    assert report["status"] == "incomplete"
    assert report["consumed_file_custody_status"] == "not_verified"
    assert any("missing manager actual consumed-file provenance" in text for text in report["failures"])


@pytest.mark.parametrize("version", ["$LATEST", "$DEFAULT", "02", "0", "٢", 2])
def test_selected_document_version_must_be_concrete_canonical_numeric_string(case, version):
    version_selected_request(case)
    selection = case[1]
    spec = F.load(selection)
    spec["deployment"]["ssm_document_version"] = version
    save(selection, spec)
    with pytest.raises(ValueError, match="canonical numeric SSM"):
        F.prepare(selection)
