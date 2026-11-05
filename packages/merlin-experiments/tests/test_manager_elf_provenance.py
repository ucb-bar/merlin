"""Local custody failures and transcript completion cannot manufacture an FPGA pass."""

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from merlin_experiments.post_freeze import chain_observation
from merlin_experiments.post_freeze import elf_custody as C
from merlin_experiments.post_freeze import firesim as F
from merlin_experiments.post_freeze import manager_result as M

from merlin.common.paths import repo_root


def save(path, value):
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def case(tmp_path):
    result = tmp_path / "result"
    result.mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    payload = b"\x7fELFstandalone harmless fixture"
    boot = runtime / "program.elf"
    boot.write_bytes(payload)
    driver = runtime / "sim-run.sh"
    driver.write_text("# original generated driver\n")
    shutil.copyfile(driver, result / "sim-run.sh")
    entrypoint = tmp_path / "manager.sh"
    entrypoint.write_text("# fixed manager entrypoint\n")
    document = {"Name": "FsimRunElf", "DocumentVersion": "2", "Content": "fixed provenance entrypoint"}
    save(result / "ssm-document.json", document)
    model = {
        "id": "example",
        "chain_steps": 1,
        "declared_output_arrays": 1,
        "transport": "all_element_comparison_v1",
        "stages": {"stage": {"steps": 1, "outputs": [{"dtype": "i8", "shape": [2]}]}},
        "output_roster": [{"stage": "stage", "step": 0, "index": 0, "dtype": "i8", "shape": [2]}],
        "assets": [{"role": "boot_elf", **C.artifact(boot)}],
    }
    request = {
        "schema": "merlin.post_freeze.firesim.request.v1",
        "status": "prepared",
        "phase": "phase1",
        "compiler": {"hash_tree_sha256": "a" * 64},
        "workload_contract": {"sha256": "b" * 64},
        "deployment": {
            "mode": "direct_elf_ssm_on_demand",
            "ssm_document": "FsimRunElf",
            "ssm_document_version": "2",
            "ssm_document_sha256": hashlib.sha256(document["Content"].encode()).hexdigest(),
            "manager_entrypoint_sha256": C.artifact(entrypoint)["sha256"],
            "manager_instance": "i-test",
            "upload_prefix": "s3://selected/firesim-elfs/",
            "agfi": "agfi-selected",
        },
        "budget": {"job_wall_s": 120},
        "models": [model],
    }
    request["manager_contract_definition"] = {
        "schema": C.CONTRACT_SCHEMA,
        "boot_argument_prefix": "+prog0=",
        "helpers": {
            name: C.identity(C.artifact(source))
            for name, source in (
                ("elf_custody", C.__file__),
                ("chain_observation", chain_observation.__file__),
                ("manager_result", M.__file__),
            )
        },
    }
    request_path = save(tmp_path / "request.json", request)
    sha = C.artifact(request_path)["sha256"]

    def fake_aws(argv):
        assert argv[:3] == ["aws", "s3api", "get-object"]
        assert argv[argv.index("--version-id") + 1] == "elf-v1"
        Path(argv[-1]).write_bytes(payload)
        return json.dumps({"ContentLength": len(payload), "VersionId": "elf-v1"}).encode()

    downloaded = C.download(
        "selected",
        "firesim-elfs/job.elf",
        "elf-v1",
        C.identity(C.artifact(boot)),
        tmp_path / "download",
        region="test-region",
        aws=fake_aws,
    )
    save(result / "elf-download.json", downloaded)
    shutil.copyfile(tmp_path / "download/get-object.json", result / "elf-get-object.json")
    argv = save(runtime / "argv.json", ["./simulator", "+prog0=program.elf"])
    before = C.before(result / "elf-download.json", request_path, sha, "example", argv, runtime, "+prog0=", driver)
    save(result / "elf-before.json", before)
    save(result / "elf-after.json", C.after(result / "elf-before.json"))
    save(result / "request-get-object.json", {"ContentLength": request_path.stat().st_size, "VersionId": "request-v1"})
    params = {
        "elfKey": "firesim-elfs/job.elf",
        "elfVersion": "elf-v1",
        "requestKey": "firesim-requests/job.json",
        "requestVersion": "request-v1",
        "requestSha256": sha,
        "model": "example",
        "elfSha256": C.artifact(boot)["sha256"],
        "elfBytes": str(len(payload)),
        "entrypointSha256": C.artifact(entrypoint)["sha256"],
        "jobWallSeconds": "120",
        "submissionId": "job",
    }
    command = {
        "Command": {
            "CommandId": "cmd1",
            "DocumentName": "FsimRunElf",
            "DocumentVersion": "2",
            "InstanceIds": ["i-test"],
            "Parameters": {key: [value] for key, value in params.items()},
        }
    }
    save(result / "ssm-command.json", command)
    save(
        result / "execution.json",
        {
            "schema": "merlin.post_freeze.firesim.manager_execution.v1",
            "ssm_command_id": "cmd1",
            "returncode": 0,
            "cleanup": {"complete": True},
        },
    )
    (result / "HW_CFG_SUMMARY").write_text("AGFI: agfi-selected\n")
    (result / "uartlog").write_text(
        "CHAIN_STEP v1 0 0 37 0\n"
        "CHAIN_ARRAY v1 0 0 0 2 0 1 0000000000000000 0000000000000000 "
        "0000000000000011 0000000000000022 0\nCHAIN_END v1 1 1 0\nDONE\n"
    )
    return request_path, sha, request, model, result, entrypoint, command, runtime, argv


def publish(case):
    request_path, sha, _, _, result, entrypoint, *_ = case
    return M.publish(request_path, sha, "example", result, "cmd1", entrypoint)


def test_publish_reopens_actual_custody_and_uses_observed_counts(case):
    manifest = publish(case)
    request_path, _, request, model, result, _, command, *_ = case
    M.verify(request_path, request, model, manifest, result, command)
    assert manifest["completed_chain_steps"] == manifest["output_array_count"] == 1
    assert manifest["status"] == "observed_complete"
    assert manifest["scientific_qualification"] is False
    assert manifest["elf_provenance"]["chain_observation"]["step_cycles"][0]["cycles"] == 37


@pytest.mark.parametrize(
    "changed",
    [
        "version",
        "request_version",
        "key",
        "command",
        "request_sha",
        "size",
        "source",
        "download",
        "after",
        "host",
        "script",
        "document",
        "entrypoint",
        "truncated",
        "extra",
    ],
)
def test_wrong_identity_or_incomplete_transport_is_refused(case, changed):
    request_path, sha, _, _, result, entrypoint, *_ = case
    if changed in {"version", "request_version", "key", "command", "request_sha", "size"}:
        path = result / "ssm-command.json"
        raw = C.read(path)
        if changed == "version":
            raw["Command"]["DocumentVersion"] = "$LATEST"
        elif changed == "command":
            raw["Command"]["CommandId"] = "foreign"
        else:
            key = {
                "request_version": "requestVersion",
                "key": "elfKey",
                "request_sha": "requestSha256",
                "size": "elfBytes",
            }[changed]
            raw["Command"]["Parameters"][key] = ["wrong"]
    elif changed in {"download", "after", "host", "script"}:
        path = result / ("elf-download.json" if changed == "download" else "elf-after.json")
        raw = C.read(path)
        if changed == "download":
            raw["downloaded_elf"]["sha256"] = "0" * 64
        elif changed == "after":
            raw["consumed_elf"]["sha256"] = "0" * 64
        elif changed == "host":
            raw["runtime"]["boot_id"] = "foreign"
        else:
            raw["driver_script"]["sha256"] = "0" * 64
    elif changed == "document":
        path = result / "ssm-document.json"
        raw = C.read(path)
        raw["Content"] = "changed code"
    elif changed == "entrypoint":
        entrypoint.write_text("changed entrypoint")
        raw = None
    elif changed == "source":
        request_path.write_text(request_path.read_text() + " ")
        raw = None
    else:
        uart = result / "uartlog"
        uart.write_text(
            uart.read_text().removesuffix("DONE\n")
            if changed == "truncated"
            else uart.read_text() + "CHAIN_END v1 1 1 0\n"
        )
        raw = None
    if raw is not None:
        save(path, raw)
    with pytest.raises(ValueError):
        M.publish(request_path, sha, "example", result, "cmd1", entrypoint)


def test_runtime_host_checks_actual_operand_and_postrun_mutation(case):
    request_path, sha, _, _, result, _, _, runtime, argv = case
    (runtime / "other.elf").write_bytes(b"wrong image")
    save(argv, ["simulator", "+prog0=other.elf"])
    with pytest.raises(ValueError, match="actual ELF bytes"):
        C.before(
            result / "elf-download.json", request_path, sha, "example", argv, runtime, "+prog0=", runtime / "sim-run.sh"
        )
    (runtime / "program.elf").write_bytes(b"changed while running")
    with pytest.raises(ValueError, match="actual ELF bytes"):
        C.after(result / "elf-before.json")


def test_complete_execution_failure_is_retained_without_success(case):
    path = case[4] / "execution.json"
    raw = C.read(path)
    raw["returncode"] = 7
    save(path, raw)
    manifest = publish(case)
    assert manifest["status"] == "observed_failure"
    with pytest.raises(ValueError, match="incomplete or failed"):
        M.verify(case[0], case[2], case[3], manifest, case[4], case[6])


@pytest.mark.parametrize("version", ["null", "", None, "\nversion"])
def test_unversioned_download_is_refused_before_aws(tmp_path, version):
    with pytest.raises(ValueError, match="S3 object version"):
        C.download(
            "selected",
            "object",
            version,
            {"sha256": "a" * 64, "bytes": 1},
            tmp_path / "download",
            region="example",
            aws=lambda _: pytest.fail("must not call AWS"),
        )


def test_collector_detects_postpublication_uart_or_custody_change(case):
    manifest = publish(case)
    (case[4] / "uartlog").write_text((case[4] / "uartlog").read_text().replace("37 0", "38 0"))
    with pytest.raises(ValueError, match="artifact bytes differ"):
        M.verify(case[0], case[2], case[3], manifest, case[4], case[6])


def test_an_audit_operand_cannot_replace_the_selected_actual_boot_operand(case):
    request_path, sha, _, _, result, _, _, runtime, argv = case
    (runtime / "other.elf").write_bytes(b"wrong actual boot")
    save(argv, ["simulator", "+audit=program.elf", "+prog0=other.elf"])
    with pytest.raises(ValueError, match="prefix differs"):
        C.before(
            result / "elf-download.json", request_path, sha, "example", argv, runtime, "+audit=", runtime / "sim-run.sh"
        )
    with pytest.raises(ValueError, match="actual ELF bytes"):
        C.before(
            result / "elf-download.json", request_path, sha, "example", argv, runtime, "+prog0=", runtime / "sim-run.sh"
        )


def test_unselected_helper_source_is_refused(case):
    request_path, _, request, *_ = case
    request["manager_contract_definition"]["helpers"]["elf_custody"]["sha256"] = "0" * 64
    save(request_path, request)
    with pytest.raises(ValueError, match="selected image"):
        C.before(
            case[4] / "elf-download.json",
            request_path,
            C.artifact(request_path)["sha256"],
            "example",
            case[8],
            case[7],
            "+prog0=",
            case[7] / "sim-run.sh",
        )


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_ambiguous_provenance_json_is_refused(tmp_path, text):
    path = tmp_path / "ambiguous.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        C.read(path)


def test_changed_evidence_during_publication_is_refused(case, monkeypatch):
    original = M.observe_compact

    def change_after_observing(model, console):
        result = original(model, console)
        path = case[4] / "HW_CFG_SUMMARY"
        path.write_text(path.read_text() + "changed while publishing\n")
        return result

    monkeypatch.setattr(M, "observe_compact", change_after_observing)
    with pytest.raises(ValueError, match="evidence changed"):
        publish(case)


def test_existing_upload_intake_reopens_new_selected_custody_records(case):
    manifest = publish(case)
    request_path, _, request, model, result, _, _, *_ = case
    downloaded = C.read(result / "elf-download.json")
    upload = save(
        result / "operator-upload.json",
        {
            "schema": F.UPLOAD_SCHEMA,
            "bucket": "selected",
            "key": downloaded["key"],
            "version_id": downloaded["version_id"],
            "downloaded_elf": F.pin(Path(downloaded["downloaded_elf"]["path"])),
            "s3_get_object": F.pin(result / "elf-get-object.json"),
        },
    )
    matches, evidence, _ = F._uploaded_image(
        request,
        model,
        manifest,
        {"CommandId": "cmd1"},
        result / "ssm-command.json",
        upload,
        request_path=request_path,
        result_dir=result,
    )
    assert matches
    assert "elf-before.json" in evidence and "execution.json" in evidence


def test_locked_document_uses_literal_parameters_and_syntax_checks_without_execution():
    path = repo_root() / "examples/gemmini/validation/FsimRunElf.provenance.yaml"
    doc = yaml.safe_load(path.read_bytes())
    assert doc["schemaVersion"] == "2.2"
    assert len(doc["parameters"]) == 11
    for name, policy in doc["parameters"].items():
        assert policy["interpolationType"] == "ENV_VAR", name
        assert re.fullmatch(policy["allowedPattern"], "x'; touch /tmp/escape; '") is None
    script = doc["mainSteps"][0]["inputs"]["runCommand"][0]
    assert "{{" not in script and "eval " not in script
    subprocess.run(["bash", "-n"], input=script.encode(), check=True, capture_output=True, timeout=5)
