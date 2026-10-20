"""Publish and reopen manager ELF custody observations after a frozen model run.

The fixed manager adapter owns execution and authenticated runtime-host file
collection. This boundary runs no commands, never upgrades historical records,
and leaves reference comparison and placement to their existing checkers.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
from pathlib import Path
from urllib.parse import urlsplit

from . import chain_observation
from . import elf_custody as custody
from .chain_observation import observe_compact

SCHEMA = "merlin.post_freeze.firesim.manager_result.v1"
PROVENANCE_SCHEMA = "merlin.post_freeze.firesim.manager_elf_provenance.v1"
FILES = (
    "elf-download.json",
    "elf-before.json",
    "elf-after.json",
    "ssm-command.json",
    "ssm-document.json",
    "request-get-object.json",
    "execution.json",
    "uartlog",
    "HW_CFG_SUMMARY",
    "sim-run.sh",
    "elf-get-object.json",
)


def deployment_identity(deployment):
    version = deployment.get("ssm_document_version")
    if (
        not isinstance(version, str)
        or not version.isascii()
        or not version.isdecimal()
        or str(int(version)) != version
        or int(version) < 1
    ):
        raise ValueError("select a concrete canonical numeric SSM document version")
    for key in ("ssm_document_sha256", "manager_entrypoint_sha256"):
        if not custody.digest(deployment.get(key)):
            raise ValueError(f"deployment requires {key}")
    return version


def _parameter(command, name):
    value = command.get("Parameters", {}).get(name)
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], str):
        raise ValueError(f"original SendCommand must retain one exact {name}")
    return value[0]


def _stable(records):
    if any(custody.artifact(record["path"]) != record for record in records.values()):
        raise ValueError("manager evidence changed while publishing or reopening")


def _runtime_observation(contract, directory, records, before):
    selected = contract.get("runtime_execution")
    if selected is None:
        return True
    path = Path(directory) / "runtime-driver.json"
    records[path.name] = custody.artifact(path)
    row = custody.read(path)
    custody.require_identity(row.get("observer", {}), selected.get("helper", {}))
    custody.require_identity(row.get("driver", {}), selected.get("driver", {}))
    if (row.get("schema") != "merlin.post_freeze.firesim.runtime_driver.v1"
            or row.get("runtime") != before["runtime"]
            or row.get("argv") != before["simulator_argv"]
            or row.get("working_directory") != before["working_directory"]
            or custody.identity(row.get("before", {})) != custody.identity(records["elf-before.json"])
            or custody.identity(row.get("after", {})) != custody.identity(records["elf-after.json"])
            or type(row.get("returncode")) is not int
            or type(row.get("timed_out")) is not bool):
        raise ValueError("runtime driver observation differs from actual consumed-file observations")
    return row.get("status") == "driver_finished" and row["returncode"] == 0 and not row["timed_out"]


def _records(request_sha256, request, model, directory, command_id, entrypoint, publisher_sources):
    directory = Path(directory)
    deployment = request["deployment"]
    version = deployment_identity(deployment)
    contract = custody.contract(request)
    if publisher_sources != contract["helpers"]:
        raise ValueError("actual publisher/helper source bytes differ from selected manager contract")
    records = {name: custody.artifact(directory / name) for name in FILES}
    downloaded, before, after, command, document, get_request, execution = (
        custody.read(directory / name) for name in FILES[:7]
    )
    command = command.get("Command", {})
    if (
        command.get("CommandId") != command_id
        or command.get("DocumentName") != deployment["ssm_document"]
        or command.get("DocumentVersion") != version
        or command.get("InstanceIds") != [deployment["manager_instance"]]
    ):
        raise ValueError("original SendCommand identity differs from selected manager/document/version")
    content = document.get("Content")
    if (
        document.get("Name") != deployment["ssm_document"]
        or document.get("DocumentVersion") != version
        or not isinstance(content, str)
        or hashlib.sha256(content.encode()).hexdigest() != deployment["ssm_document_sha256"]
    ):
        raise ValueError("retained SSM document content differs from selected version/SHA")
    expected = custody.identity(next(row for row in model["assets"] if row["role"] == "boot_elf"))
    expected_parameters = {
        "requestSha256": request_sha256,
        "model": model["id"],
        "elfSha256": expected["sha256"],
        "elfBytes": str(expected["bytes"]),
        "entrypointSha256": deployment["manager_entrypoint_sha256"],
        "jobWallSeconds": str(request["budget"]["job_wall_s"]),
    }
    if any(_parameter(command, key) != value for key, value in expected_parameters.items()):
        raise ValueError("SendCommand parameters differ from frozen request/image/entrypoint")
    submission = _parameter(command, "submissionId")
    if (
        not 1 <= len(submission) <= 64
        or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in submission)
        or submission in {".", ".."}
        or _parameter(command, "requestKey") != f"firesim-requests/{submission}.json"
    ):
        raise ValueError("request key differs from unique selected submission")
    elf_version, request_version = (_parameter(command, key) for key in ("elfVersion", "requestVersion"))
    custody.require_version(elf_version)
    custody.require_version(request_version)
    if get_request.get("VersionId") != request_version:
        raise ValueError("manager did not GET the selected request version")
    if (
        downloaded.get("schema") != custody.SCHEMA
        or downloaded.get("kind") != "versioned_download"
        or downloaded.get("expected") != expected
        or downloaded.get("version_id") != elf_version
        or downloaded.get("key") != _parameter(command, "elfKey")
    ):
        raise ValueError("manager download record differs from selected ELF version/key/bytes")
    prefix = urlsplit(deployment["upload_prefix"])
    if downloaded.get("bucket") != prefix.netloc:
        raise ValueError("manager downloaded from another bucket")
    custody.require_identity(downloaded["downloaded_elf"], expected)
    custody.require_identity(downloaded["observer"], contract["helpers"]["elf_custody"])
    get_elf = custody.read(directory / "elf-get-object.json")
    if (
        custody.identity(downloaded["s3_get_object"]) != custody.identity(records["elf-get-object.json"])
        or get_elf.get("VersionId") != elf_version
        or get_elf.get("ContentLength") != expected["bytes"]
        or (
            get_elf.get("ChecksumSHA256") is not None
            and get_elf["ChecksumSHA256"] != base64.b64encode(bytes.fromhex(expected["sha256"])).decode()
        )
    ):
        raise ValueError("actual retained GetObject response differs from selected ELF bytes/version")
    if (
        before.get("schema") != custody.SCHEMA
        or before.get("kind") != "consumed_before"
        or before.get("request_sha256") != request_sha256
        or before.get("model") != model["id"]
        or before.get("download_record_sha256") != records["elf-download.json"]["sha256"]
        or before.get("expected") != expected
    ):
        raise ValueError("before-run consumed-file observation differs from selected download/request")
    custody.require_identity(before["consumed_elf"], expected)
    custody.require_identity(before["observer"], contract["helpers"]["elf_custody"])
    argv, boot_prefix = before.get("simulator_argv"), before.get("boot_argument_prefix")
    if (
        not isinstance(argv, list)
        or not all(isinstance(row, str) for row in argv)
        or boot_prefix != contract["boot_argument_prefix"]
        or not isinstance(boot_prefix, str)
        or not boot_prefix
        or not boot_prefix.endswith("=")
    ):
        raise ValueError("missing concrete simulator argv/boot operand")
    boot_values = [row[len(boot_prefix) :] for row in argv if row.startswith(boot_prefix)]
    if len(boot_values) != 1:
        raise ValueError("ambiguous consumed boot operand")
    boot = Path(boot_values[0])
    boot = boot if boot.is_absolute() else Path(before["working_directory"]) / boot
    if str(boot) != before["consumed_elf"]["path"]:
        raise ValueError("observed consumed file is not the actual simulator boot operand")
    if (
        after.get("schema") != custody.SCHEMA
        or after.get("kind") != "consumed_after"
        or after.get("before_sha256") != records["elf-before.json"]["sha256"]
        or any(
            after.get(key) != before.get(key)
            for key in ("request_sha256", "model", "runtime", "consumed_elf", "observer", "driver_script")
        )
        or custody.identity(before["driver_script"]) != custody.identity(records["sim-run.sh"])
    ):
        raise ValueError("after-run file/runtime/script custody differs from before-run observations")
    if entrypoint["sha256"] != deployment["manager_entrypoint_sha256"]:
        raise ValueError("actual manager entrypoint differs from selected executable source")
    if (
        execution.get("schema") != "merlin.post_freeze.firesim.manager_execution.v1"
        or execution.get("ssm_command_id") != command_id
        or type(execution.get("returncode")) is not int
        or type(execution.get("cleanup", {}).get("complete")) is not bool
    ):
        raise ValueError("missing actual manager execution and cleanup outcome")
    execution["runtime_driver_complete"] = _runtime_observation(contract, directory, records, before)
    return records, downloaded, before, execution


def publish(request_path, request_sha256, model_id, result_dir, command_id, entrypoint_path):
    request, model, _ = custody.selected_model(request_path, request_sha256, model_id)
    if model.get("transport") != "all_element_comparison_v1":
        raise ValueError("this manager publisher requires explicitly selected compact chain transport")
    entrypoint = custody.artifact(entrypoint_path)
    publisher_sources = {
        name: custody.identity(custody.artifact(source))
        for name, source in (
            ("elf_custody", custody.__file__),
            ("chain_observation", chain_observation.__file__),
            ("manager_result", __file__),
        )
    }
    records, downloaded, before, execution = _records(
        request_sha256, request, model, result_dir, command_id, entrypoint, publisher_sources
    )
    response = custody.read(Path(result_dir) / "request-get-object.json")
    if response.get("ContentLength") != custody.artifact(request_path)["bytes"] or (
        response.get("ChecksumSHA256") is not None
        and response["ChecksumSHA256"] != base64.b64encode(bytes.fromhex(request_sha256)).decode()
    ):
        raise ValueError("actual downloaded request size differs from versioned response")
    observation = observe_compact(model, (Path(result_dir) / "uartlog").read_bytes())
    if request["deployment"]["agfi"] not in (Path(result_dir) / "HW_CFG_SUMMARY").read_text().replace(":", " ").split():
        raise ValueError("actual hardware summary does not observe the selected AGFI")
    complete = (
        observation["status"] == "observed_complete"
        and execution["returncode"] == 0
        and execution["cleanup"]["complete"]
        and execution["runtime_driver_complete"]
    )
    manifest = {
        "schema": SCHEMA,
        "phase": request["phase"],
        "model": model_id,
        "compiler_sha256": request["compiler"]["hash_tree_sha256"],
        "contract_sha256": request["workload_contract"]["sha256"],
        "request_sha256": request_sha256,
        "agfi": request["deployment"]["agfi"],
        "ssm_command_id": command_id,
        "assets": model["assets"],
        "transport": model["transport"],
        "elf_key": downloaded["key"],
        "uploaded_elf": {**downloaded["expected"], "version_id": downloaded["version_id"]},
        "consumed_elf": custody.identity(before["consumed_elf"]),
        "completed_chain_steps": observation["completed_chain_steps"],
        "output_array_count": observation["output_array_count"],
        "outputs": [],
        "status": "observed_complete" if complete else "observed_failure",
        "elf_provenance": {
            "schema": PROVENANCE_SCHEMA,
            "manager_entrypoint": entrypoint,
            "publisher_sources": publisher_sources,
            "artifacts": {name: custody.identity(record) for name, record in records.items()},
            "chain_observation": observation,
        },
        "scientific_qualification": False,
    }
    _stable(records)
    if custody.artifact(entrypoint_path) != entrypoint:
        raise ValueError("manager entrypoint changed during publication")
    custody.selected_model(request_path, request_sha256, model_id)
    custody.write(Path(result_dir) / "result_manifest.json", manifest)
    return manifest


def verify(request_path, request, model, observed, result_dir, command):
    """Reopen retained provenance without treating metadata as new target execution."""
    provenance = observed.get("elf_provenance", {})
    if provenance.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError("missing manager actual consumed-file provenance")
    directory = Path(result_dir)
    records, downloaded, before, execution = _records(
        custody.artifact(request_path)["sha256"],
        request,
        model,
        directory,
        observed["ssm_command_id"],
        provenance["manager_entrypoint"],
        provenance.get("publisher_sources"),
    )
    response = custody.read(directory / "request-get-object.json")
    if response.get("ContentLength") != custody.artifact(request_path)["bytes"] or (
        response.get("ChecksumSHA256") is not None
        and response["ChecksumSHA256"]
        != base64.b64encode(bytes.fromhex(custody.artifact(request_path)["sha256"])).decode()
    ):
        raise ValueError("versioned request download response differs from selected request bytes")
    if custody.read(directory / "ssm-command.json") != command:
        raise ValueError("manager and operator retained different original SendCommand records")
    if provenance.get("artifacts") != {name: custody.identity(record) for name, record in records.items()}:
        raise ValueError("retained provenance/output artifact bytes differ from manager manifest")
    observation = observe_compact(model, (directory / "uartlog").read_bytes())
    if (
        provenance.get("chain_observation") != observation
        or type(observed.get("completed_chain_steps")) is not int
        or observed["completed_chain_steps"] != observation["completed_chain_steps"]
        or type(observed.get("output_array_count")) is not int
        or observed["output_array_count"] != observation["output_array_count"]
        or observation["status"] != "observed_complete"
        or observed.get("status") != "observed_complete"
        or execution["returncode"] != 0
        or not execution["cleanup"]["complete"]
        or not execution["runtime_driver_complete"]
        or observed.get("uploaded_elf") != {**downloaded["expected"], "version_id": downloaded["version_id"]}
        or observed.get("consumed_elf") != custody.identity(before["consumed_elf"])
    ):
        raise ValueError("manager execution/completion/custody observation is incomplete or failed")
    _stable(records)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("request", "result-dir", "entrypoint"):
        parser.add_argument("--" + field, type=Path, required=True)
    for field in ("request-sha256", "model", "command-id"):
        parser.add_argument("--" + field, required=True)
    args = parser.parse_args(argv)
    result = publish(args.request, args.request_sha256, args.model, args.result_dir, args.command_id, args.entrypoint)
    return 0 if result["status"] == "observed_complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
