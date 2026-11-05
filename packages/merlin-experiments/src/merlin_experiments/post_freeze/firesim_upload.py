"""Reopen uploaded complete-chain ELF bytes and strict manager custody records."""

import base64
from pathlib import Path
from urllib.parse import urlsplit

from .firesim import UPLOAD_SCHEMA, load, pin, reopen


def upload_identity(request, model, observed, invocation, command_path, upload_path):
    if command_path is None or upload_path is None:
        return False, {}, "missing original SSM command or uploaded-ELF checksum evidence"
    command = load(command_path).get("Command", {})
    upload = load(upload_path)
    prefix = urlsplit(request["deployment"]["upload_prefix"])
    key = upload.get("key")
    parts = key.split("/") if isinstance(key, str) else []
    valid_key = (
        len(parts) == 2
        and parts[0] == "firesim-elfs"
        and 1 <= len(parts[1]) <= 128
        and parts[1].endswith(".elf")
        and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-" for c in parts[1])
    )
    if (
        upload.get("schema") != UPLOAD_SCHEMA
        or upload.get("bucket") != prefix.netloc
        or not valid_key
        or command.get("CommandId") != invocation.get("CommandId")
        or command.get("DocumentName") != "FsimRunElf"
        or command.get("InstanceIds") != [request["deployment"]["manager_instance"]]
        or command.get("Parameters", {}).get("elfKey") != [key]
    ):
        return False, {}, "uploaded ELF key differs from the original selected FsimRunElf command"
    boot = next(asset for asset in model["assets"] if asset["role"] == "boot_elf")
    downloaded = pin(reopen(upload["downloaded_elf"]))
    response_path = reopen(upload["s3_get_object"])
    response = load(response_path)
    expected = {"sha256": boot["sha256"], "bytes": boot["bytes"]}
    version = upload.get("version_id")
    checksum = response.get("ChecksumSHA256")
    matches = (
        all(downloaded[k] == v for k, v in expected.items())
        and response.get("ContentLength") == boot["bytes"]
        and isinstance(version, str)
        and bool(version)
        and version != "null"
        and response.get("VersionId") == version
        and (checksum is None or checksum == base64.b64encode(bytes.fromhex(boot["sha256"])).decode())
        and observed.get("elf_key") == key
        and observed.get("uploaded_elf") == {**expected, "version_id": version}
        and observed.get("consumed_elf") == expected
    )
    evidence = {
        "elf_upload": pin(upload_path),
        "ssm_command": pin(command_path),
        "uploaded_elf": downloaded,
        "s3_get_object": pin(response_path),
    }
    return matches, evidence, "uploaded/consumed ELF bytes do not match the selected complete-chain image"


def _uploaded_image(request, model, observed, invocation, command_path, upload_path, *, request_path, result_dir):
    if command_path is None or upload_path is None:
        return False, {}, "missing original SSM command or uploaded-ELF checksum evidence"
    matches, evidence, reason = upload_identity(request, model, observed, invocation, command_path, upload_path)
    if "ssm_document_version" in request["deployment"]:
        from .manager_result import verify

        try:
            retained = verify(request_path, request, model, observed, result_dir, load(command_path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            return False, evidence, f"manager consumed-file provenance refused: {error}"
        evidence.update({name: pin(Path(record["path"])) for name, record in retained.items()})
    return matches, evidence, reason
