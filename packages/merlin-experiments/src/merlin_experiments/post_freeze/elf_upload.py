"""Upload frozen ELF/request bytes and prepare, but never send, an SSM command."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import firesim as F

SCHEMA = "merlin.post_freeze.firesim.elf_stage.v1"
REQUEST_UPLOAD_SCHEMA = "merlin.post_freeze.firesim.request_upload.v1"
AWS = Callable[[list[str]], dict[str, Any]]


def _write(path: Path, value: Any) -> dict:
    raw = value if isinstance(value, bytes) else (json.dumps(value, indent=2) + "\n").encode()
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(raw)
    return F.pin(path)


def _document_version(value: Any) -> str:
    if (not isinstance(value, str) or not value or any(c not in "0123456789" for c in value)
            or value.startswith("0")):
        raise ValueError("select an explicit positive numeric ssm_document_version")
    return value


def _version(response: dict) -> str:
    value = response.get("VersionId")
    if not isinstance(value, str) or not value or value == "null":
        raise ValueError("S3 upload requires an actual non-null object VersionId")
    return value


def _invoke(aws: AWS, output: Path, label: str, arguments: list[str]) -> tuple[dict, dict]:
    _write(output / (label + "-argv.json"), arguments)
    try:
        response = aws(arguments)
    except Exception as error:
        streams = {}
        for stream in ("stdout", "stderr"):
            value = getattr(error, stream, None)
            if isinstance(value, str):
                value = value.encode()
            streams[stream] = _write(output / (label + "-" + stream), value) if isinstance(value, bytes) else None
        _write(output / (label + "-failure.json"), {
            "error_type": type(error).__name__, "reason": str(error), **streams,
        })
        raise
    if type(response) is not dict:
        raise ValueError("AWS callable must return the complete JSON response mapping")
    return response, _write(output / (label + "-response.json"), response)


def _upload(aws: AWS, output: Path, *, label: str, source: Path, expected: dict,
            bucket: str, key: str) -> dict:
    if F.pin(source) != expected:
        raise ValueError("upload source changed before PutObject")
    put, put_pin = _invoke(aws, output, label + "-put", [
        "s3api", "put-object", "--bucket", bucket, "--key", key,
        "--body", str(source), "--if-none-match", "*",
    ])
    version = _version(put)
    downloaded = output / ("downloaded-" + label + source.suffix)
    get, get_pin = _invoke(aws, output, label + "-get", [
        "s3api", "get-object", "--bucket", bucket, "--key", key,
        "--version-id", version, str(downloaded),
    ])
    observed = F.pin(downloaded)
    downloaded.chmod(0o600)
    checksum = get.get("ChecksumSHA256")
    if (get.get("VersionId") != version or type(get.get("ContentLength")) is not int
            or get["ContentLength"] != expected["bytes"]
            or any(observed[k] != expected[k] for k in ("sha256", "bytes"))
            or (checksum is not None and checksum != base64.b64encode(bytes.fromhex(expected["sha256"])).decode())):
        raise ValueError("version-pinned downloaded bytes differ from the frozen upload source")
    return {"bucket": bucket, "key": key, "version_id": version,
            "downloaded": observed, "s3_put_object": put_pin, "s3_get_object": get_pin}


def _aws_cli(region: str) -> AWS:
    environment = {**os.environ, "AWS_PAGER": ""}
    probe = subprocess.run(
        ["aws", "--region", region, "s3api", "put-object", "--generate-cli-skeleton", "input"],
        env={**environment, "AWS_EC2_METADATA_DISABLED": "true"},
        check=True, capture_output=True, text=True, timeout=30,
    )
    if "IfNoneMatch" not in json.loads(probe.stdout):
        raise ValueError("installed AWS CLI lacks create-only PutObject IfNoneMatch; upgrade before staging")
    def call(arguments: list[str]) -> dict:
        result = subprocess.run(
            ["aws", "--region", region, *arguments, "--output", "json"],
            env=environment, check=True, capture_output=True, text=True, timeout=120,
        )
        return json.loads(result.stdout)
    return call


def stage(request_path: Path, *, model: str, output: Path, submission_id: str | None = None,
          aws: AWS | None = None) -> dict:
    """Only upload one selected ELF and the exact request; no cloud execution."""
    request_path, output = Path(request_path).absolute(), Path(output).absolute()
    request_pin = F.pin(request_path)
    request = json.loads(request_path.read_bytes())
    F.verify_request(request)
    if request.get("status") != "prepared":
        raise ValueError("staging requires a fully prepared frozen request")
    deployment = request["deployment"]
    version = _document_version(deployment.get("ssm_document_version"))
    document_sha = deployment.get("ssm_document_sha256")
    entrypoint_sha = deployment.get("manager_entrypoint_sha256")
    if not F.is_sha256(document_sha) or not F.is_sha256(entrypoint_sha):
        raise ValueError("deployment needs ssm_document_sha256 and manager_entrypoint_sha256")
    prefix = urlsplit(deployment.get("upload_prefix", ""))
    if (deployment.get("mode") != F.DIRECT_MODE or prefix.scheme != "s3" or not prefix.netloc
            or prefix.path != "/firesim-elfs/" or prefix.query or prefix.fragment):
        raise ValueError("staging needs the selected direct-ELF versioned S3 transport")
    rows = [row for row in request["models"] if row["id"] == model]
    if len(rows) != 1 or not F._token(model):
        raise ValueError("select exactly one declared model")
    row = rows[0]
    F._direct_image(row, request["phase"], request["compiler"]["hash_tree_sha256"],
                    request["workload_contract"]["sha256"])
    boot, = [asset for asset in row["assets"] if asset["role"] == "boot_elf"]
    source = F.reopen(boot)
    source_pin = F.pin(source)
    identifier = uuid.uuid4().hex if submission_id is None else submission_id
    if not F._token(identifier):
        raise ValueError("submission_id must be a bounded safe token")
    if (output.exists() or output.resolve() != output
            or any(path.is_symlink() for path in (output, *output.parents))
            or output.is_relative_to(Path(request["compiler"]["path"]))):
        raise ValueError("staging needs a fresh ordinary output outside the frozen compiler")
    if F.pin(request_path) != request_pin:
        raise ValueError("request bytes changed before staging")
    output.mkdir(parents=True, mode=0o700)
    client = _aws_cli(deployment["region"]) if aws is None else aws
    document, document_pin = _invoke(client, output, "ssm-document", [
        "ssm", "get-document", "--name", deployment["ssm_document"],
        "--document-version", version, "--document-format", "JSON",
    ])
    content = document.get("Content")
    if (document.get("Name") != deployment["ssm_document"] or document.get("DocumentVersion") != version
            or document.get("Status") != "Active" or not isinstance(content, str)
            or hashlib.sha256(content.encode()).hexdigest() != document_sha):
        raise ValueError("selected SSM document version/content differs from deployment pins")
    content_pin = _write(output / "ssm-document-content.json", content.encode())
    bucket = prefix.netloc
    elf = _upload(client, output, label="elf", source=source, expected=source_pin,
                  bucket=bucket, key=f"firesim-elfs/{identifier}.elf")
    raw_request = _upload(client, output, label="request", source=request_path, expected=request_pin,
                          bucket=bucket, key=f"firesim-requests/{identifier}.json")
    F.verify_request(request)
    if F.pin(request_path) != request_pin or F.pin(source) != source_pin:
        raise ValueError("frozen request or ELF changed during staging")
    elf_upload = {"schema": F.UPLOAD_SCHEMA, **{k: v for k, v in elf.items() if k != "downloaded"},
                  "downloaded_elf": elf["downloaded"]}
    request_upload = {"schema": REQUEST_UPLOAD_SCHEMA,
                      **{k: v for k, v in raw_request.items() if k != "downloaded"},
                      "downloaded_request": raw_request["downloaded"]}
    parameters = {"elfKey": [elf["key"]], "elfVersion": [elf["version_id"]],
                  "requestKey": [raw_request["key"]], "requestVersion": [raw_request["version_id"]],
                  "requestSha256": [request_pin["sha256"]], "model": [model],
                  "elfSha256": [source_pin["sha256"]], "elfBytes": [str(source_pin["bytes"])],
                  "entrypointSha256": [entrypoint_sha],
                  "jobWallSeconds": [str(request["budget"]["job_wall_s"])], "submissionId": [identifier]}
    command = {"DocumentName": deployment["ssm_document"], "DocumentVersion": version,
               "DocumentHash": document_sha, "DocumentHashType": "Sha256",
               "InstanceIds": [deployment["manager_instance"]], "Parameters": parameters}
    command_pin = _write(output / "send-command.json", command)
    result = {"schema": SCHEMA, "submission_id": identifier, "model": model, "request": request_pin,
              "elf_upload": _write(output / "elf-upload.json", elf_upload),
              "request_upload": _write(output / "request-upload.json", request_upload),
              "ssm_document": document_pin, "ssm_document_content": content_pin,
              "manager_entrypoint_sha256": entrypoint_sha,
              "send_command": command_pin, "send_command_argv": [
                  "aws", "--region", deployment["region"], "ssm", "send-command",
                  "--cli-input-json", "file://" + command_pin["path"], "--output", "json"],
              "send_command_environment": {"AWS_PAGER": ""},
              "planned_command_evidence": {"bucket": bucket, "key": f"firesim-commands/{identifier}.json"},
              "paid_jobs_launched": 0, "ssm_commands_sent": 0, "manager_started": False,
              "scientific_qualification": False,
              "scope": "Uploads and unsent command plan only; no manager or consumed-ELF authority."}
    _write(output / "stage.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--submission-id")
    parser.add_argument("--upload", action="store_true", help="Explicitly allow only S3 uploads and document lookup")
    args = parser.parse_args(argv)
    if not args.upload:
        parser.error("staging performs uploads; explicitly select --upload")
    print(json.dumps(stage(args.request, model=args.model, output=args.output,
                           submission_id=args.submission_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
