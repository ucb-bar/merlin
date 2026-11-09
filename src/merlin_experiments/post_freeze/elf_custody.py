"""Observe versioned downloads and actual simulator boot files without running hardware.

The fixed manager adapter calls before/after on the runtime host holding the
actual simulator file. A staging copy on the manager is not a consumed file.
This module is standard-library-only so that the same pinned file can be
installed on the manager and runtime host. Its records are observations, not
an isolation guarantee or a numerical qualification.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import stat
import subprocess
from pathlib import Path

SCHEMA = "merlin.post_freeze.firesim.elf_custody.v1"
CONTRACT_SCHEMA = "merlin.post_freeze.firesim.manager_contract.v1"


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"non-finite JSON value: {value}")


def read(path):
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=_pairs, parse_constant=_nonfinite)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _stamp(value):
    return {key: getattr(value, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}


def artifact(path):
    path = Path(path).absolute()
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("artifact paths must not traverse symlinks")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        start = os.fstat(stream.fileno())
        if not stat.S_ISREG(start.st_mode):
            raise ValueError("artifact must be a regular file")
        hashed = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hashed.update(block)
        sha = hashed.hexdigest()
        if _stamp(start) != _stamp(os.fstat(stream.fileno())) or _stamp(start) != _stamp(path.stat()):
            raise ValueError("artifact changed while hashing")
    return {"path": str(path), "sha256": sha, "bytes": start.st_size, "stat": _stamp(start)}


def identity(record):
    return {key: record[key] for key in ("sha256", "bytes")}


def require_identity(record, expected):
    if not digest(expected.get("sha256")) or type(expected.get("bytes")) is not int or expected["bytes"] < 1:
        raise ValueError("invalid selected ELF identity")
    if identity(record) != expected:
        raise ValueError("actual ELF bytes differ from the selected image")


def require_version(version):
    if not isinstance(version, str) or not version or version == "null" or any(ord(c) < 33 for c in version):
        raise ValueError("a concrete non-null S3 object version is required")


def runtime_identity():
    return {"hostname": socket.gethostname(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def contract(request):
    selected = request.get("manager_contract_definition", {})
    if not isinstance(selected, dict):
        raise ValueError("manager contract must be a JSON object")
    prefix = selected.get("boot_argument_prefix")
    helpers = selected.get("helpers", {})
    if (
        selected.get("schema") != CONTRACT_SCHEMA
        or not isinstance(prefix, str)
        or not 2 <= len(prefix) <= 64
        or not prefix.endswith("=")
        or any(c.isspace() or c == "\x00" for c in prefix)
        or not isinstance(helpers, dict)
        or set(helpers) != {"elf_custody", "chain_observation", "manager_result"}
    ):
        raise ValueError("manager contract must select exact boot operand and helper sources")
    for source in helpers.values():
        if not isinstance(source, dict) or set(source) != {"sha256", "bytes"}:
            raise ValueError("helper source identity must contain only SHA256 and size")
        require_identity(source, source)
    return selected


def download(bucket, key, version, expected, directory, *, region, aws=None):
    """GET an exact version, retaining raw response and actual downloaded bytes."""
    require_version(version)
    require_identity(expected, expected)
    directory = Path(directory).absolute()
    directory.mkdir(parents=True, exist_ok=False)
    elf = directory / "downloaded.elf"
    argv = [
        "aws",
        "s3api",
        "get-object",
        "--bucket",
        bucket,
        "--key",
        key,
        "--version-id",
        version,
        "--region",
        region,
        "--output",
        "json",
        str(elf),
    ]
    if aws is None:
        response = subprocess.run(argv, check=True, capture_output=True, timeout=600).stdout
    else:
        response = aws(argv)
    response_path = directory / "get-object.json"
    response_path.write_bytes(response)
    observed = read(response_path)
    actual = artifact(elf)
    require_identity(actual, expected)
    checksum = observed.get("ChecksumSHA256")
    if (
        observed.get("VersionId") != version
        or observed.get("ContentLength") != actual["bytes"]
        or (checksum is not None and checksum != base64.b64encode(bytes.fromhex(actual["sha256"])).decode())
    ):
        raise ValueError("versioned GetObject response differs from downloaded ELF bytes")
    record = {
        "schema": SCHEMA,
        "kind": "versioned_download",
        "bucket": bucket,
        "key": key,
        "version_id": version,
        "expected": expected,
        "downloaded_elf": actual,
        "s3_get_object": artifact(response_path),
        "observer": artifact(__file__),
    }
    write(directory / "download.json", record)
    return record


def selected_model(request_path, request_sha256, model_id):
    request_pin = artifact(request_path)
    if not digest(request_sha256) or request_pin["sha256"] != request_sha256:
        raise ValueError("actual request bytes differ from selected request SHA256")
    request = read(request_path)
    if artifact(request_path) != request_pin:
        raise ValueError("request changed while reading selected model")
    if (
        request.get("schema") != "merlin.post_freeze.firesim.request.v1"
        or request.get("status") != "prepared"
        or request.get("deployment", {}).get("mode") != "direct_elf_ssm_on_demand"
    ):
        raise ValueError("expected a prepared direct complete-chain request")
    models = [row for row in request["models"] if row.get("id") == model_id]
    if len(models) != 1:
        raise ValueError("model is not uniquely selected by frozen request")
    boot = [row for row in models[0]["assets"] if row.get("role") == "boot_elf"]
    if len(boot) != 1:
        raise ValueError("request must select one complete-chain boot ELF")
    return request, models[0], identity(boot[0])


def before(download_path, request_path, request_sha256, model_id, argv_path, cwd, boot_prefix, driver_script):
    """Observe the concrete boot operand of the actual simulator argv on its host."""
    request, model, expected = selected_model(request_path, request_sha256, model_id)
    selected = contract(request)
    require_identity(artifact(__file__), selected["helpers"]["elf_custody"])
    if boot_prefix != selected["boot_argument_prefix"]:
        raise ValueError("boot argument prefix differs from selected manager contract")
    downloaded = read(download_path)
    if (
        downloaded.get("schema") != SCHEMA
        or downloaded.get("kind") != "versioned_download"
        or downloaded.get("expected") != expected
    ):
        raise ValueError("download receipt differs from selected boot ELF")
    require_version(downloaded.get("version_id"))
    require_identity(downloaded["downloaded_elf"], expected)
    argv = json.loads(Path(argv_path).read_bytes())
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(a, str) and a and "\x00" not in a for a in argv)
        or not isinstance(boot_prefix, str)
        or not boot_prefix
        or not boot_prefix.endswith("=")
    ):
        raise ValueError("expected the actual simulator argv and an explicit boot argument prefix")
    values = [token[len(boot_prefix) :] for token in argv if token.startswith(boot_prefix)]
    if len(values) != 1 or not values[0]:
        raise ValueError("simulator argv must select exactly one concrete boot file")
    cwd = Path(cwd).absolute()
    path = Path(values[0])
    consumed = artifact(path if path.is_absolute() else cwd / path)
    require_identity(consumed, expected)
    return {
        "schema": SCHEMA,
        "kind": "consumed_before",
        "request_sha256": request_sha256,
        "model": model["id"],
        "download_record_sha256": artifact(download_path)["sha256"],
        "expected": expected,
        "runtime": runtime_identity(),
        "working_directory": str(cwd),
        "simulator_argv": argv,
        "boot_argument_prefix": boot_prefix,
        "consumed_elf": consumed,
        "driver_script": artifact(driver_script),
        "observer": artifact(__file__),
    }


def after(before_path):
    original = read(before_path)
    if original.get("schema") != SCHEMA or original.get("kind") != "consumed_before":
        raise ValueError("expected original before-run consumed-file observation")
    if original["runtime"] != runtime_identity():
        raise ValueError("after-run observation must run on the same runtime host boot")
    actual = artifact(original["consumed_elf"]["path"])
    require_identity(actual, original["expected"])
    if (
        actual != original["consumed_elf"]
        or artifact(__file__) != original["observer"]
        or artifact(original["driver_script"]["path"]) != original["driver_script"]
    ):
        raise ValueError("consumed file, driver script or observer changed during execution")
    return {
        "schema": SCHEMA,
        "kind": "consumed_after",
        "before_sha256": artifact(before_path)["sha256"],
        "request_sha256": original["request_sha256"],
        "model": original["model"],
        "runtime": runtime_identity(),
        "consumed_elf": actual,
        "observer": artifact(__file__),
        "driver_script": original["driver_script"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download")
    for field in ("bucket", "key", "version", "sha256", "region"):
        d.add_argument("--" + field, required=True)
    d.add_argument("--size", type=int, required=True)
    d.add_argument("--directory", type=Path, required=True)
    b = sub.add_parser("before")
    for field in ("download-record", "request", "argv-json", "cwd", "driver-script", "out"):
        b.add_argument("--" + field, type=Path, required=True)
    for field in ("request-sha256", "model", "boot-argument-prefix"):
        b.add_argument("--" + field, required=True)
    a = sub.add_parser("after")
    a.add_argument("--before-record", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "download":
        download(
            args.bucket,
            args.key,
            args.version,
            {"sha256": args.sha256, "bytes": args.size},
            args.directory,
            region=args.region,
        )
    elif args.command == "before":
        write(
            args.out,
            before(
                args.download_record,
                args.request,
                args.request_sha256,
                args.model,
                args.argv_json,
                args.cwd,
                args.boot_argument_prefix,
                args.driver_script,
            ),
        )
    else:
        write(args.out, after(args.before_record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
