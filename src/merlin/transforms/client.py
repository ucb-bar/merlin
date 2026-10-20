"""Public transport for host-owned transformations and preservation checks.

Clients supply actual IR only. Semantics, coverage, proof tools and acceptance
remain with the original host broker; returned summaries create no authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import tempfile
import time
from pathlib import Path

ACTION = "common-transform"
SCHEMA = "merlin.common_transform_request.v1"
CHECK_ACTION = "common-check"
CHECK_SCHEMA = "merlin.common_check_request.v1"
MAX_REQUEST_BYTES = 256 * 1024


def _requester_identity() -> dict:
    """The existing original broker compares process leases within one namespace."""
    try:
        tail = Path("/proc/self/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        return {"pid": os.getpid(), "start_ticks": int(tail[19]), "pid_namespace": os.readlink("/proc/self/ns/pid")}
    except (OSError, ValueError, IndexError):
        return {}  # The original finite deadline remains authoritative.


def _response_identity(result, expected, *, request_id=None, optional_source=False):
    if not isinstance(result, dict):
        raise RuntimeError("invalid common transformation response")
    if request_id is not None and result.get("selfcheck_request_id") != request_id:
        raise RuntimeError("common transformation response belongs to a different request")
    for name, digest in expected.items():
        if optional_source and name == "source_sha256" and result.get(name) is None:
            continue
        if result.get(name) != digest:
            raise RuntimeError("common transformation response describes different input IR")
    return result


def _submit(workspace, payload, *, action, expected, wait_seconds, transport_command, strict):
    if type(wait_seconds) is not int or not 1 <= wait_seconds <= 540:
        raise ValueError("common transform wait must be 1..540 seconds")
    if transport_command is not None:
        if (
            not isinstance(transport_command, tuple)
            or not transport_command
            or any(not isinstance(argument, str) or not argument for argument in transport_command)
        ):
            raise ValueError("common transform requires its declared broker transport command")
        encoded = json.dumps(payload).encode()
        if len(encoded) > MAX_REQUEST_BYTES:
            raise ValueError("common transformation request exceeds the admitted byte budget")
        workspace = Path(workspace).resolve()
        directory = workspace / "build"
        if directory.is_symlink():
            raise ValueError("common transformation request directory cannot be linked")
        directory.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".json", delete=False) as temporary:
            temporary.write(encoded)
            temporary.flush()
            path = Path(temporary.name)
        try:
            done = subprocess.run(
                [*transport_command, action, "request=" + str(path.relative_to(workspace))],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=wait_seconds,
            )
            if done.returncode:
                return {"status": "unknown", "reason": "common transformation transport refused the request"}
            if len(done.stdout.encode()) > 2 * MAX_REQUEST_BYTES:
                raise RuntimeError("invalid common transformation response")
            result = json.loads(done.stdout)
            if isinstance(result, dict) and isinstance(result.get("stdout"), str):
                result = json.loads(result["stdout"])
            return _response_identity(result, expected, request_id=path.name if strict else None)
        except subprocess.TimeoutExpired:
            return {"status": "unknown", "reason": "common transformation transport timed out"}
        finally:
            path.unlink(missing_ok=True)
    channel = Path(workspace) / ".qa_channel"
    if channel.is_symlink() or not channel.is_dir() or (channel / "STOP").exists():
        raise RuntimeError("the admitted common transform broker is unavailable")
    request_id = f"{os.getpid()}_{time.time_ns()}_{secrets.token_hex(16)}"
    deadline = time.time_ns() + wait_seconds * 1_000_000_000
    request = {
        "protocol": 3,
        "request_id": request_id,
        "requester": _requester_identity(),
        "requested_at_unix_ns": time.time_ns(),
        "deadline_unix_ns": deadline,
        "action": action,
        "transform": payload,
    }
    encoded = json.dumps(request, sort_keys=True).encode()
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("common transformation request exceeds the admitted byte budget")
    destination, temporary = channel / f"req_{request_id}.json", channel / f".{request_id}.tmp"
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    response, done = channel / f"resp_{request_id}.json", channel / f"done_{request_id}"
    while time.time_ns() < deadline:
        if done.exists() and response.exists():
            if response.is_symlink() or response.stat().st_size > 2 * MAX_REQUEST_BYTES:
                raise RuntimeError("invalid common transformation response")
            result = json.loads(response.read_text(encoding="utf-8"))
            return _response_identity(result, expected, request_id=request_id, optional_source=not strict)
        if (channel / "STOP").exists():
            raise RuntimeError("common transformation broker stopped")
        time.sleep(0.05)
    return {"status": "unknown", "reason": "common transformation request timed out", "request_id": request_id}


def apply_transform(
    workspace: Path,
    source_ir: str,
    options: dict,
    *,
    wait_seconds: int = 120,
    transport_command: tuple[str, ...] | None = None,
) -> dict:
    """Apply admitted common passes; receipt summaries are descriptive only."""
    if not isinstance(source_ir, str) or not source_ir.strip() or not isinstance(options, dict):
        raise ValueError("common transformation requires actual IR and explicit options")
    return _submit(
        workspace,
        {"schema": SCHEMA, "source_ir": source_ir, "options": options},
        action=ACTION,
        expected={"source_sha256": hashlib.sha256(source_ir.encode()).hexdigest()},
        wait_seconds=wait_seconds,
        transport_command=transport_command,
        strict=False,
    )


def check_transform(
    workspace: Path,
    before_ir: str,
    after_ir: str,
    *,
    entry: str,
    wait_seconds: int = 120,
    transport_command: tuple[str, ...] | None = None,
) -> dict:
    """Check an actual rewrite under the frozen host's semantic and proof policy."""
    if (
        not isinstance(before_ir, str)
        or not before_ir.strip()
        or not isinstance(after_ir, str)
        or not after_ir.strip()
        or not isinstance(entry, str)
        or not entry.strip()
        or len(entry.encode()) > 1024
    ):
        raise ValueError("common check requires actual before/after IR and a bounded entry symbol")
    return _submit(
        workspace,
        {"schema": CHECK_SCHEMA, "before_ir": before_ir, "after_ir": after_ir, "entry": entry},
        action=CHECK_ACTION,
        expected={
            "source_sha256": hashlib.sha256(before_ir.encode()).hexdigest(),
            "target_sha256": hashlib.sha256(after_ir.encode()).hexdigest(),
        },
        wait_seconds=wait_seconds,
        transport_command=transport_command,
        strict=True,
    )
