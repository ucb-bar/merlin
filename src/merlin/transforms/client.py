"""Public channel client for host-verified common transformations.

This client supplies IR and transformation parameters only. Semantic definitions,
coverage, proof tools and acceptance remain with the original host broker.
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
MAX_REQUEST_BYTES = 256 * 1024


def apply_transform(
    workspace: Path,
    source_ir: str,
    options: dict,
    *,
    wait_seconds: int = 120,
    transport_command: tuple[str, ...] | None = None,
) -> dict:
    """Request a transformation; receipt summaries are descriptive, never proof authority."""
    if type(wait_seconds) is not int or not 1 <= wait_seconds <= 540:
        raise ValueError("common transform wait must be 1..540 seconds")
    if not isinstance(source_ir, str) or not source_ir.strip() or not isinstance(options, dict):
        raise ValueError("common transformation requires actual IR and explicit options")
    if transport_command is not None:
        if (
            not isinstance(transport_command, tuple)
            or not transport_command
            or any(not isinstance(argument, str) or not argument for argument in transport_command)
        ):
            raise ValueError("common transform requires its declared broker transport command")
        payload = {"schema": SCHEMA, "source_ir": source_ir, "options": options}
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
                [*transport_command, ACTION, "request=" + str(path.relative_to(workspace))],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=wait_seconds,
            )
            if done.returncode:
                return {"status": "unknown", "reason": "common transformation transport refused the request"}
            result = json.loads(done.stdout)
            if isinstance(result, dict) and isinstance(result.get("stdout"), str):
                result = json.loads(result["stdout"])
            if (
                not isinstance(result, dict)
                or result.get("source_sha256") != hashlib.sha256(source_ir.encode()).hexdigest()
            ):
                raise RuntimeError("common transformation response describes different input IR")
            return result
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
        "requested_at_unix_ns": time.time_ns(),
        "deadline_unix_ns": deadline,
        "action": ACTION,
        "transform": {"schema": SCHEMA, "source_ir": source_ir, "options": options},
    }
    encoded = json.dumps(request, sort_keys=True).encode()
    if len(encoded) > MAX_REQUEST_BYTES:
        raise ValueError("common transformation request exceeds the admitted byte budget")
    destination = channel / f"req_{request_id}.json"
    temporary = channel / f".{request_id}.tmp"
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
            if not isinstance(result, dict) or result.get("selfcheck_request_id") != request_id:
                raise RuntimeError("common transformation response belongs to a different request")
            if result.get("source_sha256") not in (None, hashlib.sha256(source_ir.encode()).hexdigest()):
                raise RuntimeError("common transformation response describes different input IR")
            return result
        if (channel / "STOP").exists():
            raise RuntimeError("common transformation broker stopped")
        time.sleep(0.05)
    return {"status": "unknown", "reason": "common transformation request timed out", "request_id": request_id}
