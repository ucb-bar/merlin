"""Owned ordinary Python subprocess transport for the isolated new producer.

No proof command, native tool, Git command or SDK/runtime selection is executed.
All first process output is retained. A captured output is never live authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from merlin.common.owned_process import run_owned


def ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def capture(request, destination):
    directory = Path(destination).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    worker = Path(__file__).with_name("configuration_hoist_worker.py")
    before = ref(worker)
    cache = directory / "unused-python-cache"
    if cache.exists():
        raise ValueError("new source capture unexpectedly contains Python cache input")
    command = [sys.executable, "-I", "-B", "-X", "pycache_prefix=" + str(cache), str(worker)]
    (directory / "REQUEST.json").write_text(json.dumps(request, indent=2) + "\n")
    (directory / "INTENT.json").write_text(
        json.dumps({"command": command, "worker": before, "Git_SDK_Lean_native_launches": False}, indent=2) + "\n"
    )
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "LEAN_PATH"}}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        process = run_owned(
            command,
            input=json.dumps(request, allow_nan=False),
            cwd=directory,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        for name in ("stdout", "stderr"):
            value = getattr(error, name, "") or ""
            if type(value) is bytes:
                value = value.decode(errors="replace")
            (directory / name).write_text(value)
        (directory / "PROCESS.json").write_text(
            json.dumps(
                {"status": "unavailable", "error": str(error), "cleanup": getattr(error, "__notes__", [])}, indent=2
            )
            + "\n"
        )
        raise ValueError("new hoist producer unavailable; first raw output retained") from error
    (directory / "stdout").write_text(process.stdout)
    (directory / "stderr").write_text(process.stderr)
    (directory / "PROCESS.json").write_text(json.dumps({"returncode": process.returncode}, indent=2) + "\n")
    if process.returncode or process.stderr or ref(worker) != before or cache.exists():
        raise ValueError("new hoist producer failed/changed; first raw output retained")
    record = json.loads(process.stdout)
    if record.get("proof_authority") is not False:
        raise ValueError("incorrect source-owned diagnostic response")
    for row in [record["facts"], record["software"], record["source_manifest"], *record["project_sources"]]:
        if ref(row["path"]) != row:
            raise ValueError("new producer/source/input bytes changed after capture")
    (directory / "ACTUAL.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
