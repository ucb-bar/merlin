"""Bounded owned-child recording for the independent configuration campaign."""

import json
import subprocess

from merlin.common.owned_process import run_owned
from merlin.common.paths import repo_root


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(command, destination, *, environment, request=None, timeout=120):
    destination.mkdir(parents=True, exist_ok=False)
    command = list(map(str, command))
    save(
        destination / "INTENT.json",
        dict(
            command=command,
            cwd=str(repo_root()),
            timeout=timeout,
            owned_process_group=True,
            environment={
                key: environment.get(key) for key in ("PYTHONPATH", "MERLIN_REPO_ROOT", "PYTHONDONTWRITEBYTECODE")
            },
        ),
    )
    if request is not None:
        save(destination / "INPUT.json", request)
    try:
        process = run_owned(
            command,
            cwd=repo_root(),
            env=environment,
            input=None if request is None else json.dumps(request),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        for name in ("stdout", "stderr"):
            value = getattr(error, name, "") or ""
            (destination / name).write_text(value.decode(errors="replace") if isinstance(value, bytes) else value)
        save(
            destination / "PROCESS.json",
            dict(returncode=None, error=str(error), cleanup=getattr(error, "__notes__", [])),
        )
        raise RuntimeError("owned process unavailable; first output and cleanup retained") from error
    (destination / "stdout").write_text(process.stdout)
    (destination / "stderr").write_text(process.stderr)
    save(destination / "PROCESS.json", dict(returncode=process.returncode))
    if process.returncode:
        raise RuntimeError("worker failed; first raw result retained")
    return json.loads(process.stdout)["result"]
