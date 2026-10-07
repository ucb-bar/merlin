"""Start a prepared run's authoring sessions DETACHED, and find its launcher again by its own record.

A launcher attached to the shell that started it ends with that shell, and one whose output is a PIPE
ends the first time nobody reads it: on 2026-10-01 a cell loop died when the pipe its launcher wrote to
was closed.  So every launch here is its own session (``start_new_session``), its stdin is
``/dev/null`` and its output APPENDS to ``launch.log`` in the run directory.  The pid and argv go to
``launch.json`` beside it, and :func:`launcher_alive` reads that record -- never a process-name match,
and never a bare pid (pids are reused): the live process's command line must name the run.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .identity import now, read_json, write_json_atomic

LAUNCH_RECORD = "launch.json"
LAUNCH_LOG = "launch.log"
LAUNCH_SCHEMA = "merlin.phase2.whole_model_measured.launch.v1"
#: The module whose ``start`` a launch runs.
MODULE = __package__


def spawn_process(argv: list[str], **kwargs: Any) -> Any:
    """Start one detached launcher (the single seam tests replace)."""
    return subprocess.Popen(argv, **kwargs)


def start_argv(
    run_dir: Path, *, profile: str, round_driver: str, price_table: Path | None, python: str | None = None
) -> list[str]:
    """The command a detached launch runs: this mode's ``start`` of ``run_dir``."""
    argv = [python or sys.executable, "-m", MODULE, "start", str(Path(run_dir).resolve()), "--profile", profile]
    argv += ["--round-driver", round_driver]
    if price_table is not None:
        argv += ["--price-table", str(price_table)]
    return argv


def launch(
    run_dir: Path,
    *,
    profile: str,
    round_driver: str,
    price_table: Path | None = None,
    spawn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Start ``run_dir``'s sessions detached; return the launch record (also written to ``launch.json``).

    Refuses a run whose recorded launcher is still alive: two launchers of one run would number the same
    rounds and take the same workspaces."""
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "run.json").is_file():
        raise ValueError(f"{run_dir} is not a prepared run (no run.json)")
    if launcher_alive(run_dir):
        raise ValueError(f"{run_dir} already has a live launcher (pid {(record(run_dir) or {}).get('pid')})")
    argv = start_argv(run_dir, profile=profile, round_driver=round_driver, price_table=price_table)
    log_path = run_dir / LAUNCH_LOG
    spawn = spawn or spawn_process
    with log_path.open("ab") as log:
        process = spawn(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    previous = record(run_dir)
    document = {
        "schema": LAUNCH_SCHEMA,
        "pid": int(process.pid),
        "argv": argv,
        "profile": profile,
        "round_driver": round_driver,
        "started_at": now(),
        "log": str(log_path),
        "previous": [*((previous or {}).get("previous") or ()), *([_brief(previous)] if previous else [])],
    }
    write_json_atomic(run_dir / LAUNCH_RECORD, document)
    return document


def _brief(document: dict[str, Any]) -> dict[str, Any]:
    return {k: document.get(k) for k in ("pid", "profile", "started_at")}


def record(run_dir: Path) -> dict[str, Any] | None:
    """The run's launch record, or None when it was never launched through :func:`launch`."""
    return read_json(Path(run_dir) / LAUNCH_RECORD)


def launcher_alive(run_dir: Path) -> bool | None:
    """Whether the run's recorded launcher is alive; None when no launch was recorded."""
    document = record(run_dir)
    if not document:
        return None
    pid = document.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return str(Path(run_dir).resolve()).encode() in cmdline


__all__ = ["LAUNCH_LOG", "LAUNCH_RECORD", "launch", "launcher_alive", "record", "start_argv"]
