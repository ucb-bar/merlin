"""Validate launch profiles and start detached measured-run sessions.

Profiles retain model identity and pricing checks. Launches append to their own
log and keep the existing process identity and duplicate-launch checks.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import Any

from merlin.development.backend.optimization.whole_model_measured import CLI_MODULE as MODULE
from merlin.development.backend.optimization.whole_model_measured.measurement.identity import (
    now,
    read_json,
    write_json_atomic,
)


PROFILE_SCHEMA = "merlin.phase2.whole_model_measured.launch_profiles.v1"


PROFILE_FIELDS = (
    "driver",
    "model",
    "effort",
    "max_sessions",
    "round_seconds",
    "total_authoring_seconds",
    "iteration_seconds",
    "max_tool_calls",
)


class ProfileError(ValueError):
    """The profile is unknown, malformed, or names a model that would not run as named."""


def _profile_document() -> dict[str, Any]:
    source = files("merlin.experiments").joinpath("resources", "whole_model_measured_profiles.json")
    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("schema") != PROFILE_SCHEMA:
        raise ProfileError(f"the bundled launch profiles do not declare {PROFILE_SCHEMA}")
    return document


def profile_names() -> list[str]:
    return sorted(_profile_document()["profiles"])


def load_profile(name: str) -> dict[str, Any]:
    profiles = _profile_document()["profiles"]
    if name not in profiles:
        raise ProfileError(f"no launch profile {name!r} ({sorted(profiles)})")
    profile = dict(profiles[name])
    missing = [key for key in PROFILE_FIELDS if key not in profile]
    if missing:
        raise ProfileError(f"launch profile {name!r} lacks {missing}")
    for key in PROFILE_FIELDS[3:]:
        if isinstance(profile[key], bool) or not isinstance(profile[key], int) or profile[key] <= 0:
            raise ProfileError(f"launch profile {name!r}: {key} must be a positive integer")
    return {"name": name, **profile}


def check_profile(profile: dict[str, Any], *, price_table: Path) -> dict[str, Any]:
    """Refuse a profile whose model would not run as named or is unpriced; return what was checked."""
    model = str(profile["model"])
    resolved = model
    if profile["driver"] == "codex":
        from merlin.development.backend.providers import codex_launch_codex_agent as codex_agent

        resolved = str(codex_agent.resolve_model(model) or "")
        if resolved != model:
            raise ProfileError(
                f"the codex driver resolves {model!r} to {resolved!r}: the run would not use the model it names"
            )
    from merlin.development.backend.optimization.authoring.telemetry_accounting import _declared_price_rate

    rate = _declared_price_rate(Path(price_table), resolved)
    if rate is None:
        raise ProfileError(
            f"{price_table} prices no model matching {resolved!r}; the telemetry preflight would refuse it"
        )
    return {"model": model, "resolved_model": resolved, "price_table": str(price_table), "rate": list(rate)}


LAUNCH_RECORD = "launch.json"


LAUNCH_LOG = "launch.log"


LAUNCH_SCHEMA = "merlin.phase2.whole_model_measured.launch.v1"


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
