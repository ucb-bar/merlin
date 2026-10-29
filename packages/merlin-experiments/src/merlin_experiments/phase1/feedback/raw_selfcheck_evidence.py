"""Retain raw ordinary selfchecks in an operator-selected host-only directory."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

ROOT_ENV = "MERLIN_PHASE1_SELFCHECK_EVIDENCE_ROOT"
WORKSPACE_ENV = "MERLIN_PHASE1_SELFCHECK_VISIBLE_WORKSPACE"


def _ordinary_absolute(raw: str, label: str) -> Path:
    path = Path(raw)
    if not raw or not path.is_absolute() or path.resolve() != path:
        raise ValueError(f"{label} must be a canonical absolute directory without symlinks")
    if path.exists() and not path.is_dir():
        raise ValueError(f"{label} must be an ordinary directory")
    return path


def _selected_root(environment: Mapping[str, str]) -> Path | None:
    if ROOT_ENV not in environment and WORKSPACE_ENV not in environment:
        return None
    root = _ordinary_absolute(environment.get(ROOT_ENV, ""), "selfcheck evidence root")
    workspace = _ordinary_absolute(environment.get(WORKSPACE_ENV, ""), "visible workspace")
    if not workspace.is_dir():
        raise ValueError("visible workspace must exist before selecting selfcheck evidence")
    if root == workspace or root in workspace.parents or workspace in root.parents:
        raise ValueError("selfcheck evidence root must be disjoint from the visible workspace")
    return root


def broker_environment(root: Path | None, workspace: Path, *, environ=None) -> dict[str, str]:
    """Select from trusted broker configuration; requests cannot choose raw paths."""
    environment = dict(os.environ if environ is None else environ)
    for key in (ROOT_ENV, WORKSPACE_ENV):
        environment.pop(key, None)
    if root is not None:
        environment[ROOT_ENV] = str(root)
        environment[WORKSPACE_ENV] = str(workspace)
        _selected_root(environment)
    return environment


def operator_log_dir(*, environ=None) -> Path | None:
    """Operator-only directory for broker child logs, or ``None`` when no evidence root is selected.

    A child's raw stdout/stderr can name host paths; spooled into the agent's channel they were
    readable by the agent in full. With a selected evidence root they live beside the raw runs.
    """
    environment = os.environ if environ is None else environ
    try:
        root = _selected_root(environment)
    except ValueError:
        return None  # an unusable selection keeps the historical channel location rather than failing a check
    if root is None:
        return None
    directory = root / "broker-logs"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory


def create_runs_root(*, environ=None) -> Path:
    """Keep failed and successful raw runs; legacy unselected calls use TMPDIR."""
    environment = os.environ if environ is None else environ
    root = _selected_root(environment)
    if root is not None:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        _selected_root(environment)
    return Path(tempfile.mkdtemp(prefix="selfcheck_", dir=root))
