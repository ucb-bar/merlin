"""Phase-1 authoring workspace placement and read-only historical discovery.

New mutable work lives under generated output. Archived environment records remain authoritative:
resuming an old run does not move its workspace or rewrite its scientific provenance.
"""

from __future__ import annotations

import sys
from functools import wraps
from pathlib import Path

import yaml

from merlin.common.paths import build_dir, repo_root


def _component(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    if not value or value in {".", ".."} or any(character not in allowed for character in value):
        raise ValueError(f"invalid workspace path component: {value!r}")
    return value


def workspace_parent(target: str, arm: str) -> Path:
    """Generated workspace root; honors MERLIN_OUT_ROOT without creating directories."""
    return build_dir() / "agent-workspaces" / "phase1" / _component(target) / _component(arm)


def recorded_workspace(run_dir: Path) -> Path | None:
    environment = run_dir / "environment.yaml"
    if not environment.is_file():
        return None
    data = yaml.safe_load(environment.read_text(encoding="utf-8")) or {}
    value = data.get("workspace_path")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute() or path.name != "workspace":
        raise ValueError(f"invalid archived workspace path: {value!r}")
    return path


def workspace_candidates(
    run_dir: Path,
    *,
    target: str | None = None,
    arm: str | None = None,
    experiment: Path | None = None,
    repo: Path | None = None,
) -> list[Path]:
    """Recorded, generated, then legacy workspace paths; discovery never changes artifacts."""
    run_id = _component(run_dir.name)
    candidates = []
    archived = recorded_workspace(run_dir)
    if archived is not None:
        candidates.append(archived)
    generated = build_dir() / "agent-workspaces" / "phase1"
    target_part = _component(target) if target else "*"
    arm_part = _component(arm) if arm else "*"
    candidates.extend(sorted(generated.glob(f"{target_part}/{arm_part}/{run_id}/workspace")))
    if experiment is not None:
        candidates.append(experiment / "_qa_ws" / run_id / "workspace")
    else:
        targets = (repo or repo_root()) / "merlin" / "experiments" / "capsule_bench" / "targets"
        candidates.extend(sorted(targets.glob(f"*/_qa_ws/{run_id}/workspace")))
    return list(dict.fromkeys(candidates))


def select_workspace_root(*, target: str, arm: str, run_dir: Path, experiment: Path, resume: bool) -> Path:
    if resume:
        for workspace in workspace_candidates(run_dir, target=target, arm=arm, experiment=experiment):
            if workspace.is_dir():
                return workspace.parent
    return workspace_parent(target, arm) / _component(run_dir.name)


def workspace_session(function):
    """Release workspace leases on an orderly return, not after an uncertain crash.

    The driver owns child shutdown. An exception may strand a child, so retain its lease until an
    operator has checked the crashed process tree and explicitly acknowledges abandonment. ``wraps``
    preserves the public driver's identity and source-based control-flow regression tests.
    """

    @wraps(function)
    def invoke(*args, **kwargs):
        leases = []
        try:
            result = function(*args, _workspace_leases=leases, **kwargs)
        except BaseException:
            if leases:
                print(
                    "Phase-1 workspace leases retained after interrupted execution: "
                    + ", ".join(held.path for held in leases)
                    + ". Child shutdown is uncertain. After this process exits, verify that all "
                    "child workers have stopped, then use storage_lifecycle.acknowledge_abandoned"
                    "(path, reason=...) to release crash protection. Retention pins remain intact.",
                    file=sys.stderr,
                )
            raise
        for held in reversed(leases):
            held.close("completed" if result == 0 else "failed")
        return result

    return invoke
