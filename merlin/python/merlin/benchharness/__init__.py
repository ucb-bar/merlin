"""Shared, target-parametric benchmark-harness primitives.

The gemmini/muon capsule- and perf-bench experiment harnesses historically each hand-rolled their
own repo-root discovery (`Path(__file__).parents[4]`), run/report routing, and isolation utilities
(`_common.py`, `_pbcommon.py`). This package is the single home for that shared machinery so the
per-target harnesses stay thin. It is the seam WS2 (harness unification) grows into — the QA-loop,
sandbox, perf runner, and grading dispatch move here incrementally, parameterized by a target.

Canonical output routing (see CLAUDE.md "Generated-output convention"): runs -> `runs/`,
products -> `artifacts/` — never inside the source tree.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from merlin.common.paths import repo_root

__all__ = ["repo_root", "runs_root", "reports_root", "sh", "hash_tree", "repo_sha"]

_SKIP = {"build", "__pycache__", ".git"}


def runs_root(target: str, suite: str) -> Path:
    """Canonical run root for an experiment suite: out/runs/<target>/<suite>/."""
    return repo_root() / "out" / "runs" / target / suite


def reports_root(*parts: str) -> Path:
    """Canonical generated-product root under out/artifacts/: out/artifacts/<parts...>/."""
    return repo_root().joinpath("out", "artifacts", *parts)


def sh(args: list[str], cwd: Path | None = None, timeout: int = 120) -> str:
    """Run a command, return stripped stdout ('' on any failure/timeout)."""
    try:
        return subprocess.run(args, cwd=str(cwd or repo_root()), capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except Exception:
        return ""


def hash_tree(root: Path) -> dict:
    """Content hash of a directory tree (skips build/__pycache__/.git). For isolation checks."""
    if not root.exists():
        return {"present": False, "sha256": None, "n_files": 0}
    h = hashlib.sha256()
    n = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file() or _SKIP & set(p.parts):
            continue
        h.update(p.relative_to(root).as_posix().encode()); h.update(b"\0")
        h.update(p.read_bytes()); h.update(b"\0")
        n += 1
    return {"present": True, "sha256": h.hexdigest(), "n_files": n}


def repo_sha() -> str:
    """Current git HEAD sha, or 'unknown'."""
    return sh(["git", "rev-parse", "HEAD"]) or "unknown"
