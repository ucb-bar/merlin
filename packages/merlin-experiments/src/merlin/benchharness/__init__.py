"""Shared, target-parametric benchmark-harness primitives.

The gemmini/muon capsule- and perf-bench experiment harnesses historically each hand-rolled their
own repo-root discovery (`Path(__file__).parents[4]`), run/report routing, and isolation utilities
(`_common.py`, `_pbcommon.py`). This package is the single home for that shared machinery so the
per-target harnesses stay thin. It is the seam WS2 (harness unification) grows into — the QA-loop,
sandbox, perf runner, and grading dispatch move here incrementally, parameterized by a target.

Canonical output routing uses ``merlin.common.paths`` and honors ``MERLIN_OUT_ROOT``:
runs -> ``out/runs/``, products -> ``out/artifacts/`` — never inside the source tree.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from merlin.common.paths import artifacts_dir, repo_root, runs_root
from merlin.common.tree_hash import _SKIP, hash_tree  # noqa: F401 -- stable shared identity

__all__ = ["repo_root", "runs_root", "reports_root", "sh", "hash_tree", "repo_sha"]


def reports_root(*parts: str) -> Path:
    """Canonical generated-product root under out/artifacts/: out/artifacts/<parts...>/."""
    return artifacts_dir().joinpath(*parts)


def sh(args: list[str], cwd: Path | None = None, timeout: int = 120) -> str:
    """Run a command, return stripped stdout ('' on any failure/timeout)."""
    try:
        return subprocess.run(
            args, cwd=str(cwd or repo_root()), capture_output=True, text=True, timeout=timeout
        ).stdout.strip()
    except Exception:
        return ""


def repo_sha(*, repo: Path | None = None) -> str:
    """Current git HEAD sha, or 'unknown'."""
    return sh(["git", "rev-parse", "HEAD"], cwd=repo) or "unknown"
