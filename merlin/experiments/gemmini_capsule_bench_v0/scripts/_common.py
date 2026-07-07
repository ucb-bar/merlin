"""Shared helpers for the gemmini_capsule_bench_v0 isolation harness."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
EXP = REPO / "merlin" / "experiments" / "gemmini_capsule_bench_v0"
# Generated runs live under the canonical runs/ root (see CLAUDE.md "Generated-output
# convention"); input bundles remain tracked source under the experiment dir.
RUNS = REPO / "runs" / "gemmini" / "capsule-bench"
# Generated reports/analyses live under artifacts/ (never in the source tree).
REPORTS = REPO / "artifacts" / "capsule-bench" / "gemmini"
BUNDLES = EXP / "input_bundles"
_SKIP = {"build", "__pycache__", ".git"}


def sh(args: list[str], cwd: Path | None = None, timeout: int = 120) -> str:
    try:
        return subprocess.run(args, cwd=str(cwd or REPO), capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except Exception:
        return ""


def hash_tree(root: Path) -> dict:
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
    return sh(["git", "rev-parse", "HEAD"]) or "unknown"
