"""Shared helpers for the gemmini_capsule_bench_v0 isolation harness.

Thin shim over ``merlin.benchharness`` (the shared harness primitives). This module is imported by
harness scripts BEFORE they add merlin/python to sys.path, so it bootstraps the repo root itself
(git first, parents[] fallback), puts merlin/python on the path, then re-exports the shared helpers.
Public symbols (REPO/EXP/RUNS/REPORTS/BUNDLES/sh/hash_tree/repo_sha) are preserved for callers.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Self-contained bootstrap (runs before merlin is importable).
_HERE = Path(__file__).resolve()
_git = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=str(_HERE.parent),
                      capture_output=True, text=True).stdout.strip()
REPO = Path(_git) if _git else _HERE.parents[4]
sys.path.insert(0, str(REPO / "merlin" / "python"))

from merlin.benchharness import sh, hash_tree, repo_sha, runs_root, reports_root  # noqa: E402

# EXP is DERIVED from where these scripts live (the experiment dir), and the TARGET from the experiment's
# descriptor — so a per-target experiment dir (e.g. <target>_capsule_bench_v0) works with no edits here.
EXP = _HERE.parents[1]                                 # the experiment dir the scripts live in
_desc = EXP / "target_experiment.yaml"
try:
    import yaml as _yaml
    TARGET = (_yaml.safe_load(_desc.read_text()) or {}).get("target") if _desc.is_file() else None
except Exception:  # noqa: BLE001
    TARGET = None
TARGET = TARGET or EXP.name.split("_")[0]              # fallback: the dir-name stem before _capsule_bench
RUNS = runs_root(TARGET, "capsule-bench")              # runs/<target>/capsule-bench
REPORTS = reports_root("capsule-bench", TARGET)        # artifacts/capsule-bench/<target>
BUNDLES = EXP / "input_bundles"

__all__ = ["REPO", "EXP", "TARGET", "RUNS", "REPORTS", "BUNDLES", "sh", "hash_tree", "repo_sha"]
