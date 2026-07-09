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

EXP = REPO / "merlin" / "experiments" / "gemmini_capsule_bench_v0"
RUNS = runs_root("gemmini", "capsule-bench")          # runs/gemmini/capsule-bench
REPORTS = reports_root("capsule-bench", "gemmini")    # artifacts/capsule-bench/gemmini
BUNDLES = EXP / "input_bundles"

__all__ = ["REPO", "EXP", "RUNS", "REPORTS", "BUNDLES", "sh", "hash_tree", "repo_sha"]
