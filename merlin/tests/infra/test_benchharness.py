"""merlin.benchharness shared primitives + the _common/_pbcommon shims that re-export them.

WS2 unifies the per-target bench harnesses onto merlin.benchharness. This guards the shared
run/report routing + repo-root helpers, and that the two shim modules still expose the exact public
symbols (with the same values) that the (yet-to-be-migrated) harness scripts import.
"""
from __future__ import annotations

import importlib.util
import sys

from merlin import benchharness as B
from merlin.common.paths import repo_root

ROOT = repo_root()


def test_benchharness_routing():
    assert B.repo_root() == ROOT
    assert B.runs_root("gemmini", "capsule-bench") == ROOT / "runs" / "gemmini" / "capsule-bench"
    assert B.reports_root("capsule-bench", "gemmini") == ROOT / "artifacts" / "capsule-bench" / "gemmini"
    assert B.repo_sha() and B.repo_sha() != "unknown"
    assert B.hash_tree(ROOT / "does-not-exist") == {"present": False, "sha256": None, "n_files": 0}


def _load(name: str, rel: str):
    path = ROOT / rel
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_common_shim_preserves_symbols():
    c = _load("_cbench_common", "merlin/experiments/gemmini_capsule_bench_v0/scripts/_common.py")
    assert c.REPO == ROOT
    assert c.RUNS == ROOT / "runs" / "gemmini" / "capsule-bench"
    assert c.REPORTS == ROOT / "artifacts" / "capsule-bench" / "gemmini"
    assert c.BUNDLES == c.EXP / "input_bundles"
    assert callable(c.sh) and callable(c.hash_tree) and c.repo_sha() != "unknown"


def test_pbcommon_shim_preserves_symbols():
    p = _load("_pb_common", "merlin/experiments/gemmini_perf_bench/scripts/_pbcommon.py")
    assert p.RUNS == ROOT / "runs" / "gemmini" / "perf-bench"
    assert p.REPORTS == ROOT / "artifacts" / "plots" / "gemmini" / "perf-bench"
    assert p.KERNELS == p.EXP / "kernels"
    assert p.DIM == 16 and p.PEAK_MACS_PER_CYCLE == 256
    assert p.align(17) == 32 and p.matmul_macs(2, 3, 4) == 24 and p.utilization_pct(256, 1) == 100.0
