"""merlin.benchharness shared primitives + the _common/_pbcommon shims that re-export them.

WS2 unifies the per-target bench harnesses onto merlin.benchharness. This guards the shared
run/report routing + repo-root helpers, and that the two shim modules still expose the exact public
symbols (with the same values) that the (yet-to-be-migrated) harness scripts import.
"""
from __future__ import annotations

import importlib.util
import json
import sys

from merlin import benchharness as B
from merlin.common.paths import repo_root

ROOT = repo_root()


def test_benchharness_routing():
    assert B.repo_root() == ROOT
    assert B.runs_root("gemmini", "capsule-bench") == ROOT / "out" / "runs" / "gemmini" / "capsule-bench"
    assert B.reports_root("capsule-bench", "gemmini") == ROOT / "out/artifacts" / "capsule-bench" / "gemmini"
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
    assert c.RUNS == ROOT / "out" / "runs" / "gemmini" / "capsule-bench"
    assert c.REPORTS == ROOT / "out/artifacts" / "capsule-bench" / "gemmini"
    assert c.BUNDLES == c.EXP / "input_bundles"
    assert callable(c.sh) and callable(c.hash_tree) and c.repo_sha() != "unknown"


def test_pbcommon_shim_preserves_symbols():
    p = _load("_pb_common", "merlin/experiments/gemmini_perf_bench/scripts/_pbcommon.py")
    assert p.RUNS == ROOT / "out" / "runs" / "gemmini" / "perf-bench"
    assert p.REPORTS == ROOT / "out/artifacts" / "plots" / "gemmini" / "perf-bench"
    assert p.KERNELS == p.EXP / "kernels"
    assert p.DIM == 16 and p.PEAK_MACS_PER_CYCLE == 256
    assert p.align(17) == 32 and p.matmul_macs(2, 3, 4) == 24 and p.utilization_pct(256, 1) == 100.0


# --- target-parametric bench driver (spec + selfcheck + perf), oracle-free via a stub runner -------
class _StubRunner:
    """Mimics capsule_runner/muon_capsule_runner's discover_capsules + run_capsule interface."""

    def __init__(self, results):
        self._results = results  # name -> result dict

    def discover_capsules(self, root, *, labels=None, contract=None):
        return [{"name": n} for n in sorted(self._results)]

    def run_capsule(self, cap, package_dir, *, runs_root, run_id, contract=None, timeout=0):
        return self._results[cap["name"]]


def _spec(runner):
    from merlin.benchharness.spec import BenchTargetSpec
    return BenchTargetSpec(name="Stub", runner=runner, corpus_root=ROOT, perf_tier="L2",
                           perf_fields=lambda t: {"pct_fp_peak": t.get("pct_fp_peak")})


def test_redacted_grade_is_redacted_and_aggregates():
    from merlin.benchharness.selfcheck import redacted_grade
    runner = _StubRunner({
        "k_ok": {"status": "pass", "tiers": {"L2": {"cycles": 100, "pct_fp_peak": 0.5}},
                 "numeric": {"mismatch_count": 0}},
        "k_bad": {"status": "fail", "tiers": {"L2": {}}, "numeric": {"mismatch_count": 7},
                  "failure": {"plane": "numeric", "category": "value_mismatch"}},
    })
    v = redacted_grade(_spec(runner), "sub", runs_root="/tmp/x", timeout=1)
    assert v["n_passed"] == 1 and v["n_capsules"] == 2 and v["all_pass"] is False
    bad = next(r for r in v["per_capsule"] if r["capsule"] == "k_bad")
    assert bad["fail_plane"] == "numeric" and bad["mismatch_count"] == 7
    # redaction: only a COUNT + status/plane surface — no expected/got values anywhere in the verdict
    assert "expected" not in json.dumps(v) and "golden" not in json.dumps(v)
    # only= filters to a single capsule
    assert redacted_grade(_spec(runner), "sub", runs_root="/tmp/x", timeout=1,
                          only="k_ok")["n_capsules"] == 1


def test_run_perf_writes_report(tmp_path):
    from merlin.benchharness.perf import run_perf
    runner = _StubRunner({
        "k0": {"status": "pass", "tiers": {"L2": {"cycles": 200, "pct_fp_peak": 1.0}}},
        "k1": {"status": "fail", "tiers": {"L2": {"cycles": None}}},
    })
    s = run_perf(_spec(runner), package="pkg", run_id="t", out_dir=tmp_path, timeout=1)
    assert s["passed"] == 1 and s["total"] == 2
    assert (tmp_path / "perf_results.json").is_file() and (tmp_path / "perf_table.md").is_file()
    assert "1/2 pass" in (tmp_path / "perf_table.md").read_text()
