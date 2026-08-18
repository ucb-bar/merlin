"""The capsule-bench aggregators must look for runs where runs actually are.

Several aggregators kept resolving runs at ``<experiment>/runs/`` — a directory that stopped existing
when generated output moved under the single ``out/`` root. Nothing errored: they simply iterated an
absent directory and produced an EMPTY aggregate, so every plot and comparison table silently reported
"no runs" while twenty of them sat on disk. These tests pin both halves of that fix.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from merlin.common.paths import repo_root

HARNESS = repo_root() / "merlin/experiments/capsule_bench/harness"
# Modules that enumerate run directories; each must go through the descriptor-driven root.
RUN_READERS = ["agg_ab_results.py", "agg_agentic_results.py", "timing_decomposition.py",
               "analyze_abc4.py", "verify_no_cheat.py", "abc_status.py", "agg_by_model.py",
               "plots/make_plots.py"]


@pytest.mark.parametrize("rel", RUN_READERS)
def test_no_module_resolves_runs_under_the_retired_experiment_root(rel):
    text = (HARNESS / rel).read_text()
    for bad in ('EXP / "runs"', 'EXP/"runs"', 'EXP / "reports"'):
        assert bad not in text, (
            f"{rel} resolves runs under the retired <experiment>/runs root ({bad}); use the "
            f"descriptor-driven C.RUNS / C.REPORTS or the aggregate is silently empty")


def test_run_root_is_under_the_single_out_root():
    p = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.argv=['x']; import _common as C; print(C.RUNS); print(C.REPORTS)"],
        cwd=str(HARNESS), capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    runs, reports = p.stdout.split()
    assert "/out/runs/" in runs and runs.endswith("capsule-bench"), runs
    assert "/out/artifacts/capsule-bench/" in reports, reports


def test_every_arm_bundle_variant_maps_back_to_its_arm():
    """A bundle ships in several variants (_public_v0 / _realistic_v0 / _hwbringup_v0 / _nokernel_v0).
    The CIRCT arm's id also starts with the arm-3 stem, so a prefix table in the wrong order files
    every arm-4 run as arm-3 — which would silently merge two arms in the comparison."""
    p = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.argv=['x']; import agg_agentic_results as A;"
         "print('\\n'.join(f'{b}={A.arm_from_bundle_id(b)}' for b in ["
         "'merlin_assisted_rtlchecks_public_v0','merlin_assisted_rtlchecks_hwbringup_v0',"
         "'merlin_assisted_rtlchecks_hwbringup_nokernel_v0','merlin_assisted_hwbringup_v0',"
         "'raw_baseline_hwbringup_v0','cpp_merlininfra_hwbringup_v0','not_a_bundle']))"],
        cwd=str(HARNESS), capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    got = dict(line.split("=") for line in p.stdout.split())
    assert got["merlin_assisted_rtlchecks_public_v0"] == "merlin_rtlchecks"
    assert got["merlin_assisted_rtlchecks_hwbringup_v0"] == "merlin_rtlchecks"
    assert got["merlin_assisted_rtlchecks_hwbringup_nokernel_v0"] == "merlin_rtlchecks"
    assert got["merlin_assisted_hwbringup_v0"] == "merlin"
    assert got["raw_baseline_hwbringup_v0"] == "baseline"
    assert got["cpp_merlininfra_hwbringup_v0"] == "cpp_merlininfra"
    assert got["not_a_bundle"] == "None"          # fail closed, never guess an arm


def test_collect_finds_the_runs_that_exist_on_disk():
    """The end-to-end symptom: with runs present, the aggregate must not be empty."""
    p = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.argv=['x']; import _common as C, agg_ab_results as AB;"
         "n=sum(len(v) for v in AB.collect(None).values());"
         "print(sum(1 for s in ('raw_baseline','merlin_assisted') if (C.RUNS/s).is_dir()));print(n)"],
        cwd=str(HARNESS), capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    have_dirs, n_found = (int(x) for x in p.stdout.split())
    if not have_dirs:
        pytest.skip("no capsule-bench run directories on this machine")
    assert n_found > 0, "run directories exist but collect() found none — the run root is wrong again"


def test_scores_are_ranked_by_count_not_by_string():
    """`passed` is written as "20/20"; ranking those lexicographically puts "5/20" above "20/20".

    That is not hypothetical: the by-model table reported gpt-5.6-sol's best public score as 5/20 across
    two runs where one scored 20/20, because "5" > "2" as a character. A comparison table whose headline
    number is the WORSE of two runs is worse than no table.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("agg_by_model", HARNESS / "agg_by_model.py")
    mod = importlib.util.module_from_spec(spec)
    import sys as _s
    _s.argv = ["x"]
    _s.path.insert(0, str(HARNESS))
    spec.loader.exec_module(mod)

    lo = mod._score({"public_dev": {"passed": "5/20"}}, "public_dev")
    hi = mod._score({"public_dev": {"passed": "20/20"}}, "public_dev")
    assert lo["n"] == 5 and hi["n"] == 20, "the printed score must be parsed into a count"
    assert hi["n"] > lo["n"], "20/20 must outrank 5/20"
    assert lo["passed"] == "5/20" and hi["passed"] == "20/20", "the printed form is preserved"

    rows = [{"run_id": "r_lo", "model": "m", "arm": "a", "public": lo, "hidden": {"n": 5, "passed": "5/5"},
             "billing_mode": "subscription_notional", "cost_usd": None, "codex": {},
             "first_failure_planes": {}, "tokens_total": 1, "tool_calls": 1, "n_rounds": 1},
            {"run_id": "r_hi", "model": "m", "arm": "a", "public": hi, "hidden": {"n": 5, "passed": "5/5"},
             "billing_mode": "subscription_notional", "cost_usd": None, "codex": {},
             "first_failure_planes": {}, "tokens_total": 1, "tool_calls": 1, "n_rounds": 1}]
    assert mod.by_model(rows)["m"]["best_public"] == "20/20"
