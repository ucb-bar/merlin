"""The ceiling the agent optimises toward must come from measurements it did not author.

Phase 1's corpus is not phase 2's. Measured 2026-09-04: harvested from the functional run alone the
ceiling was 80.01 MACs/cycle while four performance members already ran ABOVE it at baseline, and
that set's dispersion of 0.472 put "already at the ceiling" at 42.25 -- so **14 of 38 members were
told they had no headroom left**. Over the widened set the ceiling is 99.79, the dispersion 0.245,
and 6 members read as finished.

The safety property is the one this file exists for: BASELINE arms may move the ceiling, CANDIDATE
arms may not. A ceiling the agent can raise is a target the agent authors -- its own candidate would
sit near 1.0 by construction while every other member was pushed down.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from merlin.common.paths import merlin_dir

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import perf_agent_stage as PAS  # noqa: E402


def _plant(work_root: Path, *, call: int, arm: str, capsule: str, macs: int, cycles: int):
    """Write one measured member the way the stage's own workspace layout does."""
    run = (work_root / f"round_00" / f"call_{call:03d}"
           / PAS._ARM_WORKSPACE.format(index=0, arm=arm)
           / "unprofiled" / "capsule_runs" / "runs" / "t-capsule-bench" / f"x_{capsule}_{arm}")
    (run / "generated").mkdir(parents=True)
    (run / "capsule_result.json").write_text(json.dumps({
        "capsule": capsule, "status": "pass",
        "tiers": {"L3": {"status": "pass", "cycles": cycles,
                         "derived_from_rtl": True, "cycle_accurate": True}},
    }), encoding="utf-8")
    # one MATMUL whose declared shapes price to exactly `macs`
    (run / "generated" / "command_buffer.json").write_text(json.dumps({
        "abi_version": "0.1", "target": "t",
        "tensors": {"A": {"shape": [1, macs // 1], "dtype": "i8", "role": "input"},
                    "W": {"shape": [macs // 1, 1], "dtype": "i8", "role": "weight"},
                    "Y": {"shape": [1, 1], "dtype": "i32", "role": "output"}},
        "commands": [{"opcode": "MATMUL", "operands": {"lhs": "A", "rhs": "W", "dst": "Y"}}],
    }), encoding="utf-8")


def test_only_the_baseline_arm_is_harvested(tmp_path):
    """The whole safety argument, asserted directly."""
    work = tmp_path / "work"
    _plant(work, call=1, arm="baseline", capsule="SLOW", macs=1000, cycles=1000)
    _plant(work, call=1, arm="candidate", capsule="FAST", macs=1000, cycles=1)
    harvested = PAS.harvest_baseline_points(work)
    names = {p.capsule for p in harvested}
    assert any("SLOW" in n for n in names), "the baseline arm must be harvested"
    assert not any("FAST" in n for n in names), (
        "a candidate arm reached the ceiling: the agent can now author its own target")


def test_an_absent_work_root_harvests_nothing_rather_than_raising(tmp_path):
    assert PAS.harvest_baseline_points(tmp_path / "never-created") == []


def _evaluator(tmp_path, seed):
    from types import SimpleNamespace
    return PAS.DevelopmentGsimFeedback(
        SimpleNamespace(sha256="a" * 64), SimpleNamespace(capsules=[], capsules_sha256="a" * 64),
        Path("."), "a" * 64, SimpleNamespace(target="t"), {}, tmp_path / "work", {},
        peak_macs_per_cycle=256, peak_basis="test",
        achievable_macs_per_cycle=seed, achievable_basis="seeded", achievable_dispersion=0.5,
        seed_points=(), functional_run_id="fn")


def test_an_empty_harvest_never_downgrades_a_known_ceiling(tmp_path):
    """Absence of new evidence must not erase the evidence already in hand."""
    evaluator = _evaluator(tmp_path, 80.0)
    evaluator._refresh_achievable()
    assert evaluator.achievable_macs_per_cycle == 80.0
    assert evaluator.achievable_basis == "seeded"


def test_a_faster_baseline_raises_the_ceiling_and_says_where_it_came_from(tmp_path):
    evaluator = _evaluator(tmp_path, 80.0)
    _plant(evaluator.work_root, call=1, arm="baseline", capsule="Q", macs=1000, cycles=5)
    _plant(evaluator.work_root, call=2, arm="baseline", capsule="R", macs=1000, cycles=10)
    evaluator._refresh_achievable()
    assert evaluator.achievable_macs_per_cycle > 80.0, "a faster baseline must move the ceiling"
    basis = evaluator.achievable_basis
    assert "baseline arms only" in basis, (
        "the basis must let a reader rule out circularity without reading the code")
    assert "no candidate measurement contributes" in basis
