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


# ---------------------------------------------------------------------------------------------------
# a stop condition that cannot be answered must not look like one that keeps passing
# ---------------------------------------------------------------------------------------------------
def test_a_condition_that_cannot_be_answered_says_so():
    """`fired: False` means the same thing for "checked, no" and "could not check".

    Measured on a recorded run: `predicted_remaining_below` returned the same `missing` on 8 of 8
    calls, because in this configuration the AGENT is the candidate generator and the host never
    enumerates an unevaluated candidate to price. That is correct behaviour, and it still has to be
    visible -- otherwise a reader counts four judges when one of them has never been able to speak.
    """
    from merlin.perf import select as SELECT
    from merlin.perf.budget import Budget, unpriced_channel

    state = SELECT.SearchState(
        baseline_cycles=1000, best_cycles=900.0,
        budget=Budget(unit=unpriced_channel("q", missing="not priced in this test"),
                      limit_items=100),
        attainable_cycles=500.0, improvements=(0.5, 0.4, 0.3))
    verdicts = {v.name: v for v in SELECT.check_stop(state)}

    predicted = verdicts["predicted_remaining_below"]
    assert predicted.fired is False
    assert predicted.evaluable is False, (
        "a condition with no candidate pool to price must report that it could not be answered")
    assert predicted.to_dict()["evaluable"] is False

    # the conditions that CAN be answered here must not be tarred with the same flag
    for name in ("attainment_reached", "plateaued", "budget_exhausted"):
        assert verdicts[name].evaluable is True, f"{name} is answerable and must say so"


def test_a_missing_input_marks_the_condition_unevaluable_wherever_it_occurs():
    """`missing` in the prose is not the same as saying so in a field.

    Observed live: `attainment_reached` reported "the conservative attainable target is UNKNOWN, so
    attainment cannot be evaluated" and was NOT listed as inapplicable, while
    `predicted_remaining_below` -- the same situation -- was. Two of four conditions could not be
    answered and only one admitted it, which is the defect this flag exists to remove.
    """
    from merlin.perf import select as SELECT
    from merlin.perf.budget import Budget, unpriced_channel

    budget = Budget(unit=unpriced_channel("q", missing="not priced here"), limit_items=100)
    # attainable UNKNOWN: exactly the live case, caused by members with no declared work
    state = SELECT.SearchState(baseline_cycles=1000, best_cycles=900.0, budget=budget,
                               improvements=(0.5, 0.4, 0.3))
    verdicts = {v.name: v for v in SELECT.check_stop(state)}
    assert verdicts["attainment_reached"].evaluable is False, (
        "a condition whose own reason says it cannot be evaluated must say so in the field too")
    for name in ("plateaued", "budget_exhausted"):
        assert verdicts[name].evaluable is True


# ---------------------------------------------------------------------------------------------------
# a reused weight is still declared work
# ---------------------------------------------------------------------------------------------------
def _reuse_descriptor(*, k=64, n=64, activations=(16, 16)):
    return {
        "operation": {"op": "resident_reuse", "attributes": {
            "weight": "W",
            "matmuls": [{"lhs": f"A{i}", "out": f"Y{i}"} for i in range(len(activations))]}},
        "inputs": [{"name": "W", "shape": [k, n], "role": "weight"}]
                  + [{"name": f"A{i}", "shape": [m, k], "role": "input"}
                     for i, m in enumerate(activations)],
    }


def test_a_reused_weight_contributes_work_for_every_reuse():
    """Twelve of thirty-eight members declare one resident weight and a LIST of activations.

    Reading only a single `lhs` left every one of them with no declared work -- no utilization, no
    share of the achievable rate, no verdict -- and left the corpus attainable total UNKNOWN, which
    silently disabled the attainment stop condition. Verified against the compiler's own emitted
    work on six real capsules: the sums agree exactly.
    """
    macs, basis = PAS.declared_capsule_macs(_reuse_descriptor(activations=(16, 16)))
    assert macs == 2 * (16 * 64 * 64), "each reuse of the weight is work the specification demands"
    assert "reuse" in basis

    one, _ = PAS.declared_capsule_macs(_reuse_descriptor(activations=(16,)))
    assert one == 16 * 64 * 64, "a single reuse must price as a single matmul"


def test_a_reuse_whose_shapes_do_not_contract_is_refused_by_name():
    bad = _reuse_descriptor()
    bad["inputs"][1]["shape"] = [16, 32]          # activation K disagrees with the weight's K
    macs, basis = PAS.declared_capsule_macs(bad)
    assert macs is None and "does not contract" in basis


def test_every_shipped_perf_capsule_declares_work_it_can_be_scored_against():
    """A member with no declared work gets no headroom signal at all, and drags the corpus total
    into UNKNOWN with it. This pins the corpus, not just the function."""
    import glob as _glob

    import yaml as _yaml

    from merlin.common.paths import repo_root as _root

    undecided, elementwise = [], []
    for path in sorted(_glob.glob(str(_root() / "merlin/contract/capsules/_perf/*/capsule.yaml"))):
        descriptor = _yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        macs, basis = PAS.declared_capsule_macs(descriptor)
        # AN ELEMENTWISE MEMBER HAS NO MACs, and that is a fact about the workload rather than a gap
        # in the pricing. It ships as the "part" arm of the fusion comparison, so what scores it is
        # the paired difference against the fused member, not a utilization ratio. Demanding a MAC
        # count of it would invent a denominator; the assertion below instead pins that it stays
        # unpriced, so if it ever starts pricing, this exemption is reported as stale rather than
        # silently covering a real number.
        if (descriptor.get("operation") or {}).get("op") in PAS._ELEMENTWISE_OPERATIONS:
            elementwise.append(Path(path).parent.name)
            assert not macs, f"{Path(path).parent.name} is elementwise yet priced {macs} MACs"
            continue
        if not macs:
            undecided.append(f"{Path(path).parent.name}: {basis}")
    assert not undecided, "capsules with no declared work: " + "; ".join(undecided)
    assert elementwise, ("no elementwise member was found, so the exemption above is either stale or "
                         "matching nothing -- check it still names the shipped op spelling")
