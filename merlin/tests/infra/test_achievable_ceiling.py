"""Observed ceilings use independent public baseline measurements only.

Current-session baseline arms may contribute a rate. Candidate arms and saved
functional or sibling-run artifacts cannot author a target for their own search.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from merlin_experiments.phase2 import development_feedback as DF
from merlin_experiments.phase2 import feedback_metrics as FM

from merlin.common.paths import merlin_dir

_SCRIPTS = merlin_dir() / "experiments/gemmini_perf_bench/scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _evaluator(tmp_path, seed):
    from types import SimpleNamespace

    return DF.DevelopmentGsimFeedback(
        SimpleNamespace(sha256="a" * 64),
        SimpleNamespace(capsules=[], capsules_sha256="a" * 64),
        Path("."),
        "a" * 64,
        SimpleNamespace(target="t"),
        {},
        tmp_path / "work",
        {},
        peak_macs_per_cycle=256,
        peak_basis="test",
        achievable_macs_per_cycle=seed,
        achievable_basis="seeded",
        achievable_dispersion=0.5,
    )


def test_an_empty_harvest_never_downgrades_a_known_ceiling(tmp_path):
    """Absence of new evidence must not erase the evidence already in hand."""
    evaluator = _evaluator(tmp_path, 80.0)
    evaluator._refresh_achievable()
    assert evaluator.achievable_macs_per_cycle == 80.0
    assert evaluator.achievable_basis == "seeded"


def _matmul_descriptor(*, m: int, k: int, n: int) -> dict:
    return {
        "operation": {"op": "matmul", "attributes": {"lhs": "A", "weight": "W"}},
        "inputs": [{"name": "A", "shape": [m, k]}, {"name": "W", "shape": [k, n]}],
    }


def test_a_member_is_not_scored_against_an_unrelated_deep_k_rate(tmp_path):
    """The live PM defect: k=16 was judged against a deep-k ceiling it cannot amortise to."""
    from merlin_experiments.phase2 import calibration as PM

    evaluator = _evaluator(tmp_path, 256.0)
    evaluator._achievable_points = (
        PM.MeasuredPoint("shallow", 4096, 256, "host", (16,)),
        PM.MeasuredPoint("shallow-wide", 8192, 400, "host", (16,)),
        PM.MeasuredPoint("deep", 65536, 256, "host", (256,)),
    )
    member = SimpleNamespace(descriptor=_matmul_descriptor(m=16, k=16, n=16))
    rate, basis, dispersion = evaluator._matched_achievable(member)
    assert rate == 20.48, "the faster deep-k point must not become the shallow member's target"
    assert "[16]" in basis and "2 host-owned" in basis
    assert dispersion is not None, "the matching cohort's spread should be measured"


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
        baseline_cycles=1000,
        best_cycles=900.0,
        budget=Budget(unit=unpriced_channel("q", missing="not priced in this test"), limit_items=100),
        attainable_cycles=500.0,
        improvements=(0.5, 0.4, 0.3),
    )
    verdicts = {v.name: v for v in SELECT.check_stop(state)}

    predicted = verdicts["predicted_remaining_below"]
    assert predicted.fired is False
    assert predicted.evaluable is False, (
        "a condition with no candidate pool to price must report that it could not be answered"
    )
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
    state = SELECT.SearchState(baseline_cycles=1000, best_cycles=900.0, budget=budget, improvements=(0.5, 0.4, 0.3))
    verdicts = {v.name: v for v in SELECT.check_stop(state)}
    assert verdicts["attainment_reached"].evaluable is False, (
        "a condition whose own reason says it cannot be evaluated must say so in the field too"
    )
    for name in ("plateaued", "budget_exhausted"):
        assert verdicts[name].evaluable is True


# ---------------------------------------------------------------------------------------------------
# a reused weight is still declared work
# ---------------------------------------------------------------------------------------------------
def _reuse_descriptor(*, k=64, n=64, activations=(16, 16)):
    return {
        "operation": {
            "op": "resident_reuse",
            "attributes": {
                "weight": "W",
                "matmuls": [{"lhs": f"A{i}", "out": f"Y{i}"} for i in range(len(activations))],
            },
        },
        "inputs": [{"name": "W", "shape": [k, n], "role": "weight"}]
        + [{"name": f"A{i}", "shape": [m, k], "role": "input"} for i, m in enumerate(activations)],
    }


def test_a_reused_weight_contributes_work_for_every_reuse():
    """Twelve of thirty-eight members declare one resident weight and a LIST of activations.

    Reading only a single `lhs` left every one of them with no declared work -- no utilization, no
    share of the achievable rate, no verdict -- and left the corpus attainable total UNKNOWN, which
    silently disabled the attainment stop condition. Verified against the compiler's own emitted
    work on six real capsules: the sums agree exactly.
    """
    macs, basis = FM.declared_capsule_macs(_reuse_descriptor(activations=(16, 16)))
    assert macs == 2 * (16 * 64 * 64), "each reuse of the weight is work the specification demands"
    assert "reuse" in basis

    one, _ = FM.declared_capsule_macs(_reuse_descriptor(activations=(16,)))
    assert one == 16 * 64 * 64, "a single reuse must price as a single matmul"


def test_a_reuse_whose_shapes_do_not_contract_is_refused_by_name():
    bad = _reuse_descriptor()
    bad["inputs"][1]["shape"] = [16, 32]  # activation K disagrees with the weight's K
    macs, basis = FM.declared_capsule_macs(bad)
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
        macs, basis = FM.declared_capsule_macs(descriptor)
        # AN ELEMENTWISE MEMBER HAS NO MACs, and that is a fact about the workload rather than a gap
        # in the pricing. It ships as the "part" arm of the fusion comparison, so what scores it is
        # the paired difference against the fused member, not a utilization ratio. Demanding a MAC
        # count of it would invent a denominator; the assertion below instead pins that it stays
        # unpriced, so if it ever starts pricing, this exemption is reported as stale rather than
        # silently covering a real number.
        if (descriptor.get("operation") or {}).get("op") in FM._ELEMENTWISE_OPERATIONS:
            elementwise.append(Path(path).parent.name)
            assert not macs, f"{Path(path).parent.name} is elementwise yet priced {macs} MACs"
            continue
        if not macs:
            undecided.append(f"{Path(path).parent.name}: {basis}")
    assert not undecided, "capsules with no declared work: " + "; ".join(undecided)
    assert elementwise, (
        "no elementwise member was found, so the exemption above is either stale or "
        "matching nothing -- check it still names the shipped op spelling"
    )


def test_host_island_declared_work_prices_both_contractions_without_pricing_the_map():
    import yaml

    from merlin.common.paths import repo_root

    path = repo_root() / "merlin/contract/capsules/_perf/PB00_island_m16k32h16n16/capsule.yaml"
    descriptor = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert FM.declared_capsule_macs(descriptor)[0] == 16 * 32 * 16 + 16 * 16 * 16
    assert FM.declared_reduction_depths(descriptor)[0] == (32, 16)
    descriptor["inputs"][2]["shape"][0] = 15
    assert FM.declared_capsule_macs(descriptor)[0] is None
