"""Every capsule that demands certification must be priced, and the price must be derived.

`required_oracle_tiers: [.., L3]` asks for a cycle-accurate run, and that run costs real time. Nothing
checked the cost against a budget, so the corpus grew capsules whose certification nobody would pay
for -- which is the same as not certifying them while reporting that they must be.

The driver is MEASURED. A calibration ladder held the lhs at one tile while the weight grew 64x, and
cycle-accurate seconds scaled with the WRITTEN OUTPUT, not the operands: x1.98 then x2.06 against
output x2 and x2, r2 0.9998, at 0.347 s per element with no fixed floor. Confirmed off-ladder on a
two-commit resident-reuse capsule at 0.3409 s/element. Over the corpus: 295 capsules demand L3 for a
predicted 125.1 hours, of which TWELVE are 108.5 hours -- 87% -- while the median capsule costs 89s.

The two failure modes this pins are the ones that actually happened while building it: pricing only
`COMMIT` capsules left 85 of 295 at zero (an epilogue-only or movement capsule writes its result with
no commit at all), and a linalg-on-tensors capsule parsed by a bare builtin context left another 54 at
zero. A capsule priced at zero reads as free, which is the most dangerous possible error here.
"""
from __future__ import annotations

import importlib.util
import sys

import pytest
import yaml

from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen import cert_cost as CC

_CAPS = merlin_dir() / "contract" / "capsules"


def _gate():
    p = repo_root() / "build_tools" / "scripts" / "check_cert_affordability.py"
    spec = importlib.util.spec_from_file_location("_afford_gate", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_afford_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def _iface(name: str):
    for p in _CAPS.rglob("capsule.interface.mlir"):
        if p.parent.name == name:
            return p
    return None


def test_a_commit_capsule_is_priced_by_its_committed_extent():
    p = _iface("A2_single_tile_matmul")
    if p is None:
        pytest.skip("capsule absent")
    assert CC.capsule_output_elements(p.read_text(encoding="utf-8")) == 256


def test_two_commits_are_summed_not_maximised():
    """A capsule that writes twice pays for both; the largest single write understates it."""
    p = _iface("PC00_k64")
    if p is None:
        pytest.skip("capsule absent")
    # Two commits of 16x64 each. Measured cost of this capsule was 698.2s, and 2048 * 0.347 = 711s.
    assert CC.capsule_output_elements(p.read_text(encoding="utf-8")) == 2048


def test_a_capsule_with_no_commit_is_still_priced():
    """The 85-capsule bug: an epilogue-only or movement capsule writes its result with no COMMIT."""
    for name, expect in (("PF02_bias_add_m16k16n16", 256), ("A1_mvin_mvout", 256)):
        p = _iface(name)
        if p is None:
            continue
        got = CC.capsule_output_elements(p.read_text(encoding="utf-8"))
        assert got == expect, f"{name}: {got} != {expect}"
        assert got > 0, f"{name} priced at zero reads as free"


def test_a_linalg_capsule_is_priced_from_its_result_type():
    """The other 54: a linalg-on-tensors capsule has no merlin_iface commands at all."""
    p = _iface("SY_micro_model")
    if p is None:
        pytest.skip("capsule absent")
    assert CC.capsule_output_elements(p.read_text(encoding="utf-8")) == 1024


def test_every_l3_demanding_capsule_in_the_corpus_can_be_priced():
    """The property that matters: an unpriced capsule has not been shown to be affordable."""
    gate = _gate()
    rep = gate.audit(budget_s=900.0)
    assert rep["n_demanding_l3"] > 0, "nothing demanded L3, so this test established nothing"
    assert rep["unpriceable"] == [], (
        f"{len(rep['unpriceable'])} capsule(s) demand L3 and cannot be priced: "
        f"{[u['capsule'] for u in rep['unpriceable'][:8]]}")


def test_an_over_budget_capsule_is_excused_only_by_an_l2_cap_or_an_extends():
    """The remedy is a declaration, not deletion: a large shape is often the representative one."""
    gate = _gate()
    rep = gate.audit(budget_s=900.0)
    for row in rep["over_budget"]:
        assert not row["extends"] and str(row["max_oracle_tier"] or "").upper() != "L2", (
            f"{row['capsule']} declares a remedy yet is reported over budget")
    # And the gate must actually be able to fail, or it is decoration.
    assert gate.main(["--budget-s", "900", "--fail-on-unaffordable"]) == 1, (
        "with 12 known over-budget capsules the gate must exit non-zero")


def test_a_generous_budget_admits_the_whole_corpus():
    """Discriminating: the gate must respond to the budget rather than always reporting the same set."""
    gate = _gate()
    tight = gate.audit(budget_s=900.0)
    loose = gate.audit(budget_s=10_000_000.0)
    assert len(loose["over_budget"]) < len(tight["over_budget"]), (
        "raising the budget must shrink the over-budget set")
    assert loose["over_budget"] == [], loose["over_budget"]


def test_the_declared_remedy_fields_are_the_ones_the_corpus_uses():
    """`max_oracle_tier` and `extends` must be real capsule fields, not invented by this gate."""
    used = {"max_oracle_tier": 0, "extends": 0}
    for cy in _CAPS.rglob("capsule.yaml"):
        try:
            doc = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        for k in used:
            if doc.get(k):
                used[k] += 1
    assert used["max_oracle_tier"] or used["extends"], (
        "neither remedy field appears anywhere in the corpus, so the gate would be asking for "
        f"something no capsule can express: {used}")


def test_the_cost_law_reproduces_the_calibration_runs():
    """The four measured points, and the model must land on them rather than near them.

    512/1024/2048/4096 written elements took 177.2 / 351.6 / 723.1 / 1682.6 seconds on a
    cycle-accurate oracle. Seconds per element is 0.346 / 0.343 / 0.353 / 0.411 -- flat over the first
    three rungs and 16% higher on the fourth -- so a single constant is the wrong shape, and the flat
    figure UNDER-predicts exactly where being wrong is most expensive.
    """
    measured = {512: 177.2, 1024: 351.6, 2048: 723.1, 4096: 1682.6}
    for out, secs in measured.items():
        got, extrapolated = CC.predict_seconds_from_output(out)
        assert got is not None
        err = abs(got - secs) / secs
        assert err < 0.06, f"output {out}: model {got:.1f}s vs measured {secs}s ({err:.1%})"
        assert extrapolated is False, "the calibration points are inside the measured range"


def test_the_law_is_superlinear_and_beats_a_flat_rate_at_the_top():
    """A flat s/element is not merely imprecise; it is optimistic where it matters."""
    flat = 4096 * CC.MEASURED_S_PER_OUTPUT_ELEMENT
    law, _ = CC.predict_seconds_from_output(4096)
    assert law > flat, "the law must not be cheaper than the flat rate at the top rung"
    assert abs(law - 1682.6) < abs(flat - 1682.6), (
        f"the law ({law:.0f}s) must be closer to the measured 1682.6s than the flat rate ({flat:.0f}s)")
    # And doubling the output must cost MORE than double.
    a, _ = CC.predict_seconds_from_output(2048)
    b, _ = CC.predict_seconds_from_output(4096)
    assert b > 2 * a


def test_a_size_past_the_calibration_is_flagged_as_extrapolation():
    """90% of the corpus's predicted bill sits out here; leaving it unsaid makes it a hidden guess."""
    _, inside = CC.predict_seconds_from_output(CC.MEASURED_MAX_OUTPUT_ELEMENTS)
    _, outside = CC.predict_seconds_from_output(CC.MEASURED_MAX_OUTPUT_ELEMENTS + 1)
    assert inside is False and outside is True


def test_the_gate_reports_how_much_of_the_bill_is_extrapolated():
    gate = _gate()
    rep = gate.audit(budget_s=900.0)
    assert rep["n_extrapolated"] > 0
    assert rep["extrapolated_hours"] > 0.5 * (rep["total_predicted_s"] / 3600.0), (
        "most of this corpus's cost is beyond the calibrated range, and the report must say so")


def test_no_capsule_is_priced_at_zero():
    """Zero reads as free, which is the most dangerous error available here."""
    gate = _gate()
    rep = gate.audit(budget_s=900.0)
    assert rep["unpriceable"] == []
    # Every priced capsule must have a positive prediction.
    assert rep["total_predicted_s"] > 0


def test_a_cycle_bound_perf_capsule_is_never_told_to_cap_itself_at_l2():
    """An L2 cap is not a remedy every capsule can take, and advising it would be advising a lie.

    A performance capsule whose `gate.instrument` is a cycle count NEEDS a cycle-accurate tier:
    capping it at L2 would not make it cheap, it would delete the measurement the capsule exists to
    take. Measured here: PL01/PL03 declare `cycle_count_and_preflight`, so the remedy available to
    them is a smaller shape or an accepted cost -- never the L2 cap the other over-budget capsules
    should take. The first version of this gate lumped them together and would have advised the
    impossible fix.
    """
    gate = _gate()
    rep = gate.audit(budget_s=900.0)
    cycle_bound = rep["over_budget_needs_cycle_accurate"]
    if not cycle_bound:
        pytest.skip("no cycle-bound capsule is currently over budget")
    for row in cycle_bound:
        assert row["needs_cycle_accurate"] is True
        assert "cycle" in (row["instrument"] or ""), row
    # And they must NOT appear in the cappable list, or the advice is contradictory.
    cappable = {r["capsule"] for r in rep["over_budget"]}
    assert not (cappable & {r["capsule"] for r in cycle_bound}), (
        "a capsule cannot be both cappable at L2 and dependent on a cycle-accurate tier")


def test_a_capsule_with_no_perf_instrument_is_cappable():
    """The other side of the split: a purely functional capsule can rest on a certified sibling."""
    gate = _gate()
    rep = gate.audit(budget_s=900.0)
    if not rep["over_budget"]:
        pytest.skip("nothing over budget")
    for row in rep["over_budget"]:
        assert row["needs_cycle_accurate"] is False, row
