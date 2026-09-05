"""A phase exclusion is a DERIVED verdict, and the seal has to be checkable against the derivation.

WHY THIS EXISTS. A capsule-bench run serves one phase. ``phase_policy.phase_of`` decides which phase a
member can serve from what can be checked about it — whether its answer can be certified on this target
inside the declared certification budget, and whether its work can be priced — so "this row is not in
this run's denominator" is a fact the corpus and the target can be asked about, not an operator's
opinion. The descriptor still WRITES the list, because a graded denominator that moves silently is the
thing the seal exists to stop; these tests are what keep the written list equal to the derived one.

Three properties, each of which has a way of quietly going wrong:

* the classes stay APART. Capability (the datapath cannot hold the operand), resource (a declared cost
  policy) and phase (a derived verdict) answer different questions, and folding a derived verdict into a
  human decision loses the only thing that says which rows would return if the budget moved;
* a phase verdict without its BUDGET is not a fact. The same member is priced out at one budget and
  admitted at another, so a name list with no budget beside it cannot be re-derived by a reader;
* UNDETERMINED IS NEVER AN EXCLUSION. A target with no certification history cannot have its phase-1
  membership decided at all. Folding that into "excluded" would record a missing measurement as a
  property of the corpus — the exact failure the repo keeps re-finding, where an unanswerable question
  reads as a "no".
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen import cert_cost as CC
from merlin.targetgen import phase_policy as PP
from merlin.targetgen.target_experiment import load_target_experiment

DESCRIPTORS = sorted((merlin_dir() / "experiments/capsule_bench/targets").glob(
    "*/target_experiment.yaml"))


def _write(tmp_path: Path, grading: dict, *, target: str = "probe_target") -> Path:
    p = tmp_path / "target_experiment.yaml"
    p.write_text(yaml.safe_dump({"target": target, "grading": grading}), encoding="utf-8")
    return p


def _phase_sealed():
    """Descriptors that declare a phase partition, with their loaded experiment."""
    out = []
    for path in DESCRIPTORS:
        te = load_target_experiment(path)
        if getattr(te, "graded_phase", None) is not None:
            out.append((path, te))
    return out


# ------------------------------------------------------------------ the descriptor grammar

def test_phase_exclusions_are_parsed_apart_from_capability_and_resource(tmp_path):
    te = load_target_experiment(_write(tmp_path, {
        "capability_exclude_capsules": ["A"],
        "resource_bound": {"policy": "p", "exclude_capsules": ["B"],
                           "required_admitted_models": ["M"]},
        "phase_bound": {"phase": 1, "budget_s": 300, "policy": "q",
                        "phase2_only_capsules": ["C", "D"], "exclude_capsules": ["C"]},
    }))
    assert te.graded_capability_exclude == ("A",)
    assert te.graded_resource_exclude == ("B",)
    assert te.graded_phase2_only == ("C", "D")
    assert te.graded_phase_exclude == ("C",)
    assert te.graded_phase == 1 and te.graded_phase_budget_s == 300.0
    # and the union is what the materializer removes, with every row counted exactly once
    assert sorted(te.graded_exclude) == ["A", "B", "C"]


def test_the_frozen_arithmetic_counts_the_phase_class(tmp_path):
    """source == admitted + capability + resource + phase, or the descriptor refuses to load."""
    grading = {
        "capability_exclude_capsules": ["A"],
        "resource_bound": {"policy": "p", "exclude_capsules": ["B"],
                           "required_admitted_models": ["M"]},
        "phase_bound": {"phase": 1, "budget_s": 300, "policy": "q",
                        "phase2_only_capsules": ["C", "D"], "exclude_capsules": ["C"]},
        "expected_cohort": {"source_capsules": 13, "admitted_capsules": 10},
    }
    assert load_target_experiment(_write(tmp_path, grading)).graded_expected_admitted_capsules == 10
    # the pre-phase arithmetic (source == admitted + capability + resource) must now FAIL
    grading["expected_cohort"] = {"source_capsules": 12, "admitted_capsules": 10}
    with pytest.raises(ValueError, match="arithmetic"):
        load_target_experiment(_write(tmp_path, grading))


def test_a_phase_verdict_without_its_budget_is_refused(tmp_path):
    with pytest.raises(ValueError, match="budget_s"):
        load_target_experiment(_write(tmp_path, {
            "phase_bound": {"phase": 1, "policy": "q", "phase2_only_capsules": ["C"]}}))


def test_a_phase_class_without_a_named_policy_is_refused(tmp_path):
    with pytest.raises(ValueError, match="policy"):
        load_target_experiment(_write(tmp_path, {
            "phase_bound": {"phase": 1, "budget_s": 300, "phase2_only_capsules": ["C"]}}))


def test_a_row_may_not_be_held_out_for_a_verdict_it_did_not_fail(tmp_path):
    """`exclude_capsules` is the narrow instrument, and it must rest on the recorded verdict.

    Without this, "phase exclusion" becomes a free-form hold-out list wearing a derived verdict's name —
    which is the disguise the three separate classes exist to prevent.
    """
    with pytest.raises(ValueError, match="did not fail"):
        load_target_experiment(_write(tmp_path, {
            "phase_bound": {"phase": 1, "budget_s": 300, "policy": "q",
                            "phase2_only_capsules": ["C"], "exclude_capsules": ["NOT_C"]}}))


@pytest.mark.parametrize("other,field", [
    ("capability_exclude_capsules", None),
    ("resource_bound", "exclude_capsules"),
])
def test_a_row_may_not_be_excluded_for_two_reasons_at_once(tmp_path, other, field):
    """An overlap is not a harmless duplicate: it lets a row be reported under whichever heading reads
    best, and it double-counts in the frozen arithmetic."""
    grading = {"phase_bound": {"phase": 1, "budget_s": 300, "policy": "q",
                               "phase2_only_capsules": ["DUP"], "exclude_capsules": ["DUP"]}}
    if field:
        grading[other] = {"policy": "p", field: ["DUP"], "required_admitted_models": ["M"]}
    else:
        grading[other] = ["DUP"]
    with pytest.raises(ValueError, match="overlap"):
        load_target_experiment(_write(tmp_path, grading))


# ------------------------------------------------------------------ the derivation itself

def test_undetermined_is_never_an_exclusion():
    """A target with no certification history cannot have its phase-1 membership decided.

    This is the property that makes the seal safe to re-derive automatically: the derivation must be
    able to say "I cannot tell", and "I cannot tell" must not turn into a removed row.
    """
    capsule = {"name": "X", "kind": "isa",
               "inputs": [{"name": "A", "shape": [4, 4], "dtype": "i8"},
                          {"name": "B", "shape": [4, 4], "dtype": "i8"}],
               "operation": {"op": "matmul", "attributes": {"lhs": "A", "rhs": "B", "out": "Y"}},
               "required_oracle_tiers": ["L0", "L1", "L2", "L3"]}
    # no fit and no cycle-accurate answer -> the verdict is UNDETERMINED, not NEITHER
    v = PP.phase_of(capsule, target="probe_target", fit=None, budget_s=300.0,
                    cycle_accurate_available=None)
    assert v.phase == PP.UNDETERMINED
    assert v.phase != PP.NEITHER
    assert v.reason.strip(), "an undetermined verdict with no reason cannot be acted on"


@pytest.mark.parametrize("path", DESCRIPTORS, ids=lambda p: p.parent.name)
def test_every_targets_partition_is_derivable_and_every_verdict_carries_a_reason(path):
    """The derivation must produce an answer for every admitted row on every target — or say it cannot.

    This runs whether or not a descriptor has SEALED its partition, which is deliberate: the sweep below
    only visits descriptors that declare one, and a parametrized test over an empty list is green and
    proves nothing. This one cannot be empty, and it pins the property that makes the seal safe to
    re-derive later — every verdict is decided or explicitly UNDETERMINED, and either way it says why.
    """
    from merlin.targetgen import capsule_runner as CR

    te = load_target_experiment(path)
    contract = str(repo_root() / "merlin/contract")
    source = CR.discover_capsules(te.graded_roots(), labels={"public", "dev"}, contract=contract)
    assert source, f"{path}: resolves no public capsules; the derivation below would be vacuous"
    fit = CC.fit_for(te.target)
    cycle_accurate = PP.cycle_accurate_seen(te.target)
    # An undetermined verdict must NAME the evidence it is missing. Without that it is indistinguishable
    # from a row the predicate quietly declined on, and a nameless gap is what lets an absence be read
    # later as a verdict against the capsule.
    missing_evidence = ("no measured certification history", "was not established",
                        "achievable ceiling is not positive")
    for cap in source:
        v = PP.phase_of(cap, target=te.target, fit=fit, budget_s=300.0,
                        cycle_accurate_available=cycle_accurate)
        assert v.phase in (PP.BOTH, PP.PHASE1, PP.PHASE2, PP.NEITHER, PP.UNDETERMINED)
        assert v.reason.strip(), f"{path}: {cap.get('name')} is {v.phase} with no recorded reason"
        if v.phase == PP.UNDETERMINED:
            assert any(m in v.reason for m in missing_evidence), (
                f"{path}: {cap.get('name')} is undetermined but its reason names no missing evidence: "
                f"{v.reason!r}")


@pytest.mark.parametrize("path,te", _phase_sealed(), ids=lambda x: getattr(x, "name", ""))
def test_the_recorded_phase_partition_is_exactly_what_the_verdict_derives(path, te):
    """The written list must equal the derived one — no more, and no fewer.

    Deliberately not a cardinality: a phase partition CAN publish its names (unlike the hidden pool), so
    the check that a swap cannot hide behind an unchanged count is available here and is taken.
    """
    from merlin.targetgen import capsule_runner as CR

    contract = str(repo_root() / "merlin/contract")
    source = CR.discover_capsules(te.graded_roots(), labels={"public", "dev"}, contract=contract)
    op_source = [c for c in source if c.get("kind") != "model"]
    _eligible, ineligible = CR._split_ineligible(op_source, te.target)
    capability = {str(r.get("capsule")) for r in ineligible}
    resource = set(te.graded_resource_exclude)

    fit = CC.fit_for(te.target)
    cycle_accurate = PP.cycle_accurate_seen(te.target)
    serves = {PP.BOTH, PP.PHASE1} if te.graded_phase == 1 else {PP.BOTH, PP.PHASE2}
    derived, undetermined = set(), []
    for cap in source:
        name = str(cap.get("name"))
        if name in capability or name in resource:
            continue                      # already out, for a different and stated reason
        v = PP.phase_of(cap, target=te.target, fit=fit, budget_s=te.graded_phase_budget_s,
                        cycle_accurate_available=cycle_accurate)
        if v.phase == PP.UNDETERMINED:
            undetermined.append(name)
        elif v.phase not in serves:
            derived.add(name)

    assert not undetermined, (
        f"{path}: {len(undetermined)} row(s) have no phase verdict on this target "
        f"({undetermined[:5]}). Certify this target's corpus once; do not read the gap as an exclusion.")
    assert sorted(te.graded_phase2_only) == sorted(derived), (
        f"{path}: the recorded phase-2-only set is not what the verdict derives at "
        f"{te.graded_phase_budget_s}s.\n"
        f"  recorded but not derived: {sorted(set(te.graded_phase2_only) - derived)}\n"
        f"  derived but not recorded: {sorted(derived - set(te.graded_phase2_only))}")
