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
