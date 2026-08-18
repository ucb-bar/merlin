"""Capping ``required_oracle_tiers`` to a phase ceiling must never strip the numeric floor to nothing —
and must never paper over that by substituting a tier the capsule did not declare.

A float capsule requires an RTL cert tier with the integer L0/L1 floor marked not_applicable. If the
per-round loop caps to a ceiling below that cert tier, a naive intersection leaves only ``[L0, L1]`` —
both N/A — so the grade would enforce ZERO numeric tiers and any capsule that merely builds reads back
as pass. That crash-pass risk is real.

The fix is NOT to append the ceiling as a substitute required tier. That was tried, and it graded a
corpus against a cheaper additive tier it had never declared; when that tier's runner hung, every
capsule failed on a gate the capsule never asked for while its declared cert tier ran fine and returned
real verdicts. The fix is upstream: the loop tier is the fastest endpoint tier THE CORPUS DECLARES, so
capping never has to remove the numeric floor in the first place — and where no declared tier is
reachable, the harness fails closed and says so.
"""
from __future__ import annotations

from merlin.targetgen.contract.materialize import _cap_required, _cap_tiers


def _cap(tiers, ceiling):
    return _cap_required(tiers, set(_cap_tiers(ceiling)))


def test_float_cert_capped_below_reach_is_never_substituted():
    # A float capsule declaring the cert tier L3, capped at L2: the cert is DROPPED and REPORTED,
    # never swapped for L2. Grading a substituted tier and reporting it as the declared one is the bug.
    kept, unreachable = _cap(["L0", "L1", "L3"], "L2")
    assert kept == ["L0", "L1"]
    assert unreachable == ["L3"]


def test_checkpoint_ceiling_keeps_the_cert_mandatory():
    kept, unreachable = _cap(["L0", "L1", "L3"], "L3")
    assert kept == ["L0", "L1", "L3"]           # the cert stays required when it IS reachable
    assert unreachable == []


def test_integer_capsule_whose_loop_tier_is_declared_is_untouched():
    # its loop tier L2 is already in the required set -> capping is a plain intersection, no change.
    assert _cap(["L0", "L1", "L2"], "L2") == (["L0", "L1", "L2"], [])
    assert _cap(["L0", "L1", "L2", "L3"], "L2") == (["L0", "L1", "L2"], ["L3"])


def test_no_reachable_rtl_tier_is_not_a_fabricated_pass():
    # ceiling below every RTL tier -> no numeric tier can be invented (honest not-gradeable, never green).
    kept, unreachable = _cap(["L0", "L1", "L3"], "L1")
    assert kept == ["L0", "L1"]
    assert unreachable == ["L3"]


def test_derived_loop_set_enforces_a_declared_numeric_tier_on_every_capsule():
    """End-to-end: every target's DERIVED loop capsule set must still require a numeric (RTL) tier on
    every capsule — so the fast loop can never crash-pass a float capsule whose oracle threw — AND that
    tier must be one the capsule declared."""
    import pytest
    import yaml
    from merlin.common.paths import repo_root
    from merlin.targetgen.contract.materialize import declared_oracle_tiers, public_capsules_for
    from merlin.targetgen.target_experiment import load_target_experiment
    targets = repo_root() / "merlin/experiments/capsule_bench/targets"
    checked = 0
    for d in sorted(targets.iterdir()):
        if not (d / "target_experiment.yaml").is_file():
            continue
        te = load_target_experiment(d / "target_experiment.yaml")
        try:
            dest = public_capsules_for(te)
        except Exception:                        # noqa: BLE001 — endpoint/corpus not resolvable here
            continue
        declared = declared_oracle_tiers(*te.graded_roots())
        caps = list(dest.rglob("capsule.yaml"))
        for cap_yaml in caps:
            tiers = yaml.safe_load(cap_yaml.read_text()).get("required_oracle_tiers", [])
            assert any(t in ("L2", "L3", "L4", "L5") for t in tiers), (
                f"{te.target}/{cap_yaml.parent.name} loop grade enforces no numeric tier: {tiers} "
                f"(crash-pass risk)")
            assert set(tiers) <= declared, (
                f"{te.target}/{cap_yaml.parent.name} requires {sorted(set(tiers) - declared)}, which its "
                f"corpus never declared — that is the substitution this module exists to prevent")
            checked += 1
    if not checked:
        pytest.skip("no target's corpus materialized in this environment")
