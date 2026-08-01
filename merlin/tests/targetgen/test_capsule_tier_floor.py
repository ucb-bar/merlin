"""Capping ``required_oracle_tiers`` to a phase ceiling must never strip the numeric floor to nothing.

A float capsule requires an RTL cert tier (atlas: L3 cosim) with the integer L0/L1 floor marked
not_applicable. The per-round loop caps to its fast ceiling (L2). A naive intersection leaves only
``[L0, L1]`` — both N/A — so the grade enforces ZERO numeric tiers and any capsule that merely builds
reads back as pass (the crash-pass regression that appeared when the fast L2 tier lowered the loop
ceiling from L3 to L2). The materializer must instead make the reachable ceiling tier (L2) the mandatory
loop tier, while the cycle-accurate cert (L3) stays required at the checkpoint (ceiling L3).
"""
from __future__ import annotations

from merlin.targetgen.contract.materialize import _cap_required, _cap_tiers


def _cap(tiers, ceiling):
    return _cap_required(tiers, set(_cap_tiers(ceiling)), ceiling)


def test_float_cert_capped_below_reach_substitutes_reachable_tier():
    # atlas float capsule: integer floor N/A, L3 cosim is the cert; the fast loop reaches only L2.
    assert _cap(["L0", "L1", "L3"], "L2") == ["L0", "L1", "L2"]  # L2 is now the mandatory loop tier


def test_checkpoint_ceiling_keeps_the_cert_mandatory():
    assert _cap(["L0", "L1", "L3"], "L3") == ["L0", "L1", "L3"]  # L3 stays required at the checkpoint


def test_gemmini_style_integer_capsule_untouched():
    # its loop tier L2 is already required -> no substitution, no change (no gemmini regression)
    assert _cap(["L0", "L1", "L2"], "L2") == ["L0", "L1", "L2"]
    assert _cap(["L0", "L1", "L2", "L3"], "L2") == ["L0", "L1", "L2"]


def test_no_reachable_rtl_tier_is_not_a_fabricated_pass():
    # ceiling below every RTL tier -> cannot substitute a numeric tier (honest not-gradeable; never invent).
    assert _cap(["L0", "L1", "L3"], "L1") == ["L0", "L1"]


def test_atlas_loop_set_enforces_a_numeric_tier_on_every_capsule():
    """End-to-end: atlas's DERIVED loop capsule set must require a numeric (RTL) tier on every capsule, so
    the fast loop can never again crash-pass a float capsule whose npu oracle threw."""
    import yaml
    from merlin.common.paths import repo_root
    from merlin.targetgen.contract.materialize import public_capsules_for
    from merlin.targetgen.target_experiment import load_target_experiment
    te = load_target_experiment(
        repo_root() / "merlin/experiments/capsule_bench/targets/atlas/target_experiment.yaml")
    dest = public_capsules_for(te)  # default ceiling = the fast loop tier
    caps = list(dest.rglob("capsule.yaml"))
    assert caps, "no atlas capsules materialized"
    for cap_yaml in caps:
        tiers = yaml.safe_load(cap_yaml.read_text()).get("required_oracle_tiers", [])
        assert any(t in ("L2", "L3", "L4", "L5") for t in tiers), (
            f"{cap_yaml.parent.name} loop grade enforces no numeric tier: {tiers} (crash-pass risk)")
