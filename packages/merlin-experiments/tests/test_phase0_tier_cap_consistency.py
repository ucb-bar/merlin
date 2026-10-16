"""A Phase-0-written capsule never demands an oracle tier above its own cap.

`required_oracle_tiers: [..., L3]` beside `max_oracle_tier: L2` grades as incomplete forever when the
member is run on its own: the cap forbids the tier the requirement insists on. The writer trims the
requirement to the cap whether the cap comes from the profile entry or is already on the capsule.
"""

from merlin_experiments.phase0.writer import _cap_oracle_tiers

TIERS = ["L0", "L1", "L2", "L3"]


def test_an_entry_cap_trims_the_required_tiers():
    cap = {"name": "m", "required_oracle_tiers": list(TIERS)}
    assert _cap_oracle_tiers({"max_oracle_tier": "L2", "extends": "sib"}, cap)
    assert cap["required_oracle_tiers"] == ["L0", "L1", "L2"] and cap["max_oracle_tier"] == "L2"
    assert cap["extends"] == "sib"


def test_a_cap_already_on_the_capsule_is_honoured_too():
    cap = {"name": "m", "required_oracle_tiers": list(TIERS), "max_oracle_tier": "L2"}
    assert _cap_oracle_tiers({}, cap)
    assert cap["required_oracle_tiers"] == ["L0", "L1", "L2"]


def test_no_cap_leaves_the_requirement_alone():
    cap = {"name": "m", "required_oracle_tiers": list(TIERS)}
    assert not _cap_oracle_tiers({}, cap)
    assert cap["required_oracle_tiers"] == TIERS and "max_oracle_tier" not in cap
