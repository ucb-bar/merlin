"""A tier that cannot see a class of program never certifies a capsule in that class."""

from __future__ import annotations

import pytest

from merlin.targetgen import oracle_blind_spots as blind
from merlin.targetgen import tier_policy

_ENTRY = {
    "id": "state_the_model_pins",
    "tiers": ["L2"],
    "axes": [["instruction_class", "LOOPED"], ["mode", "carried"]],
    "what": "the functional model pins a base the RTL rotates",
    "evidence": "one executable: functional tier 0 wrong, FPGA all wrong",
}


def _capsule(*, classes=(), modes=(), tiers=("L0", "L1", "L2", "L3"), ceiling=None) -> dict:
    capsule = {
        "name": "c",
        "required_oracle_tiers": list(tiers),
        "expected": {"instruction_classes": list(classes), "modes": {m: True for m in modes}},
    }
    if ceiling:
        capsule[tier_policy.CEILING_FIELD] = ceiling
    return capsule


def test_a_capsule_outside_every_class_needs_only_its_shallowest_tier() -> None:
    spots = blind.parse([_ENTRY])
    capsule = _capsule(classes=["PLAIN"])
    assert blind.blind_tiers(spots, capsule) == {}
    assert blind.required_tier(spots, capsule, capsule["required_oracle_tiers"]) == "L0"


def test_a_capsule_in_a_blind_class_needs_the_first_tier_past_the_blind_one() -> None:
    spots = blind.parse([_ENTRY])
    capsule = _capsule(modes=["carried"])
    assert blind.blind_tiers(spots, capsule) == {"L2": [{"id": "state_the_model_pins", "via": [["mode", "carried"]]}]}
    # Shallower tiers are refused too: a ladder is ordered by what a tier observes.
    assert blind.required_tier(spots, capsule, capsule["required_oracle_tiers"]) == "L3"


def test_a_ceiling_inside_the_blind_spot_is_a_finding_and_lifting_it_clears_it() -> None:
    spots = blind.parse([_ENTRY])
    capped = _capsule(classes=["LOOPED"], ceiling="L2")
    report = blind.audit("synthetic", [capped, _capsule(classes=["PLAIN"])], spots=spots)
    assert (report.capsules, report.concerned) == (2, 1)
    (finding,) = report.findings
    assert (finding["finding"], finding["reaches"], finding["needs"]) == (blind.UNCERTIFIABLE, "L2", "L3")
    assert blind.audit("synthetic", [_capsule(classes=["LOOPED"])], spots=spots).findings == []


def test_a_ladder_with_no_seeing_tier_is_its_own_finding_never_a_pass() -> None:
    spots = blind.parse([_ENTRY])
    short = _capsule(classes=["LOOPED"], tiers=("L0", "L1", "L2"))
    (finding,) = blind.audit("synthetic", [short], spots=spots).findings
    assert (finding["finding"], finding["needs"]) == (blind.NO_SEEING_TIER, None)


def test_deleting_the_entry_is_what_stops_the_requirement() -> None:
    # The mutation: with the entry gone the same capped capsule is clean, so the entry is
    # load-bearing and nothing else in the audit is doing its work.
    capped = _capsule(classes=["LOOPED"], ceiling="L2")
    assert blind.audit("synthetic", [capped], spots=blind.parse([])).findings == []
    assert blind.audit("synthetic", [capped], spots=blind.parse([_ENTRY])).findings


def test_an_entry_nobody_exercises_is_reported() -> None:
    report = blind.audit("synthetic", [_capsule(classes=["PLAIN"])], spots=blind.parse([_ENTRY]))
    assert report.unexercised == ["state_the_model_pins"]


@pytest.mark.parametrize(
    "broken",
    [
        {**_ENTRY, "tiers": []},
        {**_ENTRY, "axes": []},
        {**_ENTRY, "axes": [["mode"]]},
        {**_ENTRY, "evidence": " "},
        {k: v for k, v in _ENTRY.items() if k != "id"},
    ],
)
def test_a_malformed_entry_fails_closed(broken) -> None:
    with pytest.raises(blind.BlindSpotError):
        blind.parse([broken])


def test_a_declared_ceiling_at_a_blind_tier_says_so_in_the_tier_record(monkeypatch) -> None:
    monkeypatch.setattr(blind, "_declared", lambda target: blind.parse([_ENTRY]))
    capped = _capsule(classes=["LOOPED"], ceiling="L2")
    verdict = tier_policy.oracle_ceiling("synthetic", capped, "L3", declared_tiers=capped["required_oracle_tiers"])
    assert not verdict.allowed
    assert [row["id"] for row in verdict.record["blind_spots_at_ceiling"]] == ["state_the_model_pins"]
    assert "does not screen it either" in verdict.reason
    plain = _capsule(classes=["PLAIN"], ceiling="L2")
    seen = tier_policy.oracle_ceiling("synthetic", plain, "L3", declared_tiers=plain["required_oracle_tiers"])
    assert seen.record["blind_spots_at_ceiling"] == []


def test_every_declared_registry_in_the_tree_parses() -> None:
    from merlin.targetgen import target_registry

    for target in target_registry.all_targets():
        blind.parse((target_registry.load_contract(target) or {}).get(blind.CONTRACT_KEY))


def test_a_program_is_judged_by_the_instructions_it_used_not_only_by_what_was_asked() -> None:
    # The capsule declares nothing in the blind class; the backend reached for the looped command.
    spots = blind.parse([_ENTRY])
    capsule = _capsule(classes=["PLAIN"])
    trace = {"instructions": [{"index": 0, "class": "PLAIN"}, {"index": 1, "class": "LOOPED"}]}
    screened = blind.judge_result(spots, capsule, {"L1": {"status": "pass"}, "L2": {"status": "pass"}}, trace)
    assert screened["seen"] is False and screened["finding"] == "passed_only_where_the_oracle_cannot_see"
    assert screened["blind"]["L2"][0]["via"] == [["instruction_class", "LOOPED"]]
    # The same program with a pass past the blind tier is seen; so is one that never used the command.
    deeper = {"L2": {"status": "pass"}, "L3": {"status": "pass"}}
    assert blind.judge_result(spots, capsule, deeper, trace)["seen"] is True
    plain = {"instructions": [{"index": 0, "class": "PLAIN"}]}
    assert blind.judge_result(spots, capsule, {"L2": {"status": "pass"}}, plain)["seen"] is True
