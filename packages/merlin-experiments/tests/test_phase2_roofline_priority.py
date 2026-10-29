"""Furthest below the roofline first; stop only when every derived floor reaches the declared target."""

from __future__ import annotations

import pytest
from merlin_experiments.phase2 import roofline_priority as RP


def _row(cycles, floor, status="derived"):
    return {"cycles": cycles, "staged_roofline": {"status": status, "floor_cycles": floor}}


def test_the_member_furthest_below_its_roofline_is_worked_first():
    result = RP.roofline_priorities(
        {
            ("PX", "near"): [_row(110, 100), _row(105, 100)],
            ("PX", "far"): [_row(400, 100), _row(410, 100)],
            ("PX", "done"): [_row(101, 100), _row(101, 100)],
        },
        attainment_min=0.95,
    )
    assert [row["member"] for row in result["ranked"]] == [("PX", "far"), ("PX", "near"), ("PX", "done")]
    assert result["next"] == [("PX", "far"), ("PX", "near")]
    assert result["status"] == RP.CONTINUE
    assert result["ranked"][0]["attainment"]["worst"] == pytest.approx(100 / 410)


def test_the_stop_is_the_declared_target_on_every_members_worst_repeat():
    members = {("PX", "a"): [_row(110, 100), _row(130, 100)], ("PX", "b"): [_row(120, 100), _row(120, 100)]}
    assert RP.roofline_priorities(members, attainment_min=0.75)["status"] == RP.CONVERGED
    assert RP.roofline_priorities(members, attainment_min=0.8)["status"] == RP.CONTINUE


def test_an_unknown_or_refuted_floor_never_reads_as_converged():
    unknown = {("PX", "a"): [_row(110, 100)], ("PX", "b"): [{"cycles": 50}]}
    result = RP.roofline_priorities(unknown, attainment_min=0.5)
    assert result["status"] == RP.UNDECIDED and result["unknown_floor"][0]["member"] == ("PX", "b")
    refuted = RP.roofline_priorities({("PX", "a"): [_row(90, 100)]}, attainment_min=0.5)
    assert refuted["status"] == RP.REFUTED and refuted["floor_refuted"]


def test_the_feedback_cell_roofline_is_read_too_and_the_target_is_declared_once():
    row = {"cycles": 200, "roofline": {"status": "derived", "roofline_cycles": 100}}
    assert RP.floor_of(row) == 100.0
    descriptors = [
        {"performance": {"capacity": {"roofline_target": {"attainment_min": 0.8}}}},
        {"performance": {"capacity": {"roofline_target": {"attainment_min": 0.8}}}},
    ]
    assert RP.declared_target(descriptors) == 0.8
    descriptors[1]["performance"]["capacity"]["roofline_target"]["attainment_min"] = 0.9
    with pytest.raises(ValueError, match="exactly one"):
        RP.declared_target(descriptors)


def test_partial_floors_can_reach_the_numeric_target_without_establishing_completion():
    row = _row(125, 100, "partial")
    row["staged_roofline"]["unresolved"] = {"accelerator_host_stage": "packing and issue are unpriced"}
    result = RP.roofline_priorities({"one": [row], "two": [_row(120, 100, "partial")]}, attainment_min=0.8)
    assert result["schema"] == RP.SCHEMA
    assert result["status"] == RP.UNDECIDED and result["completion_blocked"]
    assert result["next"] == []
    assert all(item["within_target"] for item in result["ranked"])
    assert all(not item["convergence_eligible"] for item in result["ranked"])
    assert all(item["floor_status"] == "partial" for item in result["ranked"])
    assert [item["member"] for item in result["partial_floor"]] == ["one", "two"]
    assert result["partial_floor"][0]["evidence"][0]["unresolved"] == row["staged_roofline"]["unresolved"]
    assert RP.floor_of(row) == 100  # Keep the frozen floor useful for exploration.


@pytest.mark.parametrize("partial_cycles,expected", [(120, RP.UNDECIDED), (200, RP.CONTINUE)])
def test_known_and_partial_floors_keep_exploration_and_completion_separate(partial_cycles, expected):
    result = RP.roofline_priorities(
        {"known": [_row(120, 100)], "partial": [_row(partial_cycles, 100, "partial")]}, attainment_min=0.8
    )
    assert result["status"] == expected and result["completion_blocked"]
    assert result["partial_floor"][0]["member"] == "partial"
    known = next(row for row in result["ranked"] if row["member"] == "known")
    assert known["convergence_eligible"]
    assert result["next"] == (["partial"] if partial_cycles == 200 else [])
    if partial_cycles == 200:
        assert result["ranked"][0]["member"] == "partial"
        assert result["ranked"][0]["gap_cycles"] == 100


def test_any_partial_repeat_blocks_completion_even_when_numeric_floors_agree():
    result = RP.roofline_priorities({"same": [_row(110, 100), _row(120, 100, "partial")]}, attainment_min=0.8)
    assert result["status"] == RP.UNDECIDED
    assert result["ranked"][0]["attainment"]["worst"] == pytest.approx(100 / 120)
    assert [item["status"] for item in result["partial_floor"][0]["evidence"]] == ["derived", "partial"]


def test_unknown_and_partial_reasons_are_retained_per_repeat_without_aliasing_inputs():
    missing = _row(100, None, "unknown")
    missing["staged_roofline"].update(
        unresolved={"movement": "selected DMA width is unavailable"},
        reason="no comparable floor",
        reasons=["read rate missing", "write rate missing"],
    )
    partial = _row(125, 100, "partial")
    partial["staged_roofline"].update(unresolved={"host": "issue unpriced"}, reason="serial host term omitted")
    result = RP.roofline_priorities({"missing": [missing], "partial": [partial]}, attainment_min=0.8)
    assert result["status"] == RP.UNDECIDED
    unknown = result["unknown_floor"][0]["evidence"][0]
    assert unknown["status"] == "unknown" and unknown["source"] == "staged_roofline"
    assert unknown["unresolved"] == missing["staged_roofline"]["unresolved"]
    assert unknown["reason"] == "no comparable floor"
    assert unknown["reasons"] == ["read rate missing", "write rate missing"]
    assert result["partial_floor"][0]["evidence"][0]["reason"] == "serial host term omitted"
    unknown["unresolved"]["movement"] = "edited returned evidence"
    assert missing["staged_roofline"]["unresolved"]["movement"] == "selected DMA width is unavailable"


@pytest.mark.parametrize("source", ["staged", "nested", "feedback"])
def test_unresolved_terms_cannot_be_hidden_by_a_derived_status(source):
    document = {"status": "derived", "unresolved": {"host": "unpriced accelerator-feeding work"}}
    if source == "feedback":
        row = {"cycles": 125, "roofline": {**document, "roofline_cycles": 100}}
    elif source == "nested":
        row = {"cycles": 125, "performance": {"capacity": {"roofline": {**document, "floor_cycles": 100}}}}
    else:
        row = {"cycles": 125, "staged_roofline": {**document, "floor_cycles": 100}}
    result = RP.roofline_priorities({"incomplete": [row]}, attainment_min=0.8)
    assert result["status"] == RP.UNDECIDED and result["completion_blocked"]
    assert result["ranked"][0]["floor_cycles"] == 100
    assert result["partial_floor"][0]["evidence"][0]["status"] == "derived"
    assert result["partial_floor"][0]["evidence"][0]["unresolved"] == document["unresolved"]


def test_a_measurement_below_a_partial_floor_still_refutes_it():
    row = _row(90, 100, "partial")
    row["staged_roofline"]["unresolved"] = {"host": "unpriced"}
    result = RP.roofline_priorities({"refuted": [row]}, attainment_min=0.8)
    assert result["status"] == RP.REFUTED and result["completion_blocked"]
    assert result["ranked"] == []
    assert result["floor_refuted"][0]["floor_cycles"] == 100
    assert result["floor_refuted"][0]["evidence"][0]["status"] == "partial"
    assert result["floor_refuted"][0]["evidence"][0]["unresolved"] == {"host": "unpriced"}


def test_complete_derived_floors_remain_eligible_at_the_frozen_target():
    result = RP.roofline_priorities({"complete": [_row(125, 100)]}, attainment_min=0.8)
    assert result["status"] == RP.CONVERGED and not result["completion_blocked"]
    assert result["partial_floor"] == result["unknown_floor"] == []
    assert result["ranked"][0]["convergence_eligible"]


def test_omitted_cost_terms_keep_a_derived_bound_partial():
    row = {"cycles": 125, "roofline": {
        "status": "derived", "roofline_cycles": 100,
        "omitted": ["host issue", "pipeline fill"], "unresolved": {},
    }}
    result = RP.roofline_priorities({"one": [row]}, attainment_min=0.8)
    assert result["status"] == RP.UNDECIDED and result["completion_blocked"]
    assert result["ranked"][0]["within_target"]
    assert not result["ranked"][0]["convergence_eligible"]
    assert result["partial_floor"][0]["evidence"][0]["omitted"] == row["roofline"]["omitted"]
    assert RP.floor_of(row) == 100


def test_an_explicit_empty_omission_roster_does_not_block_a_complete_floor():
    row = _row(125, 100)
    row["staged_roofline"]["omitted"] = []
    assert RP.roofline_priorities({"one": [row]}, attainment_min=0.8)["status"] == RP.CONVERGED
