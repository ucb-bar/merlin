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
