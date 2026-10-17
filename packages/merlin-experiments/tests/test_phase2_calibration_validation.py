"""The calibration-validation list: cohort points by regime, each a schedule pair to order once on RTL."""

from __future__ import annotations

import copy

import pytest
from merlin_experiments.phase0 import capacity_families as CF
from merlin_experiments.phase2 import calibration_validation as CV
from test_phase0_capacity_families import _FACTS, _px


def _plan():
    sweep = _px()
    cohort = CF.capacity_family_entries(sweep, target="synthetic", facts=_FACTS)
    basis = CF.derive_basis("synthetic", _FACTS)
    return CV.select(cohort, basis, sweep["requires_capacity_scope"]["calibration_validation"]), cohort


def test_every_declared_category_has_points_drawn_from_the_cohort_or_the_ports():
    plan, cohort = _plan()
    policy = _px()["requires_capacity_scope"]["calibration_validation"]
    assert not plan["unavailable"]
    assert {p["category"] for p in plan["points"]} == {c["name"] for c in policy["categories"]}
    names = {e["name"] for e in cohort}
    for point in plan["points"]:
        assert len(point["schedules"]) == 2
        if point["category"] != "dma_packet_width":
            assert point["member"] in names and point["floor_cycles"] > 0
    widths = sorted(p["shape"]["N"] for p in plan["points"] if p["category"] == "dma_packet_width")
    assert widths == [8, 32, 33], "one beat, the largest request, and one element past it"


def test_the_cheapest_member_of_a_regime_is_chosen_unless_the_policy_says_costliest():
    plan, cohort = _plan()
    overlap = [p for p in plan["points"] if p["category"] == "dma_compute_overlap"]
    single = [
        e
        for e in cohort
        if e["performance"]["capacity"]["family"] == "knee" and e["performance"]["capacity"]["regime"] == CF.FITS_SINGLE
    ]
    costliest = max(single, key=lambda e: e["performance"]["capacity"]["roofline"]["floor_cycles"])
    assert any(p["member"] == costliest["name"] for p in overlap)


def test_board_reconciliation_is_carried_and_qualification_never_skips_it():
    plan, _ = _plan()
    observations = {
        p["label"]: {"rtl": dict(zip(p["schedules"], (100, 200))), "spike": dict(zip(p["schedules"], (90, 210)))}
        for p in plan["points"]
    }
    verdict = CV.decide(plan, observations)["categories"]
    assert verdict["short_tile_waves"]["status"] == "qualified"
    assert verdict["dma_packet_width"]["status"] == "qualified_pending_board_reconciliation"
    flipped = copy.deepcopy(observations)
    for p in plan["points"]:
        if p["category"] == "epilogue_store_cost":
            flipped[p["label"]]["spike"] = dict(zip(p["schedules"], (210, 90)))
    assert CV.decide(plan, flipped)["categories"]["epilogue_store_cost"]["status"] == "not_qualified"
    missing = {k: v for k, v in observations.items() if "chain" not in k}
    assert CV.decide(plan, missing)["categories"]["configuration_and_fence_drains"]["status"] == "unknown"


def test_a_policy_without_its_ranking_gates_is_refused():
    sweep = _px()
    policy = sweep["requires_capacity_scope"]["calibration_validation"]
    del policy["acceptance"]["ordering_agreement_min"]
    with pytest.raises(ValueError, match="ordering_agreement_min"):
        CV.select([], CF.derive_basis("synthetic", _FACTS), policy)
