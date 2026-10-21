"""Search stop controls with synthetic arithmetic; no target qualification."""

import time
from copy import deepcopy
from dataclasses import replace

import pytest
from merlin_experiments.phase2 import component_workflow as CW
from merlin_experiments.phase2 import search_coordinator as SC
from merlin_experiments.phase2.contracts import StageGateError
from test_search_coordinator import diagnostic_coordinator, observation


@pytest.fixture(autouse=True)
def arithmetic_only(monkeypatch):
    monkeypatch.setattr(CW, "validate_component_feedback", deepcopy)


def coordinator(rows, **settings):
    result = diagnostic_coordinator(rows, beam=2)
    result.settings = replace(result.settings, **settings)
    result.settings.verify()
    return result


def with_floors(row, compute=40, host=4_000):
    for cell, floor in zip(row["result"]["evidence"]["cells"], (compute, host), strict=True):
        cell["roofline"] = {"status": "derived", "roofline_cycles": floor}
    return row


def test_goal_stops_on_complete_measured_incumbent_without_final_acceptance():
    result = coordinator([observation("c" * 64, 50, 5_000)], target_max_ratio=0.6).status()
    assert result["performance_target"]["status"] == "reached"
    assert result["performance_target"]["source_sha256"] == result["incumbent"]["source_sha256"]
    assert "performance target reached" in result["stop_reason"]
    assert result["promotion"] == "NO_FINAL_ACCEPTANCE"


def test_a_later_slow_repeat_revokes_the_target():
    rows = [observation("c" * 64, 50, 5_000)]
    search = coordinator(rows, target_max_ratio=0.6)
    assert search.status()["performance_target"]["status"] == "reached"
    rows.append(observation("c" * 64, 80, 8_000, index=1))
    result = search.status()
    assert result["performance_target"]["status"] == "continue"
    assert result["stop_reason"] is None


@pytest.mark.parametrize("kind", ["component-analytical-feedback", "corpus-selected-compile-spike-feedback"])
def test_estimates_and_selected_diagnostics_cannot_reach_a_target(kind):
    row = observation("c" * 64, 1, 1)
    row["kind"] = kind
    result = coordinator([row], target_max_ratio=0.9).status()
    assert result["performance_target"]["status"] == "unknown"
    assert result["stop_reason"] is None


def test_incorrect_complete_roster_cannot_reach_a_target():
    result = coordinator([observation("c" * 64, 1, 1, complete=False)], target_max_ratio=0.9).status()
    assert result["performance_target"]["status"] == "unknown"


def test_roofline_goal_requires_every_member_instead_of_an_average():
    row = with_floors(observation("c" * 64, 50, 5_000))
    search = coordinator([row], target_roofline_attainment=0.8)
    assert search.status()["performance_target"]["status"] == "reached"
    row["result"]["evidence"]["cells"][1]["roofline"]["roofline_cycles"] = 400
    result = search.status()
    assert result["performance_target"]["status"] == "continue"
    assert result["stop_reason"] is None
    assert result["roofline_priorities"]["next"] == [("host", "copy")]


def test_roofline_priority_follows_the_worst_repeat_and_keeps_unknowns_separate():
    rows = [with_floors(observation("c" * 64, 50, 5_000), compute=2, host=1_000),
            with_floors(observation("c" * 64, 80, 5_000, index=1), compute=2, host=1_000)]
    result = coordinator(rows, target_roofline_attainment=0.3).status()
    assert result["roofline_priorities"]["next"] == [("compute", "contraction"), ("host", "copy")]
    assert result["roofline_priorities"]["ranked"][0]["attainment"]["worst"] == 2 / 80
    for row in rows:
        del row["result"]["evidence"]["cells"][0]["roofline"]
    result = coordinator(rows, target_roofline_attainment=0.3).status()
    assert result["roofline_priorities"]["next"] == [("host", "copy")]
    assert result["roofline_priorities"]["unknown_floor"][0]["member"] == ("compute", "contraction")


@pytest.mark.parametrize("missing", ["unknown", "unresolved", "omitted", "refuted", "late_partial_repeat"])
def test_incomplete_or_refuted_rooflines_block_attainment(missing):
    rows = [with_floors(observation("c" * 64, 50, 5_000))]
    cell = rows[0]["result"]["evidence"]["cells"][0]
    if missing == "unknown":
        del cell["roofline"]
    elif missing == "unresolved":
        cell["roofline"]["unresolved"] = {"host": "command issue unpriced"}
    elif missing == "omitted":
        cell["roofline"]["omitted"] = ["command issue", "physical traffic"]
    elif missing == "refuted":
        cell["roofline"]["roofline_cycles"] = 60
    else:
        extra = with_floors(observation("c" * 64, 50, 5_000, index=1))
        extra["result"]["evidence"]["cells"][0]["roofline"]["unresolved"] = {"host": "unpriced"}
        rows.append(extra)
    result = coordinator(rows, target_roofline_attainment=0.8).status()
    assert result["performance_target"]["status"] == "unknown"
    assert result["stop_reason"] is None


def test_configured_ratio_and_roofline_are_both_required():
    row = with_floors(observation("c" * 64, 50, 5_000))
    result = coordinator([row], target_max_ratio=0.4, target_roofline_attainment=0.8).status()
    assert result["performance_target"]["conditions"]["roofline"]["reached"]
    assert result["performance_target"]["status"] == "continue"


def test_disabled_plateau_keeps_searching_but_respects_the_budget():
    rows = [observation("c" * 64, 120, 12_000), observation("d" * 64, 130, 13_000)]
    ordinary = coordinator(rows, plateau_batches=1)
    assert "plateau" in ordinary.status()["stop_reason"]
    continuous = coordinator(rows, plateau_batches=1, stop_on_plateau=False)
    assert continuous.status()["plateau_batches"] == 1
    assert continuous.status()["stop_reason"] is None
    continuous.started = time.monotonic() - continuous.settings.hard_seconds - 1
    assert continuous.status()["stop_reason"] == "hard wall budget exhausted"


def test_budget_exhaustion_is_not_reported_as_target_completion():
    search = coordinator([observation("c" * 64, 50, 5_000)], target_max_ratio=0.6)
    search.started = time.monotonic() - search.settings.hard_seconds - 1
    result = search.status()
    assert result["performance_target"]["status"] == "reached"
    assert result["stop_reason"] == "hard wall budget exhausted"


@pytest.mark.parametrize("field", ["target_max_ratio", "target_roofline_attainment"])
@pytest.mark.parametrize("value", [0, -0.1, 1.1, True, float("nan"), float("inf"), "0.8", 10**1_000])
def test_invalid_targets_refuse(field, value):
    with pytest.raises(StageGateError):
        SC.SearchSettings(**{field: value}).verify()


def test_plateau_selection_requires_an_explicit_boolean():
    with pytest.raises(StageGateError):
        SC.SearchSettings(stop_on_plateau=0).verify()
