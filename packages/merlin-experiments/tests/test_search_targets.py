"""Target stopping uses complete measured repeats and preserves budget limits."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import search_coordinator as S
from merlin_experiments.phase2.contracts import StageGateError
from test_search_coordinator import diagnostic_coordinator, observation
from test_search_coordinator import diagnostic_projection as diagnostic_projection  # noqa: F401


def observed(source="c" * 64, *, slow=False, unresolved=False):
    row = observation(source, 90 if not slow else 120, 9000 if not slow else 12000)
    for cell in row["result"]["evidence"]["cells"]:
        cell["roofline"] = {"status": "derived",
            "roofline_cycles": 80 if cell["family"] == "compute" else 8000,
            "unresolved": ["host scope UNKNOWN"] if unresolved else []}
    return row


def configured(rows, **settings):
    owner = diagnostic_coordinator(rows, beam=2)
    owner.settings = replace(owner.settings, **settings)
    owner.settings.verify()
    return owner


def test_roofline_target_stops_only_on_every_complete_measured_member():
    status = configured([observed()], target_roofline_attainment=.8).status()
    assert status["target"]["achieved"]
    assert status["stop_reason"] == "declared performance target reached"
    assert len(status["target"]["evidence"]["ranked"]) == 2


def test_partial_floor_keeps_search_running_even_above_numeric_target():
    status = configured([observed(unresolved=True)], target_roofline_attainment=.8).status()
    assert not status["target"]["achieved"]
    assert status["target"]["status"] == "undecided"
    assert status["stop_reason"] is None


def test_failed_member_and_estimates_never_satisfy_target():
    failed = observed()
    failed["result"]["evidence"]["cells"][1]["measured"] = False
    predicted = {"state": "completed", "kind": "component-analytical-feedback",
                 "source_sha256": "d" * 64, "result": {"predicted_cycles": 1}}
    status = configured([failed, predicted], target_roofline_attainment=.8).status()
    assert not status["target"]["achieved"]
    assert status["target"]["status"] == "undecided"
    assert status["stop_reason"] is None


def test_later_slower_repeat_revokes_target_and_incumbent():
    rows = [observed()]
    owner = configured(rows, target_roofline_attainment=.8)
    assert owner.status()["target"]["achieved"]
    rows.append(observed(slow=True))
    assert not owner.status()["target"]["achieved"]
    assert owner.status()["stop_reason"] is None


def test_baseline_ratio_target_uses_conservative_complete_roster():
    rows = [observation("c" * 64, 50, 5000)]
    owner = configured(rows, target_cycle_ratio=.6)
    assert owner.status()["target"]["achieved"]
    rows.append(observation("c" * 64, 70, 7000, index=1))
    assert not owner.status()["target"]["achieved"]


def test_target_mode_can_continue_past_measured_plateau_but_not_deadline():
    rows = [observation(chr(99+i) * 64, 110, 11000) for i in range(8)]
    owner = configured(rows, plateau_batches=1, stop_on_plateau=False,
                       target_roofline_attainment=.8)
    assert owner.status()["plateau_batches"] >= 1
    assert owner.status()["stop_reason"] is None
    owner.settings = replace(owner.settings, stop_on_plateau=True)
    assert "plateau" in owner.status()["stop_reason"]
    owner.settings = replace(owner.settings, stop_on_plateau=False)
    owner.started -= owner.settings.hard_seconds
    assert owner.status()["stop_reason"] == "hard wall budget exhausted"


@pytest.mark.parametrize("options", [
    {"target_roofline_attainment": float("nan")},
    {"target_roofline_attainment": True},
    {"target_roofline_attainment": 1.1},
    {"target_cycle_ratio": 0},
    {"target_cycle_ratio": .5, "target_roofline_attainment": .8},
    {"stop_on_plateau": 1},
])
def test_invalid_or_conflicting_targets_refuse(options):
    with pytest.raises(StageGateError):
        S.SearchSettings(**options).verify()


def test_refuted_or_inconsistent_floors_never_stop():
    first = observed()
    first["result"]["evidence"]["cells"][0]["roofline"]["roofline_cycles"] = 100
    assert not configured([first], target_roofline_attainment=.8).status()["target"]["achieved"]
    first = observed()
    second = observed()
    second["result"]["evidence"]["cells"][0]["roofline"]["roofline_cycles"] = 79
    assert not configured([first, second], target_roofline_attainment=.8).status()["target"]["achieved"]


def test_capsule_priority_uses_lowest_attainment_instead_of_largest_cycle_gap():
    row = observed()
    cells = row["result"]["evidence"]["cells"]
    cells[0]["roofline"]["roofline_cycles"] = 1
    cells[1]["roofline"]["roofline_cycles"] = 7000
    status = configured([row], target_roofline_attainment=.3).status()
    priority = status["capsule_priorities"]
    assert priority["ranked"][0]["member"] == ("compute", "contraction")
    assert status["priorities"]["ranked"][0]["family"] == "host"
    assert priority["source_sha256"] == row["source_sha256"]


def test_unknown_priority_keeps_all_required_members_visible():
    status = configured([], target_roofline_attainment=.3).status()
    assert len(status["capsule_priorities"]["unknown_floor"]) == 2
    assert status["capsule_priorities"]["ranked"] == []
    assert not status["target"]["achieved"]


def test_estimated_priority_view_stays_separate_from_the_measured_stop():
    owner = configured([], target_roofline_attainment=.3)
    owner.pool.policy = SimpleNamespace(_native_exploration=SimpleNamespace(exploratory_estimate_protocol="selected"))
    estimate = {"status": "EXPLORATORY", "ranked": [{"attainment": .9}], "convergence": False}
    owner.exploratory_capsule_priorities = lambda observations: estimate
    status = owner.status()
    assert status["exploratory_capsule_priorities"] is estimate
    assert status["capsule_priorities"]["ranked"] == []
    assert not status["target"]["achieved"] and status["stop_reason"] is None
