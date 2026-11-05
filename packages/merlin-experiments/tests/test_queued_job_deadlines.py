"""Original queue routing and clocks, without any target or scientific owner."""

import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import component_workflow as CW
from merlin_experiments.phase2 import ordinary_fast_feedback as FAST
from merlin_experiments.phase2 import ordinary_measurement_selection as SELECT
from merlin_experiments.phase2 import queued_search_actions as ACTIONS
from merlin_experiments.phase2 import search_coordinator as SC
from merlin_experiments.phase2.contracts import StageGateError


def coordinator(*, selected=None, now=100):
    calls = []
    value = SC.ComponentSearchCoordinator.__new__(SC.ComponentSearchCoordinator)
    value.settings = SC.SearchSettings(
        soft_seconds=5000,
        hard_seconds=5000,
        final_reserve_seconds=1000,
        queued_job_timeout_seconds=selected,
    )
    value.started, value._finalized = now, False
    value.status = lambda: {"stop_reason": None}
    value.pool = SimpleNamespace(submit=lambda *a, **kw: calls.append((a, kw)) or {"state": "queued"})
    return value, calls


@pytest.mark.parametrize("value", [0, -1, True, 1.0, 2401, "2400"])
def test_host_lifetime_selection_requires_bounded_integer(value):
    with pytest.raises(StageGateError):
        SC.SearchSettings(queued_job_timeout_seconds=value).verify()


def test_selected_lifetime_must_fit_the_declared_campaign_reserve():
    with pytest.raises(StageGateError):
        SC.SearchSettings(
            soft_seconds=2000, hard_seconds=2000, final_reserve_seconds=100, queued_job_timeout_seconds=2400
        ).verify()


@pytest.mark.parametrize("selected,expected", [(None, 600), (2400, 2400)])
def test_acknowledgment_and_queued_lifetime_are_independent_only_by_host_opt_in(monkeypatch, selected, expected):
    owner, calls = coordinator(selected=selected)
    monkeypatch.setattr(SC.time, "monotonic", lambda: 100)
    owner.submit("revision", timeout_s=600, kind=FAST.SELECTED_KIND, selected_members=(("f", "c"),))
    assert calls == [
        (
            ("revision",),
            {
                "kind": FAST.SELECTED_KIND,
                "selected_members": (("f", "c"),),
                "timeout_s": expected,
                **({"completion_deadline": 4100} if selected is not None else {}),
            },
        )
    ]


def test_metadata_time_cannot_spend_the_reserved_campaign_allowance(monkeypatch):
    owner, calls = coordinator(selected=2400)
    ticks = iter((100, 1800))
    monkeypatch.setattr(SC.time, "monotonic", lambda: next(ticks))
    owner.submit("revision", timeout_s=600)
    assert calls == [(("revision",), {"timeout_s": 2300, "completion_deadline": 4100})]


def test_late_submission_uses_the_remaining_exploration_time(monkeypatch):
    owner, calls = coordinator(selected=2400)
    monkeypatch.setattr(SC.time, "monotonic", lambda: 4010)
    owner.submit("revision", timeout_s=600)
    assert calls == [(("revision",), {"timeout_s": 90, "completion_deadline": 4100})]


@pytest.mark.parametrize("ticks", [(4099.5,), (100, 4100)])
def test_expired_exploration_never_dispatches(monkeypatch, ticks):
    owner, calls = coordinator(selected=2400)
    times = iter(ticks)
    monkeypatch.setattr(SC.time, "monotonic", lambda: next(times))
    with pytest.raises(StageGateError, match="reserves"):
        owner.submit("revision", timeout_s=600)
    assert calls == []


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf")])
def test_selected_lifetime_does_not_admit_invalid_acknowledgment_budget(monkeypatch, timeout):
    owner, calls = coordinator(selected=2400)
    monkeypatch.setattr(SC.time, "monotonic", lambda: 100)
    with pytest.raises(StageGateError, match="acknowledgment"):
        owner.submit("revision", timeout_s=timeout)
    assert calls == []


@pytest.mark.parametrize(
    "action,kind",
    [
        (CW.EVALUATION_SUBMIT_ACTION, FAST.KIND),
        (FAST.SELECTED_ACTION, FAST.SELECTED_KIND),
        (SELECT.ACTION, SELECT.KIND),
    ],
)
def test_normal_broker_actions_preserve_selected_source_kind_and_roster(monkeypatch, action, kind):
    owner, calls = coordinator(selected=2400)
    monkeypatch.setattr(SC.time, "monotonic", lambda: 100)
    members = (("family", "capsule"),)
    policy = SimpleNamespace(
        validate_search_inputs=lambda: set(members), search_coordinator=owner, evaluation_pool=owner.pool
    )
    rendered = {"revision_id": "immutable", "kind": kind, "members": "family/capsule"}
    response, _ = ACTIONS.execute(policy, action, rendered, 600, 100)
    assert json.loads(response["stdout"]) == {"state": "queued"}
    assert calls == [
        (
            ("immutable",),
            {
                "kind": kind,
                "timeout_s": 2400,
                "completion_deadline": 4100,
                **({"selected_members": members} if action != CW.EVALUATION_SUBMIT_ACTION else {}),
            },
        )
    ]


def test_search_status_keeps_its_synchronous_budget(monkeypatch):
    owner, calls = coordinator(selected=2400)
    monkeypatch.setattr(SC.time, "monotonic", lambda: 100)
    policy = SimpleNamespace(validate_search_inputs=lambda: (), search_coordinator=owner, evaluation_pool=owner.pool)
    response, _ = ACTIONS.execute(policy, CW.SEARCH_STATUS_ACTION, {}, 1, 100)
    assert response["returncode"] == 0 and calls == []


def test_stopped_campaign_cannot_submit_longer_queued_job(monkeypatch):
    owner, calls = coordinator(selected=2400)
    monkeypatch.setattr(SC.time, "monotonic", lambda: 100)
    owner._finalized = True
    with pytest.raises(StageGateError, match="reserves"):
        owner.submit("revision", timeout_s=600)
    assert calls == []
