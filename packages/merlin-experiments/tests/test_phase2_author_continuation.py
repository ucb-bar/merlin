"""Continuous queue stopping is owned by live host evidence, never model prose."""

import inspect
import time
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import authoring as A
from merlin_experiments.phase2 import stage_prompt as SP
from merlin_experiments.phase2.contracts import StageGateError
from merlin_experiments.phase2.search_coordinator import ComponentSearchCoordinator


def test_live_queue_target_unknowns_budget_and_missing_owner():
    state = {"target": {"achieved": False, "status": "UNKNOWN"}, "stop_reason": None}
    workflow = SimpleNamespace(search_coordinator=SimpleNamespace(status=lambda: state))
    broker = SimpleNamespace(calls=[], max_calls=3)
    assert A.queued_author_stop(workflow, broker, time.monotonic()+100) is None
    state.update(target={"achieved": True, "status": "qualified"}, stop_reason="declared performance target reached")
    assert A.queued_author_stop(workflow, broker, time.monotonic()+100) == "declared performance target reached"
    state.update(stop_reason="final drain and verification reserve reached")
    assert "reserve" in A.queued_author_stop(workflow, broker, time.monotonic()+100)
    broker.calls = [1, 2, 3]
    assert "tool-call budget" in A.queued_author_stop(workflow, broker, time.monotonic()+100)
    assert "wall budget" in A.queued_author_stop(workflow, broker, time.monotonic()-1)
    broker.calls = []
    with pytest.raises(StageGateError, match="original search coordinator"):
        A.queued_author_stop(SimpleNamespace(), broker, time.monotonic()+100)
    state.pop("stop_reason")
    with pytest.raises(StageGateError, match="original stopping evidence"):
        A.queued_author_stop(workflow, broker, time.monotonic()+100)


def test_continuation_default_remains_off_and_standing_prompt_is_explicit():
    assert inspect.signature(A.run_stage).parameters["continue_queued_author"].default is False
    assert inspect.signature(A._codex_round).parameters["continue_session"].default is False
    assert SP.render_queued_author_continuation(False) == ""
    text = SP.render_queued_author_continuation(True)
    assert "session history" in text and "lowest-attainment" in text and "search-status" in text
    assert "agent_selfcheck" not in text and "qa/verdict" not in text
    with pytest.raises(StageGateError):
        SP.render_queued_author_continuation(1)


def test_target_driven_queue_cannot_be_finalized_before_target_or_budget():
    queue = object.__new__(ComponentSearchCoordinator)
    from merlin_experiments.phase2.search_coordinator import SearchSettings
    queue.settings = SearchSettings(stop_on_plateau=False, target_roofline_attainment=.3,
        soft_seconds=3600, hard_seconds=3600, final_reserve_seconds=600)
    queue.started = time.monotonic()
    queue._finalized = False
    drained = []
    queue.pool = SimpleNamespace(drain=lambda **kwargs: drained.append(kwargs))
    state = {"stop_reason": None}
    queue.status = lambda: state
    with pytest.raises(StageGateError, match="target-driven"):
        queue.finalize(timeout_s=1)
    assert not drained and queue._finalized is False
    state["stop_reason"] = "declared performance target reached"
    assert queue.finalize(timeout_s=1) is state
    assert drained and queue._finalized is True
