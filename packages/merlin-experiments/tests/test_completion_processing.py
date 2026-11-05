"""Native refill and scope lifetime with delayed durable-result consumers."""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event

import pytest

from merlin.targetgen.capsule_scheduling import run_pipelined
from merlin_experiments.phase1.feedback.completion_processing import CompletionProcessing


def test_slow_consumer_does_not_block_native_refill_and_drains_inside_scope(monkeypatch):
    monkeypatch.setenv("MERLIN_CAPSULE_CALIBRATION_S", "0")
    monkeypatch.setenv("MERLIN_CAPSULE_ORDER", "legacy")
    consumer_started, refill_started = Event(), Event()
    events = []
    owner = ContextVar("test_completion_owner", default=None)

    @contextmanager
    def custody():
        token = owner.set("original-owner")
        events.append("scope-open")
        try:
            yield
        finally:
            events.append("scope-close")
            owner.reset(token)

    def consume(row):
        assert owner.get() == "original-owner"
        assert "scope-close" not in events
        if row["capsule"] == "first":
            consumer_started.set()
            assert refill_started.wait(2), "completion blocked admission of the next capsule"
        events.append("consumed-" + row["capsule"])

    def run(cap, workers):
        if cap["name"] == "second":
            assert consumer_started.wait(2)
            refill_started.set()
        return {"capsule": cap["name"], "status": "pass"}

    with custody(), CompletionProcessing(consume, max_items=2) as queue:
        result = run_pipelined([{"name": "first"}, {"name": "second"}], run,
                               max_workers=1, on_complete=queue.submit)
    assert [row["capsule"] for row in result] == ["first", "second"]
    assert events == ["scope-open", "consumed-first", "consumed-second", "scope-close"]


def test_failed_consumer_drains_all_rows_and_reports_error():
    seen = []

    def consume(row):
        seen.append(row["capsule"])
        if row["capsule"] == "first":
            raise ValueError("original receipt changed")

    with pytest.raises(RuntimeError, match="after draining") as error:
        with CompletionProcessing(consume, max_items=2) as queue:
            queue.submit({"capsule": "first"})
            queue.submit({"capsule": "second"})
    assert seen == ["first", "second"]
    assert isinstance(error.value.__cause__, ValueError)


def test_native_failure_preserved_and_callback_input_frozen():
    release, started = Event(), Event()
    seen = []

    def consume(row):
        started.set()
        assert release.wait(2)
        seen.append(row["nested"]["value"])
        raise ValueError("consumer failure")

    with pytest.raises(OSError, match="native failure") as error:
        with CompletionProcessing(consume) as queue:
            row = {"nested": {"value": "original"}}
            queue.submit(row)
            assert started.wait(2)
            row["nested"]["value"] = "later-mutation"
            release.set()
            raise OSError("native failure")
    assert seen == ["original"]
    assert "also failed" in error.value.__notes__[0]


def test_bounded_queue_refuses_overflow_without_losing_accepted_row():
    seen = []
    queue = CompletionProcessing(seen.append, max_items=1)
    with pytest.raises(ValueError, match="bounded cohort"):
        with queue:
            queue.submit({"capsule": "accepted"})
            queue.submit({"capsule": "refused"})
    assert seen == [{"capsule": "accepted"}]
    with pytest.raises(RuntimeError, match="not open"):
        queue.submit({})
    with pytest.raises(RuntimeError, match="reopened"):
        with queue:
            pass
