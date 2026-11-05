"""Slow original verifiers leave controls responsive without accepting unchecked work."""

import threading
import time
from copy import copy
from types import SimpleNamespace

import pytest
from component_baseline_fixture import synthetic_component_controller as synthetic_component_controller
from merlin_experiments.phase2 import evaluation_pool as EP
from merlin_experiments.phase2.contracts import StageGateError
from test_retained_evaluation_pool import bounded_control, completing_feedback, wait_for
from test_retained_evaluation_pool import selected_pool as selected_pool

pytestmark = pytest.mark.usefixtures("synthetic_component_controller")


def asynchronous(callback, name):
    values, errors = [], []

    def invoke():
        try:
            values.append(callback())
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=invoke, name=name, daemon=True)
    thread.start()
    return thread, values, errors


@pytest.mark.parametrize("change", ["stop", "close", "deadline", "source", "owner"])
def test_slow_submission_rechecks_controls_deadline_source_and_owner(selected_pool, monkeypatch, change):
    policy, pool, *_ = selected_pool
    revision = pool.revisions.freeze(policy.candidate)
    entered, resume = threading.Event(), threading.Event()
    original = pool._closure
    clock = SimpleNamespace(value=time.monotonic())
    clock.monotonic = lambda: clock.value
    monkeypatch.setattr(EP, "time", clock)

    def blocked():
        result = original()
        if threading.current_thread().name == "slow-submit":
            entered.set()
            assert resume.wait(6)
        return result

    monkeypatch.setattr(pool, "_closure", blocked)
    thread, values, errors = asynchronous(
        lambda: pool.submit(revision.revision_id, completion_deadline=clock.value + 60), "slow-submit"
    )
    original_inputs = pool.inputs
    try:
        assert entered.wait(2)
        assert bounded_control(pool.observations) == ()
        if change == "stop":
            bounded_control(pool.stop_submissions)
        elif change == "close":
            closing, _, close_errors = asynchronous(pool.close, "close-during-admission")
            wait_for(lambda: pool._closed)
            assert closing.is_alive() and pool._engine_lease is not None and pool._admitting == 1
        elif change == "deadline":
            clock.value += 61
        elif change == "source":
            path = revision.source / "compiler.py"
            path.chmod(0o600)
            path.write_text("def schedule():\n    return 999\n")
        else:
            pool.inputs = copy(pool.inputs)
        resume.set()
        thread.join(3)
        assert not thread.is_alive() and not values
        assert len(errors) == 1 and isinstance(errors[0], StageGateError)
        assert not pool._jobs  # No accepted job, lease or execution survived admission.
        assert pool._admitting == 0
        if change == "close":
            closing.join(3)
            assert not closing.is_alive() and not close_errors and pool._engine_lease is None
    finally:
        pool.inputs = original_inputs
        resume.set()
        thread.join(6)


def test_simultaneous_same_revision_submissions_recheck_pending_capacity(selected_pool, monkeypatch):
    policy, pool, *_ = selected_pool
    revision = pool.revisions.freeze(policy.candidate)
    original = pool._closure
    barrier = threading.Barrier(3)

    def blocked():
        result = original()
        if threading.current_thread().name.startswith("simultaneous-submit-"):
            barrier.wait(6)
        return result

    monkeypatch.setattr(pool, "_closure", blocked)
    attempts = [
        asynchronous(lambda: pool.submit(revision.revision_id), f"simultaneous-submit-{index}")
        for index in range(4)
    ]
    for thread, _, _ in attempts:
        thread.join(8)
        assert not thread.is_alive()
    values = [value for _, values, _ in attempts for value in values]
    errors = [error for _, _, errors in attempts for error in errors]
    assert len(values) == 3 and len({row["job_id"] for row in values}) == 3
    assert len(errors) == 1 and isinstance(errors[0], StageGateError)
    assert "pending grant" in str(errors[0])
    assert len(pool._jobs) == pool.max_pending


@pytest.mark.parametrize("selected_pool", [completing_feedback], indirect=True)
@pytest.mark.parametrize("change", ["cancel", "close", "source", "source_after_cleanup", "deadline", "duplicate"])
def test_finalization_releases_control_lock_but_keeps_original_ownership(selected_pool, monkeypatch, change):
    policy, pool, *_ = selected_pool
    revision = pool.revisions.freeze(policy.candidate)
    entered, resume = threading.Event(), threading.Event()
    verify, release = pool._verify, pool._release
    released, final_checks = [], [0]
    clock = SimpleNamespace(value=time.monotonic())
    clock.monotonic = lambda: clock.value
    monkeypatch.setattr(EP, "time", clock)

    def blocked(job):
        if job.state == "finalizing":
            final_checks[0] += 1
            if final_checks[0] == (2 if change == "source_after_cleanup" else 1):
                entered.set()
                assert resume.wait(6)
        return verify(job)

    def releasing(job, **kwargs):
        released.append(job.job_id)
        return release(job, **kwargs)

    monkeypatch.setattr(pool, "_verify", blocked)
    monkeypatch.setattr(pool, "_release", releasing)
    submitted = pool.submit(revision.revision_id)
    identity = submitted["job_id"]
    try:
        assert entered.wait(3)
        job = pool._jobs[identity]
        status = bounded_control(lambda: pool.status(identity))
        assert status["state"] == "finalizing" and not status["terminal"] and status["result"] is None
        assert job.pins
        assert (job.guardian_fd is None) is (change == "source_after_cleanup")
        # Another admission must not wait for this original completion verifier.
        second = bounded_control(lambda: pool.submit(revision.revision_id))
        bounded_control(lambda: pool.cancel(second["job_id"]))
        if change == "cancel":
            cancelled = bounded_control(lambda: pool.cancel(identity))
            assert cancelled["state"] == "finalizing" and job.cancel_requested
        elif change == "close":
            closing, _, close_errors = asynchronous(pool.close, "close-during-finalization")
            wait_for(lambda: pool._closed)
            assert closing.is_alive() and job.pins and pool._engine_lease is not None
            assert bounded_control(lambda: pool.status(identity))["state"] == "finalizing"
        elif change in {"source", "source_after_cleanup"}:
            path = revision.source / "compiler.py"
            path.chmod(0o600)
            path.write_text("def schedule():\n    return 999\n")
        elif change == "deadline":
            clock.value = job.deadline + 1
        else:
            bounded_control(lambda: pool._finish(job))
            assert identity not in released  # A duplicate finisher cannot reap or release its owner.
        resume.set()
        final = wait_for(lambda: pool.status(identity) if pool.status(identity)["terminal"] else None)
        expected = "cancelled" if change in {"cancel", "close"} else "completed" if change == "duplicate" else "refused"
        assert final["state"] == expected
        assert (final["result"] is not None) is (expected == "completed")
        assert released.count(identity) == 1 and job.guardian_fd is None and not job.pins
        assert (job.root / "cleanup.json").is_file()
        assert (job.root / "feedback.json").exists() is (expected == "completed")
        if change == "close":
            closing.join(3)
            assert not closing.is_alive() and not close_errors and pool._engine_lease is None
    finally:
        resume.set()
