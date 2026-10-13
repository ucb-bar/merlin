"""Parent-hook controls with fake runtime/cleanup boundaries; no pool or engine runs."""

import json
import os
import time
from copy import copy, deepcopy
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import evaluation_costs as COST
from merlin_experiments.phase2 import evaluation_pool as EP
from merlin_experiments.phase2 import ordinary_fast_feedback as FAST
from merlin_experiments.phase2.contracts import StageGateError, canonical_json


class Clock:
    def __init__(self):
        self.value = 10.0

    def monotonic(self):
        return self.value

    def advance(self, label, events):
        events.append(label)
        self.value += 1


@pytest.fixture
def parent(tmp_path, monkeypatch):
    clock, events = Clock(), []
    monkeypatch.setattr(COST, "time", clock)
    member = SimpleNamespace(family="compute", capsule="tail", source_sha256="a" * 64, descriptor={"performance": {}})
    corpus = SimpleNamespace(capsules=(member,), capsules_sha256="b" * 64, manifest_sha256="c" * 64)
    revision = SimpleNamespace(source_sha256="d" * 64, public_record=lambda: {"source_sha256": "d" * 64})
    pool = EP.ComponentEvaluationPool.__new__(EP.ComponentEvaluationPool)
    pool.inputs = SimpleNamespace(
        ordinary=True,
        corpus=corpus,
        rtl=SimpleNamespace(certificate=SimpleNamespace(sha256="e" * 64)),
        validate_result=lambda result, *_: deepcopy(result),
    )
    pool.baseline_revision = SimpleNamespace(source_sha256="f" * 64)
    pool.grant = EP.JobResources(1, 1 << 20, 1 << 20)
    pool.resources = EP.AssignedResources((1, 2), 1 << 22, 1 << 22, 1, author_cores=1)
    pool.enforcement, pool._external_boundary = {"hard_aggregate_limits": "UNKNOWN"}, None
    pool._feedback_bytes, pool._feedback_budget = 0, 1 << 20
    pool._jobs = {}
    pool._evaluation_costs = COST.EvaluationCosts(pool)
    publication = {"status": "UNKNOWN", "published": 0}
    pool.policy = SimpleNamespace(
        publish_completed_artifacts=lambda _: (clock.advance("publish", events), publication)[1],
        discard_cleaned_artifacts=lambda _: (clock.advance("discard", events), publication)[1],
    )
    pool._save = lambda _: clock.advance("save", events)
    pool._verify = lambda _: clock.advance("verify", events)
    monkeypatch.setattr(EP, "executing_revisions", lambda *_: __import__("contextlib").nullcontext())
    descriptors = []

    def job(identity="job"):
        root = tmp_path / identity
        root.mkdir()
        value = EP._Job(identity, revision, FAST.SELECTED_KIND, 100, "1" * 64, "2" * 64, root)
        value.selected_members = ((member.family, member.capsule),)
        value.cpus = (2,)
        value.deadline = time.monotonic() + 100
        value.control = SimpleNamespace(close=lambda: clock.advance("control-close", events))
        value.guardian_pid = 123
        value.guardian_fd, writer = os.pipe()
        descriptors.extend((value.guardian_fd, writer))
        value.lease = SimpleNamespace(close=lambda _: clock.advance("release", events))
        pool._jobs[identity] = value
        COST.note(pool, value, "queued", requested=clock.value - 1)
        value.state = "running"
        return value

    def receive(*_, **__):
        clock.advance("receive", events)
        return {"result": {"cells": []}}, 1

    def reap(*_, **__):
        clock.advance("reap", events)
        return {"error": None}, True, 0

    monkeypatch.setattr(EP, "receive_feedback_value", receive)
    monkeypatch.setattr(EP, "_close_guardian", reap)
    monkeypatch.setattr(EP, "_complete_cleanup", lambda *_: True)
    yield SimpleNamespace(pool=pool, clock=clock, events=events, job=job, publication=publication, member=member)
    for fd in descriptors:
        try:
            os.close(fd)
        except OSError:
            pass


def complete(parent, job):
    COST.note(parent.pool, job, "start")
    parent.clock.advance("worker", parent.events)
    parent.pool._finish(job)
    return COST.view(parent.pool, job)


def test_original_finish_prices_setup_through_release_and_publication(parent):
    job = parent.job()
    parent.clock.value = 14
    cost = complete(parent, job)
    assert cost["status"] == "observed" and job.state == "completed"
    assert cost["wall_seconds"] == 10
    assert cost["admission_seconds"] == 1
    assert cost["queue_wait_seconds"] == 4
    assert cost["request_wall_seconds"] == 15
    assert parent.events[-3:] == ["release", "publish", "save"]
    assert cost["artifact_publication"] == parent.publication
    assert cost["cache_state"] == "UNKNOWN"
    assert cost["cpu_seconds"] is cost["per_arm_seconds"] is None
    assert cost["context"]["selected_members"] == [["compute", "tail"]]
    assert cost["context"]["member"]["source_sha256"] == parent.member.source_sha256
    assert json.loads((job.root / "evaluation-cost.json").read_bytes()) == cost


def test_start_brackets_original_verification_before_any_fork(parent):
    job = parent.job()
    job.root.rmdir()
    job.guardian_pid, job.guardian_fd = None, None

    def refuses(_):
        assert parent.pool._evaluation_costs._entries[job.job_id].started == parent.clock.value
        assert job.cpus == (3,)
        raise StageGateError("synthetic verification boundary")

    parent.pool._verify = refuses
    with pytest.raises(StageGateError, match="synthetic verification"):
        parent.pool._start(job, (3,))
    assert COST.view(parent.pool, job)["status"] == "UNKNOWN"


def test_interleaved_jobs_retain_individual_wall_ranges(parent):
    first, second = parent.job("first"), parent.job("second")
    COST.note(parent.pool, first, "start")
    parent.clock.value += 4
    COST.note(parent.pool, second, "start")
    parent.clock.value += 2
    parent.pool._finish(second)
    parent.pool._finish(first)
    assert COST.view(parent.pool, second)["wall_seconds"] == 11
    assert COST.view(parent.pool, first)["wall_seconds"] == 24


@pytest.mark.parametrize("product", ["feedback.json", "cleanup.json", "evaluation-cost.json"])
def test_changed_products_invalidate_cost_without_new_execution(parent, product):
    job = parent.job()
    assert complete(parent, job)["status"] == "observed"
    path = job.root / product
    path.chmod(0o600)
    path.write_bytes(path.read_bytes() + b"\n")
    assert COST.view(parent.pool, job)["status"] == "UNKNOWN"


def test_copy_saved_data_fork_or_changed_assignment_cannot_reopen_cost(parent, monkeypatch):
    job = parent.job()
    assert complete(parent, job)["status"] == "observed"
    assert COST.view(parent.pool, copy(job))["status"] == "UNKNOWN"
    new = COST.EvaluationCosts(parent.pool)
    assert new.view(job)["status"] == "UNKNOWN"
    pid = os.getpid()
    with monkeypatch.context() as patch:
        patch.setattr(COST, "os", SimpleNamespace(getpid=lambda: pid + 1))
        assert COST.view(parent.pool, job)["status"] == "UNKNOWN"
    job.cpus = (3,)
    assert COST.view(parent.pool, job)["status"] == "UNKNOWN"


@pytest.mark.parametrize("boundary", ["cleanup", "release"])
def test_unknown_cleanup_or_release_cannot_publish_complete_cost(parent, monkeypatch, boundary):
    job = parent.job()
    if boundary == "cleanup":
        monkeypatch.setattr(EP, "_complete_cleanup", lambda *_: False)
    else:

        def fails(_):
            raise OSError("synthetic release failure")

        job.lease.close = fails
    cost = complete(parent, job)
    assert job.state == "cleanup_unknown"
    assert cost["status"] == "UNKNOWN" and cost["wall_seconds"] is None
    assert "publish" not in parent.events


def test_cancelled_started_job_reports_failure_cost_without_success(parent):
    job = parent.job()
    job.cancel_requested = True
    cost = complete(parent, job)
    assert job.state == cost["outcome"] == "cancelled"
    assert cost["status"] == "observed" and cost["wall_seconds"] > 0
    assert job.result is None


def test_cost_io_failure_preserves_grade_and_releases_only_once(parent, monkeypatch):
    job = parent.job()

    def fails(*_, **__):
        raise OSError("synthetic cost disk failure")

    monkeypatch.setattr(COST, "write_json", fails)
    cost = complete(parent, job)
    assert job.state == "completed" and job.result is not None
    assert cost["status"] == "UNKNOWN"
    assert parent.events.count("reap") == parent.events.count("release") == 1
    assert (job.root / "feedback.json").read_bytes() == canonical_json(job.result)


def test_preexisting_cost_product_cannot_select_or_overwrite_parent_observation(parent):
    job = parent.job()
    path = job.root / "evaluation-cost.json"
    path.write_bytes(b"untrusted preexisting cost")
    cost = complete(parent, job)
    assert cost["status"] == "UNKNOWN"
    assert path.read_bytes() == b"untrusted preexisting cost"
    assert job.state == "completed"
    assert parent.events.count("release") == 1
