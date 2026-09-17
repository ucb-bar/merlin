"""Restart-safe request handling for the sandbox self-check channel.

The channel is intentionally file based, so its durable files outlive the broker.  A restarted broker
must distinguish an outstanding request from the completed and abandoned requests left by earlier
clients; otherwise one fresh self-check can sit behind hours of replayed grading work.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

from merlin.common.paths import merlin_dir


HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _module(name: str):
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _request(ch: Path, rid: str, *, mtime: float, timeout: int = 1800,
             extra: dict | None = None) -> Path:
    req = ch / f"req_{rid}.json"
    body = {"sim": "spike", "capsules": "all", "timeout": timeout}
    body.update(extra or {})
    req.write_text(json.dumps(body))
    os.utime(req, (mtime, mtime))
    return req


def test_request_ids_remain_distinct_when_pid_and_clock_repeat(monkeypatch):
    """Namespace PIDs repeat and the old millisecond clock wrapped every 16m40s."""
    shim = _module("selfcheck_shim")
    monkeypatch.setattr(shim.os, "getpid", lambda: 4734)
    monkeypatch.setattr(shim.time, "time_ns", lambda: 123_456_789)

    ids = {shim._request_id() for _ in range(100)}

    assert len(ids) == 100
    assert all(rid.startswith("4734_123456789_") for rid in ids)


def test_completed_requests_are_not_replayed_after_broker_restart(tmp_path):
    broker = _module("selfcheck_broker")
    ch = tmp_path / ".qa_channel"
    ch.mkdir()
    old = _request(ch, "old", mtime=10)
    (ch / "resp_old.json").write_text('{"all_pass": false}')
    (ch / "done_old").write_text("ok")
    fresh = _request(ch, "fresh", mtime=20)

    assert broker._pending_requests(ch, set(), now=20) == [fresh]
    assert old.exists(), "restart recovery must not mutate the durable audit trail"


def test_abandoned_requests_expire_instead_of_blocking_the_live_fifo(tmp_path):
    broker = _module("selfcheck_broker")
    ch = tmp_path / ".qa_channel"
    ch.mkdir()
    abandoned = _request(ch, "abandoned", mtime=10, timeout=30)
    fresh = _request(ch, "fresh", mtime=100, timeout=1800)

    # A legacy request remains accepted through its declared timeout plus the shim's 240 s grace.  Once
    # its client must already have timed out, replaying it can only delay a live caller.
    assert broker._pending_requests(ch, set(), now=279) == [abandoned, fresh]
    assert broker._pending_requests(ch, set(), now=281) == [fresh]


def test_dead_requester_is_not_replayed_even_when_suite_deadline_is_still_live(tmp_path):
    """A full-suite budget may outlive its client by hours; process identity is the lease."""
    broker = _module("selfcheck_broker")
    ch = tmp_path / ".qa_channel"
    ch.mkdir()
    abandoned = _request(
        ch, "999999_123_nonce", mtime=100, timeout=1800,
        extra={"protocol": 3, "requester": {"pid": 999999, "start_ticks": 1,
                                             "pid_namespace": broker._pid_namespace()},
               "deadline_unix_ns": 10_000_000_000_000},
    )
    legacy = _request(ch, "legacy", mtime=101, timeout=1800)

    assert broker._pending_requests(ch, set(), now=102) == [legacy]
    assert abandoned.exists(), "restart recovery must preserve abandoned requests as audit evidence"

    old_v3 = _request(
        ch, "999998_456_nonce", mtime=103, timeout=1800,
        extra={"protocol": 3, "request_id": "999998_456_nonce",
               "deadline_unix_ns": 10_000_000_000_000},
    )
    assert broker._pending_requests(ch, set(), now=104) == [legacy]
    assert old_v3.exists()


def test_shim_request_identity_matches_its_live_process():
    shim = _module("selfcheck_shim")
    broker = _module("selfcheck_broker")

    identity = shim._requester_identity()

    assert identity["pid"] == os.getpid()
    assert identity["pid_namespace"] == broker._pid_namespace()
    assert broker._requester_alive({"requester": identity}) is True


def test_foreign_pid_namespace_cannot_be_judged_by_the_brokers_proc(monkeypatch):
    """A sandbox PID is not a host PID; its request keeps the deadline lease."""
    broker = _module("selfcheck_broker")
    monkeypatch.setattr(broker, "_pid_namespace", lambda: "pid:[host]")
    request = {"requester": {
        "pid": 999999,
        "start_ticks": 1,
        "pid_namespace": "pid:[sandbox]",
    }}

    assert broker._requester_alive(request) is None


def test_pre_namespace_identity_keeps_its_deadline_contract():
    """Protocol-v3 requests already on disk cannot safely be mapped to host /proc."""
    broker = _module("selfcheck_broker")
    request = {"requester": {"pid": 999999, "start_ticks": 1}}

    assert broker._requester_alive(request) is None


def test_full_suite_channel_budget_cannot_expire_before_per_capsule_work(tmp_path):
    """The grader's --timeout applies to every capsule, not to the whole parallel suite."""
    shim = _module("selfcheck_shim")
    per_capsule = 1800
    workers = 8
    n_capsules = 57

    budget = shim._request_budget_seconds(
        per_capsule, capsules="all", workers=workers, suite_size=n_capsules)

    # The suite has a short serial calibration head and then parallel worker waves.  The channel must
    # remain alive for at least the parallel work; the old `timeout + 240` budget was only 2,040 s.
    assert budget >= math.ceil(n_capsules / workers) * per_capsule + 240
    assert budget > per_capsule + 240


def test_broker_child_deadline_honors_the_request_deadline():
    broker = _module("selfcheck_broker")
    request = {"timeout": 1800, "deadline_unix_ns": 20_000_000_000}

    deadline = broker._child_deadline_monotonic(request, now_wall=10.0, now_monotonic=100.0)

    assert deadline == 110.0


def test_atomic_publish_replaces_content_and_leaves_no_partial_file(tmp_path):
    broker = _module("selfcheck_broker")
    path = tmp_path / "resp_id.json"
    path.write_text("stale response from a colliding legacy request")

    broker._atomic_write(path, '{"all_pass": true}')

    assert path.read_text() == '{"all_pass": true}'
    assert not list(tmp_path.glob(".*.tmp"))


def test_protocol_v3_binds_a_verdict_to_the_graded_submission(tmp_path):
    broker = _module("selfcheck_broker")
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "backend.py").write_text("version = 1\n")
    digest = broker._submission_digest(submission)
    req = {
        "protocol": 3,
        "request_id": "req-1",
        "requested_at_unix_ns": 10,
        "submission_sha256": digest,
    }

    doc = broker._decorate_response('{"all_pass": true}', req, "req-1", digest,
                                    started_ns=20, completed_ns=30)

    assert doc["all_pass"] is True
    assert doc["selfcheck_protocol"] == 3
    assert doc["selfcheck_request_id"] == "req-1"
    assert doc["submission_sha256"] == digest
    assert doc["requested_at_unix_ns"] == 10
    assert doc["graded_at_unix_ns"] == 30


def test_submission_digest_ignores_build_state_but_tracks_authored_sources(tmp_path):
    broker = _module("selfcheck_broker")
    submission = tmp_path / "submission"
    submission.mkdir()
    source = submission / "backend.py"
    source.write_text("version = 1\n")
    first = broker._submission_digest(submission)

    build = submission / "build/cache.bin"
    build.parent.mkdir()
    build.write_bytes(b"generated")
    assert broker._submission_digest(submission) == first

    source.write_text("version = 2\n")
    assert broker._submission_digest(submission) != first


def test_feedback_health_distinguishes_completed_expired_and_stranded(tmp_path):
    loop = _module("run_baseline_qa_loop")
    ch = tmp_path / ".qa_channel"
    ch.mkdir()
    _request(ch, "complete", mtime=10, timeout=30)
    (ch / "resp_complete.json").write_text("{}")
    (ch / "done_complete").write_text("ok")
    _request(ch, "expired", mtime=10, timeout=30)
    _request(ch, "live", mtime=100, timeout=1800)

    health = loop._feedback_health(tmp_path, now=500)

    assert health["requests"] == 3
    assert health["completed"] == 1
    assert health["expired"] == 1
    assert health["stranded"] == 1
    assert health["healthy"] is False


def test_feedback_health_refuses_an_expired_unanswered_request(tmp_path):
    loop = _module("run_baseline_qa_loop")
    ch = tmp_path / ".qa_channel"
    ch.mkdir()
    _request(ch, "expired", mtime=10, timeout=30)

    health = loop._feedback_health(tmp_path, now=500)

    assert health["expired"] == 1
    assert health["stranded"] == 0
    assert health["healthy"] is False


def test_broker_hides_the_childs_partial_response_until_it_is_complete(tmp_path, monkeypatch):
    """Exercise the real broker loop, not just the atomic-write helper."""
    broker = _module("selfcheck_broker")
    ws = tmp_path / "ws"
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True)
    fake = tmp_path / "fake_selfcheck.py"
    fake.write_text(
        "import argparse, os, time\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--out'); p.add_argument('--submission'); p.add_argument('--sim')\n"
        "p.add_argument('--capsules'); p.add_argument('--workers'); p.add_argument('--timeout')\n"
        "p.add_argument('--progress-out')\n"
        "a=p.parse_args()\n"
        "with open(a.progress_out, 'w') as f:\n"
        " f.write('{\\\"status\\\":\\\"running\\\",\\\"n_finished\\\":1}')\n"
        "with open(a.out, 'w') as f:\n"
        " f.write('{\\\"all_pass\\\":'); f.flush(); os.fsync(f.fileno()); time.sleep(.25)\n"
        " f.write(' true}'); f.flush(); os.fsync(f.fileno())\n"
    )
    monkeypatch.setattr(broker, "SELFCHECK", fake)
    req = _request(ch, "live", mtime=time.time(), timeout=30)
    thread = threading.Thread(target=broker.main, args=(["--ws", str(ws), "--poll", "0.01"],))
    thread.start()
    try:
        limit = time.monotonic() + 3
        while time.monotonic() < limit and not list(ch.glob(".resp_live.*.tmp")):
            time.sleep(.01)
        assert list(ch.glob(".resp_live.*.tmp")), "fake child never reached its partial write"
        assert json.loads((ch / "progress_live.json").read_text())["n_finished"] == 1
        assert not (ch / "resp_live.json").exists()
        assert not (ch / "done_live").exists()

        while time.monotonic() < limit and not (ch / "done_live").exists():
            time.sleep(.01)
        published = json.loads((ch / "resp_live.json").read_text())
        assert published["all_pass"] is True
        assert published["selfcheck_request_id"] == "live"
        assert published["submission_sha256"] == broker._submission_digest(ws / "submission")
        assert (ch / "done_live").read_text() == "ok"
    finally:
        (ch / "STOP").write_text("stop")
        # Promotion is intentionally off the response path but may already have started in the narrow
        # interval between publishing ``done`` and this STOP. Give that optional cleanup time to return.
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert req.exists()
