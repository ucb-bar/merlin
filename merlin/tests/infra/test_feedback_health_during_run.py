"""The feedback channel's health must be checked WHILE the run can still act on it.

`_feedback_health` already computed the right thing — expired requests, stranded ones, orphan
responses, broker restarts. It was called once, in the end-of-run summary, where it gates whether the
official grade counts as complete. By then the budget is spent.

Measured on merlincirct_atlas_feedback_v3_20260906 by hand while it was still running: `healthy:
false`, 16 expired, 3 stranded, 6 broker starts, and the last completed self-check 5.5 h earlier while
the agent kept editing. Every one of those was computable hours earlier by a function already written.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import pytest
from merlin_experiments.phase1.feedback import lifecycle as FL

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments" / "capsule_bench" / "harness"


@pytest.fixture
def loop():
    spec = importlib.util.spec_from_file_location("run_baseline_qa_loop", HARNESS / "run_baseline_qa_loop.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_baseline_qa_loop", m)
    sys.path.insert(0, str(HARNESS))
    spec.loader.exec_module(m)
    return m


def _req(ch, rid, *, mtime, timeout=30, done=False):
    p = ch / f"req_{rid}.json"
    p.write_text(json.dumps({"sim": "spike", "capsules": "all", "timeout": timeout}))
    os.utime(p, (mtime, mtime))
    if done:
        (ch / f"resp_{rid}.json").write_text("{}")
        (ch / f"done_{rid}").write_text("ok")
    return p


def test_an_unhealthy_channel_is_recorded_and_announced(loop, tmp_path, capsys):
    ws, run_dir = tmp_path / "ws", tmp_path / "run"
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True)
    run_dir.mkdir()
    _req(ch, "expired", mtime=10)  # long past its timeout, never answered
    FL.record_channel_health(ws, run_dir)

    doc = json.loads((run_dir / "feedback_health.json").read_text())
    assert doc["healthy"] is False and doc["expired"] >= 1
    assert "UNHEALTHY" in capsys.readouterr().out


def test_a_healthy_channel_is_recorded_and_quiet(loop, tmp_path, capsys):
    """The paired direction: a check that always warns is noise and gets ignored."""
    ws, run_dir = tmp_path / "ws", tmp_path / "run"
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True)
    run_dir.mkdir()
    _req(ch, "ok1", mtime=time.time(), done=True)
    FL.record_channel_health(ws, run_dir)

    doc = json.loads((run_dir / "feedback_health.json").read_text())
    assert doc["healthy"] is True
    assert "UNHEALTHY" not in capsys.readouterr().out


def test_it_is_checked_during_the_run_not_only_at_the_end(loop):
    """The whole defect. Both operator-side grade paths must record it, or a continuous run — which
    has no rounds — reports nothing until it is over."""
    src = (HARNESS / "run_baseline_qa_loop.py").read_text()
    assert src.count("FL.record_channel_health(ws, run_dir)") >= 2, (
        "health must be recorded where grades land, not only in the end-of-run summary"
    )
    # and the end-of-run use must still be there: it gates whether the official grade counts
    assert "feedback_health = FL.channel_health(ws)" in src


def test_it_never_reaches_the_agent(loop):
    """Telling the agent its feedback channel is broken is feedback, and feedback defines an arm."""
    import inspect

    body = inspect.getsource(FL.record_channel_health)
    assert 'run_dir / "feedback_health.json"' in body
    assert 'ws / "qa"' not in body, "it must never be written into the agent's workspace"
    assert "feedback defines an arm" in body


def test_a_diagnostic_failure_cannot_fail_a_grade(loop, tmp_path):
    """It runs inside the grade path, so it must be incapable of breaking one."""
    FL.record_channel_health(tmp_path / "absent", tmp_path / "also_absent")


def test_broker_restarts_are_surfaced(loop, tmp_path, capsys):
    """Six restarts is the signal that distinguished 'slow' from 'broken' on the run that prompted
    this — a channel can look merely busy while its server keeps dying."""
    ws, run_dir = tmp_path / "ws", tmp_path / "run"
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True)
    run_dir.mkdir()
    _req(ch, "expired", mtime=10)
    (ch / "broker_health.json").write_text(json.dumps({"broker_starts": 6, "processed": 3}))
    FL.record_channel_health(ws, run_dir)
    out = capsys.readouterr().out
    assert "broker start" in out
    assert json.loads((run_dir / "feedback_health.json").read_text())["broker_starts"] == 6


# --- the denominator is not a directory listing -----------------------------------------------------
# `requests` globbed the req_ files still on disk and `completed` was a SUBSET of it, so an exchange
# whose request file was removed left the numerator and the denominator together and became invisible.
# MEASURED on the live rcp_model_layers_20260917_r1 channel: the recorded file read
# "requests: 61, completed: 61" -- a perfect score -- while the directory held 67 req_, 72 resp_ and
# 72 done_. Five delivered exchanges the ratio could not see.


def _delivered_without_request(ch, rid):
    """A completed exchange whose req_ file is gone: the shape the old count could not represent."""
    (ch / f"resp_{rid}.json").write_text("{}")
    (ch / f"done_{rid}").write_text("ok")


def test_an_exchange_survives_the_loss_of_its_request_file(loop, tmp_path):
    ws = tmp_path / "ws"
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True)
    _req(ch, "kept", mtime=time.time(), done=True)
    _delivered_without_request(ch, "vanished")

    h = FL.channel_health(ws)
    assert h["requests"] == 2, "a response or a done proves a request existed"
    assert h["completed"] == 2
    assert h["requests_on_disk"] == 1, "the listing is still reported, as a listing"
    assert h["completed_without_request"] == 1


def test_a_deleted_request_cannot_manufacture_a_perfect_score(loop, tmp_path):
    """The failure this hides: a stranded exchange whose request file is removed must not vanish."""
    ws = tmp_path / "ws"
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True)
    _req(ch, "ok", mtime=time.time(), done=True)
    (ch / "resp_halfway.json").write_text("{}")  # answered, never marked done, request gone

    h = FL.channel_health(ws)
    assert h["requests"] == 2 and h["completed"] == 1
    assert h["stranded"] == 1
    assert h["healthy"] is False, "an undelivered exchange must still fail the channel"


def test_a_completed_orphan_does_not_fail_the_channel(loop, tmp_path):
    """A response and a done together ARE the delivery this gate checks. It stays reported, because
    something removed a request file and that is worth seeing, but it is not an undelivered exchange."""
    ws = tmp_path / "ws"
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True)
    _delivered_without_request(ch, "vanished")

    h = FL.channel_health(ws)
    assert h["orphan_responses"] == 1 and h["completed_without_request"] == 1
    assert h["healthy"] is True


def test_a_done_with_no_response_is_still_unhealthy(loop, tmp_path):
    """The mutation: the other malformed pair must keep failing."""
    ws = tmp_path / "ws"
    ch = ws / ".qa_channel"
    ch.mkdir(parents=True)
    (ch / "done_lonely").write_text("ok")

    h = FL.channel_health(ws)
    assert h["done_without_response"] == 1 and h["healthy"] is False
