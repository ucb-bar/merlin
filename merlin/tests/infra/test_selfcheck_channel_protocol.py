"""Restart-safe request handling for the sandbox self-check channel.

The channel is intentionally file based, so its durable files outlive the broker.  A restarted broker
must distinguish an outstanding request from the completed and abandoned requests left by earlier
clients; otherwise one fresh self-check can sit behind hours of replayed grading work.
"""
from __future__ import annotations

import importlib.util
import json
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


def _request(ch: Path, rid: str, *, mtime: float, timeout: int = 1800) -> Path:
    req = ch / f"req_{rid}.json"
    req.write_text(json.dumps({"sim": "spike", "capsules": "all", "timeout": timeout}))
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


def test_atomic_publish_replaces_content_and_leaves_no_partial_file(tmp_path):
    broker = _module("selfcheck_broker")
    path = tmp_path / "resp_id.json"
    path.write_text("stale response from a colliding legacy request")

    broker._atomic_write(path, '{"all_pass": true}')

    assert path.read_text() == '{"all_pass": true}'
    assert not list(tmp_path.glob(".*.tmp"))


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
        "a=p.parse_args()\n"
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
        assert not (ch / "resp_live.json").exists()
        assert not (ch / "done_live").exists()

        while time.monotonic() < limit and not (ch / "done_live").exists():
            time.sleep(.01)
        assert json.loads((ch / "resp_live.json").read_text()) == {"all_pass": True}
        assert (ch / "done_live").read_text() == "ok"
    finally:
        (ch / "STOP").write_text("stop")
        thread.join(timeout=3)
    assert not thread.is_alive()
    assert req.exists()
