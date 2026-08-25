"""The broker must actually promote — the policy being right is not the same as the wiring firing.

`oracle_schedule` is tested as pure policy and `cert_capsule_cover` as pure selection, but neither proves
that a real verdict landing in the broker's reap produces a real cert-tier job. That gap is exactly where
this kind of change dies quietly: promotion is wrapped in a `try/except` so it can never gate a run, so a
broken wiring would show up as *nothing happening* — indistinguishable from a corpus where nothing needed
promoting.

These tests drive the broker's own functions against a temp workspace, no oracle and no agent.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _broker():
    """Import simjob_broker.py by path — it is a harness script, not an installed module."""
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location("simjob_broker", HARNESS / "simjob_broker.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001 — harness deps absent in this env
        pytest.skip(f"simjob_broker not importable here: {type(e).__name__}: {e}")
    return mod


def _ws(tmp_path, files=(("submission/manifest.yaml", "x: 1"),)):
    ws = tmp_path / "ws"
    for rel, body in files:
        f = ws / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body)
    (ws / ".qa_channel").mkdir(parents=True, exist_ok=True)
    return ws


def _verdict(rows):
    return {"per_capsule": [{"capsule": n, "pass": p} for n, p in rows]}


# ---------------------------------------------------------------------------------------------
def test_a_passing_capsule_produces_a_cert_job(tmp_path):
    """The whole point: an L2 pass enqueues an L3 job, with no agent involvement."""
    B = _broker()
    ws, ch = _ws(tmp_path), None
    ch = ws / ".qa_channel"
    promoted = B._promote(ws, ch, _verdict([("A", True)]), "L2", "L3", None, sys.stderr)
    assert promoted == ["A"]
    reqs = list(ch.glob("simreq_*.json"))
    assert len(reqs) == 1
    r = json.loads(reqs[0].read_text())
    assert r["capsules"] == "A" and r["tiers"] == "L3" and r["promoted"] is True


def test_a_failing_capsule_buys_no_cert_time(tmp_path):
    """A capsule whose numerics are wrong cannot be rescued by RTL, and RTL costs minutes."""
    B = _broker()
    ws = _ws(tmp_path)
    assert B._promote(ws, ws / ".qa_channel", _verdict([("A", False)]), "L2", "L3", None, sys.stderr) == []
    assert list((ws / ".qa_channel").glob("simreq_*.json")) == []


def test_outside_the_cover_is_not_promoted(tmp_path):
    B = _broker()
    ws = _ws(tmp_path)
    got = B._promote(ws, ws / ".qa_channel", _verdict([("A", True), ("B", True)]),
                     "L2", "L3", {"A"}, sys.stderr)
    assert got == ["A"]


def test_the_same_bytes_are_never_certified_twice(tmp_path):
    """Content-addressing is what makes continuous grading affordable: a second identical verdict must
    enqueue nothing, or the loop re-certifies forever."""
    B = _broker()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    first = B._promote(ws, ch, _verdict([("A", True)]), "L2", "L3", None, sys.stderr)
    second = B._promote(ws, ch, _verdict([("A", True)]), "L2", "L3", None, sys.stderr)
    assert first == ["A"] and second == []


def test_changed_bytes_re_certify(tmp_path):
    """...and the converse: an edit must invalidate the cert verdict it earned, or a stale RTL pass
    stands for code that no longer exists."""
    B = _broker()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    assert B._promote(ws, ch, _verdict([("A", True)]), "L2", "L3", None, sys.stderr) == ["A"]
    (ws / "submission" / "manifest.yaml").write_text("x: 2")          # the submission moved on
    assert B._promote(ws, ch, _verdict([("A", True)]), "L2", "L3", None, sys.stderr) == ["A"]


def test_the_digest_tracks_content_not_time(tmp_path):
    B = _broker()
    ws = _ws(tmp_path)
    d1 = B._submission_digest(ws)
    assert B._submission_digest(ws) == d1                              # stable for identical bytes
    (ws / "submission" / "manifest.yaml").write_text("x: 2")
    assert B._submission_digest(ws) != d1


def test_tier_state_records_both_tiers(tmp_path):
    """The state file is what a continuous loop reads to decide what is left to do; a promotion that does
    not record `pending` would be re-enqueued on every reap."""
    B = _broker()
    ws = _ws(tmp_path)
    B._promote(ws, ws / ".qa_channel", _verdict([("A", True), ("B", False)]), "L2", "L3", None, sys.stderr)
    st = json.loads((ws / "qa" / "tier_state.json").read_text())
    assert st["A"]["L2"]["status"] == "pass"
    assert st["A"]["L3"]["status"] == "pending"
    assert st["B"]["L2"]["status"] == "fail"
    assert "L3" not in st["B"]


def test_promotion_never_gates_a_run(tmp_path):
    """It is an optimisation. A malformed verdict must cost nothing but a log line -- the reap wraps this,
    and this asserts the function itself does not explode on junk."""
    B = _broker()
    ws = _ws(tmp_path)
    for junk in ({}, {"per_capsule": None}, {"per_capsule": [{}]}, {"per_capsule": [{"capsule": None}]}):
        assert B._promote(ws, ws / ".qa_channel", junk, "L2", "L3", None, sys.stderr) == []
