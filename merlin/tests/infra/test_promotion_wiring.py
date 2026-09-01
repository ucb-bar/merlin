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
    """Import the shared promotion module by path — it is a harness script, not installed."""
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location("tier_promote", HARNESS / "tier_promote.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001 — harness deps absent in this env
        pytest.skip(f"tier_promote not importable here: {type(e).__name__}: {e}")
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
    promoted = B.promote(ws, ch, _verdict([("A", True)]), "L2", "L3", None, sys.stderr)
    assert promoted == ["A"]
    reqs = list(ch.glob("simreq_*.json"))
    assert len(reqs) == 1
    r = json.loads(reqs[0].read_text())
    # The job asks for the LOOP tier beside the cert tier. Requesting "L3" alone dropped the loop tier
    # the capsule declares mandatory, and an unreached mandatory tier is scored NOT_RUN_IS_NOT_PASS -- so
    # the response came back `pass: false` however the cert itself went, and no capsule could ever be
    # recorded as certified. See test_mandatory_tiers_are_never_dropped.py for the measured case.
    assert r["capsules"] == "A" and r["tiers"] == "L2,L3" and r["promoted"] is True


def test_a_failing_capsule_buys_no_cert_time(tmp_path):
    """A capsule whose numerics are wrong cannot be rescued by RTL, and RTL costs minutes."""
    B = _broker()
    ws = _ws(tmp_path)
    assert B.promote(ws, ws / ".qa_channel", _verdict([("A", False)]), "L2", "L3", None, sys.stderr) == []
    assert list((ws / ".qa_channel").glob("simreq_*.json")) == []


def test_outside_the_cover_is_not_promoted(tmp_path):
    B = _broker()
    ws = _ws(tmp_path)
    got = B.promote(ws, ws / ".qa_channel", _verdict([("A", True), ("B", True)]),
                     "L2", "L3", {"A"}, sys.stderr)
    assert got == ["A"]


def test_the_same_bytes_are_never_certified_twice(tmp_path):
    """Content-addressing is what makes continuous grading affordable: a second identical verdict must
    enqueue nothing, or the loop re-certifies forever."""
    B = _broker()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    first = B.promote(ws, ch, _verdict([("A", True)]), "L2", "L3", None, sys.stderr)
    second = B.promote(ws, ch, _verdict([("A", True)]), "L2", "L3", None, sys.stderr)
    assert first == ["A"] and second == []


def test_changed_bytes_re_certify(tmp_path):
    """...and the converse: an edit must invalidate the cert verdict it earned, or a stale RTL pass
    stands for code that no longer exists."""
    B = _broker()
    ws = _ws(tmp_path)
    ch = ws / ".qa_channel"
    assert B.promote(ws, ch, _verdict([("A", True)]), "L2", "L3", None, sys.stderr) == ["A"]
    (ws / "submission" / "manifest.yaml").write_text("x: 2")          # the submission moved on
    assert B.promote(ws, ch, _verdict([("A", True)]), "L2", "L3", None, sys.stderr) == ["A"]


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
    B.promote(ws, ws / ".qa_channel", _verdict([("A", True), ("B", False)]), "L2", "L3", None, sys.stderr)
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
        assert B.promote(ws, ws / ".qa_channel", junk, "L2", "L3", None, sys.stderr) == []


# ---------------------------------------------------------------------------------------------
# both brokers must promote — hooking one was the original bug
# ---------------------------------------------------------------------------------------------
def test_both_brokers_call_promotion():
    """Promotion was first wired into the async oracle only. A live run then showed the agent using the
    SYNC self-check 7 times to the async path's 2, so eight verdicts completed and promotion fired zero
    times. Whichever broker produces a verdict must consider promotion, or the feature is dead on the
    path that matters."""
    for name in ("simjob_broker.py", "selfcheck_broker.py", "run_baseline_qa_loop.py"):
        src = (HARNESS / name).read_text(encoding="utf-8")
        assert "tier_promote" in src, f"{name} does not reach the shared promotion module"
        assert "promote" in src, f"{name} never calls promotion"


def test_the_round_grade_promotes_too():
    """A broker only sees a verdict the agent ASKED for, and a converged agent stops asking: measured 24
    self-checks in round 0, then ZERO in rounds 1 and 2 once the corpus ceiling was reached. The round
    grade was then the only verdict produced, and promotion had nothing to fire on -- so the deeper tier
    would only ever be reached while the agent was still struggling, which is backwards. A converged
    submission is the one worth certifying."""
    src = (HARNESS / "run_baseline_qa_loop.py").read_text(encoding="utf-8")
    i = src.index("def qa_grade(")
    j = src.index("def _write_stage_ledger(")
    assert "tier_promote" in src[i:j], "qa_grade does not promote"


def test_the_policy_is_sourced_not_reimplemented():
    """One policy, one place. The brokers are plumbing; `oracle_schedule` decides."""
    src = (HARNESS / "tier_promote.py").read_text(encoding="utf-8")
    assert "oracle_schedule" in src


# ---------------------------------------------------------------------------------------------
# a rejection must carry a remedy
# ---------------------------------------------------------------------------------------------
def test_a_rejection_names_the_field_and_the_remedy():
    """An agent submitted to the async oracle twice, was rejected twice with "bad sim or capsule
    (constrained runner)", and never used it again -- while the arm that DID reach the async path used it
    98 times in the round its compiler-earned score moved 17 -> 26. The check is load-bearing isolation
    and stays; what changes is that the refusal says which field was wrong and what would be accepted.
    Asserted on the source because the branch needs a live broker loop to reach."""
    src = (HARNESS / "simjob_broker.py").read_text(encoding="utf-8")
    assert "rejected: bad sim or capsule (constrained runner)" not in src, "the remedy-free message is back"
    assert "rejected_field" in src, "a rejection must say WHICH field it refused"
    assert "--tiers" in src, "a neutral-sim target must be told how to choose a tier"
