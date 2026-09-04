"""The recommended run shape, and the per-capsule ladder that makes it work, are enforced.

An operator should not have to know a flag to get the behaviour the experiment is designed around:

  * the CERTIFIED continuous path (`--schedule continuous` with a long `--round-timeout`), where the
    round count is not a terminator and the post-freeze public+hidden L3 grade still runs -- NOT the
    legacy `--continuous` single-session path, which reports progress and can never report a formal
    success;
  * per-capsule tiering, so a capsule that clears the loop tier goes to the certifying tier
    IMMEDIATELY -- capsule 2's L3 starts while the agent is still working on capsule 1's L2;
  * and the certificate that costs minutes of RTL is KEPT, not discarded.

Each of those was broken or off-by-default at some point on merlincirct_arm4_func_20260901_v4, and
each failure was invisible: a round barrier looks like slow progress, a promotion that never fires
looks like "nothing needed promoting", and a discarded cert looks like a capsule that is still pending.

The MECHANISM of the first property -- that `--schedule continuous` actually runs one session with a
background grader under it -- is gated in `test_continuous_schedule_grades_under_the_agent.py`. The
tests here check the FLAGS, and for a long time that was the whole gate: `--continuous` stayed opt-in
and the round cap stayed lifted, while the certified schedule quietly ran the round loop with no
grader at all. A flag test cannot see that; keep both files.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys

import pytest

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _mod(name: str):
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 -- harness deps absent in this env
        pytest.skip(f"{name} not importable here: {type(exc).__name__}: {exc}")
    return mod


def test_the_legacy_single_session_path_is_not_the_default():
    """`--continuous` must stay opt-in, because it can never report a formal success.

    It keeps one agent session and re-grades underneath it, which sounds like the shape we want -- but
    it does NOT run the post-freeze public+hidden L3 grade: it returns 1 and hardcodes
    `formal_complete=False`. Measured 2026-09-01: launched with `--continuous`, both gemmini sessions
    closed after ~1.5h at 18/33 with `grades=2` when the AGENT stopped, well inside a 12h
    --round-timeout, and no formal verdict was reachable. Defaulting to it would make formal success
    impossible for every run.
    """
    src = (HARNESS / "run_baseline_qa_loop.py").read_text(encoding="utf-8")
    decl = src[src.index('"--continuous"'):][:400]
    assert "action=\"store_true\"" in decl, (
        "--continuous is no longer opt-in; the legacy progress-only path must not be the default "
        "because it cannot run the post-freeze public+hidden L3 grade")


def test_the_certified_continuous_path_still_ignores_the_round_cap():
    """`--schedule continuous` is the certified path: rounds are not a terminator.

    A run must stop on EVIDENCE (converged, plateaued) or on a declared budget -- never because an
    arithmetic round cap ran out while the submission was still improving.
    """
    src = (HARNESS / "run_baseline_qa_loop.py").read_text(encoding="utf-8")
    assert 'a.schedule == "rounds"' in src, "the rounds/continuous distinction is gone"
    assert "1_000_000" in src, (
        "continuous mode no longer lifts the round cap, so a productive run can be cut at --max-rounds")


def test_a_loop_pass_still_enqueues_the_cert_tier_immediately():
    """The per-capsule promise: clearing the loop tier launches the cert tier, not next round."""
    TP = _mod("tier_promote")
    assert hasattr(TP, "promote"), "promotion entry point is gone"
    assert hasattr(TP, "cert_sim"), (
        "cert_sim is gone: promotion would fall back to a sim the broker rejects")
    assert hasattr(TP, "record_cert"), (
        "record_cert is gone: a completed promotion's certificate would be discarded again")


def test_the_broker_records_a_completed_promotion():
    """Wiring, not just presence -- promotion's try/except hides a missing call as 'nothing to do'."""
    src = (HARNESS / "simjob_broker.py").read_text(encoding="utf-8")
    assert "_TP.record_cert(" in src, "the broker never records a completed promotion"
    reap = src[src.index('if not j.get("promoted")'):]
    assert 'elif j.get("promoted")' in reap[:800], (
        "record_cert is not on the promoted-job branch of the reap")


def test_the_cert_sim_is_one_the_broker_accepts():
    """A cert job must name a sim the broker will run; it used to name a rejected sentinel."""
    TP = _mod("tier_promote")
    SB = _mod("simjob_broker")
    allowed = tuple(SB._allowed_sims())
    sim = TP.cert_sim("L3")
    assert sim is None or sim in allowed, (
        f"promotion would enqueue --sim {sim!r}, which this broker rejects; it accepts {allowed}")
