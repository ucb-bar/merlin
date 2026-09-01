"""The recommended run shape is the DEFAULT run shape, and the ladder that makes it work is enforced.

An operator should not have to know a flag to get the behaviour the experiment is designed around:

  * ONE long-lived agent session, re-graded underneath it -- not round relaunches that discard the
    agent's context at every barrier and defer every verdict to the next one;
  * per-capsule tiering, so a capsule that clears the loop tier goes to the certifying tier
    IMMEDIATELY -- capsule 2's L3 starts while the agent is still working on capsule 1's L2;
  * and the certificate that costs minutes of RTL is KEPT, not discarded.

Each of those was broken or off-by-default at some point on merlincirct_arm4_func_20260901_v4, and
each failure was invisible: a round barrier looks like slow progress, a promotion that never fires
looks like "nothing needed promoting", and a discarded cert looks like a capsule that is still pending.
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


def test_continuous_mode_is_on_without_asking_for_it():
    """A run launched with no mode flag must be the continuous one."""
    src = (HARNESS / "run_baseline_qa_loop.py").read_text(encoding="utf-8")
    idx = src.index('"--continuous"')
    decl = src[idx:idx + 400]
    assert "BooleanOptionalAction" in decl and "default=True" in decl, (
        "--continuous is not default-on, so an operator who does not know the flag gets round "
        "relaunches: the agent's context is discarded at every barrier and each verdict waits for "
        "the next one")


def test_the_legacy_round_mode_is_still_reachable():
    """Default-on must not remove the ability to reproduce an old round-based run."""
    src = (HARNESS / "run_baseline_qa_loop.py").read_text(encoding="utf-8")
    assert "BooleanOptionalAction" in src[src.index('"--continuous"'):][:400], (
        "there is no --no-continuous escape hatch")


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
