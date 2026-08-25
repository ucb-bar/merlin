"""The scheduler that replaces the round barrier: run what buys the most information per second.

Measured on a live A/B, the round loop wasted time three ways at once: it re-certified capsules whose
bytes had not changed, it paid the cycle-accurate tier for capsules whose numerics had already failed,
and it batched every verdict to a round boundary hours away. Together those were 80% of an agent round,
and 34% of that round returned no verdict at all.

These tests pin the policy, not the plumbing. The three rules:

  1. a verdict is keyed by the BYTES that produced it, so unchanged work is never re-run and changed work
     invalidates exactly what it touched;
  2. a deep tier is gated on the shallow tier PASSING, so RTL seconds only go to work that could certify;
  3. the deep tier runs a representative cover, because the hardware cannot tell two capsules in the same
     (family, dtype) cell apart.

And the rule that keeps it honest: nothing is dropped silently. A scheduler that quietly stops scheduling
is indistinguishable from one that has finished.
"""
from __future__ import annotations

from merlin.targetgen.oracle_schedule import CapsuleState, Verdict, explain, schedule

TIERS = ["L2", "L3"]
CERT = ("L3",)
COST = {"L2": 2.5, "L3": 300.0}          # measured order of magnitude: seconds vs minutes


def _st(name, digest="d1", **verdicts):
    return CapsuleState(name=name, digest=digest,
                        verdicts={t: Verdict(s, digest) for t, s in verdicts.items()})


# ---------------------------------------------------------------------------------------------
# 1. never re-run what did not change
# ---------------------------------------------------------------------------------------------
def test_unchanged_bytes_are_never_rescheduled():
    """The single largest saving. Today every sweep rebuilds and re-grades all 35 capsules because one
    changed; content-addressing the verdict makes that cost disappear."""
    states = [_st("A", L2="pass", L3="pass"), _st("B", L2="fail")]
    assert schedule(states, tier_order=TIERS, cert_tiers=CERT, cost_s=COST) == []


def test_changed_bytes_invalidate_only_that_capsule():
    done = _st("A", L2="pass", L3="pass")
    edited = CapsuleState("B", digest="d2", verdicts={"L2": Verdict("pass", "d1")})   # bytes moved on
    q = schedule([done, edited], tier_order=TIERS, cert_tiers=CERT, cost_s=COST)
    assert [(w.capsule, w.tier) for w in q] == [("B", "L2")]


def test_an_l3_driven_edit_reopens_l2():
    """The ratchet guard: a fix made to satisfy RTL can break numerics, so changed bytes must re-open the
    functional tier rather than letting a stale L2 pass stand as if it still held."""
    s = CapsuleState("A", digest="d2",
                     verdicts={"L2": Verdict("pass", "d1"), "L3": Verdict("fail", "d1")})
    q = schedule([s], tier_order=TIERS, cert_tiers=CERT, cost_s=COST)
    assert [(w.capsule, w.tier) for w in q] == [("A", "L2")]


# ---------------------------------------------------------------------------------------------
# 2. the cheap tier gates the expensive one
# ---------------------------------------------------------------------------------------------
def test_a_failed_shallow_tier_never_buys_rtl_time():
    """The short-circuit, at the scheduling level: a capsule that is already failed cannot be rescued by
    a cycle-accurate verdict, and that verdict costs minutes."""
    q = schedule([_st("A", L2="fail")], tier_order=TIERS, cert_tiers=CERT, cost_s=COST)
    assert q == []


def test_an_unknown_shallow_tier_defers_the_deep_one():
    q = schedule([_st("A")], tier_order=TIERS, cert_tiers=CERT, cost_s=COST)
    assert [(w.capsule, w.tier) for w in q] == [("A", "L2")]


def test_a_passing_shallow_tier_promotes_immediately():
    """The user-facing behaviour: the moment a capsule passes the functional tier it is promoted, without
    waiting for a round boundary or for any other capsule."""
    q = schedule([_st("A", L2="pass")], tier_order=TIERS, cert_tiers=CERT, cost_s=COST)
    assert [(w.capsule, w.tier) for w in q] == [("A", "L3")]
    assert "passed" in q[0].reason


# ---------------------------------------------------------------------------------------------
# 3. the deep tier runs a representative cover
# ---------------------------------------------------------------------------------------------
def test_outside_the_cover_is_not_certified():
    q = schedule([_st("A", L2="pass"), _st("B", L2="pass")],
                 tier_order=TIERS, cert_tiers=CERT, cert_cover={"A"}, cost_s=COST)
    assert [(w.capsule, w.tier) for w in q] == [("A", "L3")]


def test_no_cover_is_permissive_not_silent():
    """`cert_cover=None` means 'no cover computed'. It must certify anything eligible: a missing cover is
    a caller bug, and certifying NOTHING would look exactly like everything already being done."""
    q = schedule([_st("A", L2="pass")], tier_order=TIERS, cert_tiers=CERT, cert_cover=None, cost_s=COST)
    assert [(w.capsule, w.tier) for w in q] == [("A", "L3")]


# ---------------------------------------------------------------------------------------------
# ordering and budget
# ---------------------------------------------------------------------------------------------
def test_cheap_gating_work_outranks_expensive_work():
    """Information per second: an unknown L2 costs ~2.5 s and can unlock a promotion; an unknown L3 costs
    minutes. Doing the cheap one first strictly dominates."""
    q = schedule([_st("Slow", L2="pass"), _st("Fast")],
                 tier_order=TIERS, cert_tiers=CERT, cost_s=COST)
    assert [(w.capsule, w.tier) for w in q] == [("Fast", "L2"), ("Slow", "L3")]


def test_the_order_is_deterministic():
    states = [_st(n) for n in ("C", "A", "B")]
    a = [w.key for w in schedule(states, tier_order=TIERS, cert_tiers=CERT, cost_s=COST)]
    b = [w.key for w in schedule(states, tier_order=TIERS, cert_tiers=CERT, cost_s=COST)]
    assert a == b == [("A", "L2"), ("B", "L2"), ("C", "L2")]


def test_a_budget_defers_rather_than_truncates_silently():
    states = [_st("A", L2="pass"), _st("B", L2="pass"), _st("C", L2="pass")]
    rep = explain(states, tier_order=TIERS, cert_tiers=CERT, cost_s=COST, budget_s=650.0)
    assert len(rep["queue"]) == 2
    assert rep["deferred_over_budget"] == [("C", "L3")]      # named, not vanished


# ---------------------------------------------------------------------------------------------
# nothing disappears quietly
# ---------------------------------------------------------------------------------------------
def test_every_exclusion_is_named_and_counted():
    """A quiet scheduler and a finished scheduler look identical in a log. They must not."""
    states = [_st("done", L2="pass", L3="pass"),
              _st("failed", L2="fail"),
              _st("uncovered", L2="pass")]
    rep = explain(states, tier_order=TIERS, cert_tiers=CERT, cert_cover={"done"}, cost_s=COST)
    assert rep["queue"] == []
    assert rep["unchanged"] == ["done"]
    assert ("failed", "L3") in rep["blocked_on_shallower_tier"]
    assert ("uncovered", "L3") in rep["outside_cert_cover"]


def test_queued_cost_is_reported_so_a_run_can_be_priced():
    rep = explain([_st("A"), _st("B", L2="pass")], tier_order=TIERS, cert_tiers=CERT, cost_s=COST)
    assert rep["queued_cost_s"] == 302.5
