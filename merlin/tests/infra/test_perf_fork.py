"""The fork protocol: what a performance run forked from, and what it is forbidden to spend.

Phase F proves a compiler functionally complete; Phase P forks that exact compiler and optimizes it.
Two things then have to be true and neither is true by default:

  * an edit made by the optimizer must re-prove the SMALL functional capsules that ride on the
    component it touched -- not the corpus (a whole-submission digest kills every certificate per
    edit, and the cycle-accurate tier is minutes each) and not the large performance workload, which
    is not a functional capsule at all;
  * a Phase-F invariant is HELD only when the candidate's OWN bytes re-earned it. "Nobody re-ran it"
    is a third state, and folding it into "held" lets a certificate earned by the functional compiler
    stand for an optimizer that has since rewritten the component it depended on.

``test_the_invalidation_matrix`` is the falsifier here: it touches each component in turn and asserts
the exact set of capsules that lost their certificate. A matrix where every touch invalidates
everything -- or nothing -- would prove nothing, so the assertion is on the whole matrix.
"""
from __future__ import annotations

import pytest

from merlin.perf.falsifier import (
    ACCEPT, REJECT, UNDETERMINABLE as ETA_UNDETERMINABLE, ab_decision, eta_from_occupancy,
)
from merlin.perf.fork import (
    HELD, UNDETERMINABLE, WEAKENED, ForkPoint, candidate_states, changed_components,
    check_invariants, fork_from, fork_from_dict, requeue,
)
from merlin.targetgen.oracle_schedule import (
    CHANGED, FAIL, PASS, UNATTRIBUTED, CapsuleState, Verdict,
)

TIERS = ["L2", "L3"]
CERT = ("L3",)

# Component names are the submission manifest's own command keys; they are DERIVED there and merely
# quoted here (a test may name the contract it is testing). The performance workload below is
# deliberately not one of the functional capsules.
C1, C2, C3 = "parse", "emit_command_buffer", "lower_target_to_llvm"
FORK_COMPONENTS = {C1: "aaa1", C2: "bbb1", C3: "ccc1", UNATTRIBUTED: "zzz1"}
FORK_DIGEST = "sub-v1"

# Small functional capsules with declared dependencies, plus one that declares nothing (which means
# "depends on everything", the fail-closed reading).
DEPENDS = {"cap_parse": (C1,), "cap_cmdbuf": (C2,), "cap_codegen": (C3,), "cap_undeclared": None}
#: Minutes-per-capsule at the cert tier; the workload is an order of magnitude worse.
COST = {"L2": 3.0, "L3": 120.0}
WORKLOAD = "workload_gemm"


def _phase_f(digest=FORK_DIGEST, components=None, statuses=None) -> list[CapsuleState]:
    """The Phase-F capsules as they stood when the compiler was declared functionally complete."""
    comps = dict(FORK_COMPONENTS if components is None else components)
    out = []
    for name, deps in DEPENDS.items():
        st = (statuses or {}).get(name, {"L2": PASS, "L3": PASS})
        out.append(CapsuleState(
            name=name, digest=digest, components=comps, depends_on=deps,
            verdicts={t: Verdict(status=s, digest=digest, components=comps)
                      for t, s in st.items()}))
    return out


def _fork(**kw) -> ForkPoint:
    return fork_from(_phase_f(**kw), tier_order=TIERS, recorded_at="20260830T000000Z")


def _touch(component: str, digest="sub-v2") -> tuple[str, dict]:
    """The candidate's bytes after the optimizer edited exactly one component."""
    comps = dict(FORK_COMPONENTS)
    comps[component] = comps[component] + "-edited"
    return digest, comps


# -------------------------------------------------------------------------------------------------
# 1. the record: what was forked, pinned by content
# -------------------------------------------------------------------------------------------------
def test_the_fork_pins_the_submission_by_content_and_by_component():
    f = _fork()
    assert f.digest == FORK_DIGEST
    assert f.components == FORK_COMPONENTS
    assert f.capsules == tuple(sorted(DEPENDS))
    assert f.invariants["cap_parse"] == {"L2": PASS, "L3": PASS}
    assert f.depends_on["cap_parse"] == (C1,)
    assert "cap_undeclared" not in f.depends_on          # declared nothing -> nothing recorded


def test_a_stale_certificate_is_not_promoted_to_an_invariant_by_forking():
    """The verdict sits in the Phase-F state but was earned by OTHER bytes. Inheriting it would hand
    Phase P a correctness budget Phase F never actually established."""
    states = _phase_f()
    stale = states[0]
    stale.verdicts["L3"] = Verdict(status=PASS, digest="some-older-submission",
                                   components={C1: "different"})
    f = fork_from(states, tier_order=TIERS, recorded_at="20260830T000000Z")
    assert f.invariants[stale.name] == {"L2": PASS}      # L3 dropped: it is not about these bytes


def test_a_failing_capsule_contributes_no_invariant():
    f = _fork(statuses={"cap_parse": {"L2": FAIL}})
    assert f.invariants["cap_parse"] == {}


def test_a_fork_over_two_submissions_raises():
    states = _phase_f()
    states[1].digest = "some-other-submission"
    with pytest.raises(ValueError, match="different submission"):
        fork_from(states, tier_order=TIERS)


def test_the_record_round_trips():
    f = _fork()
    assert fork_from_dict(f.to_dict()) == f


def test_an_undecomposable_submission_falls_back_to_the_whole_digest():
    """Empty components is UNDETERMINABLE, never 'this submission declares no components'."""
    moved = changed_components(_fork(components={}), {})
    assert [(s.component, s.reason) for s in moved] == [("<whole-submission>", UNDETERMINABLE)]


def test_changed_components_names_what_moved():
    digest, comps = _touch(C2)
    moved = changed_components(_fork(), comps)
    assert [(s.component, s.reason) for s in moved] == [(C2, CHANGED)]


# -------------------------------------------------------------------------------------------------
# 2. the requeue: small capsules, never the workload
# -------------------------------------------------------------------------------------------------
def _queued(component):
    digest, comps = _touch(component)
    r = requeue(_fork(), digest=digest, components=comps, tier_order=TIERS, cert_tiers=CERT,
                cert_cover={WORKLOAD, *DEPENDS}, cost_s=COST)
    return r, {q["capsule"] for q in r["queue"]}


def test_the_invalidation_matrix():
    """Touch each component in turn; assert the EXACT set that lost its certificate. Every row must
    differ from the others, or the decomposition is buying nothing."""
    matrix = {c: _queued(c)[1] for c in (C1, C2, C3, UNATTRIBUTED)}
    assert matrix == {
        C1: {"cap_parse", "cap_undeclared"},
        C2: {"cap_cmdbuf", "cap_undeclared"},
        C3: {"cap_codegen", "cap_undeclared"},
        # Bytes no component claims could be anything -- the entrypoint script, a shared helper, the
        # manifest itself -- so they are a dependency of every capsule.
        UNATTRIBUTED: set(DEPENDS),
    }
    assert len({frozenset(v) for v in matrix.values()}) == 4


def test_an_untouched_capsule_keeps_its_certificate():
    r, queued = _queued(C1)
    assert set(r["unchanged"]) == {"cap_cmdbuf", "cap_codegen"}
    assert "cap_cmdbuf" not in queued


def test_the_requeue_never_emits_the_large_workload():
    """The performance workload is not a functional capsule, so it is not in the fork's capsule set
    and no edit to the compiler can put it in the queue -- even though it IS in the cert cover."""
    r, queued = _queued(C1)
    assert WORKLOAD not in queued
    assert WORKLOAD not in r["capsules_in_scope"]
    # Two small capsules at the shallow tier, not two cycle-accurate runs and not the workload.
    assert r["queued_cost_s"] == pytest.approx(2 * COST["L2"])


def test_the_requeue_reports_what_moved_and_what_it_covers():
    r, _ = _queued(C3)
    assert r["components_moved"] == [{"component": C3, "reason": CHANGED}]
    assert "never the large workload" in r["scope"]


def test_a_capsule_that_declares_nothing_is_invalidated_by_any_edit():
    for component in (C1, C2, C3, UNATTRIBUTED):
        assert "cap_undeclared" in _queued(component)[1]


def test_re_earned_verdicts_take_the_capsule_back_out_of_the_queue():
    digest, comps = _touch(C1)
    r = requeue(_fork(), digest=digest, components=comps, tier_order=TIERS, cert_tiers=CERT,
                cost_s=COST, verdicts={"cap_parse": {"L2": PASS, "L3": PASS},
                                       "cap_undeclared": {"L2": PASS, "L3": PASS}})
    assert r["queue"] == [] and set(r["unchanged"]) == set(DEPENDS)


# -------------------------------------------------------------------------------------------------
# 3. the invariant check: three states, and the third does not promote
# -------------------------------------------------------------------------------------------------
def _check(component=C1, verdicts=None, **kw):
    f = _fork()
    digest, comps = _touch(component)
    return check_invariants(f, candidate_states(f, digest=digest, components=comps,
                                                verdicts=verdicts), **kw)


def test_nothing_re_run_is_undeterminable_not_held():
    """The candidate edited a component two capsules ride on and re-ran nothing. Their Phase-F
    certificates are not evidence about these bytes."""
    c = _check(C1)
    assert c.state == UNDETERMINABLE and c.ok is None
    assert {(n, t) for n, t, _ in c.unproven} == {("cap_parse", "L2"), ("cap_parse", "L3"),
                                                  ("cap_undeclared", "L2"), ("cap_undeclared", "L3")}
    assert ("cap_cmdbuf", "L2") in c.held               # untouched, so still about these bytes


def test_a_re_earned_invariant_holds():
    c = _check(C1, verdicts={"cap_parse": {"L2": PASS, "L3": PASS},
                             "cap_undeclared": {"L2": PASS, "L3": PASS}})
    assert c.state == HELD and c.ok is True and not c.unproven
    assert len(c.held) == 8


def test_a_regression_weakens_the_fork_and_dominates_the_unproven():
    c = _check(C1, verdicts={"cap_parse": {"L2": FAIL}})
    assert c.state == WEAKENED and c.ok is False
    assert ("cap_parse", "L2") in c.weakened
    assert c.unproven                                   # present, and does not change the verdict


def test_a_capsule_the_candidate_never_reported_is_unproven():
    f = _fork()
    states = [s for s in candidate_states(f, digest="sub-v2", components=FORK_COMPONENTS)
              if s.name != "cap_codegen"]
    c = check_invariants(f, states)
    assert c.missing == ("cap_codegen",) and c.state == UNDETERMINABLE


def test_the_three_states_are_three_distinct_values():
    assert len({HELD, WEAKENED, UNDETERMINABLE}) == 3
    assert {HELD: True, WEAKENED: False, UNDETERMINABLE: None}[UNDETERMINABLE] is None


# -------------------------------------------------------------------------------------------------
# 4. the hardware revision the invariants were earned on
# -------------------------------------------------------------------------------------------------
PIN = {"target_rev": "0" * 40}


def test_a_fork_that_pinned_hardware_refuses_a_candidate_that_states_none():
    f = fork_from(_phase_f(), tier_order=TIERS, provenance=PIN, recorded_at="20260830T000000Z")
    states = candidate_states(f, digest=FORK_DIGEST, components=FORK_COMPONENTS)
    assert check_invariants(f, states).state == UNDETERMINABLE
    assert check_invariants(f, states, provenance={"target_rev": "1" * 40}).state == UNDETERMINABLE
    held = check_invariants(f, states, provenance=PIN)
    assert held.state == HELD and "hardware revision the fork pinned" in held.reason


# -------------------------------------------------------------------------------------------------
# 5. the two halves joined: the fork's tri-state IS the A/B gate's invariant input
# -------------------------------------------------------------------------------------------------
BIND = {"mv": "E_move", "ar": "E_arith"}


def _eta(label, mover, arith):
    return eta_from_occupancy(label, {"mv": [c == "1" for c in mover],
                                      "ar": [c == "1" for c in arith]},
                              unit_of=BIND, work="w")


BASE = _eta("base", "11110000", "00001111")
BETTER = _eta("better", "11110000", "00111111")


def test_an_unproven_fork_does_not_promote_however_good_eta_is():
    """The eta rose, the answer is bit-exact -- and the functional invariants were never re-proven.
    That is undeterminable, and undeterminable does not promote."""
    c = _check(C1)
    d = ab_decision(BASE, BETTER, bit_exact=True, invariants_held=c.ok)
    assert d.state == ETA_UNDETERMINABLE and "forked compiler is still functionally complete" in d.reason


def test_a_weakened_fork_is_rejected_however_good_eta_is():
    c = _check(C1, verdicts={"cap_parse": {"L2": FAIL}})
    assert ab_decision(BASE, BETTER, bit_exact=True, invariants_held=c.ok).state == REJECT


def test_held_invariants_plus_a_risen_eta_accept():
    c = _check(C1, verdicts={"cap_parse": {"L2": PASS, "L3": PASS},
                             "cap_undeclared": {"L2": PASS, "L3": PASS}})
    assert ab_decision(BASE, BETTER, bit_exact=True, invariants_held=c.ok).state == ACCEPT
    # and the same held fork still rejects a schedule that bought no overlap
    assert ab_decision(BASE, _eta("same", "11110000", "00001111"), bit_exact=True,
                       invariants_held=c.ok).state == REJECT
