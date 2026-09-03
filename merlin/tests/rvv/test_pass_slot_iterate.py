"""One turn is not a loop.

The gate produces a precise, machine-checked refusal -- "inert: byte-identical to the unpatched build,
the pass was imported but its matching never fired" -- and a one-shot slot throws it away. MEASURED on
the first real turn through this slot: an 18-minute, 611-line proposal was refused for a reason its
author could have acted on immediately, and nothing carried it forward.

The hard constraint on the feedback is that it is FACTS ONLY. If it named a fix, the resulting pass
would be the author's idea typed by a model, and it would carry the author's blind spots into every
future run -- which is exactly what "auto discovery" must not become.
"""
import pytest

from merlin.mining.pass_slot import (PassProposal, PassVerdict, iterate_pass_slot, verdict_feedback)


class _Action:
    divergence_axis = "compute.activation_vectorization"
    intended_facet = {"compute.activation_vectorization": "vectorized_polynomial"}
    change = "extend the polynomial emitter's op coverage"
    target_seam = "pass:llvmlower/act_poly.py"
    action_class = "CODEGEN"


def _checks(verdicts):
    """Gate seams that yield a scripted verdict per turn, by driving the FIRST check."""
    seq = list(verdicts)
    state = {"i": 0}

    def frozen(_p):
        return True

    def bit_exact(_p):
        v = seq[min(state["i"], len(seq) - 1)]
        state["i"] += 1
        return (v is None), ("ok" if v is None else v)

    return {"frozen_baseline_ok": frozen, "bit_exact_ok": bit_exact,
            "lift_cca": lambda _p: None}


# --------------------------------------------------------------------------- the feedback itself

def test_the_feedback_is_the_measurement_and_never_a_suggested_fix():
    v = PassVerdict(False, "inert",
                    "byte-identical to the unpatched build (digest 0358237f); the pass was imported "
                    "but its matching never fired",
                    detail={"control_digest": "0358237f"})
    fb = verdict_feedback(v)
    assert "inert" in fb and "byte-identical" in fb and "0358237f" in fb
    # the words a suggested fix would use
    for leading in ("you should", "try ", "instead of", "consider ", "probably"):
        assert leading not in fb.lower(), f"the feedback is prescribing a fix: {leading!r}"


def test_outstanding_axes_are_reported_when_the_facet_check_refused():
    v = PassVerdict(False, "facet", "promised X but did not achieve Y",
                    residual=("compute.activation_vectorization",))
    assert "compute.activation_vectorization" in verdict_feedback(v)


# --------------------------------------------------------------------------- the loop

def test_a_refusal_is_fed_into_the_next_turn():
    seen = []

    def propose(action, *, feedback=None):
        seen.append(feedback)
        return PassProposal(module="m", source=f"X = {len(seen)}\n")

    iterate_pass_slot(_Action(), propose=propose, max_turns=2,
                      **_checks(["numerics moved", "numerics moved again"]))
    assert seen[0] is None, "the first turn has nothing to learn from"
    assert seen[1] and "numerics moved" in seen[1], "the refusal did not reach turn 2"


def test_the_loop_stops_the_moment_a_proposal_is_accepted():
    """Uses the REAL escalated action and a CCA that achieves its promise, so acceptance is reached
    through the same promise audit production uses rather than a stand-in."""
    from merlin.kernels import action_catalog as ac, cca
    from merlin.kernels.cca_compare import Divergence

    d = Divergence(axis="compute.activation_vectorization", expert="vectorized_polynomial",
                   ours="scalar_libm_call", backend="rvv")
    action = ac.route_escalated(d, ac.route(d).action_class)
    achieved = cca.CCA(op="matmul", backend=["rvv"],
                       compute=cca.ComputeFacet(op="matmul",
                                                activation_vectorization="vectorized_polynomial"))
    calls = {"n": 0}

    def propose(a, *, feedback=None):
        calls["n"] += 1
        return PassProposal(module="m", source="X = 1\n")

    out = iterate_pass_slot(action, propose=propose, max_turns=5,
                            frozen_baseline_ok=lambda _p: True,
                            bit_exact_ok=lambda _p: (True, "ok"),
                            inert_ok=lambda _p: (True, "changed"),
                            lift_cca=lambda _p: achieved)
    assert out[-1][1].accepted, out[-1][1].reason
    assert calls["n"] == 1, f"an accepted proposal must end the loop, ran {calls['n']} turns"


def test_two_identical_verdicts_stop_the_loop():
    """The feedback is not landing, so more turns are spend without signal. Board and agent time are
    the scarce resources here, not attempts."""
    calls = {"n": 0}

    def propose(action, *, feedback=None):
        calls["n"] += 1
        return PassProposal(module="m", source=f"X = {calls['n']}\n")

    out = iterate_pass_slot(_Action(), propose=propose, max_turns=5,
                            **_checks(["same failure"] * 5))
    assert calls["n"] == 2, f"expected to stop after the repeat, ran {calls['n']} turns"
    assert len(out) == 2


def test_a_verdict_that_moved_to_a_LATER_stage_is_progress_and_does_not_stop_the_loop():
    """Still a refusal, but a different one -- the turn learned something."""
    calls = {"n": 0}

    def propose(action, *, feedback=None):
        calls["n"] += 1
        return PassProposal(module="m", source=f"X = {calls['n']}\n")

    iterate_pass_slot(_Action(), propose=propose, max_turns=3,
                      **_checks(["numerics moved", "still inert", "facet unmet"]))
    assert calls["n"] == 3, "distinct refusals must keep the loop running"


def test_every_turn_is_returned_including_the_refusals():
    def propose(action, *, feedback=None):
        return PassProposal(module="m", source="X = 1\n")

    out = iterate_pass_slot(_Action(), propose=propose, max_turns=3,
                            **_checks(["a", "b", "c"]))
    assert len(out) == 3
    assert all(isinstance(v, PassVerdict) for _p, v in out)
    assert all(p is not None for p, _v in out), "a refused proposal is evidence, not noise"


def test_no_proposal_ends_the_loop_honestly():
    out = iterate_pass_slot(_Action(), propose=lambda a, *, feedback=None: None, max_turns=3,
                            **_checks([None]))
    assert len(out) == 1 and out[0][0] is None and out[0][1].stage == "no_proposal"


def test_no_proposer_is_a_refusal_not_a_crash():
    out = iterate_pass_slot(_Action(), propose=None, max_turns=3, **_checks([None]))
    assert len(out) == 1 and out[0][1].stage == "no_proposal"


def test_a_proposer_that_predates_the_feedback_argument_still_works():
    """Backward compatibility: it simply cannot learn between turns."""
    calls = {"n": 0}

    def old_style(action):          # no feedback kwarg
        calls["n"] += 1
        return PassProposal(module="m", source=f"X = {calls['n']}\n")

    out = iterate_pass_slot(_Action(), propose=old_style, max_turns=2,
                            **_checks(["a", "b"]))
    assert calls["n"] == 2 and len(out) == 2


def test_the_turn_hook_sees_each_turn_as_it_lands():
    seen = []
    iterate_pass_slot(_Action(),
                      propose=lambda a, *, feedback=None: PassProposal(module="m", source="X=1\n"),
                      max_turns=2, on_turn=lambda n, p, v: seen.append((n, v.stage)),
                      **_checks(["a", "b"]))
    assert [n for n, _ in seen] == [1, 2]


def test_a_retry_does_not_inherit_the_previous_turns_proposal_file(tmp_path):
    """A turn that writes nothing must not be credited with its predecessor's source -- the gate would
    re-refuse the same bytes while the record showed two independent attempts."""
    from merlin.mining import pass_agent as pa
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    (ws / pa.PROPOSAL_FILENAME).write_text("STALE = 1\n")
    att = pa.propose_pass(_Action(), module="m", current_source="x=1\n", workspace=ws,
                          require_sandbox=False,
                          runner=lambda **kw: {"text": "I could not do it.", "usage": {}})
    assert att.proposal is None, "a stale file was served as this turn's proposal"
    assert "no usable python block" in (att.error or "")
