"""The pass slot's GATE — the deterministic part, tested with no agent and no toolchain.

This is the leaf the escalation ladder terminates in, and the gate is the part that must be right: it
is what makes an agentic step safe to run at all. Every check is injected, so these tests exercise the
real ordering and the real fail-closed semantics without a board, a build, or an agent.
"""
from __future__ import annotations

import pytest

from merlin.kernels import action_catalog as ac
from merlin.kernels.cca_compare import Divergence
from merlin.mining import pass_slot as ps

AXIS = "compute.activation_vectorization"


def _action():
    d = Divergence(axis=AXIS, backend="rvv", ours="scalar_libm_call",
                   expert="vectorized_polynomial")
    return ac.route(d)


class _CCA:
    """Minimal stand-in exposing the one facet path the promise names."""

    def __init__(self, value):
        self.compute = type("C", (), {"activation_vectorization": value})()


def _ok_kwargs(cca_value="vectorized_polynomial", **over):
    kw = dict(frozen_baseline_ok=lambda p: True,
              bit_exact_ok=lambda p: (True, ""),
              lift_cca=lambda p: _CCA(cca_value),
              models=("bitvla", "small_llama"))
    kw.update(over)
    return kw


def _proposal(src="def emit_rsqrt(): return 'vfmacc chain'"):
    return ps.PassProposal(module="merlin.llvmlower.act_poly", source=src)


# ---- ordering: the cheapest disqualifier must fire first -------------------------------

def test_a_cheating_proposal_is_refused_before_anything_is_built():
    """If the scan ran last, a proposal that reads the golden would still have consumed a build and a
    board run first. It must be first, and it must not need any of the injected checks."""
    def explode(_p):
        raise AssertionError("must not be reached: the cheat scan runs first")

    v = ps.gate(_proposal("x = open('golden.npy')"), _action(),
                **_ok_kwargs(frozen_baseline_ok=explode, bit_exact_ok=explode, lift_cca=explode))
    assert not v.accepted and v.stage == "cheat"
    assert "golden.npy" in v.detail["tokens"]


def test_a_model_name_in_a_compiler_pass_is_a_cheat():
    """A pass that names a model is overfit by construction -- the loop exists because a lever found on
    one model must transfer. The token list is DERIVED from the captures on disk, not hardcoded."""
    v = ps.gate(_proposal("if model == 'small_llama': coeffs = TUNED"), _action(), **_ok_kwargs())
    assert not v.accepted and v.stage == "cheat"
    assert any(t.startswith("model:") for t in v.detail["tokens"])
    # ...and the derivation covers captures without this file being edited
    assert "bitvla" in ps.model_name_tokens()


def test_asserting_its_own_verdict_is_a_cheat():
    """A proposal that imports the gate's own helpers could return an empty residual directly."""
    for src in ("from merlin.kernels.action_catalog import achieved_residual",
                "action.intended_facet = {}", "F._REGISTRY['x'] = y", "monkeypatch.setattr(...)"):
        v = ps.gate(_proposal(src), _action(), **_ok_kwargs())
        assert not v.accepted and v.stage == "cheat", src


def test_the_frozen_baseline_is_checked_before_the_expensive_numerics():
    def explode(_p):
        raise AssertionError("bit-exactness must not run once the baseline moved")

    v = ps.gate(_proposal(), _action(),
                **_ok_kwargs(frozen_baseline_ok=lambda p: False, bit_exact_ok=explode))
    assert not v.accepted and v.stage == "frozen_baseline"
    assert "control" in v.reason


def test_changed_numerics_are_refused_before_the_facet_is_credited():
    def explode(_p):
        raise AssertionError("the facet must not be credited on a miscompile")

    v = ps.gate(_proposal(), _action(),
                **_ok_kwargs(bit_exact_ok=lambda p: (False, "cos 0.91"), lift_cca=explode))
    assert not v.accepted and v.stage == "bit_exact" and "cos 0.91" in v.reason


# ---- the promise ----------------------------------------------------------------------

def test_the_promise_comes_from_the_router_not_from_the_gate():
    """The gate delegates to achieved_residual so it cannot disagree with the router about what was
    asked for. Here the emitted code still shows the scalar call, so the residual names the axis."""
    v = ps.gate(_proposal(), _action(), **_ok_kwargs(cca_value="scalar_libm_call"))
    assert not v.accepted and v.stage == "facet"
    assert v.residual == (AXIS,)


def test_a_delivered_facet_is_accepted():
    v = ps.gate(_proposal(), _action(), **_ok_kwargs())
    assert v.accepted and v.stage == "accepted" and v.residual == ()


def test_an_action_with_no_promise_is_refused_not_credited():
    """Accepting an unverifiable change would credit something nothing checked -- the exact failure this
    loop exists to remove. `checkable` must report it, and `accepted` must stay False."""
    class _NoPromise:
        intended_facet = None

    v = ps.verify_promise(_NoPromise(), _CCA("vectorized_polynomial"))
    assert not v.accepted and v.stage == "unverifiable" and not v.checkable
    assert "no intended_facet" in v.reason


# ---- held-out certification ----------------------------------------------------------

def test_holding_on_visible_but_not_held_out_is_refused():
    """The anti-overfit step: a proposal tuned to what it could see is exactly what held-out
    certification exists to catch."""
    v = ps.gate(_proposal(), _action(),
                **_ok_kwargs(), heldout_ok=lambda p: (False, "regressed on 2 unseen captures"))
    assert not v.accepted and v.stage == "heldout" and "unseen" in v.reason


def test_the_accept_reason_says_whether_held_out_was_actually_run():
    """"Accepted" must not read the same with and without held-out certification, or a visible-only
    pass gets quoted as if it generalised."""
    visible_only = ps.gate(_proposal(), _action(), **_ok_kwargs())
    certified = ps.gate(_proposal(), _action(), **_ok_kwargs(), heldout_ok=lambda p: (True, ""))
    assert visible_only.accepted and certified.accepted
    assert "held out" not in visible_only.reason
    assert "held out" in certified.reason


# ---- the default: no agent -----------------------------------------------------------

def test_with_no_proposer_the_slot_refuses_honestly():
    """The gate must be usable and testable with no agent budget, and must not report success when it
    had nothing to check."""
    proposal, v = ps.run_pass_slot(_action())
    assert proposal is None and not v.accepted and v.stage == "no_proposal"


def test_a_refused_proposal_is_returned_alongside_its_verdict():
    """A refused proposal is evidence about the seam, not noise -- the beam records non-actionable
    outcomes as work-items rather than dropping them."""
    bad = _proposal("read golden.npy here")
    proposal, v = ps.run_pass_slot(_action(), propose=lambda a: bad, **_ok_kwargs())
    assert proposal is bad and not v.accepted and v.stage == "cheat"
