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


# --------------------------------------------------------- cheat scan: strict without false alarms

_MODELS = ("openvla", "small", "small_llama", "rdt", "rdt2")


def _sc(src):
    from merlin.mining.pass_slot import scan_cheats
    return scan_cheats(src, models=_MODELS)


def test_a_pass_that_dispatches_on_a_model_name_is_caught():
    """The thing the scan is for: a lever found on one model has to transfer, so a pass that can tell
    which model it is compiling is overfit by construction."""
    assert "model:openvla" in _sc('if model == "openvla":\n    pass\n')
    assert "model:openvla" in _sc('TABLE = {"openvla": 4}\n')


def test_a_model_name_embedded_in_a_longer_identifier_is_caught():
    """`small_llama_hack` is the pattern -- a multi-word model name inside an identifier."""
    assert "model:small_llama" in _sc("small_llama_hack = 1\n")


def test_a_model_name_in_a_comment_or_docstring_is_not_a_cheat():
    """A comment cannot special-case a pass, and this repo WANTS the provenance: act_poly.py records
    that a blanket rewrite "drove openvla whole-model cos to 0.541" -- the measurement motivating the
    fix. Rejecting that would make every honest proposal for that module unacceptable."""
    assert _sc("# openvla cos 0.541\nx = 1\n") == []
    assert _sc('"""measured on openvla and small_llama."""\nx = 1\n') == []


def test_a_short_model_token_does_not_flag_an_ordinary_identifier():
    """The corpus model list contains short tokens (`small`, `rdt`, `pi05`). Span-matching those would
    flag `small_m_fallback` -- a real concept in this repo -- and `rdtime`, the K1 cycle counter. So a
    single-word token matches only exactly. A gate that rejects honest proposals is broken, not strict.
    """
    assert _sc("def small_m_fallback(x):\n    return x\n") == []
    assert _sc("t = rdtime()\n") == []
    assert _sc("rdt2_shape = 1\n") == []          # `rdt` must not match inside `rdt2_shape`
    assert "model:small" in _sc('small = 1\n')    # but the bare token still counts


def test_spliced_source_is_scanned_as_source_not_as_a_string():
    """These passes splice generated source as a string literal (act_poly, accum_microkernel's
    rewriter, perop_blocks), so a proposal's real content usually lives inside one. Comments in that
    nested source must stay exempt, while a dispatch inside it must still be caught."""
    assert _sc('SRC = """\n# openvla regressed here\nx = 1\n"""\n') == []
    assert "model:openvla" in _sc('SRC = """\nif m == "openvla":\n    pass\n"""\n')


def test_the_real_act_poly_module_passes_its_own_cheat_scan():
    """The regression that motivated all of the above: scanning raw text rejected this module's own
    bytes over model names in its provenance comments, so the CODEGEN rung the ladder escalates to
    could never have an acceptable proposal."""
    from merlin.common.paths import merlin_dir
    from merlin.mining.pass_slot import scan_cheats
    src = (merlin_dir() / "python" / "merlin" / "llvmlower" / "act_poly.py").read_text()
    assert scan_cheats(src) == [], "the module under improvement must pass the gate's own cheat scan"


def test_unparseable_source_is_reported_rather_than_passed():
    """It cannot be gated, so it is a finding. Silently returning [] would send a broken proposal on
    to the expensive checks and blame the failure on the build."""
    found = _sc("def broken(:\n")
    assert found and found[0].startswith("unparseable:")


def test_the_answer_reading_tokens_are_still_caught():
    assert "golden.npy" in _sc('open("golden.npy")\n')
    assert "achieved_residual" in _sc("from x import achieved_residual\n")
