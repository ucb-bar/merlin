"""A promoted cert tier must be one the executor can actually run.

The promotion policy and the promotion wiring were both tested, and both passed, while promotion could
not fire at all: `tier_promote.resolve_tiers` names its cert tier as `oracle_adapters - qa_loop_adapters`,
the broker forwards it as `--tiers <cert>`, and `agent_selfcheck` validated that request against the LOOP
map -- the one map guaranteed not to contain it. Every promoted job was refused as "unreachable" by an
endpoint that reaches it.

Nothing caught this because each side was tested against itself. These tests assert the JOIN: that the
tier one module hands over is a tier the other module accepts.
"""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import pytest
from phase1_feedback import feedback_context, feedback_source

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _mod(name: str):
    """Import a harness script by path — they are scripts, not an installed package."""
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location(name, feedback_source(name, HARNESS / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # noqa: BLE001 — harness deps absent in this env
        pytest.skip(f"{name} not importable here: {type(e).__name__}: {e}")
    return mod


FULL = {"L2": object(), "L3": object()}
LOOP = {"L2": FULL["L2"]}


# --- the pure selection contract ---------------------------------------------------------------
def test_no_request_runs_the_cheap_default():
    """The cost win: a sweep that asks for nothing pays the loop ladder, never the cert tier."""
    sel = _mod("agent_selfcheck")
    got, err = sel.select_tiers(FULL, LOOP, "")
    assert err is None
    assert set(got) == {"L2"}


def test_a_deeper_tier_is_accepted_when_the_endpoint_reaches_it():
    """The defect, stated directly: L3 is absent from the loop map and present in the full one, and
    naming it must be honoured rather than refused."""
    sel = _mod("agent_selfcheck")
    got, err = sel.select_tiers(FULL, LOOP, "L2,L3")
    assert err is None, f"a reachable tier was refused: {err}"
    assert set(got) == {"L2", "L3"}


def test_a_deeper_tier_alone_is_accepted():
    """Promotion asks for the cert tier ON ITS OWN (the capsule already passed the loop tier)."""
    sel = _mod("agent_selfcheck")
    got, err = sel.select_tiers(FULL, LOOP, "L3")
    assert err is None, f"a promoted cert-tier job would be refused: {err}"
    assert set(got) == {"L3"}


def test_an_unreachable_tier_is_named_not_dropped():
    """Failing closed still matters: silently grading fewer tiers than asked reads as a pass."""
    sel = _mod("agent_selfcheck")
    got, err = sel.select_tiers(FULL, LOOP, "L2,L9")
    assert got == {}
    assert err and "L9" in err


def test_the_error_reports_what_is_REACHABLE_not_merely_the_default():
    """The old message said `reachable: ['L2']` on an endpoint that reaches L3 — it sent the reader off
    to look for a missing simulator instead of at the tier resolution."""
    sel = _mod("agent_selfcheck")
    _got, err = sel.select_tiers(FULL, LOOP, "L9")
    assert "L3" in err, f"error hides a reachable tier: {err}"


def test_case_and_whitespace_are_tolerated():
    sel = _mod("agent_selfcheck")
    got, err = sel.select_tiers(FULL, LOOP, " l3 , l2 ")
    assert err is None and set(got) == {"L2", "L3"}


# --- the cross-module join ----------------------------------------------------------------------
def test_the_promoted_cert_tier_is_one_the_selector_accepts(tmp_path, monkeypatch):
    """The join that was missing. Whatever `resolve_tiers` nominates as the cert tier, `select_tiers`
    must accept against the same endpoint's full map — otherwise promotion is inert by construction."""
    tp = _mod("tier_promote")
    sel = _mod("agent_selfcheck")
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen import target_experiment
    from merlin.targetgen.contract import materialize

    # Synthetic actual adapter factories keep this join non-vacuous without requiring a simulator.
    descriptor = SimpleNamespace(target="fixture", sim_via="contract", graded_roots=lambda: (tmp_path,))
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda path: descriptor)
    monkeypatch.setattr(materialize, "declared_oracle_tiers", lambda *roots: ["L2"])
    monkeypatch.setattr(CR, "qa_loop_adapters", lambda *args, **kwargs: LOOP)
    monkeypatch.setattr(CR, "oracle_adapters", lambda *args, **kwargs: FULL)
    monkeypatch.setattr(tp, "_cert_cover", lambda *args, **kwargs: None)
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    ws = tmp_path / "ws"
    (ws / "submission").mkdir(parents=True)
    (ws / "submission/manifest.yaml").write_text("x: 1")
    loop_tier, cert_tier, _cover = tp.resolve_tiers(ws, context=feedback_context())
    assert (loop_tier, cert_tier) == ("L2", "L3")
    full, _sim = sel._adapters("contract", descriptor.target, descriptor.sim_via)
    assert loop_tier in full, f"loop tier {loop_tier} unreachable; full={sorted(full)}"
    _got, err = sel.select_tiers(full, {loop_tier: full[loop_tier]}, cert_tier)
    assert err is None, f"promotion nominates {cert_tier!r} but the executor refuses it: {err}"
