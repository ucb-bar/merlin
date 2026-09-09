"""A converged run must be able to finish.

Conformance is computed PER ROUND from that round's tool calls, but the checks it asks about are
start-of-work activities: `cca_used` wants the lever set enumerated, and an agent that has converged has
nothing left to enumerate. Gating the only success exit on the CURRENT round's flag therefore demanded
evidence of exploration in the same round that proves completion.

Observed: an assisted run reached 27/27 in round 3 and repeated it in rounds 4-7 without ever exiting.
Round 1 had all five checks at 25/27; rounds 3-7 had the score and not `cca_used`. Never both at once,
and never able to be, because tool calls fall 300 -> 37 -> 30 -> 20 as the agent runs out of work.

The rule these pin is the one `agg_by_model` already reports: every mandated check satisfied in SOME
round, which is what "the agent developed the way this arm mandates" actually means over a run.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir

LOOP = merlin_dir() / "experiments/capsule_bench/harness/run_baseline_qa_loop.py"


def _ever(rounds):
    """Mirror of the loop's accumulator + gate, exercised on round-by-round check dicts."""
    ever: dict = {}
    for checks in rounds:
        for k, v in checks.items():
            if v is None:
                ever.setdefault(k, None)
            else:
                ever[k] = bool(ever.get(k)) or bool(v)
    applicable = [v for v in ever.values() if v is not None]
    return all(applicable) if applicable else True


def test_the_observed_run_would_now_be_able_to_exit():
    """The real shape: one early round has the tooling, the passing rounds do not."""
    rounds = [
        {"no_regex_ok": True, "isa_tools_used": True, "asm_used": True, "cca_used": False, "full_selfcheck": False},
        {"no_regex_ok": True, "isa_tools_used": True, "asm_used": True, "cca_used": True, "full_selfcheck": True},
        {"no_regex_ok": True, "isa_tools_used": True, "asm_used": False, "cca_used": False, "full_selfcheck": False},
        {"no_regex_ok": True, "isa_tools_used": True, "asm_used": True, "cca_used": False, "full_selfcheck": False},
    ]
    assert _ever(rounds) is True
    assert all(not all(v for v in r.values()) for r in rounds[2:]), "fixture must have no single conformant late round"


def test_a_check_never_satisfied_still_blocks():
    """Not vacuous: an arm whose agent NEVER used the mandated tooling must not be certified."""
    rounds = [
        {"no_regex_ok": True, "isa_tools_used": True, "cca_used": False},
        {"no_regex_ok": True, "isa_tools_used": True, "cca_used": False},
    ]
    assert _ever(rounds) is False


def test_inapplicable_checks_never_gate():
    """A check that does not apply to this arm is None every round and must not block it."""
    assert _ever([{"no_regex_ok": None, "isa_tools_used": True}]) is True


def test_the_loop_gates_on_the_run_level_view_not_the_current_round():
    src = LOOP.read_text()
    assert "_conformant_over_run()" in src
    assert "if _run_l3 and _conformant_over_run()" in src, "the barrier must use the run-level view"
    assert "_conf_ever" in src


def test_the_per_round_flag_is_still_recorded():
    """The per-round signal stays: it is how you see WHICH round did the work."""
    src = LOOP.read_text()
    assert '"workflow_conformant": workflow_conformant' in src


def _loop():
    import sys
    sys.path.insert(0, str(merlin_dir() / "experiments/capsule_bench/harness"))
    import run_baseline_qa_loop  # noqa: PLC0415
    return run_baseline_qa_loop


def test_the_run_level_view_is_rebuilt_from_the_round_records():
    """A --resume must recover the accumulator, not restart it empty.

    `_conf_ever` lives in a closure, so a resumed run began with `{}` and a run that had ALREADY
    evidenced every mandated check was blocked by its own fresh accumulator -- the same defect as
    gating on the current round, moved into the resume path.
    """
    rounds = [
        {"conformance": {"checks": {"cca_used": True, "full_selfcheck": False, "asm_used": None}}},
        {"conformance": {"checks": {"cca_used": False, "full_selfcheck": True, "asm_used": None}}},
    ]
    ever = _loop()._conformance_ever(rounds)
    assert ever == {"cca_used": True, "full_selfcheck": True, "asm_used": None}


def test_a_check_that_never_held_is_false_after_the_fold():
    rounds = [{"conformance": {"checks": {"cca_used": False}}},
              {"conformance": {"checks": {"cca_used": False}}}]
    assert _loop()._conformance_ever(rounds) == {"cca_used": False}


def test_the_fold_tolerates_missing_and_malformed_round_records():
    """Checkpoints written before this field existed, and partially-written rounds, must not crash."""
    assert _loop()._conformance_ever([]) == {}
    assert _loop()._conformance_ever(None) == {}
    assert _loop()._conformance_ever([{}, {"conformance": {}}, {"conformance": {"checks": None}}]) == {}


def test_the_resume_path_persists_and_restores_the_run_level_view():
    src = LOOP.read_text()
    assert '"conformance_ever": dict(_conf_ever)' in src, "the run-level view must be persisted"
    assert 'st.get("conformance_ever")' in src, "a resume must read it back"
    assert "_conf_ever.update(_conformance_ever(rounds_summary))" in src, \
        "and must fall back to re-folding the round records for pre-existing checkpoints"
