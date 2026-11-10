from __future__ import annotations

import importlib.util

from merlin.common.paths import repo_root

GATE_PATH = repo_root() / "build_tools" / "checks" / "check_schedule_ir_coverage.py"


def _gate():
    """Load the gate BY PATH, the way the other gate-regression tests do -- it is a script, not a
    module, and importing it as one would test a different thing than what runs in the hook."""
    spec = importlib.util.spec_from_file_location("_schedule_ir_coverage_gate", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROSTER = {"t_alpha", "t_beta", "t_gamma"}


def _row(**overrides):
    """A row with nothing wrong with it, for a test to break in exactly one way."""
    row = {
        "id": "axis_under_test",
        "axis": "something a schedule must be able to say",
        "exo": "EXO_ABSENT",
        "exo_evidence": "verified by inspection of its grammar",
        "forced_by": ["t_alpha", "t_beta"],
        "generic": True,
        "owner": "sched",
        "surfaced_by": "a design note",
        "surfaced_on": "2026-09-15",
        "blocks": "what does not work today",
    }
    row.update(overrides)
    return row


def test_a_primitive_with_no_row_is_flagged():
    """The direction a hand-maintained register loses.

    A row naming a vanished primitive is noticed the moment someone reads the register; a primitive
    nobody had to justify is invisible, and is how an IR acquires one per kernel.
    """
    gate = _gate()
    _s, problems, rc = gate.verdict([_row()], {"a_new_primitive"}, ROSTER)
    assert rc == 1
    assert any("is exported but no row declares it" in p for p in problems)
