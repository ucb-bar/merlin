"""An instruction whose real write-set exceeds its declared destination is nearly invisible.

The emitted program decodes cleanly, every instruction is of the right class for its op, and the kernel
still returns a wrong answer -- so a decode census, a role census and a lint pass all report clean. The
only thing that sees it is running the hardware with the instruction inserted where it can do damage.

This is the shape of a measured atlas defect: `VLI_ALL` declares a `vd` field and its shipped reference
implementation writes exactly that register, but on the elaborated RTL it clears the whole matrix
register file. A backend staging two operands emitted the fill twice and the second wiped the first
operand, so the kernel returned its second input unchanged.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import isa_side_effects as SE
from merlin.targetgen.isa_model import isa_model_for_target


@pytest.fixture(scope="module")
def model():
    return isa_model_for_target("atlas")


def _words(model, spec):
    from merlin.targetgen import isa_asm
    return [isa_asm.assemble_line(model, m, ops) for m, ops in spec]


def test_the_window_is_between_the_last_load_and_the_first_consumer(model):
    """Both `VLOAD` and `VSTORE` carry the `memory` role, so taking the last memory op outright picks a
    store that runs AFTER the compute -- not a live window. That reported 'no window' for every real
    program and made the probe silently inert."""
    w = _words(model, [("VLOAD", {"rs1": 21, "imm": 0, "vd": 0}),
                       ("VADD_BF16", {"vd": 8, "vs1": 0, "vs2": 4}),
                       ("VSTORE", {"rs1": 21, "imm": 0, "vd": 8})])
    assert SE.live_injection_index(model, w) == 1


def test_no_window_when_nothing_is_loaded_or_nothing_computed(model):
    """A verdict from a program with no live window would not be attributable to the instruction."""
    only_compute = _words(model, [("VADD_BF16", {"vd": 8, "vs1": 0, "vs2": 4})])
    only_moves = _words(model, [("VLOAD", {"rs1": 21, "imm": 0, "vd": 0}),
                                ("VSTORE", {"rs1": 21, "imm": 0, "vd": 0})])
    assert SE.live_injection_index(model, only_compute) is None
    assert SE.live_injection_index(model, only_moves) is None


def test_an_instruction_that_changes_the_result_is_reported_as_writing_beyond_its_field(model):
    w = _words(model, [("VLOAD", {"rs1": 21, "imm": 0, "vd": 0}),
                       ("VADD_BF16", {"vd": 8, "vs1": 0, "vs2": 4}),
                       ("VSTORE", {"rs1": 21, "imm": 0, "vd": 8})])
    destructive = SE.probe_instruction(
        model, w, "VLI_ALL", {"imm": 0, "vd": 63},
        run=lambda words: "clobbered" if len(words) > len(w) else "good")
    assert destructive["perturbs_live_state"] is True
    assert destructive["verdict"] == "writes_beyond_declared_destination"
    assert destructive["declared_destinations"] == ["vd"]      # what it ADVERTISES, contradicted above
    assert destructive["injected_at"] == 1


def test_an_instruction_that_respects_its_destination_is_reported_clean(model):
    """Not vacuous: the same probe over an oracle whose output does NOT move must say so, or every
    instruction would be reported destructive."""
    w = _words(model, [("VLOAD", {"rs1": 21, "imm": 0, "vd": 0}),
                       ("VADD_BF16", {"vd": 8, "vs1": 0, "vs2": 4}),
                       ("VSTORE", {"rs1": 21, "imm": 0, "vd": 8})])
    clean = SE.probe_instruction(model, w, "VLI_ALL", {"imm": 0, "vd": 63},
                                 run=lambda words: "same-every-time")
    assert clean["perturbs_live_state"] is False
    assert clean["verdict"] == "respects_declared_destination"


def test_a_settle_word_is_placed_after_the_injection(model):
    """Without it, a difference could be blamed on not having waited for the pipeline. Passing the
    program's own longest settle excludes that explanation by construction."""
    w = _words(model, [("VLOAD", {"rs1": 21, "imm": 0, "vd": 0}),
                       ("VADD_BF16", {"vd": 8, "vs1": 0, "vs2": 4}),
                       ("VSTORE", {"rs1": 21, "imm": 0, "vd": 8})])
    seen = []
    SE.probe_instruction(model, w, "VLI_ALL", {"imm": 0, "vd": 63},
                         run=lambda words: seen.append(list(words)) or "x", settle=0xDEAD)
    assert len(seen[1]) == len(w) + 2 and seen[1][2] == 0xDEAD


def test_no_window_yields_no_verdict_rather_than_a_clean_bill(model):
    """Fail closed: a program the probe cannot use must not come back as 'respects its destination'."""
    r = SE.probe_instruction(model, _words(model, [("VADD_BF16", {"vd": 8, "vs1": 0, "vs2": 4})]),
                             "VLI_ALL", {"imm": 0, "vd": 63}, run=lambda w: "x")
    assert r["verdict"] == "no_live_window"
    assert "perturbs_live_state" not in r
