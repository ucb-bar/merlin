"""Operand direction, derived by probing -- against a tiny machine whose truth is known.

The fixture is a five-instruction ISA plus a functional model of it. Nothing about it resembles any
real target, which is the point: the derivation is supposed to work on whatever the ISA model and the
oracle publish, so the test gives it a machine no code anywhere has seen and checks that it recovers
the directions the fixture actually implements -- and refuses the ones the fixture deliberately makes
unobservable.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import isa_direction as ID
from merlin.targetgen import isa_disasm
from merlin.targetgen.isa_model import IsaModel


def _bits(lo: int, width: int) -> list[int]:
    return [lo + i for i in range(width)]


def _entry(name: str, opcode: int, fields: dict) -> dict:
    variable = 0
    for bits in fields.values():
        for b in bits:
            variable |= 1 << b
    return {"class": name, "fields": fields, "fixed_value": opcode,
            "fixed_mask": (~variable) & 0xFFFFFFFF}


#: rd/vd at [7..12], rs1/vs1 at [13..18], vs2 at [19..24], imm at [20..31] -- laid out so the only
#: overlap in the fixture is the one the ALIASED instruction has on purpose.
_RD = _bits(7, 6)
_RS1 = _bits(13, 5)
_VS1 = _bits(13, 6)
_VS2 = _bits(19, 6)
_IMM = _bits(20, 12)

FIXTURE = IsaModel(target="fixture", by_mnemonic={
    "ADDI": _entry("ADDI", 0x01, {"rd": _RD, "rs1": _RS1, "imm": _IMM}),
    "UPPER": _entry("UPPER", 0x02, {"rd": _RD, "imm": _bits(13, 19)}),
    "WAIT": _entry("WAIT", 0x03, {"imm": _IMM}),
    "STOP": _entry("STOP", 0x04, {}),
    "VSET": _entry("VSET", 0x05, {"vd": _RD, "imm": _IMM}),
    "VMOV": _entry("VMOV", 0x06, {"vd": _RD, "vs1": _VS1, "vs2": _VS2}),
    "VADD": _entry("VADD", 0x07, {"vd": _RD, "vs1": _VS1, "vs2": _VS2}),
    "ACCPUSH": _entry("ACCPUSH", 0x08, {"vd": _RD, "vs1": _VS1}),
    "ALIASED": _entry("ALIASED", 0x09, {"vd": _RD, "vs1": _bits(13, 7), "vs2": _bits(19, 6)}),
    # every bit of both source fields is shared, so neither can be varied on its own at all
    "FUSED": _entry("FUSED", 0x0B, {"vd": _RD, "vs1": _bits(19, 6), "vs2": _bits(19, 6)}),
    "INVISIBLE": _entry("INVISIBLE", 0x0A, {"rd": _RD, "rs1": _RS1}),
})

_SCALAR_SLOTS = 32
_TENSOR_SLOTS = 24
_ACC_SLOTS = 2


def _sign(value: int, width: int) -> int:
    return value - (1 << width) if value >= (1 << (width - 1)) else value


def run_probe(kernel_s: str) -> dict:
    """A functional model of the fixture: run the emitted words, then publish architectural state.

    The published state deliberately mirrors what a real oracle offers -- scalar registers as VALUES,
    the tensor and accumulator files as a per-slot "holds something" flag -- so the derivation is
    exercised against both kinds of observability at once.
    """
    words = [int(line.split(".word")[1].split()[0], 16)
             for line in kernel_s.splitlines() if ".word" in line]
    x = [0] * _SCALAR_SLOTS
    mrf = [0] * _TENSOR_SLOTS
    acc = [0] * _ACC_SLOTS
    for rec in isa_disasm.disassemble(FIXTURE, words):
        mn, op = rec.get("mnemonic"), rec.get("operands") or {}
        if mn == "ADDI":
            if op["rd"]:
                x[op["rd"] % _SCALAR_SLOTS] = x[op["rs1"] % _SCALAR_SLOTS] + _sign(op["imm"], 12)
        elif mn == "UPPER":
            if op["rd"]:
                x[op["rd"] % _SCALAR_SLOTS] = op["imm"] << 12
        elif mn == "VSET":
            mrf[op["vd"] % _TENSOR_SLOTS] = op["imm"]
        elif mn == "VMOV":                      # unary: vs2 is declared by the format and unread
            mrf[op["vd"] % _TENSOR_SLOTS] = mrf[op["vs1"] % _TENSOR_SLOTS]
        elif mn == "VADD":                      # a wide result: it lands in TWO consecutive slots
            total = mrf[op["vs1"] % _TENSOR_SLOTS] + mrf[op["vs2"] % _TENSOR_SLOTS]
            mrf[op["vd"] % _TENSOR_SLOTS] = total
            mrf[(op["vd"] + 1) % _TENSOR_SLOTS] = total
        elif mn == "ACCPUSH":
            if op["vd"] < _ACC_SLOTS:
                acc[op["vd"]] = mrf[op["vs1"] % _TENSOR_SLOTS]
        elif mn == "STOP":
            break
    return {"halted": True, "halt_reason": "finished", "regs": x,
            "on_chip": {"mrf": [bool(v) for v in mrf], "acc": {"u0": [bool(v) for v in acc]}}}


OPS = ID.ProbeOps(scalar_imm="ADDI", scalar_upper="UPPER", stall="WAIT", halt="STOP",
                  seed_slot_field="vd", attribution_value=2, settle=4,
                  seeders=(("VSET", {"vd": 1, "imm": 0x111}), ("VSET", {"vd": 2, "imm": 0x222})))


@pytest.fixture(scope="module")
def model() -> ID.DirectionModel:
    return ID.derive_directions(FIXTURE, OPS, run_probe)


def test_scalar_destination_is_a_definition_and_its_sources_are_uses(model):
    assert model.defs_of("ADDI") == ("rd",)
    assert "rs1" in model.uses_of("ADDI")
    assert model.file_of("ADDI", "rd") == "scalar"


def test_a_source_is_attributed_to_the_file_it_actually_reads(model):
    # ADDI reads its source from the scalar file, and the probe establishes that by taking the
    # content of the named scalar slot away rather than by reading the operand's name.
    assert model.file_of("ADDI", "rs1") == "scalar"


def test_tensor_destination_and_the_width_of_a_wide_result(model):
    assert model.defs_of("VADD") == ("vd",)
    assert model.file_of("VADD", "vd") == "mrf"
    written = model.by_mnemonic["VADD"]["vd"].written_slots
    assert len(written) == 2, f"a two-slot result should be measured as two slots, got {written}"


def test_a_declared_but_unread_operand_is_not_reported_as_a_use(model):
    # VMOV's format declares vs2; the instruction never reads it. Calling it a use would invent a
    # dependence edge, so the probe must not, and it must not call it a definition either.
    assert model.by_mnemonic["VMOV"]["vs2"].direction == ID.UNKNOWN
    assert "vs1" in model.uses_of("VMOV")
    assert model.defs_of("VMOV") == ("vd",)


def test_presence_only_file_still_evidences_a_definition(model):
    assert model.defs_of("ACCPUSH") == ("vd",)
    assert model.file_of("ACCPUSH", "vd") == "acc.u0"


def test_presence_only_file_refuses_the_read_modify_write_question(model):
    reason = model.by_mnemonic["ACCPUSH"]["vd"].reason
    assert "presence only" in reason


def test_an_overlapping_operand_field_is_refused_rather_than_probed_through(model):
    # Every bit of FUSED's two source fields is shared, so no two distinct values of either can be
    # encoded without moving the other. The probe must refuse rather than attribute whichever moved.
    for operand in ("vs1", "vs2"):
        verdict = model.by_mnemonic["FUSED"][operand]
        assert verdict.direction == ID.UNKNOWN
        assert "overlapping operand" in verdict.reason


def test_a_partly_overlapping_field_still_probes_through_its_clear_values(model):
    # ALIASED shares only its lowest source bit, so even values remain encodable and the operand is
    # probed rather than refused. Refusing here would throw away evidence the encoding does allow.
    assert len(ID.candidate_values(FIXTURE.fields_of("ALIASED"), "vs2", (0, 1, 16))) >= 2


def test_an_instruction_with_no_observable_effect_is_unknown_not_empty(model):
    verdicts = model.by_mnemonic["INVISIBLE"]
    assert all(v.direction == ID.UNKNOWN for v in verdicts.values())
    assert all("nothing about it is established" in v.reason or "no dependence" in v.reason
               for v in verdicts.values())
    assert not model.resolved("INVISIBLE")


def test_shared_bits_are_computed_from_the_layout_not_assumed():
    fields = FIXTURE.fields_of("ALIASED")
    shared = ID.shared_bits(fields)
    assert 19 in shared["vs1"] and 19 in shared["vs2"]
    assert shared["vd"] == frozenset()


def test_candidate_values_drop_anything_that_would_disturb_a_sibling():
    fields = FIXTURE.fields_of("ALIASED")
    # vs2 bit 0 lands on the bit it shares with vs1, so every odd value is dropped.
    assert all(v % 2 == 0 for v in ID.candidate_values(fields, "vs2", (0, 1, 2, 3, 4)))


def test_a_zero_settle_is_refused_because_it_makes_every_write_invisible():
    with pytest.raises(ID.DirectionError):
        ID.ProbeOps(scalar_imm="ADDI", scalar_upper="UPPER", stall="WAIT", halt="STOP",
                    settle=0).validate(FIXTURE)


def test_an_undefined_probe_op_is_refused_before_anything_is_emitted():
    with pytest.raises(ID.DirectionError):
        ID.ProbeOps(scalar_imm="NOPE", scalar_upper="UPPER", stall="WAIT",
                    halt="STOP").validate(FIXTURE)


def test_merging_two_preambles_keeps_the_stronger_evidence():
    # A preamble that pre-filled the destination CANNOT see a definition into it, so its weaker
    # verdict is a failure to observe rather than a contradiction; keeping the definition is what
    # makes two preambles resolve between them what neither resolves alone.
    a = ID.DirectionModel(target="t", by_mnemonic={
        "OP": {"x": ID.OperandDirection("OP", "x", ID.UNKNOWN, None, reason="not seen"),
               "y": ID.OperandDirection("OP", "y", ID.DEF, "f", reason="seen")}})
    b = ID.DirectionModel(target="t", by_mnemonic={
        "OP": {"x": ID.OperandDirection("OP", "x", ID.USE, "f", reason="seen"),
               "y": ID.OperandDirection("OP", "y", ID.USE, "f", reason="seen")}})
    assert a.merge(b).by_mnemonic["OP"]["x"].direction == ID.USE
    assert a.merge(b).by_mnemonic["OP"]["y"].direction == ID.DEF
    assert b.merge(a).by_mnemonic["OP"]["y"].direction == ID.DEF


def test_round_trips_through_json(model):
    again = ID.DirectionModel.from_json(model.to_json())
    assert again.summary()["operands"] == model.summary()["operands"]
    assert again.defs_of("VADD") == model.defs_of("VADD")


def test_an_oracle_that_publishes_no_state_is_a_refusal_not_a_silent_zero():
    with pytest.raises(ID.DirectionError):
        ID.state_from_debug_result({"halted": True})
