"""The derived assembler serves BOTH encoding shapes an IsaModel can carry.

A target's ISA arrives in one of two derived shapes, and which one it is follows from that target's own
RTL, never from its name:

  * **signature** — per-instruction fixed bits + operand bit maps (``by_mnemonic``), from a shipped ISA
    definition;
  * **fixed-format** — one field layout selected by an opcode field (``field_layout`` + ``opcode_table``),
    what an mlc/RTL decoder derivation produces for a wide-word core. ``by_mnemonic`` is EMPTY by design.

``assemble_text`` used to gate on ``is_empty()`` — "has no ``by_mnemonic``" — which is true of every
fixed-format model, so it refused them with "this target ships no ISA definition" while that target's own
encoder (``assemble_fixed``) sat in the same module, working. The tools that read the same model
(``disasm``/``lint``) branch on ``is_fixed_format()`` and were unaffected, so the ISA derivation looked
healthy while the one tool that turns it into emittable words was unreachable — and the refusal named the
HARDWARE as the thing at fault.

Hermetic: hand-built models, no real target, no model venv, no mlc.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import isa_asm as A
from merlin.targetgen.isa_model import IsaModel

#: A synthetic fixed-format ISA: 64-bit word, opcode in [6:0], three operand fields. Deliberately NOT any
#: real target's layout — the point is the SHAPE, not a particular device.
_LAYOUT = {"opcode": (6, 0), "rd": (16, 9), "rs1": (27, 20), "rs2": (35, 28)}
_OPCODES = {"OPX": 0x33, "LOADX": 0x03, "CUSTOMX": 0x0B}


def _fixed(width: int = 64) -> IsaModel:
    return IsaModel(target="fake_fixed", by_mnemonic={}, asm_mnemonics={},
                    inst_width=width, field_layout=dict(_LAYOUT), opcode_table=dict(_OPCODES))


def _signature() -> IsaModel:
    by_mnem = {"FakeOp": {"class": "FakeOp", "role": "matmul", "opcode": 0x2B, "fixed_value": 0x2B,
                          "fixed_mask": 0xFF07C07F,
                          "fields": {"rd": [7, 8, 9, 10, 11], "rs1": [15, 16, 17, 18, 19]}}}
    return IsaModel(target="fake_sig", by_mnemonic=by_mnem, asm_mnemonics={"FOP": "FakeOp"})


def _truth(op: str, **ops) -> int:
    word = _OPCODES[op] << _LAYOUT["opcode"][1]
    for name, value in ops.items():
        word |= value << _LAYOUT[name][1]
    return word


def test_a_fixed_format_model_is_empty_but_still_assembles():
    """The precondition that made this bug invisible: the model IS ``is_empty()`` and IS assemblable."""
    m = _fixed()
    assert m.is_empty()                       # no by_mnemonic — by design, not a defect
    assert m.is_fixed_format()
    assert A.assemble_text(m, "OPX rd=3, rs1=1, rs2=2\n") == [_truth("OPX", rd=3, rs1=1, rs2=2)]


def test_a_model_with_neither_shape_is_still_refused():
    """The refusal must survive for a target that genuinely ships no ISA — this is not a blanket opening."""
    empty = IsaModel(target="fake_none", by_mnemonic={}, asm_mnemonics={})
    assert empty.is_empty() and not empty.is_fixed_format()
    with pytest.raises(A.AssembleError, match="ships no ISA definition"):
        A.assemble_text(empty, "ANYTHING rd=1\n")


def test_the_signature_shape_is_unchanged():
    """The pre-existing path must encode bit-for-bit as before — this fix adds a shape, it replaces none."""
    m = _signature()
    assert A.assemble_text(m, "FOP rd=5, rs1=3\n") == [0x2B | (5 << 7) | (3 << 15)]


def test_a_signature_entry_wins_over_a_same_named_opcode():
    """A model may carry both shapes; the per-instruction table is more specific, so it is tried first."""
    m = _fixed()
    m = IsaModel(target="fake_both", by_mnemonic=_signature().by_mnemonic,
                 asm_mnemonics={"OPX": "FakeOp"},        # alias 'OPX' also names a fixed-format opcode
                 inst_width=64, field_layout=dict(_LAYOUT), opcode_table=dict(_OPCODES))
    # Resolves through by_mnemonic (signature), NOT the opcode table.
    assert A.assemble_text(m, "OPX rd=5, rs1=3\n") == [0x2B | (5 << 7) | (3 << 15)]


def test_an_unknown_mnemonic_names_the_vocabulary_the_model_actually_has():
    """A fixed-format model's vocabulary is its opcode table; reporting the empty by_mnemonic instead told
    the agent "defined: (none)" for a model defining three instructions."""
    m = _fixed()
    with pytest.raises(A.AssembleError) as e:
        A.assemble_text(m, "NOSUCHOP rd=1\n")
    msg = str(e.value)
    assert "NOSUCHOP" in msg
    for op in _OPCODES:
        assert op in msg, f"error names no vocabulary the agent can use: {msg}"


def test_a_hand_placed_literal_keeps_the_models_width():
    """``.word``/``.quad`` passthrough was masked to 32 bits, so a wide literal silently lost its high half
    and was emitted as a different instruction."""
    wide = 0x0000000B_00000033
    assert A.assemble_text(_fixed(64), f".quad 0x{wide:016x}\n") == [wide]
    assert A.assemble_text(_fixed(64), f".word 0x{wide:016x}\n") == [wide]
    # A 32-bit model still truncates to its own width — the mask follows the model, not a constant.
    assert A.assemble_text(_signature(), f".word 0x{wide:016x}\n") == [wide & 0xFFFFFFFF]


def test_emitted_directives_match_the_width_llvm_mc_lays_out():
    """A wide word must be emitted as ``.quad`` (8 LE bytes); as ``.word`` the oracle would read back two
    unrelated half-instructions."""
    words = [_truth("OPX", rd=3), _truth("LOADX", rd=5)]
    wide = A.to_data_lines(words, 64)
    assert wide.splitlines() == [f".quad 0x{w:016x}" for w in words]
    narrow = A.to_data_lines([0x2B, 0x13], 32)
    assert narrow == ".word 0x0000002b\n.word 0x00000013\n"


def test_the_assembler_accepts_every_model_the_broker_dispatches():
    """The broker refuses only a model that is BOTH signature-empty and not fixed-format. The assembler
    must not apply a STRICTER test one layer down — that mismatch is what made a dispatched request fail
    with a message about the hardware."""
    def broker_would_dispatch(m: IsaModel) -> bool:
        return not (m.is_empty() and not m.is_fixed_format())

    for m in (_fixed(64), _fixed(32), _signature()):
        assert broker_would_dispatch(m)
        A.assemble_text(m, "")               # must not raise the "ships no ISA" refusal
