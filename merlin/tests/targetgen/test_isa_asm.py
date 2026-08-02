"""The derived assembler packs operands into exactly the model's field bits, and REFUSES rather than emit a
wrong word. Hermetic: a hand-built IsaModel with a synthetic field map (opcode 0x2B; rd[7:12] rs1[15:20]
rs2[20:25]) — no real target, no model venv.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.isa_model import IsaModel
from merlin.targetgen import isa_asm as A


def _model(fields=None) -> IsaModel:
    fields = fields or {"rd": [7, 8, 9, 10, 11], "rs1": [15, 16, 17, 18, 19], "rs2": [20, 21, 22, 23, 24]}
    by_mnem = {"FakeMatMul": {"class": "FakeMatMul", "role": "matmul", "opcode": 0x2B,
                              "fixed_value": 0x2B, "fixed_mask": 0xFF07C07F, "fields": fields}}
    return IsaModel(target="fake", by_mnemonic=by_mnem, asm_mnemonics={"MATMUL": "FakeMatMul"})


def _truth(rd, rs1, rs2) -> int:
    return 0x2B | (rd << 7) | (rs1 << 15) | (rs2 << 20)


def test_assemble_line_matches_the_field_layout():
    m = _model()
    assert A.assemble_line(m, "FakeMatMul", {"rd": 5, "rs1": 3, "rs2": 7}) == _truth(5, 3, 7)
    assert A.assemble_line(m, "MATMUL", {"rd": 1}) == _truth(1, 0, 0)      # alias + omitted fields -> 0
    assert A.assemble_line(m, "MATMUL", {}) == 0x2B                        # bare opcode


def test_assemble_text_with_comments_and_word_passthrough():
    m = _model()
    src = ("# a small kernel\n"
           "MATMUL rd=5, rs1=3, rs2=7\n"
           "  \n"
           "MATMUL rd=1  // second\n"
           ".word 0x13\n")
    assert A.assemble_text(m, src) == [_truth(5, 3, 7), _truth(1, 0, 0), 0x13]
    assert A.to_word_lines([0x2B, 0x13]) == ".word 0x0000002b\n.word 0x00000013\n"


def test_refuses_unknown_op_and_operand_and_overflow():
    m = _model()
    with pytest.raises(A.AssembleError, match="unknown instruction"):
        A.assemble_line(m, "NOPE", {})
    with pytest.raises(A.AssembleError, match="no operand"):
        A.assemble_line(m, "MATMUL", {"rd2": 1})
    with pytest.raises(A.AssembleError, match="does not fit"):
        A.assemble_line(m, "MATMUL", {"rd": 32})                          # rd is 5 bits, 32 overflows
    with pytest.raises(A.AssembleError, match="line 2"):
        A.assemble_text(m, "MATMUL rd=1\nMATMUL rd=99\n")                 # error carries the line number


def test_refuses_nonlinear_field():
    m = _model(fields={"rd": [7, -1, 9, 10, 11]})                          # bit 1 is non-linear
    with pytest.raises(A.AssembleError, match="non-linear"):
        A.assemble_line(m, "MATMUL", {"rd": 2})


def test_empty_model_has_no_assembler():
    with pytest.raises(A.AssembleError, match="no ISA definition"):
        A.assemble_text(IsaModel(target="bare"), "MATMUL rd=1\n")
