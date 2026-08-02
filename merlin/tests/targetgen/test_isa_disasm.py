"""The derived disassembler decodes words back to {mnemonic, operands} and coverage diffs present-vs-required
instruction classes. Hermetic: a synthetic 2-op model (a matmul + a load), assembled with isa_asm and decoded
back — proving the encode/decode pair round-trips and illegal words + coverage gaps surface. No real target.
"""
from __future__ import annotations

from merlin.targetgen.isa_model import IsaModel
from merlin.targetgen import isa_asm as A
from merlin.targetgen import isa_disasm as D


def _sig(opcode: int, fields: dict) -> tuple[int, int]:
    variable = 0
    for bits in fields.values():
        for b in bits:
            if isinstance(b, int) and b >= 0:
                variable |= (1 << b)
    mask = (~variable) & 0xFFFFFFFF
    return mask, opcode & mask


def _model() -> IsaModel:
    mm_fields = {"rd": [7, 8, 9, 10, 11], "rs1": [15, 16, 17, 18, 19], "rs2": [20, 21, 22, 23, 24]}
    ld_fields = {"rd": [7, 8, 9, 10, 11], "rs1": [15, 16, 17, 18, 19]}
    mm_mask, mm_val = _sig(0x2B, mm_fields)
    ld_mask, ld_val = _sig(0x03, ld_fields)
    by_mnem = {
        "MatMul": {"class": "MatMul", "role": "matmul", "fixed_mask": mm_mask, "fixed_value": mm_val,
                   "fields": mm_fields},
        "Load": {"class": "Load", "role": "memory", "fixed_mask": ld_mask, "fixed_value": ld_val,
                 "fields": ld_fields},
    }
    return IsaModel(target="fake", by_mnemonic=by_mnem,
                    roles={"matmul": ["MatMul"], "memory": ["Load"]})


def test_disassemble_round_trips_encode():
    m = _model()
    src = "Load rd=4, rs1=2\nMatMul rd=5, rs1=3, rs2=7\n"
    words = A.assemble_text(m, src)
    recs = D.disassemble(m, words)
    assert [r["mnemonic"] for r in recs] == ["Load", "MatMul"]
    assert recs[0]["operands"] == {"rd": 4, "rs1": 2}
    assert recs[1]["operands"] == {"rd": 5, "rs1": 3, "rs2": 7}
    assert recs[1]["role"] == "matmul"
    # and re-assembling each decoded record reproduces the same word (encode/decode are inverses)
    for r, w in zip(recs, words):
        assert A.assemble_line(m, r["mnemonic"], r["operands"]) == w


def test_invented_encoding_is_illegal():
    m = _model()
    recs = D.disassemble(m, [0xFFFFFFFF, 0xDEADBEEF])
    assert all(r["illegal"] and r["mnemonic"] is None for r in recs)


def test_coverage_flags_missing_class_and_illegal():
    m = _model()
    # a "matmul" capsule needs both the memory load and the matmul; a kernel with only MatMul is short.
    only_mm = D.disassemble(m, A.assemble_text(m, "MatMul rd=1, rs1=1, rs2=1\n"))
    cov = D.coverage(m, only_mm, op="matmul")
    assert set(cov["required"]) == {"Load", "MatMul"}
    assert cov["present"] == ["MatMul"]
    assert cov["missing"] == ["Load"]
    assert cov["n_illegal"] == 0
    # explicit required list + an illegal word
    cov2 = D.coverage(m, D.disassemble(m, [0x0]), required=["MatMul"])
    assert cov2["missing"] == ["MatMul"] and cov2["n_illegal"] == 1


def test_empty_model_everything_illegal():
    recs = D.disassemble(IsaModel(target="bare"), [0x2B, 0x03])
    assert all(r["illegal"] for r in recs)
