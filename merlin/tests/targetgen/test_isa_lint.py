"""The static ISA linter: illegal-opcode (always) and halt-present (once the halt op set is derived), with an
honest INFO when termination can't be verified. Hermetic synthetic model, no real target.
"""
from __future__ import annotations

from merlin.targetgen.isa_model import IsaModel
from merlin.targetgen import isa_asm as A
from merlin.targetgen import isa_lint as L


def _sig(opcode: int, fields: dict) -> tuple[int, int]:
    var = 0
    for bits in fields.values():
        for b in bits:
            if isinstance(b, int) and b >= 0:
                var |= (1 << b)
    mask = (~var) & 0xFFFFFFFF
    return mask, opcode & mask


def _model(halt=()) -> IsaModel:
    mm_f = {"rd": [7, 8, 9, 10, 11], "rs1": [15, 16, 17, 18, 19]}
    hl_f: dict = {}                                          # terminator takes no operands
    mm_mask, mm_val = _sig(0x2B, mm_f)
    hl_mask, hl_val = _sig(0x73, hl_f)
    by_mnem = {
        "MatMul": {"class": "MatMul", "role": "matmul", "fixed_mask": mm_mask, "fixed_value": mm_val,
                   "fields": mm_f},
        "Halt": {"class": "Halt", "role": "scalar", "fixed_mask": hl_mask, "fixed_value": hl_val,
                 "fields": hl_f},
    }
    return IsaModel(target="fake", by_mnemonic=by_mnem, roles={"matmul": ["MatMul"]},
                    halt_mnemonics=halt)


def _rules(findings):
    return {f["rule"] for f in findings}


def test_illegal_opcode_is_flagged():
    m = _model()
    findings = L.lint(m, [0xFFFFFFFF])                       # matches no signature
    assert any(f["rule"] == "illegal_opcode" and f["severity"] == "error" for f in findings)


def test_halt_present_when_halt_set_known():
    m = _model(halt=("Halt",))
    good = A.assemble_text(m, "MatMul rd=1, rs1=1\nHalt\n")
    assert "no_halt" not in _rules(L.lint(m, good))
    bad = A.assemble_text(m, "MatMul rd=1, rs1=1\n")         # never terminates
    f = L.lint(m, bad)
    assert any(x["rule"] == "no_halt" and x["severity"] == "error" for x in f)


def test_halt_not_last_is_a_warning():
    m = _model(halt=("Halt",))
    words = A.assemble_text(m, "Halt\nMatMul rd=1, rs1=1\n")  # terminator present but not last
    assert "halt_not_last" in _rules(L.lint(m, words))


def test_halt_unknown_is_info_not_a_false_error():
    m = _model(halt=())                                       # halt set not derived
    f = L.lint(m, A.assemble_text(m, "MatMul rd=1, rs1=1\n"))
    assert "halt_unknown" in _rules(f)
    assert "no_halt" not in _rules(f)                        # never a false termination verdict
    assert all(x["severity"] != "error" for x in f)


def test_empty_model_is_info():
    f = L.lint(IsaModel(target="bare"), [0x2B])
    assert f == [{"rule": "no_isa_model", "severity": "info",
                  "detail": "this target ships no ISA definition; static ISA lint is unavailable"}]


def test_format_findings_orders_by_severity():
    m = _model(halt=("Halt",))
    txt = L.format_findings(L.lint(m, [0xFFFFFFFF]))
    assert txt.startswith("[ERROR]")
