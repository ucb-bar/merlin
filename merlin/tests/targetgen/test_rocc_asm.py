"""The derived RoCC assembler emits canonical, DECODABLE llvm.inline_asm MLIR.

It exists so an inline_asm_insn agent (e.g. gemmini) can't hand-roll the invalid inline-integer-literal
operand form that neither assembles nor decodes. Every emitted program must round-trip through
rocc_decode to exactly the requested classes (no UNKNOWN), and the encoder must refuse a class/funct or a
CONFIG subtype the target's derived facts don't permit.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.rocc import asm as A
from merlin.targetgen.rocc import decode as RD

_MATMUL = [
    ("CONFIG_EX", 0x0, 0), ("CONFIG_LD", 0x1, 16),
    ("MVIN", 0x80000000, 16), ("MVIN", 0x80000100, 16),
    ("CONFIG_ST", 0x2, 16), ("PRELOAD", 0x100, 0),
    ("COMPUTE_PRELOADED", 0, 0), ("MVOUT", 0xA0000000, 16),
    ("FLUSH", 0, 0), ("FENCE", 0, 0),
]


def test_emitted_program_roundtrips_to_requested_classes():
    mlir = A.assemble_program("gemmini", _MATMUL)
    got = [i["class"] for i in RD.decode_text(mlir, source="t", target="gemmini")["instructions"]]
    assert got == [c for c, _, _ in _MATMUL]
    assert "UNKNOWN" not in got


def test_operands_are_ssa_constants_not_inline_literals():
    mlir = A.assemble_program("gemmini", [("MVIN", 0x80000000, 16)])
    assert "llvm.mlir.constant" in mlir
    # the operands passed to inline_asm are %-names, never bare integer literals in the operand list
    asm_line = [l for l in mlir.splitlines() if "inline_asm" in l][0]
    after = asm_line.rsplit('"', 1)[1]      # text after the constraints string
    assert "%" in after and "(0x80000000" not in after and "(2147483648" not in after


def test_refuses_wrong_config_subtype_bits():
    with pytest.raises(A.AsmError):
        A.assemble_program("gemmini", [("CONFIG_EX", 0x1, 0)])   # rs1&3==1 but asked EX


def test_refuses_unknown_class():
    with pytest.raises(A.AsmError):
        A.assemble_program("gemmini", [("FROBNICATE", 0, 0)])


def test_assemble_text_parses_listing():
    got = [i["class"] for i in RD.decode_text(
        A.assemble_text("gemmini", "# a config\nCONFIG_EX 0 0\nMVIN 0x80000000 16\nFENCE"),
        source="t", target="gemmini")["instructions"]]
    assert got == ["CONFIG_EX", "MVIN", "FENCE"]
