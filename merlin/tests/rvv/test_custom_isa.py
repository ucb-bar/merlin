"""Custom ISA via inline asm — no LLVM fork (``merlin-lower-inline-asm``).

The pass rewrite runs everywhere xDSL is present; the demonstrator compiles a genuinely
custom instruction into an rv64gcv object and confirms it in the disassembly (auto-skips
without the clang / chipyard toolchain).
"""
from __future__ import annotations

import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

MARKER = """
builtin.module {
  func.func @forward(%a: i32, %b: i32) -> i32 {
    %r = "merlin.inline_asm"(%a, %b)
        {asm_string = "add $0, $1, $2", constraints = "=r,r,r", has_side_effects}
        : (i32, i32) -> i32
    func.return %r : i32
  }
}
"""


def test_pass_lowers_marker_to_llvm_inline_asm():
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.custom_isa import lower_inline_asm

    m = parse_mlir_text(MARKER)
    assert lower_inline_asm(m) == 1
    names = [op.name for op in m.walk()]
    assert "llvm.inline_asm" in names              # 1:1 lowering
    assert all(getattr(op, "op_name", None) is None
               or op.op_name.data != "merlin.inline_asm" for op in m.walk())


def _toolchain():
    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import spike

    return toolchain.available() and spike.available()


@pytest.mark.skipif(not _toolchain(), reason="clang-23 / chipyard objdump missing")
def test_custom_instruction_emitted_without_llvm_fork(tmp_path):
    """A CUSTOM-0 instruction the toolchain has no mnemonic for lands in the binary."""
    from merlin.llvmlower.custom_isa import build_rvv_object, disassemble

    # .insn raw-encodes CUSTOM-0 (opcode 0x0b); rd=a0, rs1=a0, rs2=a1 -> word 0x00b5050b
    obj = build_rvv_object("merlin_vcix", ".insn r 0x0b, 0, 0, $0, $1, $2", "=r,r,r",
                           ["i32", "i32"], "i32", tmp_path)
    dis = disassemble(obj)
    assert "merlin_vcix" in dis
    assert "00b5050b" in dis                       # the exact custom encoding is present
    assert ".insn" in dis                          # disassembler has no mnemonic for it


@pytest.mark.skipif(not _toolchain(), reason="clang-23 / chipyard objdump missing")
def test_inline_asm_emits_named_instruction(tmp_path):
    """A standard instruction emitted 1:1 from inline asm shows its mnemonic."""
    from merlin.llvmlower.custom_isa import build_rvv_object, disassemble

    obj = build_rvv_object("merlin_xor", "xor $0, $1, $2", "=r,r,r",
                           ["i32", "i32"], "i32", tmp_path)
    dis = disassemble(obj)
    assert "merlin_xor" in dis
    assert "xor" in dis
