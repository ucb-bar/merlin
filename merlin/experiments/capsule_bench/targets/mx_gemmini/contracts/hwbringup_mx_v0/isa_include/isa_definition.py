"""MX-Gemmini command-word ISA definition — the shipped ISA doc merlin's IsaModel introspects so the
assembler/disassembler/linter round-trip the MX MMIO/RoCC trigger word. DERIVED (mxgemmini_mmio.h:66),
FAIL-CLOSED. Shared by the standalone ``mx_gemmini`` target and the radiance composite (which embeds the MX
PE as its contained low-bit unit).

The MX endpoint is command-buffer/MMIO driven: the only decodable 32-bit word is the composed RoCC INST word
(one variable field, the command selector). Everything else about MX — the MMIO register map, the per-format
microscaling dtypes, the E8M0 block scale, the bf16 accumulate — is DATA, not instruction encoding, and is
carried in the sibling ``mmio_abi.py`` (also derived, with provenance)."""
from __future__ import annotations

from isa_patterns import MxRoccWord


class MX_INVOKE(MxRoccWord):
    """The MX-Gemmini RoCC trigger word (mxgemmini_mmio.h:66). Operand ``imm`` = the 7-bit funct7 command
    selector at word bits [31:25]; opcode/rd/rs1/rs2/funct3 are fixed by the header composition."""
    opcode = 0x7B


class _IsaSpec:
    operations = {"mx.invoke": MX_INVOKE}


ISA = _IsaSpec()
