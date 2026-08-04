"""Encoding FORMAT + semantic pattern base for the MX-Gemmini MMIO/RoCC command word — the shipped ISA doc
merlin's IsaModel introspects to give the assembler/disassembler/linter to the MX target. PER-TARGET edge
artifact (may name its own ISA); merlin core holds none of this and derives the taxonomy by probing the
``to_bytecode`` below.

The MX PE is driven by a COMMAND-BUFFER / MMIO endpoint (not a decoded RoCC instruction stream): the driver
writes the RS1/RS2 MMIO regs and then stores one composed 32-bit RoCC word to the INST reg to trigger. The
word layout is fixed by the shipped header ``radiance-kernels/lib/include/mxgemmini_mmio.h:66``:

    (0x7B) | (0<<7) | (3<<12) | (1<<15) | (2<<20) | (funct<<25)
     opcode  rd=0     funct3=3  rs1=1     rs2=2     funct7=<command>

Only ``funct7`` (the command selector) varies per invocation; rd/rs1/rs2/funct3 are constants baked by the
header. We model that faithfully: the single variable field is the command, exposed as the ``imm`` operand at
word bits [31:25]; everything else is fixed (so a malformed word is flagged by the derived decode signature).
"""
from __future__ import annotations

# The RoCC opcode + fixed operand constants, transcribed from mxgemmini_mmio.h:66 (the header's own word
# composition). custom-3 == 0x7B matches OpcodeSet.custom3 in the chipyard gemmini config.
_ROCC_OPCODE = 0x7B      # bits [6:0]
_FIXED_RD = 0            # bits [11:7]
_FIXED_FUNCT3 = 3        # bits [14:12]  (xd=0,xs1=1,xs2=1)
_FIXED_RS1 = 1           # bits [19:15]
_FIXED_RS2 = 2           # bits [24:20]


class CommandField(int):
    """The 7-bit funct7 command selector (word bits [31:25]) — the one variable field of the MX RoCC word."""


class MxRoccWord:
    """Semantic + format base: composes exactly the mxgemmini_mmio.h:66 RoCC INST word. It carries NO
    ``opcode`` class attr (so the base itself is not mistaken for a concrete op); the concrete op class in
    isa_definition sets ``opcode``. The fixed rd/rs1/rs2/funct3 are baked (not read from the instance) so they
    decode as fixed bits, and ``self.imm`` (the command) is the single operand, placed at [31:25]."""
    imm: CommandField

    def to_bytecode(self) -> int:
        cmd = int(getattr(self, "imm", 0) or 0) & 0x7F
        return ((_ROCC_OPCODE & 0x7F)
                | ((_FIXED_RD & 0x1F) << 7)
                | ((_FIXED_FUNCT3 & 0x7) << 12)
                | ((_FIXED_RS1 & 0x1F) << 15)
                | ((_FIXED_RS2 & 0x1F) << 20)
                | (cmd << 25)) & 0xFFFFFFFF
