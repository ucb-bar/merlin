"""Semantic instruction PATTERNS + RISC-V encoding FORMAT bases for the Radiance/Muon (Vortex) SIMT
custom-ISA control surface — the shipped ISA doc the merlin IsaModel introspects (the Muon analog of
the atlas ``npu_model.isa_patterns`` + ``npu_model.isa`` format bases).

This module is a PER-TARGET edge artifact (it legitimately names its own target's ISA); it is NOT shared
library code, so it may carry the concrete RISC-V field layout its own encoder uses — exactly as the atlas
``isa_definition.py`` carries ``opcode=0b1110111`` etc. Merlin core holds none of this; it derives the
taxonomy by differential-probing the ``to_bytecode`` below (see ``oracle_helpers/isa_introspect.py``).

Scope + provenance: this module is a mnemonic→ROLE taxonomy for the Vortex CUSTOM0 warp-control ops
(tmc / split / join / pred / wspawn / barrier / rast), each cited to the ``.insn r CUSTOM0, funct3,
funct7, rd, rs1, rs2`` form the shipped header ``radiance-kernels/lib/include/vx_intrinsics.h`` composes.
The 32-bit R-type layout below is the taxonomy substrate isa_introspect probes for roles — it is NOT the
target's emit encoding.

The emit encoding for EVERY op (warp-control AND compute AND memory) is the target's 64-bit FIXED-FORMAT
word, whose field layout + opcode table are DERIVED from the RTL decoder and exposed by
``merlin.targetgen.isa_model.isa_model_for_target("radiance")`` (opcodes incl. ``MADD``/``OP_FP`` compute,
``LOAD``/``STORE`` with an ``ext2`` address-space field global=0/shared=1). That derived word is what the
shipped ``isa_tools.py disasm``/``lint`` decode against and what merlin's FORK-FREE backend emits into
(stock rv32 compile → ``isa_transcode`` re-encode; no Muon-LLVM fork needed). See the corrected residue
note in ``isa_definition.py`` and the worked ``example_kernel/gemm_tile.S`` — a real, assembled,
disassembly-clean compute kernel. Compute here is DERIVABLE, not a fork-only capability.
"""
from __future__ import annotations


# --- register/operand type markers (carry the ISA-declared concept isa_introspect reads for roles) ------
class ScalarReg:
    """A general-purpose (x) register operand. ``reg_name`` carries the concept word isa_introspect maps to
    a semantic role (here: the SIMT warp-control ops are scalar-register-driven → the 'scalar' role)."""
    reg_name = "scalar register"


# --- RISC-V standard 32-bit R-type encoding (the format the vx_intrinsics.h ``.insn r`` forms assemble to
#     on stock LLVM). opcode/funct3/funct7 are captured as class attrs from the subclass kwargs; rd/rs1/rs2
#     are the free operand fields the merlin assembler packs. This is the universal RV32 R-type field layout
#     (opcode[6:0] rd[11:7] funct3[14:12] rs1[19:15] rs2[24:20] funct7[31:25]) — not a target-specific fact.
class RType:
    opcode: int | None = None
    funct3: int | None = None
    funct7: int | None = None

    def __init_subclass__(cls, *, opcode=None, funct3=None, funct7=None, **kw):
        super().__init_subclass__(**kw)
        if opcode is not None:
            cls.opcode = int(opcode)
        if funct3 is not None:
            cls.funct3 = int(funct3)
        if funct7 is not None:
            cls.funct7 = int(funct7)

    def to_bytecode(self) -> int:
        op = int(self.opcode or 0) & 0x7F
        f3 = int(self.funct3 or 0) & 0x7
        f7 = int(self.funct7 or 0) & 0x7F
        rd = int(getattr(self, "rd", 0) or 0) & 0x1F
        rs1 = int(getattr(self, "rs1", 0) or 0) & 0x1F
        rs2 = int(getattr(self, "rs2", 0) or 0) & 0x1F
        return (op | (rd << 7) | (f3 << 12) | (rs1 << 15) | (rs2 << 20) | (f7 << 25)) & 0xFFFFFFFF


# --- semantic patterns (typed operands → the DERIVED role). All Vortex warp-control ops are scalar-driven
#     control (thread-mask / warp-split / barrier); they carry no tensor/accumulator datapath, so the derived
#     role is 'scalar' (control), and merlin's matmul/memory role checks correctly do not require them. -----
class WarpMaskControl:
    """A warp/thread-mask control op reading one scalar register (tmc / join)."""
    rs1: ScalarReg


class WarpSpawn:
    """Warp spawn: count + entry, two scalar registers (wspawn)."""
    rs1: ScalarReg
    rs2: ScalarReg


class WarpSplit:
    """Predicated split: dest mask + predicate register (split)."""
    rd: ScalarReg
    rs1: ScalarReg


class WarpPredicate:
    """Predicate update: condition + mask, two scalar registers (pred)."""
    rs1: ScalarReg
    rs2: ScalarReg


class WarpBarrier:
    """Barrier: barrier id + participant count, two scalar registers (barrier/bar)."""
    rs1: ScalarReg
    rs2: ScalarReg


class WarpRaster:
    """Raster fetch: writes one scalar register (rast)."""
    rd: ScalarReg
