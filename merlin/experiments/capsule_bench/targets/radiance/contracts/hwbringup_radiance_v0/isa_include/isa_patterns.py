"""Semantic instruction PATTERNS + RISC-V encoding FORMAT bases for the Radiance/Muon (Vortex) SIMT
custom-ISA control surface — the shipped ISA doc the merlin IsaModel introspects (the Muon analog of
the atlas ``npu_model.isa_patterns`` + ``npu_model.isa`` format bases).

This module is a PER-TARGET edge artifact (it legitimately names its own target's ISA); it is NOT shared
library code, so it may carry the concrete RISC-V field layout its own encoder uses — exactly as the atlas
``isa_definition.py`` carries ``opcode=0b1110111`` etc. Merlin core holds none of this; it derives the
taxonomy by differential-probing the ``to_bytecode`` below (see ``oracle_helpers/isa_introspect.py``).

Scope + provenance (fail-closed): we model the **stock-LLVM-emittable, standard 32-bit RISC-V ``.insn r``
warp-control surface** — the Vortex CUSTOM0 ops whose raw encoding the shipped header
``radiance-kernels/lib/include/vx_intrinsics.h`` composes as ``.insn r CUSTOM0, funct3, funct7, rd, rs1,
rs2``. That is precisely the SIMT-divergence surface merlin's derive-don't-fork lowering targets on stock
LLVM (tmc / split / join / pred / wspawn / barrier / rast). The Muon-LLVM fork additionally defines these
ops in a 64-bit instruction format (``RISCVInstrInfoVX.td``) with 8-bit register fields and an ``ext2``
address-space qualifier; the mnemonic-only ops (``sw.shared`` / ``fexp.h`` / ``nu.invoke`` / ``vx_tex``)
are expressible ONLY in that 64-bit fork format and are recorded as honest residue in ``isa_definition.py``
(they are not emitted here rather than emitted with a fabricated 32-bit encoding).
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
