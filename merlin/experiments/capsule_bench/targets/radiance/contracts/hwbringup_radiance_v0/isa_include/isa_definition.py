"""Radiance/Muon (Vortex) custom-ISA definition — the shipped ISA doc merlin's IsaModel introspects to give
the raw-ISA developer tools (assembler / disassembler / linter) to the SIMT target, exactly as the atlas
``isa_definition.py`` does for its self-hosted core.

DERIVED (with provenance), FAIL-CLOSED. Each op below is the standard 32-bit RISC-V ``.insn r`` form the
shipped header ``radiance-kernels/lib/include/vx_intrinsics.h`` composes for the Vortex CUSTOM0 warp-control
surface — i.e. what stock LLVM emits for merlin's derive-don't-fork SIMT-divergence lowering. Fields cited:

  * CUSTOM opcode constants — ``vx_intrinsics.h:36-39`` (RISCV_CUSTOM0=0x0B .. CUSTOM3=0x7B), matching
    ``RISCVInstrInfoVX.td:3-6``.
  * per-op funct3/funct7 — the ``.insn r CUSTOM0, <funct3>, <funct7>, ...`` strings in ``vx_intrinsics.h``
    (line refs per op) cross-checked against the ``RVInstR<funct7,funct3,opcode>`` defs in
    ``RISCVInstrInfoVX.td`` (line refs per op).

RESIDUE recorded as UNKNOWN (not fabricated) — these need the Muon-LLVM fork's 64-bit instruction format
(8-bit reg fields + ``ext2`` address-space bits) and have NO faithful standard-32-bit ``.insn r`` encoding,
so emitting them here would be a fabricated encoding:
  * ``sw.shared`` / ``lw.shared`` / ``fence.s`` — the ``.shared`` qualifier lives in ``ext2`` (Inst{8-7}),
    which overlaps rd in the 32-bit layout (RISCVInstrInfo.td:675-699,771).
  * ``fexp.h`` / ``fnexp.h`` / ``fexp.s`` / ``fnexp.s`` — 64-bit ``NuCustomFPUnaryFrm`` with ext2=0b01
    (RISCVInstrInfoNU.td:45-51,86-109).
  * ``nu.invoke*`` — 64-bit ``NeutrinoInst`` (RISCVInstrInfoNU.td:3-81).
  * ``vx_tex`` / ``vx_rop`` / ``vx_cmov`` — R4 (4-register) forms (RISCVInstrInfoVX.td:52-54; vx_cmov is
    header-only, no .td def → doubly UNKNOWN).
  * ``vx_bar`` has TWO conflicting encodings across the toolchain (a commented-out CUSTOM0/funct3=4 native
    def in VX.td:34-37 vs an InstAlias to a CUSTOM2 ``nu.invoke`` in NU.td:114); the header still emits the
    native CUSTOM0/funct3=4 form (vx_intrinsics.h:154), which is what we model as VX_BARRIER below, and the
    nu.invoke alias is recorded as residue.
"""
from __future__ import annotations

from isa_patterns import (
    RType,
    WarpBarrier,
    WarpMaskControl,
    WarpPredicate,
    WarpRaster,
    WarpSpawn,
    WarpSplit,
)

# opcode constant, derived from the header's own literal (vx_intrinsics.h:36) == RISCVInstrInfoVX.td:3
CUSTOM0 = 0x0B


class VX_TMC(WarpMaskControl, RType, opcode=CUSTOM0, funct3=0, funct7=0):
    """Set thread mask. vx_intrinsics.h:110 `.insn r CUSTOM0,0,0,x0,mask,x0`; VX.td:10 RVInstR<0,0,CUSTOM0>."""


class VX_WSPAWN(WarpSpawn, RType, opcode=CUSTOM0, funct3=1, funct7=0):
    """Spawn warps. vx_intrinsics.h:136 `.insn r CUSTOM0,1,0,x0,n,fn`; VX.td:15 RVInstR<0,1,CUSTOM0>."""


class VX_SPLIT(WarpSplit, RType, opcode=CUSTOM0, funct3=2, funct7=0):
    """Predicated split. vx_intrinsics.h:142 `.insn r CUSTOM0,2,0,rd,pred,x0`; VX.td:19 RVInstR<0,2,CUSTOM0>."""


class VX_JOIN(WarpMaskControl, RType, opcode=CUSTOM0, funct3=3, funct7=0):
    """Reconverge. vx_intrinsics.h:148 `.insn r CUSTOM0,3,0,x0,sp,x0`; VX.td:27 RVInstR<0,3,CUSTOM0>."""


class VX_BARRIER(WarpBarrier, RType, opcode=CUSTOM0, funct3=4, funct7=0):
    """Barrier. vx_intrinsics.h:154 `.insn r CUSTOM0,4,0,x0,id,n` (native form; see module residue note)."""


class VX_PRED(WarpPredicate, RType, opcode=CUSTOM0, funct3=5, funct7=0):
    """Predicate. vx_intrinsics.h:129 `.insn r CUSTOM0,5,0,x0,cond,mask`; VX.td:39 RVInstR<0,5,CUSTOM0>."""


class VX_RAST(WarpRaster, RType, opcode=CUSTOM0, funct3=0, funct7=1):
    """Raster fetch. VX.td:47 RVInstR<1,0,CUSTOM0> (funct7=1)."""


class _IsaSpec:
    """The assembler-mnemonic → op-class map isa_introspect discovers (an object exposing ``operations``),
    so a kernel written in Vortex mnemonics maps back to semantic classes. Mnemonics per vx_intrinsics.h."""
    operations = {
        "vx_tmc": VX_TMC,
        "vx_wspawn": VX_WSPAWN,
        "vx_split": VX_SPLIT,
        "vx_join": VX_JOIN,
        "vx_barrier": VX_BARRIER,
        "vx_bar": VX_BARRIER,
        "vx_pred": VX_PRED,
        "vx_rast": VX_RAST,
    }


ISA = _IsaSpec()
