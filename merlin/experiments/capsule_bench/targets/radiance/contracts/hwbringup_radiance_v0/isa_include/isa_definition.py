"""Radiance/Muon (Vortex) custom-ISA definition — the shipped ISA doc merlin's IsaModel introspects to map
each SIMT WARP-CONTROL mnemonic to its semantic ROLE, exactly as the atlas ``isa_definition.py`` does for
its self-hosted core.

Scope of THIS file: only the CUSTOM0 warp-control surface (tmc / wspawn / split / join / barrier / pred /
rast). It is a mnemonic→role taxonomy, not the emit encoding. Each op is cited to the standard RISC-V
``.insn r`` form the shipped header ``radiance-kernels/lib/include/vx_intrinsics.h`` composes for that op;
merlin's introspection differential-probes the ``to_bytecode`` below to recover the role. Fields cited:

  * CUSTOM opcode constants — ``vx_intrinsics.h:36-39`` (RISCV_CUSTOM0=0x0B .. CUSTOM3=0x7B), matching
    ``RISCVInstrInfoVX.td:3-6``.
  * per-op funct3/funct7 — the ``.insn r CUSTOM0, <funct3>, <funct7>, ...`` strings in ``vx_intrinsics.h``
    (line refs per op) cross-checked against the ``RVInstR<funct7,funct3,opcode>`` defs in
    ``RISCVInstrInfoVX.td`` (line refs per op).

WHERE THE REAL COMPUTE ENCODING LIVES (read this before concluding "compute is unencodable"): this file
is NOT the source of truth for how you EMIT an instruction. The target's true instruction is a 64-bit
FIXED-FORMAT word, and its exact field layout + opcode table are DERIVED from the RTL decoder and exposed
by ``merlin.targetgen.isa_model.isa_model_for_target("radiance")`` — 24 opcodes incl. compute
(``MADD`` / ``MSUB`` / ``NM_ADD`` / ``NM_SUB`` = fused FP multiply-add, ``OP_FP``, ``OP``, ``OP_IMM``),
memory (``LOAD`` / ``STORE`` with the ``ext2`` field selecting the address space: global=0, shared=1),
and the neutrino ops (``NU_INVOKE`` …). That is the layout the shipped ``isa_tools.py disasm``/``lint``
decode your emitted ``.word``/``.quad`` against, and the layout merlin's own FORK-FREE backend emits into
(stock rv32 compile → ``isa_transcode.FixedFormatTranscoder`` re-encode). There is NO Muon-LLVM fork in
that path and none is needed. A ready-made, assembled, disassembly-clean proof is shipped alongside this
file at ``example_kernel/gemm_tile.S`` (a real f32 GEMM tile with ``LOAD`` / ``MADD`` / ``STORE``).

So the following ops — which an earlier note wrongly recorded as "no faithful encoding, needs the fork" —
ARE encodable in that derived 64-bit word; the accurate statement is only about what stock-LLVM
compilation currently LOWERS to, not about the encoding's existence:
  * ``sw.shared`` / ``lw.shared`` / ``fence.s`` — the ``.shared`` qualifier is the derived ``ext2`` field
    (global=0, shared=1); a scratchpad access sets ext2=1. Fully encodable and checkable via the tools.
  * ``nu.invoke*`` — the ``NU_INVOKE`` opcode family is in the derived opcode table; encodable.
  * ``fexp.h`` / ``fnexp.h`` / ``fexp.s`` / ``fnexp.s`` — SFU transcendentals. The opcode word is
    derivable; what is genuinely residual is that stock LLVM emits these as libcalls (not a single
    instruction), so merlin's fork-free codegen does not auto-lower them to one HW op yet — a lowering
    gap, NOT an unencodable instruction.
  * ``vx_tex`` / ``vx_rop`` / ``vx_cmov`` — R4 (4-register) graphics forms (they use the derived ``rs3``
    field); not exercised by the compute capsules. ``vx_cmov`` is header-only (no .td def).
  * ``vx_bar`` has TWO spellings across the toolchain (a native CUSTOM0/funct3=4 def commented out in
    VX.td:34-37 vs an InstAlias to a CUSTOM2 ``nu.invoke`` in NU.td:114); the header emits the native
    CUSTOM0/funct3=4 form (vx_intrinsics.h:154), which is what we model as VX_BARRIER below.
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
