"""Semantic instruction PATTERNS + RISC-V R-type FORMAT base for the Vortex SIMT CUSTOM0 control surface —
the shipped ISA doc merlin's IsaModel introspects (the Vortex analog of the radiance/atlas isa_patterns).

This module is a PER-TARGET edge artifact (it legitimately names its own target's ISA); it is NOT shared
library code, so it may carry the concrete RISC-V field layout its own encoder uses. Merlin core holds none
of this — it derives the taxonomy by differential-probing the ``to_bytecode`` below (see
``oracle_helpers/isa_introspect.py``).

Scope + provenance: a mnemonic->ROLE taxonomy for the Vortex CUSTOM0 ops — funct7=0 warp/thread control
(tmc / wspawn / split / join / barrier / pred / wsync) and funct7=1 cooperative-thread primitives
(vote_all / vote_any / vote_uni / vote_ballot, shfl_up / shfl_down / shfl_bfly / shfl_idx), each cited to
the ``.insn r 0x0B, funct3, funct7, rd, rs1, rs2`` form in ``VORTEX_ISA_SPEC.md`` §3. The 32-bit R-type
layout below is the taxonomy substrate isa_introspect probes for roles — it is NOT the target's emit
encoding. The compute/memory emit word is the RTL-derived fixed-format word exposed by
``merlin.targetgen.isa_model.isa_model_for_target("vortex")``; a stock, unmodified LLVM assembles the
CUSTOM0 ops via inline ``.insn r`` (VORTEX_ISA_SPEC.md §2 — no compiler fork required or permitted).
"""
from __future__ import annotations


# --- register/operand type marker (carries the ISA-declared concept isa_introspect reads for roles) --------
class ScalarReg:
    """A general-purpose (x) register operand. ``reg_name`` carries the concept word isa_introspect maps to
    a semantic role (the SIMT CUSTOM0 ops are scalar-register-driven control/collectives -> 'scalar')."""
    reg_name = "scalar register"


# --- RISC-V standard 32-bit R-type encoding (opcode[6:0] rd[11:7] funct3[14:12] rs1[19:15] rs2[24:20]
#     funct7[31:25]) — the layout the ``.insn r`` forms assemble to on stock LLVM. Universal RV32 R-type,
#     not a target-specific fact. opcode/funct3/funct7 are captured from the subclass kwargs; rd/rs1/rs2 are
#     the free operand fields the merlin assembler packs. -----------------------------------------------------
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


# --- funct7=0 warp/thread control (VORTEX_ISA_SPEC.md §3.1). All are scalar-register-driven control ops with
#     no tensor/accumulator datapath, so the derived role is 'scalar' — merlin's matmul/memory role checks
#     correctly do not require them. Each pattern declares only the operand fields the op actually reads. ------
class WarpMaskControl:
    """A warp/thread-mask control op reading one scalar register (tmc: thread mask; join: split token)."""
    rs1: ScalarReg


class WarpSpawn:
    """Warp spawn: warp count + entry PC, two scalar registers (wspawn)."""
    rs1: ScalarReg
    rs2: ScalarReg


class WarpSplit:
    """Predicated split: token written to rd, predicate in rs1 (split)."""
    rd: ScalarReg
    rs1: ScalarReg


class WarpPredicate:
    """Predicate update: condition + thread mask, two scalar registers (pred)."""
    rs1: ScalarReg
    rs2: ScalarReg


class WarpBarrier:
    """Barrier: barrier id + participant warp count, two scalar registers (barrier)."""
    rs1: ScalarReg
    rs2: ScalarReg


class WarpSync:
    """Pipeline synchronise / drain — a bare fence with no register operands (wsync: rd=rs1=rs2=x0)."""
    # no rd/rs1/rs2 operands: wsync reads and writes no register (VORTEX_ISA_SPEC.md §3.1 funct3=7).


# --- funct7=1 cooperative-thread primitives (VORTEX_ISA_SPEC.md §3.2). Warp-level collectives whose operands
#     are scalar registers (result/predicate/value) -> 'scalar' role. --------------------------------------
class WarpVote:
    """Warp vote: result written to rd, per-lane predicate in rs1 (vote_all / vote_any / vote_uni /
    vote_ballot)."""
    rd: ScalarReg
    rs1: ScalarReg


class WarpShuffle:
    """Warp shuffle: value written to rd, source value in rs1, broadcast/control in rs2 (shfl_up /
    shfl_down / shfl_bfly / shfl_idx)."""
    rd: ScalarReg
    rs1: ScalarReg
    rs2: ScalarReg
