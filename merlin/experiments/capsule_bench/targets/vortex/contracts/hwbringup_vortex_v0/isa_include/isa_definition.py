"""Vortex custom-ISA definition — the shipped ISA doc merlin's IsaModel introspects to map each SIMT
CUSTOM0 mnemonic to its semantic ROLE. Analog of the radiance (Muon/Vortex) and atlas isa_definition.py.

Scope of THIS file: the CUSTOM0 SIMT control surface (opcode 0x0B), a mnemonic->role taxonomy — NOT the
emit encoding. The encoding facts are the CUSTOM0 R-type funct7/funct3 map from ``VORTEX_ISA_SPEC.md`` §3,
cross-checked against the ``target_contract.yaml`` provenance (the decoder's ``comb.icmp-eq`` fan-out in the
HW-dialect import: 24 distinct 7-bit equality compares in @Vortex, including CUSTOM0 0x0B and CUSTOM1 0x2B).

CUSTOM0 is TWO families keyed by ``funct7``; ``funct3`` selects within a family (funct3=0 is ``tmc`` under
funct7=0 but ``vote_all`` under funct7=1):

  * ``funct7 = 0`` — warp/thread control: tmc / wspawn / split / join / barrier / pred / wsync
    (funct3 0-5, 7; funct3=6 is reserved-not-observed, do not emit).
  * ``funct7 = 1`` — cooperative-thread primitives: vote_all / vote_any / vote_uni / vote_ballot
    (funct3 0-3) + shfl_up / shfl_down / shfl_bfly / shfl_idx (funct3 4-7).

WHERE THE REAL COMPUTE ENCODING LIVES: this file is NOT the source of truth for how you EMIT a compute or
memory instruction — those are ordinary RV64IMAFD (VORTEX_ISA_SPEC.md §2), emitted by stock LLVM, and the
target's fixed-format decode is exposed by ``merlin.targetgen.isa_model.isa_model_for_target("vortex")``.
The CUSTOM0 ops here are what ``isa_tools.py asm/disasm/lint`` reason about (warp control + collectives).
CUSTOM1 (0x2B) is present in the decoder but is not modelled here (the compute capsules do not emit it).

NOTE on store visibility (VORTEX_ISA_SPEC.md §5): the CUSTOM0 surface has NO cache-flush instruction —
global stores reach the shared L2 coherence point by the machine's own write-back path; ``wsync`` (funct3=7)
drains the warp pipeline but is a pipeline fence, not a dcache flush. Cross-core visibility is a memory-model
property (§5), not a single ISA op — do not invent a flush opcode.
"""
from __future__ import annotations

from isa_patterns import (
    RType,
    WarpBarrier,
    WarpMaskControl,
    WarpPredicate,
    WarpShuffle,
    WarpSpawn,
    WarpSplit,
    WarpSync,
    WarpVote,
)

# opcode constant — RISC-V custom-0 (VORTEX_ISA_SPEC.md §3; target_contract provenance CUSTOM0 == 0x0B).
CUSTOM0 = 0x0B


# --- funct7 = 0 : warp and thread control (VORTEX_ISA_SPEC.md §3.1) -----------------------------------------
class VX_TMC(WarpMaskControl, RType, opcode=CUSTOM0, funct3=0, funct7=0):
    """tmc: set the current warp's thread mask from rs1 (a zero mask retires the warp). SPEC §3.1 funct3=0."""


class VX_WSPAWN(WarpSpawn, RType, opcode=CUSTOM0, funct3=1, funct7=0):
    """wspawn: activate rs1 warps on this core, each beginning at address rs2. SPEC §3.1 funct3=1."""


class VX_SPLIT(WarpSplit, RType, opcode=CUSTOM0, funct3=2, funct7=0):
    """split: begin per-thread divergence on the rs1 predicate; returns an IPDOM token in rd. SPEC §3.1 funct3=2."""


class VX_JOIN(WarpMaskControl, RType, opcode=CUSTOM0, funct3=3, funct7=0):
    """join: end the divergence region identified by the split token in rs1. SPEC §3.1 funct3=3."""


class VX_BARRIER(WarpBarrier, RType, opcode=CUSTOM0, funct3=4, funct7=0):
    """barrier: block until rs2 warps have arrived at barrier rs1. SPEC §3.1 funct3=4."""


class VX_PRED(WarpPredicate, RType, opcode=CUSTOM0, funct3=5, funct7=0):
    """pred: set the thread mask to rs2 for threads where rs1 holds. SPEC §3.1 funct3=5."""


class VX_WSYNC(WarpSync, RType, opcode=CUSTOM0, funct3=7, funct7=0):
    """wsync: drain outstanding operations / warp-pipeline fence (rd=rs1=rs2=x0). SPEC §3.1 funct3=7."""


# --- funct7 = 1 : cooperative-thread primitives (VORTEX_ISA_SPEC.md §3.2) -----------------------------------
class VX_VOTE_ALL(WarpVote, RType, opcode=CUSTOM0, funct3=0, funct7=1):
    """vote_all: rd = 1 iff every active lane's rs1 predicate is non-zero. SPEC §3.2 funct3=0."""


class VX_VOTE_ANY(WarpVote, RType, opcode=CUSTOM0, funct3=1, funct7=1):
    """vote_any: rd = 1 iff any active lane's rs1 predicate is non-zero. SPEC §3.2 funct3=1."""


class VX_VOTE_UNI(WarpVote, RType, opcode=CUSTOM0, funct3=2, funct7=1):
    """vote_uni: rd = 1 iff the rs1 predicate is uniform across active lanes. SPEC §3.2 funct3=2."""


class VX_VOTE_BALLOT(WarpVote, RType, opcode=CUSTOM0, funct3=3, funct7=1):
    """vote_ballot: rd = bitmask of the lanes whose rs1 predicate is non-zero. SPEC §3.2 funct3=3."""


class VX_SHFL_UP(WarpShuffle, RType, opcode=CUSTOM0, funct3=4, funct7=1):
    """shfl_up: shift rs1 values up the warp lanes (rs2 = broadcast/control). SPEC §3.2 funct3=4."""


class VX_SHFL_DOWN(WarpShuffle, RType, opcode=CUSTOM0, funct3=5, funct7=1):
    """shfl_down: shift rs1 values down the lanes (rs2 = broadcast/control). SPEC §3.2 funct3=5."""


class VX_SHFL_BFLY(WarpShuffle, RType, opcode=CUSTOM0, funct3=6, funct7=1):
    """shfl_bfly: butterfly exchange — the pattern a log-depth warp reduction wants. SPEC §3.2 funct3=6."""


class VX_SHFL_IDX(WarpShuffle, RType, opcode=CUSTOM0, funct3=7, funct7=1):
    """shfl_idx: read another lane's value by index (rs2 = broadcast/control). SPEC §3.2 funct3=7."""


class _IsaSpec:
    """The assembler-mnemonic -> op-class map isa_introspect discovers (an object exposing ``operations``),
    so a kernel written in Vortex mnemonics maps back to semantic classes. Mnemonics per VORTEX_ISA_SPEC.md §3."""
    operations = {
        # funct7=0 warp/thread control
        "vx_tmc": VX_TMC,
        "vx_wspawn": VX_WSPAWN,
        "vx_split": VX_SPLIT,
        "vx_join": VX_JOIN,
        "vx_barrier": VX_BARRIER,
        "vx_bar": VX_BARRIER,
        "vx_pred": VX_PRED,
        "vx_wsync": VX_WSYNC,
        # funct7=1 cooperative-thread primitives
        "vx_vote_all": VX_VOTE_ALL,
        "vx_vote_any": VX_VOTE_ANY,
        "vx_vote_uni": VX_VOTE_UNI,
        "vx_vote_ballot": VX_VOTE_BALLOT,
        "vx_shfl_up": VX_SHFL_UP,
        "vx_shfl_down": VX_SHFL_DOWN,
        "vx_shfl_bfly": VX_SHFL_BFLY,
        "vx_shfl_idx": VX_SHFL_IDX,
    }


ISA = _IsaSpec()
