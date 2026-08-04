"""MX-Gemmini MMIO / RoCC ABI + microscaling-datatype FACTS — DERIVED, PROVENANCE-STAMPED, FAIL-CLOSED.

These are DATA facts about the MX-Gemmini command-buffer endpoint and its numeric datapath, transcribed from
the target's own sources so nothing is fitted or assumed. Sources:
  * MMIO map + RoCC word composition — ``radiance-kernels/lib/include/mxgemmini_mmio.h``
  * per-format microscaling dtypes, block scale, accumulate — the RTL Scala:
    ``chipyard/generators/gemmini/src/main/scala/gemmini/MxRequantizer.scala`` (MxFloatFormat enum,
    E8M0 scale) + ``ConfigsFP.scala::GemminiMxFPConfigs.defaultMxFPConfig`` + ``QuantLut.scala``.

Any value not literally stated in a source is recorded as ``None`` / ``"UNKNOWN"`` (fail-closed), never a
baked default. This module is the interim fail-closed source for the MX PE dtype until a dedicated mlc
``mx_gemmini`` FIRRTL subtree extraction recognizes the nested Mesh->Tile->PE mesh directly.
"""
from __future__ import annotations

# --- MMIO register map (absolute GPU-local addresses), from mxgemmini_mmio.h:18-49 ----------------------
CTRL_BASE = 0x00084000            # GEMMINI_CTRL (mxgemmini_mmio.h:18)
MMIO_REGISTERS = {                # name: absolute address (offset from CTRL_BASE per the header)
    "INST": 0x00084000,           # :19,25  (store the composed RoCC word here to trigger)
    "READY": 0x00084008,          # :20,26
    "RS1": 0x00084010,            # :21,27  (write operand rs1 before INST)
    "RS2": 0x00084018,            # :22,28  (write operand rs2 before INST)
    "BUSY": 0x00084020,           # :23,29
    "OCCUPANCY": 0x00084028,      # :24,30
    "LUT0_WEIGHT": 0x00084080,    # :33,36
    "LUT1_ACTIVATION": 0x00084380,  # :34,37
    "LUT2_OUTPUT": 0x00084680,    # :35,38
}
SF_MEM = {                        # scale-factor memory, mxgemmini_mmio.h:41-45
    "SF_MEM_B": 0x00088000,       # base
    "SF_MEM_A": 0x0008A000,       # SF_MEM + 0x2000
    "SF_MEM_SIZE": 0x2000,
    "SF_MEM_BUFFER_OFFSET": 0x800,
}
REQUANT = {"base": 0x00040000, "size": 0x40000}  # mxgemmini_mmio.h:48-49

# The header's GPU-local bases (0x84000 / 0x88000 / 0x40000) are a separately-mapped view; they do NOT
# equal the Scala config baseAddrs (requantizer 0x10000000, scale_mem 0x10008000 in ConfigsFP.scala:327/338).
# The address translation between the two is not stated in either source -> UNKNOWN (recorded, not guessed).
SCALA_BASEADDR = {"requantizer": 0x10000000, "scale_mem": 0x10008000, "translation_to_gpu_local": "UNKNOWN"}

# --- RoCC INST word composition, from mxgemmini_mmio.h:66 -----------------------------------------------
ROCC_WORD_LAYOUT = {              # field: (lo_bit, width, fixed_value or None-if-variable)
    "opcode": (0, 7, 0x7B),       # custom-3
    "rd": (7, 5, 0),
    "funct3": (12, 3, 3),         # xd=0, xs1=1, xs2=1
    "rs1": (15, 5, 1),
    "rs2": (20, 5, 2),
    "funct7": (25, 7, None),      # the caller-supplied command selector (the one variable field)
}


def compose_rocc_word(funct7: int) -> int:
    """Compose the MX-Gemmini RoCC INST word for a given 7-bit command selector — exactly the header's
    ``(0x7B)|(0<<7)|(3<<12)|(1<<15)|(2<<20)|(funct<<25)``."""
    return (0x7B | (0 << 7) | (3 << 12) | (1 << 15) | (2 << 20) | ((funct7 & 0x7F) << 25)) & 0xFFFFFFFF


# --- microscaling datatypes, from MxRequantizer.scala MxFloatFormat (lines 7-44) + ConfigsFP.scala ------
# exp/mant per mx_format value; bias per-format is NOT literally asserted in MxFloatFormat (only the generic
# MxFloat.bias = (1<<(expWidth-1))-1 formula exists in Arithmetic.scala) -> recorded as None (fail-closed).
MX_FORMATS = {
    0: {"name": "FP8", "exp": 4, "mant": 3, "total_bits": 8, "pmax": 448, "bias": None},   # e4m3
    1: {"name": "FP6", "exp": 3, "mant": 2, "total_bits": 6, "pmax": 28, "bias": None},     # e3m2
    2: {"name": "FP4", "exp": 2, "mant": 1, "total_bits": 4, "pmax": 6, "bias": None},      # e2m1
    3: {"name": "BF16", "exp": 7, "mant": 8, "total_bits": 16, "pmax": 65024, "bias": None},
}
# block scale: E8M0 (8-bit exponent-only), biased exponent with bias 127 (MxRequantizer.scala:450,458-468)
BLOCK_SCALE = {"kind": "e8m0", "bits": 8, "bias": 127}
# accumulate: bf16 (accType = MxFloat(8,8,4); requantizer treats accumulator inputs as 16-bit BF16,
# MxRequantizer.scala:442,456,461). No int32 accumulate is present in the MX datapath.
ACCUMULATE = {"dtype": "bf16", "exp": 8, "mant": 8, "int32_accumulate_present": False}

# --- mesh geometry, from ConfigsFP.scala::GemminiMxFPConfigs.defaultMxFPConfig (lines 219-267) ----------
MESH = {
    "tileRows": 1, "tileColumns": 1, "meshRows": 16, "meshColumns": 16, "dataflow": "WS",
    "inputType": "MxFloat(3,3,2)", "weightType": "MxFloat(3,3,2)", "accType": "MxFloat(8,8,4)",
    "scaleSize": 32, "enable_lut": True,
}

# Honest residue (recorded, not fabricated):
UNKNOWN = {
    "per_format_exponent_bias": "only generic MxFloat.bias formula in source; not asserted per mx_format",
    "per_format_operand_widths_in_header": "mxgemmini_mmio.h carries no per-format width macros",
    "gpu_local_to_scala_baseaddr_translation": "relationship between 0x84000-family and 0x10000000-family",
    "funct7_command_enumeration": "the header composes funct7 dynamically; the command set is not enumerated"
                                  " in mxgemmini_mmio.h (would come from the driver op layer)",
}
