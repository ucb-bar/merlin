"""Gemmini RoCC instruction encoding, derived from the shipped ISA definition.

Field layout (from the RTL decoder / `xcustom.h` ROCC_INSTRUCTION_RS1_RS2):

    funct[31:25] rs2[24:20] rs1[19:15] xd[14] xs1[13] xs2[12] rd[11:7] opcode[6:0]

which for a two-source, no-destination accelerator command is exactly

    .insn r <CUSTOM_OPCODE>, <FUNCT3>, <funct>, x0, $0, $1

with FUNCT3 = 0b011 (xd=0, xs1=1, xs2=1).  The rs1/rs2 bit packings below are transcribed
one-for-one from the `gemmini_*` macros in `gemmini.h`; every shift is quoted in
`docs/public_facts_used.md`.
"""
from __future__ import annotations

import struct

from . import rtl_facts as F

DIM = F.DIM
ADDR_LEN = 32                      # gemmini_params.h

# ---- funct7 values (gemmini.h `#define k_*`) ------------------------------------------------
K_CONFIG = 0
K_MVIN2 = 1
K_MVIN = 2
K_MVOUT = 3
K_COMPUTE_PRELOADED = 4
K_COMPUTE_ACCUMULATE = 5
K_PRELOAD = 6
K_FLUSH = 7

#: CONFIG subtypes, carried in rs1[1:0].
CONFIG_EX = 0
CONFIG_LD = 1
CONFIG_ST = 2

#: dataflow bit (rs1[2] of a CONFIG_EX)
OUTPUT_STATIONARY = 0
WEIGHT_STATIONARY = 1

#: activation selector (rs1[4:3] of CONFIG_EX / rs1[3:2] of CONFIG_ST)
NO_ACTIVATION = 0
RELU = 1

GARBAGE_ADDR = 0xFFFFFFFF

#: local-address metadata bits (LocalAddr.scala bundle order, MSB first)
ACC_ADDR_BIT = 1 << (ADDR_LEN - 1)          # bit 31 -- address names the accumulator
ACC_ACCUMULATE_BIT = 1 << (ADDR_LEN - 2)    # bit 30 -- add into the accumulator row
ACC_FULL_ROW_BIT = 1 << (ADDR_LEN - 3)      # bit 29 -- read the full i32 accumulator row

MASK64 = (1 << 64) - 1


def f32_bits(value: float) -> int:
    """IEEE-754 single-precision bit pattern (`acc_scale_t_to_acc_scale_t_bits`)."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _u64(value: int) -> int:
    return int(value) & MASK64


def assert_legal(funct: int) -> int:
    if funct not in F.LEGAL_FUNCTS:
        raise ValueError(f"funct {funct} is not in the decoder's legal set")
    return funct


def asm_string(funct: int) -> str:
    """The `.insn` directive text for one RoCC command."""
    assert_legal(funct)
    return f".insn r {hex(F.CUSTOM_OPCODE)}, {hex(F.FUNCT3)}, {hex(funct)}, x0, $0, $1"


# ---- rs1/rs2 packers -------------------------------------------------------------------------
# Each returns (funct, rs1, rs2).  A packer whose rs1 carries a DRAM address returns `None` in
# that slot: the caller substitutes the runtime pointer value.

def flush(skip: int = 0):
    return K_FLUSH, _u64(skip), 0


def config_ex(*, dataflow: int = WEIGHT_STATIONARY, sys_act: int = NO_ACTIVATION,
              sys_shift: int = 0, acc_scale: float = 1.0, c_stride: int = 1,
              a_stride: int = 1, a_transpose: bool = False, b_transpose: bool = False,
              set_only_strides: bool = False):
    """`gemmini_extended3_config_ex`.

    rs1: [63:32] acc_scale | [31:16] a_stride | [9] b_transpose | [8] a_transpose |
         [7] set_only_strides | [5] uselut | [4:3] activation | [2] dataflow | [1:0] cmd_type
    rs2: [63:48] c_stride | [31:0] in_shift
    """
    rs1 = ((f32_bits(acc_scale) << 32)
           | ((a_stride & 0xFFFF) << 16)
           | ((1 if b_transpose else 0) << 9)
           | ((1 if a_transpose else 0) << 8)
           | ((1 if set_only_strides else 0) << 7)
           | ((sys_act & 0x3) << 3)
           | ((dataflow & 0x1) << 2)
           | CONFIG_EX)
    rs2 = ((c_stride & 0xFFFF) << 48) | (int(sys_shift) & 0xFFFFFFFF)
    return K_CONFIG, _u64(rs1), _u64(rs2)


def config_ld(*, stride: int, scale: float = 1.0, shrunk: bool = False,
              block_stride: int = DIM, pixel_repeats: int = 1, load_id: int = 0):
    """`gemmini_extended5_config_ld`.

    rs1: [63:32] mvin scale | [31:16] block_mvin_stride | [15:8] pixel_repeats |
         [4:3] load id | [2] shrunk | [1:0] cmd_type
    rs2: DRAM row stride in bytes
    """
    rs1 = ((f32_bits(scale) << 32)
           | ((block_stride & 0xFFFF) << 16)
           | ((pixel_repeats & 0xFF) << 8)
           | ((load_id & 0x3) << 3)
           | ((1 if shrunk else 0) << 2)
           | CONFIG_LD)
    return K_CONFIG, _u64(rs1), _u64(stride)


def config_st(*, stride: int, acc_act: int = NO_ACTIVATION, acc_scale: float = 1.0,
              pool_stride: int = 0, pool_size: int = 0, pool_out_dim: int = 0,
              porows: int = 0, pocols: int = 0, orows: int = 0, ocols: int = 0,
              upad: int = 0, lpad: int = 0):
    """`gemmini_extended2_config_st`.

    rs1: [63:56] ocols | [55:48] orows | [47:40] pocols | [39:32] porows |
         [31:24] pool_out_dim | [15:10] lpad | [9:8] upad | [7:6] pool_size |
         [5:4] pool_stride | [3:2] acc_act | [1:0] cmd_type
    rs2: [63:32] acc_scale | [31:0] DRAM row stride in bytes
    """
    rs1 = (((ocols & 0xFF) << 56)
           | ((orows & 0xFF) << 48)
           | ((pocols & 0xFF) << 40)
           | ((porows & 0xFF) << 32)
           | ((pool_out_dim & 0xFF) << 24)
           | ((lpad & 0x3F) << 10)
           | ((upad & 0x3) << 8)
           | ((pool_size & 0x3) << 6)
           | ((pool_stride & 0x3) << 4)
           | ((acc_act & 0x3) << 2)
           | CONFIG_ST)
    rs2 = (f32_bits(acc_scale) << 32) | (int(stride) & 0xFFFFFFFF)
    return K_CONFIG, _u64(rs1), _u64(rs2)


def _mem_rs2(local_addr: int, cols: int, rows: int) -> int:
    return _u64(((rows & 0xFFFF) << (ADDR_LEN + 16))
                | ((cols & 0xFFFF) << ADDR_LEN)
                | (local_addr & 0xFFFFFFFF))


def mvin(*, local_addr: int, cols: int, rows: int, load_id: int = 0):
    """`gemmini_extended_mvin`; rs1 is the DRAM address (runtime pointer)."""
    funct = {0: K_MVIN, 1: K_MVIN2}[load_id]
    return funct, None, _mem_rs2(local_addr, cols, rows)


def mvout(*, local_addr: int, cols: int, rows: int):
    """`gemmini_extended_mvout`; rs1 is the DRAM address (runtime pointer)."""
    return K_MVOUT, None, _mem_rs2(local_addr, cols, rows)


def preload(*, bd_addr: int, c_addr: int, bd_cols: int, bd_rows: int,
            c_cols: int, c_rows: int):
    """`gemmini_extended_preload`."""
    return K_PRELOAD, _mem_rs2(bd_addr, bd_cols, bd_rows), _mem_rs2(c_addr, c_cols, c_rows)


def compute(*, a_addr: int, bd_addr: int, a_cols: int, a_rows: int,
            bd_cols: int = DIM, bd_rows: int = DIM, accumulate: bool = False):
    """`gemmini_extended_compute_{preloaded,accumulated}`."""
    funct = K_COMPUTE_ACCUMULATE if accumulate else K_COMPUTE_PRELOADED
    return funct, _mem_rs2(a_addr, a_cols, a_rows), _mem_rs2(bd_addr, bd_cols, bd_rows)


def acc_addr(row: int, *, accumulate: bool = False, full_row: bool = False) -> int:
    a = ACC_ADDR_BIT | (row & 0x3FFF)
    if accumulate:
        a |= ACC_ACCUMULATE_BIT
    if full_row:
        a |= ACC_FULL_ROW_BIT
    return a
