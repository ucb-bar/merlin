#!/usr/bin/env python3
"""Encode a valid gemmini RoCC command stream for the GSIM M2 harness.

Produces one C header (`gemmini_cmd_stream.h`) that the command-driving harness
(`gsim_gemmini_cmd_harness.cpp`) #includes: a table of {funct, rs1, rs2, xd,
xs1, xs2, opcode} RoCC command bundles decoded straight from the gemmini ISA.

Source of truth (NO magic numbers invented here):
  third_party/gemmini/src/main/scala/gemmini/GemminiISA.scala   (funct codes)
  third_party/gemmini/software/gemmini-rocc-tests/include/gemmini.h  (rs1/rs2 packing)

We PREFER commands that never touch DRAM (no TLB assertion):
  * config_ex  (funct=CONFIG_CMD=0, rs1[1:0]=CONFIG_EX=0) -- processed inline in
                the ex-controller's waiting_for_cmd state; exercises the cmd path.
  * preload    (funct=PRELOAD_CMD=6) on a scratchpad source + accumulator dest.
  * compute    (funct=COMPUTE_AND_FLIP_CMD=4) reading scratchpad -- drives the
                ex-controller FSM control_state from waiting_for_cmd(0) -> compute(1).
None of these issue a DMA request, so the load/store controllers never hit the
TLB (the wall the prior work documented for mvin/mvout).

No regex (repo rule): pure integer/bit ops.

Usage:  gsim_gemmini_cmd_encode.py <out_dir>   # writes <out_dir>/gemmini_cmd_stream.h
"""
import os
import sys

# ---- funct codes (GemminiISA.scala) --------------------------------------
CONFIG_CMD = 0
LOAD_CMD = 2
STORE_CMD = 3
COMPUTE_AND_FLIP_CMD = 4
COMPUTE_AND_STAY_CMD = 5
PRELOAD_CMD = 6
FLUSH_CMD = 7

# rs1[1:0] config sub-type
CONFIG_EX = 0
CONFIG_LOAD = 1
CONFIG_STORE = 2

# ---- geometry: DERIVED from the pinned ISA header, never assumed -------------
def _dim_from_pinned_header():
    """The systolic mesh dimension, read from the pinned ``gemmini_params.h``.

    DIM is a real target fact (it sizes every preload/compute row and column count below), so it is
    extracted from the target's own header rather than written here. Fails closed: a build that
    cannot establish the mesh dimension emits nothing rather than a command stream shaped for a
    mesh the hardware may not have.

    No regex, per the repo rule -- ``#define DIM <n>`` is split structurally.
    """
    from merlin.common import provenance as _prov

    root = _prov.verify("gemmini_isa_headers").observed.path
    header = os.path.join(root, "include", "gemmini_params.h")
    with open(header) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "#define" and parts[1] == "DIM":
                return int(parts[2], 0)
    raise ValueError("no `#define DIM` in the pinned gemmini_params.h at %s" % header)


DIM = _dim_from_pinned_header()
ADDR_LEN = 32          # local-address width inside rs1/rs2

# The RISC-V BASE ISA custom-3 opcode (0b1111011). This is not a Gemmini fact and there is nothing
# in the target's RTL facts to derive it from -- `asm_audit._derived_opcodes("gemmini")` returns {}
# because the gemmini extraction emits no `funct_decode_table` at all. It is also not decoded: RoCC
# dispatch has already selected the accelerator by the time this field is seen, and the value is
# only formatted into a comment in the generated header below, never executed.
OPCODE_CUSTOM3 = 0x7B  # derived-ok: RISC-V unprivileged ISA custom-3 encoding, not a target fact

MASK64 = (1 << 64) - 1


def _u64(x):
    return x & MASK64


def local_addr(row, *, is_acc=False, accumulate=False):
    """Gemmini LocalAddr packing inside a 32-bit field.
    bit31 = is_acc_addr, bit30 = accumulate (acc only); low bits = row/bank index."""
    a = row & ((1 << 14) - 1)
    if is_acc:
        a |= (1 << 31)
        if accumulate:
            a |= (1 << 30)
    return a & ((1 << 32) - 1)


def config_ex(dataflow=1, act=0, sys_shift=0, a_stride=1, c_stride=1,
              a_transpose=0, b_transpose=0, set_only_strides=0,
              acc_scale_bits=0x3F800000):
    """gemmini_extended3_config_ex packing (gemmini.h).
    acc_scale_bits default = float32 bits of 1.0 (ACC_SCALE_IDENTITY)."""
    rs1 = ((acc_scale_bits & 0xFFFFFFFF) << 32) \
        | ((a_stride & 0xFFFF) << 16) \
        | ((b_transpose & 1) << 9) \
        | ((a_transpose & 1) << 8) \
        | ((set_only_strides & 1) << 7) \
        | ((act & 0xF) << 3) \
        | ((dataflow & 1) << 2) \
        | CONFIG_EX
    rs2 = ((c_stride & 0xFFFF) << 48) | (sys_shift & 0xFFFFFFFF)
    return (CONFIG_CMD, _u64(rs1), _u64(rs2))


def _rows_cols_addr(rows, cols, addr):
    return ((rows & 0xFFFF) << (ADDR_LEN + 16)) | ((cols & 0xFFFF) << ADDR_LEN) | (addr & 0xFFFFFFFF)


def preload(bd_addr, c_addr, bd_rows=DIM, bd_cols=DIM, c_rows=DIM, c_cols=DIM):
    """gemmini_extended_preload: rs1=BD(weights) source, rs2=C dest."""
    rs1 = _rows_cols_addr(bd_rows, bd_cols, bd_addr)
    rs2 = _rows_cols_addr(c_rows, c_cols, c_addr)
    return (PRELOAD_CMD, _u64(rs1), _u64(rs2))


def compute_preloaded(a_addr, bd_addr, a_rows=DIM, a_cols=DIM, bd_rows=DIM, bd_cols=DIM):
    """gemmini_extended_compute_preloaded (k_COMPUTE_PRELOADED = COMPUTE_AND_FLIP)."""
    rs1 = _rows_cols_addr(a_rows, a_cols, a_addr)
    rs2 = _rows_cols_addr(bd_rows, bd_cols, bd_addr)
    return (COMPUTE_AND_FLIP_CMD, _u64(rs1), _u64(rs2))


def compute_accumulated(a_addr, bd_addr, a_rows=DIM, a_cols=DIM, bd_rows=DIM, bd_cols=DIM):
    """k_COMPUTE_ACCUMULATE = COMPUTE_AND_STAY; single-mul path (no preload needed)."""
    rs1 = _rows_cols_addr(a_rows, a_cols, a_addr)
    rs2 = _rows_cols_addr(bd_rows, bd_cols, bd_addr)
    return (COMPUTE_AND_STAY_CMD, _u64(rs1), _u64(rs2))


def build_stream():
    """The on-chip (DRAM-free) sequence we drive.
    All source operands live in scratchpad bank 0; the preload destination is an
    accumulator address. No mvin/mvout -> no DMA -> no TLB."""
    spadA = local_addr(0)                         # scratchpad row 0 (A operand)
    spadB = local_addr(DIM)                        # scratchpad row 16 (B/weights)
    accC = local_addr(0, is_acc=True)             # accumulator row 0 (C dest)
    # Names are used as the CSV phase label by the harness -- keep them comma-free.
    stream = [
        ("config_ex_WS_act0_shift0", config_ex(dataflow=1, act=0, sys_shift=0)),
        ("preload_BD-spad16_C-acc0", preload(spadB, accC)),
        ("compute_preloaded_A-spad0_BD-spad16", compute_preloaded(spadA, spadB)),
        ("compute_accumulated_A-spad0_BD-spad16", compute_accumulated(spadA, spadB)),
    ]
    return stream


def _cmd_literal(name, funct, rs1, rs2):
    # _0_R_R form: rd=x0 (xd=0), two source regs (xs1=xs2=1)
    return ("{ %du, 0x%016xull, 0x%016xull, 0u, 1u, 1u, 0x%02xu, \"%s\" }"
            % (funct, rs1, rs2, OPCODE_CUSTOM3, name))


def emit_header(out_dir):
    stream = build_stream()
    # Named singleton bundles for the shape/size sweep harness (build blocks it
    # replays K times). Same DRAM-free scratchpad/accumulator operands as build_stream().
    spadA = local_addr(0)
    spadB = local_addr(DIM)
    accC = local_addr(0, is_acc=True)
    singletons = [
        ("GEMMINI_CONFIG_EX", "config_ex_WS", config_ex(dataflow=1, act=0, sys_shift=0)),
        ("GEMMINI_PRELOAD", "preload", preload(spadB, accC)),
        ("GEMMINI_COMPUTE_STAY", "compute_stay", compute_accumulated(spadA, spadB)),
        ("GEMMINI_COMPUTE_FLIP", "compute_flip", compute_preloaded(spadA, spadB)),
    ]
    lines = []
    lines.append("// AUTO-GENERATED by gsim_gemmini_cmd_encode.py -- do not edit.")
    lines.append("// Valid gemmini RoCC command stream (DRAM-free: config_ex + preload + compute).")  # target-ok: a per-target model-build helper; this target IS its subject
    lines.append("#pragma once")
    lines.append("#include <cstdint>")
    lines.append("struct GemminiCmd {")
    lines.append("  uint8_t  funct;   // inst.funct (7b)")
    lines.append("  uint64_t rs1;     // bits.rs1")
    lines.append("  uint64_t rs2;     // bits.rs2")
    lines.append("  uint8_t  xd, xs1, xs2, opcode;")
    lines.append("  const char* name;")
    lines.append("};")
    lines.append("static const GemminiCmd GEMMINI_CMDS[] = {")
    for name, (funct, rs1, rs2) in stream:
        lines.append("  " + _cmd_literal(name, funct, rs1, rs2) + ",")
    lines.append("};")
    lines.append("static const int GEMMINI_NCMDS = %d;" % len(stream))
    lines.append("")
    lines.append("// Named singleton bundles (for the shape/size sweep harness).")
    for cname, label, (funct, rs1, rs2) in singletons:
        lines.append("static const GemminiCmd %s = %s;"
                     % (cname, _cmd_literal(label, funct, rs1, rs2)))
    text = "\n".join(lines) + "\n"
    path = os.path.join(out_dir, "gemmini_cmd_stream.h")
    with open(path, "w") as f:
        f.write(text)
    print("wrote %s (%d commands + %d singletons)" % (path, len(stream), len(singletons)))
    for name, (funct, rs1, rs2) in stream:
        print("  funct=%d rs1=0x%016x rs2=0x%016x  %s" % (funct, rs1, rs2, name))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    emit_header(sys.argv[1])
