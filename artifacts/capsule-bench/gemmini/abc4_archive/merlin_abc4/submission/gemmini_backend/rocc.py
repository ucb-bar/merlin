"""Lower the program to ``llvm.func @gemmini_kernel`` of RoCC ``.insn r 0x7b`` ops.

We perform genuine instruction selection: each high-level op (matmul, commit, movement)
is tiled to a 16x16 weight/output-stationary micro-sequence of real Gemmini RoCC
instructions (CONFIG_EX/LD/ST, MVIN, PRELOAD, COMPUTE_PRELOADED/ACCUMULATE, MVOUT),
mirroring the ``sp_tiled_matmul`` reference in the ISA header. The instruction encodings
(funct7 + rs1/rs2 bit layouts) are taken from ``gemmini.h``.

Each Gemmini instruction is emitted as an LLVM inline-asm op using the exact xcustom.h
template for a no-rd CUSTOM_3 instruction:
    .insn r 0x7b, 0x3, <funct7>, x0, $0, $1     (inputs: rs1, rs2 in "r,r")

The runner links this module's @gemmini_kernel against its own harness, which passes
``ptr weight, ptr lhs_0.., ptr out_0..`` (leaf data, edge-padded to a multiple of 16).
"""
from __future__ import annotations

import struct
from typing import Any

from . import iface_ir as IR
from . import passes as P

DIM = 16
BANK_NUM = 4
BANK_ROWS = 4096
SPAD_ROWS = BANK_NUM * BANK_ROWS  # 16384
ADDR_LEN = 32
GARBAGE = 0xFFFFFFFF

# funct7 opcodes
F_CONFIG = 0
F_MVIN = 2
F_MVOUT = 3
F_COMPUTE_PRELOADED = 4
F_COMPUTE_ACCUMULATE = 5
F_PRELOAD = 6
F_FLUSH = 7

CONFIG_EX = 0
CONFIG_LD = 1
CONFIG_ST = 2

NO_ACTIVATION = 0
RELU = 1
DATAFLOW_OS = 0

ACC_BIT = 1 << (ADDR_LEN - 1)        # bit31: accumulator region
ACCUMULATE_BIT = 1 << (ADDR_LEN - 2)  # bit30: accumulate (vs overwrite)
FULL_BIT = 1 << (ADDR_LEN - 3)        # bit29: full-width (i32) readout


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


IDENT_BITS = _f32_bits(1.0)


def _ceil16(n: int) -> int:
    return ((n + DIM - 1) // DIM) * DIM


class _Instr:
    __slots__ = ("f7", "rs1", "rs2", "comment")

    def __init__(self, f7, rs1, rs2, comment=""):
        self.f7 = f7
        self.rs1 = rs1   # int OR ("addr", arg_index, byte_offset)
        self.rs2 = rs2   # int OR ("addr", ...)
        self.comment = comment


def _rs2_mvmem(rows, cols, spad):
    return (rows << (ADDR_LEN + 16)) | (cols << ADDR_LEN) | (spad & 0xFFFFFFFF)


# --------------------------------------------------------------------------- planning


class KernelPlan:
    def __init__(self):
        self.args: list[str] = []          # arg name order
        self.arg_index: dict[str, int] = {}
        self.instrs: list[_Instr] = []

    def add_arg(self, name):
        if name not in self.arg_index:
            self.arg_index[name] = len(self.args)
            self.args.append(name)

    def emit(self, f7, rs1, rs2, comment=""):
        self.instrs.append(_Instr(f7, rs1, rs2, comment))


def _config_ex(plan):
    rs1 = (IDENT_BITS << 32) | (1 << 16) | (DATAFLOW_OS << 2) | CONFIG_EX
    rs2 = (1 << 48)  # C_stride=1, in_shift=0
    plan.emit(F_CONFIG, rs1, rs2, "config_ex OS")


def _config_ld(plan, stride_bytes, comment=""):
    rs1 = (IDENT_BITS << 32) | (DIM << 16) | (1 << 8) | CONFIG_LD
    plan.emit(F_CONFIG, rs1, stride_bytes, "config_ld " + comment)


def _config_st(plan, stride_bytes, acc_act, acc_scale):
    rs1 = (acc_act << 2) | CONFIG_ST
    rs2 = (_f32_bits(acc_scale) << 32) | (stride_bytes & 0xFFFFFFFF)
    plan.emit(F_CONFIG, rs1, rs2, "config_st")


def _matmul_tiles(plan, weight_name, lhs_name, out_name, M, K, N,
                  epilogue, output_dtype, acc_scale):
    """Emit the tiled OS matmul + epilogue readout for one matmul/commit pair."""
    Mp, Kp, Np = _ceil16(M), _ceil16(K), _ceil16(N)
    Mt, Kt, Nt = Mp // DIM, Kp // DIM, Np // DIM

    full_C = 1 if output_dtype == "i32" else 0
    sizeof_C = 4 if output_dtype == "i32" else 1
    acc_act = RELU if "relu" in epilogue else NO_ACTIVATION
    scale = float(acc_scale) if (acc_scale is not None and "acc_scale" in epilogue) else 1.0

    w_arg = plan.arg_index[weight_name]
    a_arg = plan.arg_index[lhs_name]
    o_arg = plan.arg_index[out_name]

    A_sp0 = 0
    B_sp0 = SPAD_ROWS - Kt * Nt * DIM

    _config_st(plan, Np * sizeof_C, acc_act, scale)

    # mvin B (weight, the RHS): stride Np
    _config_ld(plan, Np, "B")
    for j in range(Nt):
        for k in range(Kt):
            off = (k * DIM * Np + j * DIM)  # i8 bytes
            spad = B_sp0 + (k * Nt + j) * DIM
            plan.emit(F_MVIN, ("addr", w_arg, off), _rs2_mvmem(DIM, DIM, spad), "mvin B")

    # mvin A (lhs): stride Kp
    _config_ld(plan, Kp, "A")
    for i in range(Mt):
        for k in range(Kt):
            off = (i * DIM * Kp + k * DIM)
            spad = A_sp0 + (i * Kt + k) * DIM
            plan.emit(F_MVIN, ("addr", a_arg, off), _rs2_mvmem(DIM, DIM, spad), "mvin A")

    # compute (output-stationary, accumulate over K in the array)
    cbits = ACC_BIT | (FULL_BIT if full_C else 0)
    for i in range(Mt):
        for j in range(Nt):
            c_off = (i * Nt + j) * DIM
            c_acc = cbits | c_off
            for k in range(Kt):
                out = c_acc if k == Kt - 1 else GARBAGE
                a_sp = A_sp0 + (i * Kt + k) * DIM
                b_sp = B_sp0 + (k * Nt + j) * DIM
                # preload(BD=GARBAGE, C=out)
                plan.emit(F_PRELOAD,
                          (DIM << 48) | (DIM << 32) | GARBAGE,
                          (DIM << 48) | (DIM << 32) | (out & 0xFFFFFFFF), "preload")
                f7 = F_COMPUTE_PRELOADED if k == 0 else F_COMPUTE_ACCUMULATE
                plan.emit(f7,
                          (DIM << 48) | (DIM << 32) | a_sp,
                          (DIM << 48) | (DIM << 32) | b_sp, "compute")

    # mvout C
    for i in range(Mt):
        for j in range(Nt):
            off = (i * DIM * Np + j * DIM) * sizeof_C
            c_acc = cbits | ((i * Nt + j) * DIM)
            plan.emit(F_MVOUT, ("addr", o_arg, off),
                      _rs2_mvmem(DIM, DIM, c_acc), "mvout C")


def _movement_tiles(plan, src_name, dst_name, M, N, dtype):
    """Pure data movement: mvin tiles to scratchpad, mvout back. No compute."""
    Mp, Np = _ceil16(M), _ceil16(N)
    Mt, Nt = Mp // DIM, Np // DIM
    sz = 4 if dtype == "i32" else 1
    s_arg = plan.arg_index[src_name]
    o_arg = plan.arg_index[dst_name]

    _config_ld(plan, Np * sz, "mv")
    _config_st(plan, Np * sz, NO_ACTIVATION, 1.0)
    for i in range(Mt):
        for j in range(Nt):
            off = (i * DIM * Np + j * DIM) * sz
            spad = (i * Nt + j) * DIM
            plan.emit(F_MVIN, ("addr", s_arg, off), _rs2_mvmem(DIM, DIM, spad), "mvin")
    for i in range(Mt):
        for j in range(Nt):
            off = (i * DIM * Np + j * DIM) * sz
            spad = (i * Nt + j) * DIM
            plan.emit(F_MVOUT, ("addr", o_arg, off), _rs2_mvmem(DIM, DIM, spad), "mvout")


def plan_kernel(prog: IR.Program) -> KernelPlan:
    plan = KernelPlan()

    # ---- argument order: [weight] ++ [matmul lhs] ++ [commit outputs] ----
    weight_name = None
    for op in prog.ops:
        if isinstance(op, IR.Pack):
            weight_name = op.src
            break
    if weight_name is not None:
        plan.add_arg(weight_name)
    for op in prog.ops:
        if isinstance(op, IR.Matmul):
            plan.add_arg(op.lhs)
        elif isinstance(op, IR.Conv2d):
            plan.add_arg(op.ifm)
        elif isinstance(op, IR.Movement):
            plan.add_arg(op.src)
    for op in prog.ops:
        if isinstance(op, IR.Commit):
            plan.add_arg(op.dst)
        elif isinstance(op, IR.Conv2d):
            plan.add_arg(op.dst)
        elif isinstance(op, IR.Movement):
            plan.add_arg(op.dst)

    # ---- instruction stream ----
    plan.instrs.append(_Instr("fence", 0, 0, "fence"))  # trace must open with a FENCE
    plan.emit(F_FLUSH, 0, 0, "flush")
    _config_ex(plan)

    # index commits by their accumulator source
    commit_by_acc = {c.src: c for c in prog.ops if isinstance(c, IR.Commit)}

    for op in prog.ops:
        if isinstance(op, IR.Matmul):
            commit = commit_by_acc.get(op.dst)
            M, N = P.matmul_out_shape(prog, op)
            K = prog.tensors[op.lhs].shape[1]
            _matmul_tiles(plan, weight_name, op.lhs, commit.dst, M, K, N,
                          commit.epilogue, commit.output_dtype, commit.acc_scale)
        elif isinstance(op, IR.Movement):
            t = prog.tensors[op.src]
            M, N = (t.shape + [1])[:2] if len(t.shape) >= 2 else (t.shape[0], 1)
            _movement_tiles(plan, op.src, op.dst, t.shape[0], t.shape[1], t.dtype)
        elif isinstance(op, IR.Conv2d):
            # im2col conv: the activation is materialized as the [M=patches, K=kh*kw*ci]
            # im2col matrix and the resident weight as [K, N=out_ch]; lower to the same
            # tiled OS matmul as a plain MATMUL_RESIDENT (K-accumulated, optional relu).
            M, N = P.conv_out_shape(prog, op)
            ifm_t = prog.tensors[op.ifm]
            kh, kw = int(op.kernel[0]), int(op.kernel[1])
            ci = int(op.kernel[2]) if len(op.kernel) > 2 else int(ifm_t.shape[-1])
            K = kh * kw * ci
            _matmul_tiles(plan, weight_name, op.ifm, op.dst, M, K, N,
                          op.epilogue, op.output_dtype, op.acc_scale)

    plan.instrs.append(_Instr("fence", 0, 0, "fence"))
    return plan


# --------------------------------------------------------------------------- emit MLIR


def _const(lines, ctr, val):
    n = ctr[0]; ctr[0] += 1
    lines.append(f"    %{n} = llvm.mlir.constant({val} : i64) : i64")
    return f"%{n}"


def _operand_ssa(lines, ctr, operand):
    if isinstance(operand, int):
        return _const(lines, ctr, operand)
    # ("addr", arg_index, offset)
    _, arg_index, off = operand
    p = ctr[0]; ctr[0] += 1
    lines.append(f"    %{p} = llvm.ptrtoint %arg{arg_index} : !llvm.ptr to i64")
    if off == 0:
        return f"%{p}"
    c = _const(lines, ctr, off)
    a = ctr[0]; ctr[0] += 1
    lines.append(f"    %{a} = llvm.add %{p}, {c} : i64")
    return f"%{a}"


def emit_llvm(prog: IR.Program) -> str:
    plan = plan_kernel(prog)
    nargs = len(plan.args)
    arg_decls = ", ".join(f"%arg{i}: !llvm.ptr" for i in range(nargs))

    lines: list[str] = []
    lines.append("module {")
    lines.append(f"  llvm.func @gemmini_kernel({arg_decls}) {{")
    ctr = [0]
    for ins in plan.instrs:
        if ins.f7 == "fence":
            lines.append('    llvm.inline_asm has_side_effects "fence", "" : () -> ()')
            continue
        rs1 = _operand_ssa(lines, ctr, ins.rs1)
        rs2 = _operand_ssa(lines, ctr, ins.rs2)
        asm = f".insn r 0x7b, 0x3, {ins.f7}, x0, $0, $1"
        lines.append(
            f'    llvm.inline_asm has_side_effects "{asm}", "r,r" {rs1}, {rs2} '
            f': (i64, i64) -> ()    // {ins.comment}')
    lines.append("    llvm.return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"
