"""Emit ``llvm.func @gemmini_kernel`` (LLVM-dialect MLIR + RoCC inline-asm) from the gemmini module.

The kernel ABI (``mlir_oot_backend_contract.yaml``): the runner harness embeds the deterministic
leaf tensors (row-major, edge tiles zero-padded to a multiple of DIM=16) and calls
``gemmini_kernel(weight, lhs_0.., out_0..)`` (or ``gemmini_kernel(src, dst)`` for pure movement).
We drive Gemmini with raw RoCC custom-3 instructions ``.insn r 0x7b, 3, <funct7>, x0, rs1, rs2``,
fully unrolled per 16x16 tile (zero padding contributes nothing, so partial edges need no masking).
Matmul uses the output-stationary dataflow: per output tile accumulate over K in the array, write the
accumulator once (overwrite), then mvout (applying the configured activation + acc_scale).
"""
from __future__ import annotations

import io
import struct

from xdsl.dialects.builtin import ModuleOp, IntegerAttr, i64
from xdsl.ir import Block, Region, SSAValue
from xdsl.printer import Printer
import xdsl.dialects.llvm as llvm

from . import dialects as D

DIM = 16
A_BASE = 0
B_BASE = 8192
GARBAGE = 0xFFFFFFFF
ACC = 1 << 31
ACC_FULL = 1 << 29

# RoCC funct7 codes (gemmini_params.h / gemmini.h)
F_CONFIG, F_MVIN, F_MVOUT, F_PRELOAD = 0, 2, 3, 6
F_COMPUTE_PRELOADED, F_COMPUTE_ACCUMULATE, F_FLUSH = 4, 5, 7
CONFIG_EX, CONFIG_LD, CONFIG_ST = 0, 1, 2
NO_ACT, RELU = 0, 1
SCALE_ONE = 0x3F800000  # f32 bits of 1.0


def _ceil(x: int) -> int:
    return ((x + DIM - 1) // DIM) * DIM


def _fbits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _esz(dtype: str) -> int:
    return {"i8": 1, "i16": 2, "i32": 4}[dtype]


class _Builder:
    def __init__(self, nargs: int):
        self.ptr = llvm.LLVMPointerType()
        self.blk = Block(arg_types=[self.ptr] * nargs)
        self.ops: list = []
        self._const: dict[int, SSAValue] = {}
        self._base: dict[int, SSAValue] = {}

    def const(self, v: int) -> SSAValue:
        v &= (1 << 64) - 1
        if v not in self._const:
            c = llvm.ConstantOp(IntegerAttr(v, 64), i64)
            self.ops.append(c)
            self._const[v] = c.result
        return self._const[v]

    def base(self, arg: int) -> SSAValue:
        if arg not in self._base:
            p = llvm.PtrToIntOp(self.blk.args[arg], i64)
            self.ops.append(p)
            self._base[arg] = p.output
        return self._base[arg]

    def dram(self, arg: int, off: int) -> SSAValue:
        if off == 0:
            return self.base(arg)
        a = llvm.AddOp(self.base(arg), self.const(off))
        self.ops.append(a)
        return a.res

    def insn(self, funct: int, rs1: SSAValue, rs2: SSAValue) -> None:
        self.ops.append(llvm.InlineAsmOp(
            f".insn r 0x7b, 0x3, {funct}, x0, $0, $1", "r,r", [rs1, rs2],
            res_types=[], has_side_effects=True))

    def fence(self):
        self.ops.append(llvm.InlineAsmOp("fence", "", [], res_types=[],
                                         has_side_effects=True))

    # convenience wrappers (rs1/rs2 are constants unless they carry a dram addr)
    def flush(self):
        self.insn(F_FLUSH, self.const(0), self.const(0))

    def config_ex_os(self):
        rs1 = (SCALE_ONE << 32) | (1 << 16) | CONFIG_EX
        self.insn(F_CONFIG, self.const(rs1), self.const(1 << 48))

    def config_ld(self, stride: int):
        rs1 = (SCALE_ONE << 32) | (DIM << 16) | (1 << 8) | CONFIG_LD
        self.insn(F_CONFIG, self.const(rs1), self.const(stride))

    def config_st(self, stride: int, act: int, scale_bits: int):
        rs1 = (act << 2) | CONFIG_ST
        self.insn(F_CONFIG, self.const(rs1), self.const((scale_bits << 32) | stride))

    def mvin(self, dram: SSAValue, sp: int):
        self.insn(F_MVIN, dram, self.const((DIM << 48) | (DIM << 32) | sp))

    def mvout(self, dram: SSAValue, sp: int):
        self.insn(F_MVOUT, dram, self.const((DIM << 48) | (DIM << 32) | sp))

    def preload(self, bd: int, c: int):
        self.insn(F_PRELOAD, self.const((DIM << 48) | (DIM << 32) | bd),
                  self.const((DIM << 48) | (DIM << 32) | c))

    def compute(self, a_sp: int, bd_sp: int, first: bool):
        f = F_COMPUTE_PRELOADED if first else F_COMPUTE_ACCUMULATE
        self.insn(f, self.const((DIM << 48) | (DIM << 32) | a_sp),
                  self.const((DIM << 48) | (DIM << 32) | bd_sp))


def _name(value) -> str:
    return value.owner.sym.data


def build_kernel_module(gem: ModuleOp) -> ModuleOp:
    ops = list(gem.body.block.ops)
    move = next((o for o in ops if isinstance(o, D.GMoveOp)), None)

    if move is not None:
        return _movement_kernel(move)

    pack = next(o for o in ops if isinstance(o, D.GPackOp))
    commits = [o for o in ops if isinstance(o, D.GCommitOp)]
    matmuls = [o for o in ops if isinstance(o, D.GMatmulOp)]

    weight_name = pack.sym.data
    lhs_order = [_name(m.lhs) for m in matmuls]
    out_order = [c.sym.data for c in commits]
    arg_names = [weight_name] + lhs_order + out_order
    arg_index = {n: i for i, n in enumerate(arg_names)}

    K, N = (int(d) for d in pack.src.type.get_shape())
    kp, npw = _ceil(K), _ceil(N)

    b = _Builder(len(arg_names))
    b.fence()
    b.flush()
    b.config_ex_os()
    # mvin the resident weight once (reused across matmuls)
    b.config_ld(npw)
    Kt, Nt = kp // DIM, npw // DIM
    w_arg = arg_index[weight_name]
    for k in range(Kt):
        for j in range(Nt):
            off = (k * DIM * npw) + (j * DIM)
            b.mvin(b.dram(w_arg, off), B_BASE + (k * Nt + j) * DIM)

    for commit in commits:
        mm = commit.acc.owner
        lhs_name = _name(mm.lhs)
        M, Kc = (int(d) for d in mm.lhs.type.get_shape())
        mp, kpc = _ceil(M), _ceil(Kc)
        Mt, Ktc = mp // DIM, kpc // DIM
        a_arg = arg_index[lhs_name]
        # mvin lhs tiles
        b.config_ld(kpc)
        for i in range(Mt):
            for k in range(Ktc):
                off = (i * DIM * kpc) + (k * DIM)
                b.mvin(b.dram(a_arg, off), A_BASE + (i * Ktc + k) * DIM)
        # output config: activation + acc_scale applied at accumulator readout
        odt = commit.output_dtype.data
        epi = [e.data for e in commit.epilogue]
        act = RELU if "relu" in epi else NO_ACT
        scale_bits = _fbits(float(commit.acc_scale.value.data)) \
            if (commit.acc_scale is not None and "acc_scale" in epi) else SCALE_ONE
        esz = _esz(odt)
        npo = _ceil(N)
        Nto = npo // DIM
        b.config_st(npo * esz, act, scale_bits)
        o_arg = arg_index[commit.sym.data]
        full = ACC_FULL if odt == "i32" else 0
        for i in range(Mt):
            for j in range(Nto):
                row = (i * Nto + j) * DIM
                for k in range(Ktc):
                    last = (k == Ktc - 1)
                    out_sp = (ACC | row) if last else GARBAGE
                    b.preload(GARBAGE, out_sp)
                    b.compute(A_BASE + (i * Ktc + k) * DIM,
                              B_BASE + (k * Nto + j) * DIM, first=(k == 0))
                off = ((i * DIM * npo) + (j * DIM)) * esz
                b.mvout(b.dram(o_arg, off), ACC | full | row)

    b.fence()
    return _finish(b)


def _movement_kernel(move) -> ModuleOp:
    M, N = (int(d) for d in move.src.type.get_shape())
    mp, np_ = _ceil(M), _ceil(N)
    Mt, Nt = mp // DIM, np_ // DIM
    b = _Builder(2)  # (src, dst)
    b.fence()
    b.flush()
    b.config_ld(np_)
    b.config_st(np_, NO_ACT, SCALE_ONE)
    for i in range(Mt):
        for j in range(Nt):
            sp = (i * Nt + j) * DIM
            off = (i * DIM * np_) + (j * DIM)
            b.mvin(b.dram(0, off), sp)
            b.mvout(b.dram(1, off), sp)
    b.fence()
    return _finish(b)


def _finish(b: _Builder) -> ModuleOp:
    b.ops.append(llvm.ReturnOp())
    b.blk.add_ops(b.ops)
    fnty = llvm.LLVMFunctionType([b.ptr] * len(b.blk.args), None)
    fn = llvm.FuncOp("gemmini_kernel", fnty, linkage=llvm.LinkageAttr("external"),
                     body=Region([b.blk]))
    return ModuleOp([fn])


def kernel_text(gem: ModuleOp) -> str:
    mod = build_kernel_module(gem)
    buf = io.StringIO()
    Printer(stream=buf).print_op(mod)
    return buf.getvalue() + "\n"
