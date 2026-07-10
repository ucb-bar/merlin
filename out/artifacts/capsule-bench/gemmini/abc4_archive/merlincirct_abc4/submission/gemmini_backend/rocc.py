"""Tiling + Gemmini RoCC instruction synthesis, and LLVM-dialect emission.

We model the device program as a flat list of abstract RoCC instructions
(funct + rs1/rs2, where rs1/rs2 are either constants or pointer+offset), then
render it as an ``llvm.func @gemmini_kernel`` of ``.insn r 0x7b`` ops (the shared
decoder reads these; the runner links it into its harness and runs it on
spike/verilator).

ISA encodings are derived from the public ``gemmini.h`` / ``gemmini_params.h``
headers and the worked example kernels (matmul_ws.c / padded.c / mvin_mvout.c).
"""
from __future__ import annotations

import struct
from typing import Any

from .program import Program

DIM = 16
ADDR_LEN = 32

# functs
K_CONFIG, K_MVIN, K_MVOUT = 0, 2, 3
K_COMPUTE_PRELOADED, K_PRELOAD, K_FLUSH = 4, 6, 7
# config selectors
CONFIG_EX, CONFIG_LD, CONFIG_ST = 0, 1, 2
WEIGHT_STATIONARY = 1
NO_ACTIVATION, RELU = 0, 1
GARBAGE = 0xFFFFFFFF
ACC_BIT = 1 << (ADDR_LEN - 1)       # 0x80000000  : address is an accumulator
ACC_ACCUMULATE = 1 << (ADDR_LEN - 2)  # 0x40000000 : accumulate (vs overwrite)
ACC_FULL = 1 << (ADDR_LEN - 3)      # 0x20000000  : full-width (i32) readout

A_SP = 0
W_SP = 2048
MASK64 = (1 << 64) - 1


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _ceil_tiles(n: int) -> int:
    return (n + DIM - 1) // DIM


def _m(n: int, k: int) -> int:
    """Tile extent: min(DIM, n - k*DIM)."""
    return min(DIM, n - k * DIM)


def _pad16(n: int) -> int:
    """Round up to a multiple of DIM (the harness pads edge tiles / strides)."""
    return ((n + DIM - 1) // DIM) * DIM


# An instruction: dict(funct, rs1, rs2) where each rs is ("c", value) or
# ("p", param_index, byte_offset).
def _c(v: int):
    return ("c", v & MASK64)


def _config_ex():
    bits = _f32_bits(1.0)
    rs1 = (bits << 32) | (1 << 16) | (WEIGHT_STATIONARY << 2) | CONFIG_EX
    rs2 = (1 << 48)
    return {"funct": K_CONFIG, "rs1": _c(rs1), "rs2": _c(rs2)}


def _config_ld(stride_bytes: int):
    bits = _f32_bits(1.0)
    rs1 = (bits << 32) | (DIM << 16) | (1 << 8) | CONFIG_LD
    return {"funct": K_CONFIG, "rs1": _c(rs1), "rs2": _c(stride_bytes)}


def _config_st(stride_bytes: int, acc_act: int, acc_scale: float):
    rs1 = (acc_act << 2) | CONFIG_ST
    rs2 = (_f32_bits(acc_scale) << 32) | (stride_bytes & 0xFFFFFFFF)
    return {"funct": K_CONFIG, "rs1": _c(rs1), "rs2": _c(rs2)}


def _flush():
    return {"funct": K_FLUSH, "rs1": _c(0), "rs2": _c(0)}


def _mvin(param_idx: int, off: int, cols: int, rows: int, sp: int):
    rs2 = (rows << (ADDR_LEN + 16)) | (cols << ADDR_LEN) | sp
    return {"funct": K_MVIN, "rs1": ("p", param_idx, off), "rs2": _c(rs2)}


def _mvout(param_idx: int, off: int, cols: int, rows: int, sp: int):
    rs2 = (rows << (ADDR_LEN + 16)) | (cols << ADDR_LEN) | sp
    return {"funct": K_MVOUT, "rs1": ("p", param_idx, off), "rs2": _c(rs2)}


def _preload(bd, bd_cols, bd_rows, c, c_cols, c_rows):
    rs1 = (bd_rows << (ADDR_LEN + 16)) | (bd_cols << ADDR_LEN) | bd
    rs2 = (c_rows << (ADDR_LEN + 16)) | (c_cols << ADDR_LEN) | c
    return {"funct": K_PRELOAD, "rs1": _c(rs1), "rs2": _c(rs2)}


def _compute(a, a_cols, a_rows, bd, bd_cols, bd_rows):
    rs1 = (a_rows << (ADDR_LEN + 16)) | (a_cols << ADDR_LEN) | a
    rs2 = (bd_rows << (ADDR_LEN + 16)) | (bd_cols << ADDR_LEN) | bd
    return {"funct": K_COMPUTE_PRELOADED, "rs1": _c(rs1), "rs2": _c(rs2)}


# --------------------------------------------------------------------------- #
#  Argument layout
# --------------------------------------------------------------------------- #
def kernel_params(prog: Program) -> list[str]:
    """[resident weights] ++ [streamed inputs] ++ [outputs], in op order."""
    weights: list[str] = []
    inputs: list[str] = []
    outputs: list[str] = []
    for rec in prog.ops:
        if rec["kind"] == "pack":
            if rec["src"] not in weights:
                weights.append(rec["src"])
        elif rec["kind"] == "matmul":
            inputs.append(rec["lhs"])
        elif rec["kind"] == "movement":
            inputs.append(rec["src"]); outputs.append(rec["dst"])
        elif rec["kind"] == "conv2d":
            # conv weight is packed-resident; the device activation is the
            # derived im2col matrix (the runner embeds it under that name).
            from .conv import conv_geometry
            if rec["weight"] not in weights:
                weights.append(rec["weight"])
            inputs.append(conv_geometry(rec)["im2col"])
            outputs.append(rec["dst"])
        elif rec["kind"] == "commit":
            outputs.append(rec["dst"])
    return weights + inputs + outputs


# --------------------------------------------------------------------------- #
#  Instruction synthesis
# --------------------------------------------------------------------------- #
def build_instructions(prog: Program):
    params = kernel_params(prog)
    idx = {name: i for i, name in enumerate(params)}
    instrs: list[dict] = [_flush()]

    has_matmul = any(r["kind"] in ("matmul", "conv2d") for r in prog.ops)
    if has_matmul:
        instrs.append(_config_ex())

    # pre-index: weight name and shape per resident handle (from pack records)
    wshape: dict[str, list[int]] = {}
    for r in prog.ops:
        if r["kind"] == "pack":
            wshape[r["dst"]] = r["shape"]

    # matmul/commit are paired through the accumulator handle
    commits = {r["acc"]: r for r in prog.ops if r["kind"] == "commit"}

    for rec in prog.ops:
        if rec["kind"] == "matmul":
            commit = commits[rec["dst"]]
            _emit_matmul(instrs, idx, rec, commit)
        elif rec["kind"] == "movement":
            _emit_movement(instrs, idx, rec)
        elif rec["kind"] == "conv2d":
            _emit_conv(instrs, idx, rec)

    return params, instrs


def _emit_matmul(instrs, idx, rec, commit):
    M, K = rec["lhs_shape"]
    K2, N = rec["weight_shape"]
    out_i8 = commit["output_dtype"] == "i8"
    epi = commit["epilogue"]
    acc_act = RELU if "relu" in epi else NO_ACTIVATION
    acc_scale = commit["acc_scale"] if commit.get("acc_scale") is not None else 1.0
    _emit_ws_matmul(instrs, idx[rec["lhs"]], idx[rec["weight"]], idx[commit["dst"]],
                    M, K, N, out_i8, acc_act, acc_scale)


def _emit_conv(instrs, idx, rec):
    """im2col conv -> WS matmul over the derived [M,K] activation x [K,Co] weight."""
    from .conv import conv_geometry
    geo = conv_geometry(rec)
    out_i8 = rec["output_dtype"] == "i8"
    acc_act = RELU if "relu" in rec["epilogue"] else NO_ACTIVATION
    _emit_ws_matmul(instrs, idx[geo["im2col"]], idx[rec["weight"]], idx[rec["dst"]],
                    geo["M"], geo["K"], geo["N"], out_i8, acc_act, 1.0)


def _emit_ws_matmul(instrs, a_idx, w_idx, o_idx, M, K, N, out_i8, acc_act, acc_scale):
    esize = 1 if out_i8 else 4

    Mt, Kt, Nt = _ceil_tiles(M), _ceil_tiles(K), _ceil_tiles(N)
    # DRAM buffers are row-major with each dim zero-padded to a multiple of DIM,
    # so the row stride is the padded (not logical) extent.
    Kp, Np = _pad16(K), _pad16(N)

    # mvin A (row stride = Kp elements)
    instrs.append(_config_ld(Kp))
    for it in range(Mt):
        for kt in range(Kt):
            cols, rows = _m(K, kt), _m(M, it)
            off = (it * DIM) * Kp + kt * DIM
            instrs.append(_mvin(a_idx, off, cols, rows, A_SP + (it * Kt + kt) * DIM))
    # mvin W (row stride = Np elements)
    instrs.append(_config_ld(Np))
    for kt in range(Kt):
        for nt in range(Nt):
            cols, rows = _m(N, nt), _m(K, kt)
            off = (kt * DIM) * Np + nt * DIM
            instrs.append(_mvin(w_idx, off, cols, rows, W_SP + (kt * Nt + nt) * DIM))

    instrs.append(_config_st(Np * esize, acc_act, acc_scale))

    for it in range(Mt):
        for nt in range(Nt):
            mrows, ncols = _m(M, it), _m(N, nt)
            for kt in range(Kt):
                kcols = _m(K, kt)
                acc_addr = ACC_BIT | (ACC_ACCUMULATE if kt > 0 else 0)
                instrs.append(_preload(W_SP + (kt * Nt + nt) * DIM, ncols, kcols,
                                       acc_addr, ncols, mrows))
                instrs.append(_compute(A_SP + (it * Kt + kt) * DIM, kcols, mrows,
                                       GARBAGE, ncols, mrows))
            out_addr = ACC_BIT | (0 if out_i8 else ACC_FULL)
            # DRAM output buffer is zero-padded to mp x Np (the harness allocates
            # and prints with the padded row stride Np), so use Np here, not N.
            off = ((it * DIM) * Np + nt * DIM) * esize
            instrs.append(_mvout(o_idx, off, ncols, mrows, out_addr))


def _emit_movement(instrs, idx, rec):
    R, C = rec["shape"]
    s_idx = idx[rec["src"]]
    o_idx = idx[rec["dst"]]
    instrs.append(_config_ld(C))
    instrs.append(_config_st(C, NO_ACTIVATION, 1.0))
    Rt, Ct = _ceil_tiles(R), _ceil_tiles(C)
    for it in range(Rt):
        for jt in range(Ct):
            cols, rows = _m(C, jt), _m(R, it)
            off = (it * DIM) * C + jt * DIM
            sp = A_SP + (it * Ct + jt) * DIM
            instrs.append(_mvin(s_idx, off, cols, rows, sp))
            instrs.append(_mvout(o_idx, off, cols, rows, sp))


# --------------------------------------------------------------------------- #
#  LLVM-dialect emission
# --------------------------------------------------------------------------- #
def emit_llvm(prog: Program) -> str:
    params, instrs = build_instructions(prog)
    n = len(params)
    args = ", ".join(f"%arg{i}: !llvm.ptr" for i in range(n))
    lines = ["module {",
             f"  llvm.func @gemmini_kernel({args}) {{"]
    ssa = [0]

    def fresh() -> str:
        ssa[0] += 1
        return f"%v{ssa[0]}"

    # open the RoCC region with a fence (the decoder requires the trace to
    # open with a FENCE; it is also closed with one below).
    lines.append('    llvm.inline_asm has_side_effects "fence", "" : () -> ()')

    # ptrtoint each param once
    pint = []
    for i in range(n):
        v = fresh()
        lines.append(f"    {v} = llvm.ptrtoint %arg{i} : !llvm.ptr to i64")
        pint.append(v)

    def operand(rs) -> str:
        if rs[0] == "c":
            v = fresh()
            lines.append(f"    {v} = llvm.mlir.constant({rs[1]} : i64) : i64")
            return v
        _, pidx, off = rs
        base = pint[pidx]
        if off == 0:
            return base
        o = fresh()
        lines.append(f"    {o} = llvm.mlir.constant({off} : i64) : i64")
        r = fresh()
        lines.append(f"    {r} = llvm.add {base}, {o} : i64")
        return r

    for ins in instrs:
        rs1 = operand(ins["rs1"])
        rs2 = operand(ins["rs2"])
        asm = f".insn r 0x7b, 0x3, {ins['funct']}, x0, $0, $1"
        lines.append(
            f'    llvm.inline_asm has_side_effects "{asm}", "r,r" {rs1}, {rs2} '
            f": (i64, i64) -> ()")
    # final fence
    lines.append('    llvm.inline_asm has_side_effects "fence", "" : () -> ()')
    lines.append("    llvm.return")
    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"
