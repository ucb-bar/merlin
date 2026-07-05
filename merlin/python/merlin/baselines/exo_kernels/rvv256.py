"""8-wide (VLEN=256) f32 RVV register class + intrinsics for the SpacemiT K1 X60.

EXO ships ``exo.platforms.rvv`` with a 4-wide ``m1`` register (VLEN=128). The K1 is VLEN=256, so
one ``vfloat32m1_t`` holds 8 f32. We mirror the shipped intrinsics at width 8 so EXO's scheduler
blocks the reduction/lane dimension by 8 and the emitted C fills a full 256-bit register.

The C templates use the standard ``__riscv_*`` vector intrinsics with a runtime ``vl``; the
SpacemiT clang (``-march=rv64gcv``) lowers them to ``vsetvli`` + ``vle32``/``vse32``/``vfmacc``.
"""
from __future__ import annotations

from exo import Memory, DRAM, instr
from exo.core.memory import MemGenError

_W = 8  # f32 lanes in one m1 register at VLEN=256


class RVV256(Memory):
    """A vector register holding 8 f32 (K1 VLEN=256, m1)."""

    @classmethod
    def global_(cls):
        return "#include <riscv_vector.h>"

    @classmethod
    def can_read(cls):
        return False

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if not shape:
            raise MemGenError(f"{srcinfo}: RVV vectors are not scalar values")
        if prim_type != "float":
            raise MemGenError(f"{srcinfo}: RVV256 vectors must be f32, got {prim_type}")
        if not (shape[-1].isdecimal() and int(shape[-1]) == _W):
            raise MemGenError(f"{srcinfo}: RVV256 register width must be {_W}, got {shape}")
        shape = shape[:-1]
        if shape:
            if not all(s.isdecimal() and int(s) > 0 for s in shape):
                raise MemGenError(f"{srcinfo}: cannot allocate a variable number of RVV registers")
            return f'vfloat32m1_t {new_name}[{"][".join(map(str, shape))}];'
        return f"vfloat32m1_t {new_name};"

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        assert strides[-1] == "1"
        idxs = indices[:-1] or ""
        if idxs:
            idxs = "[" + "][".join(idxs) + "]"
        return f"{baseptr}{idxs}"


@instr("{dst_data} = __riscv_vle32_v_f32m1(&{src_data}, {vl});")
def rvv256_vld(dst: [f32][8] @ RVV256, src: [f32][8] @ DRAM, vl: size):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert vl >= 0
    assert vl <= 8
    for i in seq(0, vl):
        dst[i] = src[i]


@instr("__riscv_vse32_v_f32m1(&{dst_data}, {src_data}, {vl});")
def rvv256_vst(dst: [f32][8] @ DRAM, src: [f32][8] @ RVV256, vl: size):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert vl >= 0
    assert vl <= 8
    for i in seq(0, vl):
        dst[i] = src[i]


@instr("{dst_data} = __riscv_vfmv_v_f_f32m1(0.0f, {vl});")
def rvv256_zero(dst: [f32][8] @ RVV256, vl: size):
    assert stride(dst, 0) == 1
    assert vl >= 0
    assert vl <= 8
    for i in seq(0, vl):
        dst[i] = 0.0


@instr("{dst_data} = __riscv_vfmacc_vf_f32m1({dst_data}, {lhs_data}, {rhs_data}, {vl});")
def rvv256_vfmacc_vf(dst: [f32][8] @ RVV256, lhs: [f32][1] @ DRAM, rhs: [f32][8] @ RVV256, vl: size):
    # dst[i] += lhs * rhs[i]  (broadcast scalar lhs[0] across the rhs vector via vfmacc.vf)
    assert stride(dst, 0) == 1
    assert stride(rhs, 0) == 1
    assert vl >= 0
    assert vl <= 8
    for i in seq(0, vl):
        dst[i] += lhs[0] * rhs[i]
