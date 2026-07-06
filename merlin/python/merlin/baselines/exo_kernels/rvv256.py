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


@instr("{dst_data} = __riscv_vfadd_vv_f32m1({lhs_data}, {rhs_data}, {vl});")
def rvv256_vfadd(dst: [f32][8] @ RVV256, lhs: [f32][8] @ RVV256, rhs: [f32][8] @ RVV256, vl: size):
    # dst[i] = lhs[i] + rhs[i]  (element-wise vector add — residual add)
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1
    assert vl >= 0
    assert vl <= 8
    for i in seq(0, vl):
        dst[i] = lhs[i] + rhs[i]


@instr("{dst_data} = __riscv_vfmul_vv_f32m1({lhs_data}, {rhs_data}, {vl});")
def rvv256_vfmul(dst: [f32][8] @ RVV256, lhs: [f32][8] @ RVV256, rhs: [f32][8] @ RVV256, vl: size):
    # dst[i] = lhs[i] * rhs[i]  (element-wise vector multiply — SwiGLU gate*up)
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1
    assert vl >= 0
    assert vl <= 8
    for i in seq(0, vl):
        dst[i] = lhs[i] * rhs[i]


# --------------------------------------------------------------------------------------------- #
#   int8 widening path (vwmacc): i16 x i16 -> i32 accumulate, VLEN=256
#
#   The K1 int8 GEMM accumulates i8xi8 products in i32. RVV ``vwmacc`` widens 2x per op, so we run
#   the widening MAC at i16 inputs -> i32 accumulator (inputs are i8 sign-extended to i16 by the
#   glue, so the product i16*i16 is exact for the |i8| range and the i32 accumulator never
#   overflows for K up to ~5632). One ``m1`` i16 register holds 16 lanes at VLEN=256; the widened
#   i32 result lives in an ``m2`` register (also 16 lanes). We vectorise the output axis by 16.
# --------------------------------------------------------------------------------------------- #

_W16 = 16  # i32 accumulator lanes (m2 at VLEN=256); the i16 input slab is the same 16 lanes (m1)


class RVV256_I16(Memory):
    """A vector register holding 16 i16 (K1 VLEN=256, m1) — the vwmacc input width.

    EXO's frontend has no signed ``i16`` primitive, so the EXO-visible element type is ``ui16``
    (the reference semantics only need the 16-lane shape); the emitted C uses the signed
    ``vint16m1_t`` and signed ``__riscv_vwmacc`` so the arithmetic is a true signed widening MAC.
    """

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
        if prim_type not in ("uint16_t", "int16_t"):
            raise MemGenError(f"{srcinfo}: RVV256_I16 vectors must be (u)i16, got {prim_type}")
        if not (shape[-1].isdecimal() and int(shape[-1]) == _W16):
            raise MemGenError(f"{srcinfo}: RVV256_I16 width must be {_W16}, got {shape}")
        shape = shape[:-1]
        if shape:
            if not all(s.isdecimal() and int(s) > 0 for s in shape):
                raise MemGenError(f"{srcinfo}: cannot allocate a variable number of RVV registers")
            return f'vint16m1_t {new_name}[{"][".join(map(str, shape))}];'
        return f"vint16m1_t {new_name};"

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


class RVV256_I32(Memory):
    """A widened accumulator register holding 16 i32 (K1 VLEN=256, m2) — the vwmacc output."""

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
        if prim_type != "int32_t":
            raise MemGenError(f"{srcinfo}: RVV256_I32 vectors must be i32, got {prim_type}")
        if not (shape[-1].isdecimal() and int(shape[-1]) == _W16):
            raise MemGenError(f"{srcinfo}: RVV256_I32 width must be {_W16}, got {shape}")
        shape = shape[:-1]
        if shape:
            if not all(s.isdecimal() and int(s) > 0 for s in shape):
                raise MemGenError(f"{srcinfo}: cannot allocate a variable number of RVV registers")
            return f'vint32m2_t {new_name}[{"][".join(map(str, shape))}];'
        return f"vint32m2_t {new_name};"

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


@instr("{dst_data} = __riscv_vle16_v_i16m1((const int16_t*)&{src_data}, {vl});")
def rvv256_vld_i16(dst: [ui16][16] @ RVV256_I16, src: [ui16][16] @ DRAM, vl: size):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert vl >= 0
    assert vl <= 16
    for i in seq(0, vl):
        dst[i] = src[i]


@instr("{dst_data} = __riscv_vle32_v_i32m2(&{src_data}, {vl});")
def rvv256_vld_i32(dst: [i32][16] @ RVV256_I32, src: [i32][16] @ DRAM, vl: size):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert vl >= 0
    assert vl <= 16
    for i in seq(0, vl):
        dst[i] = src[i]


@instr("{dst_data} = __riscv_vmv_v_x_i32m2(0, {vl});")
def rvv256_zero_i32(dst: [i32][16] @ RVV256_I32, vl: size):
    assert stride(dst, 0) == 1
    assert vl >= 0
    assert vl <= 16
    for i in seq(0, vl):
        dst[i] = 0


@instr("__riscv_vse32_v_i32m2(&{dst_data}, {src_data}, {vl});")
def rvv256_vst_i32(dst: [i32][16] @ DRAM, src: [i32][16] @ RVV256_I32, vl: size):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert vl >= 0
    assert vl <= 16
    for i in seq(0, vl):
        dst[i] = src[i]


@instr("{dst_data} = __riscv_vwmacc_vx_i32m2({dst_data}, (int16_t){lhs_data}, {rhs_data}, {vl});")
def rvv256_vwmacc_vx(dst: [i32][16] @ RVV256_I32, lhs: [ui16][1] @ DRAM, rhs: [ui16][16] @ RVV256_I16,
                     vl: size):
    # dst[o] += (i32)lhs[0] * (i32)rhs[o]  — widening MAC: i16 scalar x i16 vector -> i32 accumulate
    assert stride(dst, 0) == 1
    assert stride(rhs, 0) == 1
    assert vl >= 0
    assert vl <= 16
    for o in seq(0, vl):
        dst[o] += lhs[0] * rhs[o]
