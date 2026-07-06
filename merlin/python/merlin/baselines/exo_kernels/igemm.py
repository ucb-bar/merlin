"""EXO int8 RVV GEMM kernel (vwmacc widening MAC) for the K1 whole-model glue runtime.

The int8 (W8A8) Linear is an integer inner product accumulated in i32::

    acc_i32[m, o] = sum_i  A_i8[m, i] * W_i8[o, i]
    Y_f32[m, o]   = a_scale[m] * w_scale[o] * acc_i32[m, o]

RVV ``vwmacc`` widens 2x per op, so the widening MAC runs at **i16 inputs -> i32 accumulator**:
the glue sign-extends the i8 activation and i8 weight to i16 (exact for the |i8| range), and
``vwmacc.vx`` computes ``acc_i32[o] += (i32)A_i16 * (i32)Wt_i16[o]``. The i32 accumulator never
overflows for K up to a few thousand (max |i8*i8|*K = 127*127*5632 ~ 9e7 << 2^31).

We vectorise the output-feature axis ``o`` by 16 (one ``m1`` i16 register / one ``m2`` i32
accumulator at VLEN=256). Weight is laid out ``[K, N]`` (i16) so ``Wt[k, o0:o0+16]`` is contiguous
— the glue transposes + widens each i8 weight once at load time (a scalar-glue pre-pass, off the
timed path). The requant (a_scale * w_scale * acc) is scalar glue.

This mirrors ``gemm.py`` (fp32 vfmacc.vf) exactly, on the integer widening path — proving EXO can
schedule the K1 int8 datapath, RVV-audited honestly.
"""
from __future__ import annotations

from exo import proc, DRAM
from exo.stdlib.scheduling import (
    stage_mem,
    set_memory,
    simplify,
    replace_all,
)

from merlin.baselines.exo_kernels.rvv256 import (
    RVV256_I16,
    RVV256_I32,
    rvv256_vld_i16,
    rvv256_vld_i32,
    rvv256_vst_i32,
    rvv256_zero_i32,
    rvv256_vwmacc_vx,
)

NB = 16  # output-feature tile (i16 m1 / i32 m2 lanes at VLEN=256)


@proc
def igemm_nt_ref(
    M: size,
    N: size,
    K: size,
    Y: i32[M, N] @ DRAM,
    X: ui16[M, K] @ DRAM,     # activation, i8 sign-extended to i16 by the glue (EXO ui16 -> C int16)
    Wt: ui16[K, N] @ DRAM,    # weight transposed to [K, N] and i8->i16 by the glue
):
    # pragma: no cover
    assert N % 16 == 0
    for m in seq(0, M):
        for no in seq(0, N / 16):
            for ni in seq(0, 16):
                Y[m, ni + 16 * no] = 0.0
            for k in seq(0, K):
                for ni in seq(0, 16):
                    Y[m, ni + 16 * no] += X[m, k] * Wt[k, ni + 16 * no]


def _schedule(p=igemm_nt_ref):
    # Accumulator-resident i32 register across the fused (init + k-reduction) region under one `no`.
    init = p.find("for ni in _: Y[m, ni+16*no] = 0.0")
    block = init.as_block().expand(0, 1)  # extend forward over the following `for k` loop
    p = stage_mem(p, block, "Y[m, 16*no:16*no+16]", "Y_reg")
    p = simplify(p)  # fold 16*no+16-16*no -> 16 for the fixed-width alloc
    p = set_memory(p, "Y_reg", RVV256_I32)

    # Stage the contiguous 16-wide i16 weight slab per k.
    p = stage_mem(p, "for ni in _: Y_reg[ni] += _", "Wt[k, 16*no:16*no+16]", "W_reg")
    p = simplify(p)
    p = set_memory(p, "W_reg", RVV256_I16)

    # Replace the 16-wide leaf loops with the RVV int intrinsics: zero-init i32, weight vld i16,
    # widening vwmacc, i32 store. (The Y copy-in becomes a redundant, immediately-zeroed vld path —
    # here init is a pure store, so stage_mem lays down zero + accumulate + store.)
    p = replace_all(p, [rvv256_vld_i32, rvv256_zero_i32, rvv256_vld_i16, rvv256_vwmacc_vx,
                        rvv256_vst_i32])

    return simplify(p)


igemm_nt_rvv = _schedule()
