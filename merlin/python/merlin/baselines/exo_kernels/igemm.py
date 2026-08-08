"""EXO int8 RVV GEMM kernel (vwmacc widening MAC) for the K1 whole-model glue runtime.

The int8 (W8A8) Linear is an integer inner product accumulated in i32::

    acc_i32[m, o] = sum_i  A_i8[m, i] * W_i8[o, i]
    Y_f32[m, o]   = a_scale[m] * w_scale[o] * acc_i32[m, o]

RVV ``vwmacc`` widens 2x per op, so the widening MAC runs at **i16 inputs -> i32 accumulator**:
the glue sign-extends the i8 activation and i8 weight to i16 (exact for the |i8| range), and
``vwmacc.vx`` computes ``acc_i32[o] += (i32)A_i16 * (i32)Wt_i16[o]``. The i32 accumulator never
overflows for K up to a few thousand (max |i8*i8|*K = 127*127*5632 ~ 9e7 << 2^31).

We vectorise the output-feature axis ``o`` by 16 (one ``m1`` i16 register / one ``m2`` i32
accumulator at VLEN=256). Weight is laid out ``[K, N]`` (i16) so ``Wt[k, o0:o0+16]`` is contiguous.

**Autoscheduling (VLEN=256):** the un-tuned kernel does 1 vwmacc per scalar A-load + loop branch
per k, so RVV fraction is low (~0.17) and the ``vsetvli e16↔e32`` toggles each k. The tunable knob
is ``U`` = how many 16-wide output tiles share one A[m,k] scalar broadcast: unrolling the output
loop by ``U`` keeps U i32 accumulators live and issues U×(vle16+vwmacc) vector ops per scalar
A-load + branch, raising RVV% and cutting per-MAC overhead. :func:`build_igemm` emits the kernel
for a given ``U``; :mod:`merlin.baselines.exo` searches U on-board and keeps the best-measured one.
"""
from __future__ import annotations

from exo import proc, DRAM
from exo.stdlib.scheduling import (
    divide_loop,
    stage_mem,
    set_memory,
    simplify,
    replace_all,
    unroll_loop,
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


def _schedule_base(p):
    """Vectorise a single 16-wide output tile: i32 accumulator resident across k, vwmacc inner."""
    init = p.find("for ni in _: Y[m, ni+16*no] = 0.0")
    block = init.as_block().expand(0, 1)  # extend forward over the following `for k` loop
    p = stage_mem(p, block, "Y[m, 16*no:16*no+16]", "Y_reg")
    p = simplify(p)  # fold 16*no+16-16*no -> 16 for the fixed-width alloc
    p = set_memory(p, "Y_reg", RVV256_I32)
    p = stage_mem(p, "for ni in _: Y_reg[ni] += _", "Wt[k, 16*no:16*no+16]", "W_reg")
    p = simplify(p)
    p = set_memory(p, "W_reg", RVV256_I16)
    p = replace_all(p, [rvv256_vld_i32, rvv256_zero_i32, rvv256_vld_i16, rvv256_vwmacc_vx,
                        rvv256_vst_i32])
    return simplify(p)


def _make_uref(U: int):
    """Metaprogram an int8 GEMM reference with an explicit U-tile block whose tile loop ``nu`` lives
    INSIDE the k reduction — so once ``nu`` is unrolled the single scalar A-load ``X[m,k]`` is
    reused across the U widening MACs (the output-register-blocking that lifts the RVV ceiling).
    The block width ``16*U`` must be a compile-time literal (EXO folds div/mod only on constants).
    """
    BW = 16 * U

    @proc
    def igemm_nt_ref(M: size, N: size, K: size, Y: i32[M, N] @ DRAM, X: ui16[M, K] @ DRAM,
                     Wt: ui16[K, N] @ DRAM):
        # pragma: no cover
        assert N % BW == 0
        for m in seq(0, M):
            for nb in seq(0, N / BW):
                for nu in seq(0, U):
                    for ni in seq(0, 16):
                        Y[m, ni + 16 * (U * nb + nu)] = 0.0
                for k in seq(0, K):
                    for nu in seq(0, U):
                        for ni in seq(0, 16):
                            Y[m, ni + 16 * (U * nb + nu)] += X[m, k] * Wt[k, ni + 16 * (U * nb + nu)]

    return igemm_nt_ref


def _schedule_u(U: int):
    """Vectorise the U-tile-blocked reference into U *distinct* i32 accumulator registers resident
    across k, each fed by one shared A[m,k] broadcast per k (U vwmacc.vx per scalar A-load).

    EXO note: RVV vector C types are *sizeless*, so ``vint32m2_t Yb[U]`` (one arrayed register
    buffer) is illegal C — EXO's default Memory would emit exactly that and fail to compile. So we
    do NOT reshape into one ``[U,16]`` buffer; instead we unroll ``nu`` first (U separate tile
    bodies sharing one k-loop) then stage EACH tile into its own named 16-wide register (``Yr{j}``),
    which emits U separate ``vint32m2_t`` declarations. That is the schedule EXO *can* express; the
    thing it can't is arraying the sizeless register type."""
    BW = 16 * U
    p = _make_uref(U)
    p = unroll_loop(p, "nu")   # init tile loop -> U init loops
    p = unroll_loop(p, "nu")   # k-body tile loop -> U accumulate loops inside the shared k-loop
    p = simplify(p)
    # stage each tile j into its own 16-wide i32 register across the whole nb-body (U inits + k);
    # the per-tile window only redirects that tile's 16 columns.
    for j in range(U):
        base = f"16*{j}+{BW}*nb" if j else f"{BW}*nb"
        first_init = p.find("for ni in _: Y[m,_] = 0.0 #0")
        blk = first_init.as_block().expand(0, U)   # U init loops (incl this) ... incl the k-loop
        p = stage_mem(p, blk, f"Y[m, {base}:{base}+16]", f"Yr{j}")
        p = simplify(p)
        p = set_memory(p, f"Yr{j}", RVV256_I32)
    # stage each tile's per-k weight slab into its own i16 register (one W load per tile per k).
    for j in range(U):
        base = f"16*{j}+{BW}*nb" if j else f"{BW}*nb"
        p = stage_mem(p, f"for ni in _: Yr{j}[ni] += _", f"Wt[k, {base}:{base}+16]", f"Wr{j}")
        p = simplify(p)
        p = set_memory(p, f"Wr{j}", RVV256_I16)
    p = replace_all(p, [rvv256_vld_i32, rvv256_zero_i32, rvv256_vld_i16, rvv256_vwmacc_vx,
                        rvv256_vst_i32])
    return simplify(p)


def build_igemm(KU: int = 1, U: int = 1):
    """Emit an int8 vwmacc GEMM proc, autotune knobs:

    * ``U`` (output-register blocking) — U 16-wide i32 accumulators live across k; each k issues ONE
      scalar A[m,k] load reused across U ``vwmacc.vx``. This is the RVV-ceiling lever: it amortises
      the per-MAC scalar A-load that otherwise bounds GEMM RVV at ~0.19.
    * ``KU`` (k-unroll) — only for U=1 (the single-tile path): KU vle16+vwmacc per branch, amortising
      loop bookkeeping.

    U>1 uses the metaprogrammed U-blocked schedule (``_schedule_u``); U=1 uses the single-tile
    ``_schedule_base`` (optionally k-unrolled). The emitted C function is always ``igemm_nt_ref``.
    """
    if U > 1:
        return _schedule_u(U)
    p = _schedule_base(igemm_nt_ref)
    if KU > 1:
        p = divide_loop(p, "k", KU, ["ko", "ki"], tail="cut")
        p = unroll_loop(p, "ki")
    return simplify(p)


# Default kernel = baseline (U=1, KU=1); the autotuner rebuilds with the best (U,KU) via exocc.
igemm_nt_rvv = build_igemm(1, 1)
