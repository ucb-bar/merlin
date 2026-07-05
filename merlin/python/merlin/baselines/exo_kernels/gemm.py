"""EXO RVV GEMM kernel for the K1 whole-model glue runtime.

The dominant op in the LLM workloads is a linear layer ``Y = X @ W^T`` with the weight stored
row-major as ``W[N, K]`` (torch ``nn.Linear`` convention, matching the ``model.mlir`` operand
shapes, e.g. ``2048x2048``). So the compute is::

    Y[m, n] = sum_k  X[m, k] * W[n, k]

We vectorise the **output-feature** axis ``n`` by 8 (K1 VLEN=256 = 8 f32): for each ``(m, k)`` the
scalar ``X[m,k]`` is broadcast and FMA'd across an 8-wide slab of ``W[:, k]``. That needs ``W``
laid out as ``[K, N]`` (so ``W[k, n0:n0+8]`` is contiguous) — the glue transposes each weight once
at load time (recorded as a scalar-glue pre-pass, not on the timed path).

This is deliberately the *simplest* correct RVV GEMM — one EXO ``@instr`` per lane op, no packing /
register-blocking beyond the 8-wide n tile. It exists to prove the EXO→RVV→K1→audit path end to
end; ``rvv_audit`` will report its coverage honestly.
"""
from __future__ import annotations

from exo import proc, DRAM
from exo.stdlib.scheduling import (
    divide_loop,
    stage_mem,
    set_memory,
    simplify,
    reorder_loops,
    replace_all,
)

from merlin.baselines.exo_kernels.rvv256 import (
    RVV256,
    rvv256_vld,
    rvv256_vst,
    rvv256_zero,
    rvv256_vfmacc_vf,
)

# N is the vectorised (output-feature) axis; must be a multiple of 8. M and K are runtime sizes.
NB = 8


@proc
def gemm_nt_ref(
    M: size,
    N: size,
    K: size,
    Y: f32[M, N] @ DRAM,
    X: f32[M, K] @ DRAM,
    Wt: f32[K, N] @ DRAM,  # weight ALREADY transposed to [K, N] by the glue
):
    # pragma: no cover
    assert N % 8 == 0
    for m in seq(0, M):
        for no in seq(0, N / 8):
            for ni in seq(0, 8):
                Y[m, ni + 8 * no] = 0.0
            for k in seq(0, K):
                for ni in seq(0, 8):
                    Y[m, ni + 8 * no] += X[m, k] * Wt[k, ni + 8 * no]


def _schedule(p=gemm_nt_ref):
    # Accumulator-resident: stage the 8-wide Y slab into an RVV register across the fused
    # (init + k-reduction) region under one `no`. Build a block cursor spanning both the zero-init
    # loop and the k loop via .expand().
    init = p.find("for ni in _: Y[m, ni+8*no] = 0.0")
    block = init.as_block().expand(0, 1)  # extend forward to include the following `for k` loop
    p = stage_mem(p, block, "Y[m, 8*no:8*no+8]", "Y_reg")
    p = simplify(p)  # fold 8*no+8-8*no -> 8 so the RVV256 alloc sees a literal width
    p = set_memory(p, "Y_reg", RVV256)

    # Stage the contiguous 8-wide weight slab Wt[k, 8*no:8*no+8] into a register, hoisted above
    # the ni fmacc loop (one load per k).
    p = stage_mem(p, "for ni in _: Y_reg[ni] += _", "Wt[k, 8*no:8*no+8]", "W_reg")
    p = simplify(p)
    p = set_memory(p, "W_reg", RVV256)

    # Replace the 8-wide leaf loops (Y copy-in, zero, weight load, fmacc, store copy-out) with RVV
    # intrinsics. The Y copy-in becomes a (redundant, immediately-zeroed) vld — harmless, all-RVV.
    p = replace_all(p, [rvv256_zero, rvv256_vld, rvv256_vfmacc_vf, rvv256_vst])

    return simplify(p)


gemm_nt_rvv = _schedule()
