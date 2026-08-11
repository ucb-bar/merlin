"""Two activations against one shared weight — the workload an accelerator is built for.

    merlin-compile-kernel examples/triton/matmul_simple.py:repeated_rhs_matmul \
        --target-package out/artifacts/targets/gemmini/hand_v0 \
        --arg 'a0_ptr=*i8:16x32:read' --arg 'a1_ptr=*i8:16x32:read' \
        --arg 'w_ptr=*i8:32x16:read' \
        --arg 'c0_ptr=*i32:16x16:write' --arg 'c1_ptr=*i32:16x16:write' \
        --constexpr BM=16 --constexpr BN=16 --constexpr BK=32 --grid 1 \
        --emit all --verify

The shared `w_ptr` is what makes this worth accelerating: Merlin infers that the weight is immutable
and reused, proves it fits in the target's resident storage, and makes it stationary — RES_PACK once,
two matmuls against it, then EVICT. Give each matmul its own weight and that inference correctly
disappears.

The tile is 16x32x16 because `tl.dot` refuses anything smaller (M >= 16, N >= 16, K >= 32).
"""
import triton
import triton.language as tl


@triton.jit
def repeated_rhs_matmul(a0_ptr, a1_ptr, w_ptr, c0_ptr, c1_ptr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    w = tl.load(w_ptr + offs_k[:, None] * BN + offs_n[None, :])
    a0 = tl.load(a0_ptr + offs_m[:, None] * BK + offs_k[None, :])
    a1 = tl.load(a1_ptr + offs_m[:, None] * BK + offs_k[None, :])
    out = offs_m[:, None] * BN + offs_n[None, :]
    tl.store(c0_ptr + out, tl.dot(a0, w, out_dtype=tl.int32))
    tl.store(c1_ptr + out, tl.dot(a1, w, out_dtype=tl.int32))
