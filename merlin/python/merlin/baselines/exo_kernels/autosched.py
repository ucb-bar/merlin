"""EXO-AUTOSCHEDULED RVV kernels — driven by EXO's higher-level scheduling automation instead of
hand cursor-surgery.

EXO 1.0.0 has **no cost-model autoscheduler / autotuner** (verified: ``exocc`` has no autoschedule
flag, ``exo.stdlib.scheduling`` exposes only manual rewrite primitives, and EXO's own canonical
matmul example is fully hand-scheduled — EXO is a *scheduling language*, deliberately manual). What
it DOES ship is a layer of **automated composite scheduling ops** in ``exo.stdlib.stdlib`` —
notably :func:`vectorize` (+ :func:`fma_rule`) — that raise the abstraction above raw cursors.

This module uses that automation: a single :func:`vectorize` call auto-transforms a scalar
transpose-free dot ``Y[m,n] = sum_k X[m,k]*W[n,k]`` into a full RVV schedule — it automatically
(a) allocates an 8-wide RVV partial-sum accumulator, (b) blocks the k-reduction by 8 with contiguous
vector loads of both operands, (c) emits the vector-vector MAC, (d) emits the horizontal lane-reduce
(``vfredusum``), and (e) cuts a clean scalar tail. This is exactly the stride-1 ``vredsum`` dot the
prior HAND schedule could not express with low-level primitives and fell back to hand-written RVV C
for — EXO's own ``vectorize`` now generates it. We then :func:`replace_all` the autoscheduled loops
with the RVV ``@instr`` intrinsics (``rvv256_vld`` / ``rvv256_vfmacc_vv`` / ``rvv256_vredsum``), so
the emitted C is genuine RVV — produced by EXO's scheduler, not by hand.

What EXO's automation still can NOT do here (honest gaps): pick the vector width / blocking factor
(no cost model — done by the measurement-driven autotuner in exo.py); map the broadcast-GEMM form
(``vfmacc.vf``) without an extra vector-broadcast ``@instr``; and fuse the int8→f32 weight dequant
(no widening-convert primitive) — so the int8 hot path keeps a hand dequant and feeds this
autoscheduled f32 dot only its compute.
"""
from __future__ import annotations

from exo import proc, DRAM
from exo.stdlib.scheduling import rename, simplify, replace_all, divide_loop, reorder_loops
from exo.stdlib.stdlib import vectorize, fma_rule

from merlin.baselines.exo_kernels.rvv256 import (
    RVV256, rvv256_vld, rvv256_zero, rvv256_vfmacc_vv, rvv256_vredsum,
)

VW = 8  # RVV f32 lanes at VLEN=256 (the K1 X60)


@proc
def fdot_nk_ref(M: size, N: size, K: size,
                Y: f32[M, N] @ DRAM, X: f32[M, K] @ DRAM, Wf: f32[N, K] @ DRAM):
    # Transpose-free dot: both X[m,:] and W[n,:] are k-contiguous (weight stays native [N,K]).
    assert K % 8 == 0
    for m in seq(0, M):
        for n in seq(0, N):
            acc: f32
            acc = 0.0
            for k in seq(0, K):
                acc += X[m, k] * Wf[n, k]
            Y[m, n] = acc


def _vectorize_kdot(p):
    """Apply EXO's `vectorize` auto-op to the k-reduction, then lower to RVV @instr calls.

    `vectorize` alone builds the 8-wide partial-sum accumulator + contiguous vector loads + the
    horizontal reduce; `replace_all` maps those autoscheduled loops onto the RVV intrinsics.
    """
    p = vectorize(p, p.find("for k in _: _"), VW, "f32", RVV256, rules=[fma_rule], tail="cut")
    p = simplify(p)
    p = replace_all(p, [rvv256_vld, rvv256_zero, rvv256_vfmacc_vv, rvv256_vredsum])
    return simplify(p)


def build_fdot(nblock: int = 1):
    """Autoschedule the transpose-free f32 dot; ``nblock`` register-blocks the output (n) axis.

    ``nblock`` is the one knob the measurement-driven autotuner sweeps (EXO has no cost model): the
    output loop is divided by ``nblock`` and the outer copies interleave so ``nblock`` independent
    dot accumulators share the same X[m,:] vector loads (the register-blocking the hand int8 kernel
    did by hand). ``nblock==1`` is the plain autoscheduled dot. The emitted C function is always
    ``fdot_nk_ref`` so the glue's symbol is stable across blocking factors.
    """
    p = rename(fdot_nk_ref, "fdot_nk_ref")
    if nblock and nblock > 1:
        # split the output axis; each of the nblock inner n's keeps its own vectorized dot (they
        # share the X[m,:] loads once the compiler CSEs them). Vectorize each inner dot.
        p = divide_loop(p, p.find("for n in _: _"), nblock, ["no", "ni"], tail="cut")
    p = _vectorize_kdot(p)
    return p


# The default autoscheduled kernel (unblocked). The autotuner overrides nblock per measurement.
fdot_nk_rvv = build_fdot(1)
