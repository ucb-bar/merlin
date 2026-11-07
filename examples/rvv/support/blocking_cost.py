"""RVV register-block legality and area ranking."""

from __future__ import annotations


def _rvv_blocking_lowers(MR: int, NR: int, M: int, N: int) -> bool:
    """Does the v3 register block ``[MR, NR]`` LOWER on a contraction with parallel extents (M, N)?

    MEASURED, not assumed — 57 synthetic ``linalg.matmul`` cells through
    ``llvmlower.lower.lower_model_file`` on the int8 (W8A8) pipeline, MR in {1,2,4,8}, NR in
    {8,16,32}, M in {2,6,8,17}, N in {8,24,40,128,344}; plus 18 whole-model lowerings
    (small_llama / openvla / bitvla) that the rule predicts with no exception. The three terms:

      * ``M % MR`` — masking the M (row) parallel dim always fails. This is the constraint the
        ``accumulator_resident_wholemodel_vf`` MR_mm=1 clamp exists to dodge (openvla M=17/20).
      * ``N % NR`` — masking the N (column) parallel dim fails too, EXCEPT at ``MR == 1``, where the
        C tile is effectively rank-1 (``vector<1xNR>``) and the masked ``transfer_write`` does
        lower. This is the half the ``impr_tuned_microkernel_v3_int8`` manifest misses: it blames
        "small-M matmuls", but small_llama has M=8 (a perfect multiple of MR=4) and trips the SAME
        failure through N=344 % 16.
      * ``NR <= N`` — the rank-1 escape above stops working once the tile EXCEEDS the extent
        (measured: MR=1, NR=16, N=8 fails; N=24 lowers; MR=1, NR=32 fails at N=24 and lowers at
        N=40). A fully-masked single iteration is not the same as a masked tail.

    The reduction extent does not appear: masking K is harmless (M=8, N=128, K=344 lowers at the
    champion block). On the fp32 pipeline the masked cells do not hard-fail — they degrade (spike
    instret at M=17,N=128,K=128: 159,999 unmasked vs 5,435,000 masked), so this predicate is a
    correctness gate on int8 and a ~34x performance gate on fp32."""
    if int(M) % int(MR):
        return False
    if int(N) % int(NR) == 0:
        return True
    return int(MR) == 1 and int(NR) <= int(N)


def _rvv_best_block(
    MR_cap: int, NR_cap: int, extents: list[tuple[int, int]]
) -> tuple[int, int]:
    """Largest legal ``(MR, NR)`` for EVERY contraction in ``extents`` (list of (M, N) pairs).

    Derived from the knob space, never from a shape table: the requested ``(MR_cap, NR_cap)`` is an
    upper bound (a bigger register block is better up to the register file), and the candidates are
    the divisors of the ``gcd`` of the observed extents plus the rank-1 escape ``MR = 1``. Ranked by
    register-block AREA (``MR * NR`` — the arithmetic intensity the block buys, i.e. how many
    accumulator lanes one A/B load feeds), tie-broken on the wider vector.

    Measured cost of the two runner-up shapes at 128^3 f32 (spike instret, both legal there):
    ``(4, 16)`` 1,065,000; ``(4, 8)`` 1,465,000; ``(1, 16)`` 1,465,000 — i.e. giving up either half
    of the block costs ~38%, and the area ranking picks whichever half survives."""
    from functools import reduce
    from math import gcd

    from merlin.kernels.microkernel import largest_divisor_at_most

    if not extents:
        return int(MR_cap), int(NR_cap)
    gM = reduce(gcd, [m for m, _ in extents])
    gN = reduce(gcd, [n for _, n in extents])
    minN = min(n for _, n in extents)
    cands = {
        (largest_divisor_at_most(gM, MR_cap), largest_divisor_at_most(gN, NR_cap)),
        (1, largest_divisor_at_most(gN, NR_cap)),
        (1, min(int(NR_cap), minN)),  # the rank-1 masked-N escape
        (1, 1),
    }  # always legal (never masks anything)
    legal = [
        (mr, nr)
        for (mr, nr) in cands
        if all(_rvv_blocking_lowers(mr, nr, m, n) for m, n in extents)
    ]
    if not legal:  # unreachable while (1, 1) is a candidate
        return 1, 1
    return max(legal, key=lambda b: (b[0] * b[1], b[1]))
