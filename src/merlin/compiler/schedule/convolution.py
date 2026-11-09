"""Enumerate bounded gathered convolution panels under explicit storage budgets."""

from __future__ import annotations

from dataclasses import (
    dataclass,
)

from .block import (
    _regions,
    _schedule_ops,
)
from .model import (
    BANK_ALIGNED,
    LHS,
    WEIGHT,
    BlockSchedule,
    BlockScheduleError,
    ConvContraction,
    Geometry,
    K,
    Knobs,
    Load,
    M,
    N,
    _ceil_div,
)
from .residency import (
    check_residency,
)


@dataclass(frozen=True)
class ConvolutionPanel:
    """Operation range after which operand rows may be reused.

    Accumulators stay live across K panels until the final K panel stores them.
    This logical boundary supplies no target fence or asynchronous completion.
    """

    m_start: int
    n_start: int
    k_start: int
    m_blocks: int
    n_blocks: int
    k_blocks: int
    first_operation: int
    end_operation: int


@dataclass(frozen=True)
class TiledConvolutionSchedule:
    schedule: BlockSchedule
    panels: tuple[ConvolutionPanel, ...]
    m_blocks: int
    n_blocks: int
    k_blocks: int
    candidates_considered: int
    requested_operand_elements: int


def schedule_tiled_convolution(
    conv: ConvContraction,
    geometry: Geometry,
    knobs: Knobs | None = None,
    *,
    candidate_budget: int = 4096,
    max_operations: int = 1_000_000,
) -> TiledConvolutionSchedule:
    """Keep output panels live while streaming bounded, reusable reduction panels.

    This target-independent family preserves NHWC gathers, tails and original
    tap/channel reduction order. Operand rows are reused between K panels; live
    output rows are reused only after stores. Candidates minimize requested
    operand ELEMENTS (including padding zeros), not bytes, traffic or cycles.
    For each M/N candidate, the largest fitting K panel minimizes panel count.
    Bias, output conversion and physical completion stay with the caller. A
    target must establish exact continuation and completion before consumption.
    This finite schedule is bounded before expansion; it emits no hardware loop.
    """
    knobs = knobs or Knobs()
    knobs.validate()
    cv, d = conv, geometry.block
    sizes = (
        candidate_budget,
        max_operations,
        d,
        geometry.operand_rows,
        geometry.accumulator_rows,
        cv.batch,
        cv.in_h,
        cv.in_w,
        cv.ci,
        cv.kh,
        cv.kw,
        cv.co,
    )
    if any(type(x) is not int or x <= 0 for x in sizes):
        raise BlockScheduleError("tiled convolution requires positive integer geometry and budgets")
    if (
        len(cv.stride) != 2
        or len(cv.dilation) != 2
        or len(cv.padding) != 4
        or any(type(x) is not int or x <= 0 for x in (*cv.stride, *cv.dilation))
        or any(type(x) is not int or x < 0 for x in cv.padding)
    ):
        raise BlockScheduleError("tiled convolution needs positive stride/dilation and nonnegative padding")
    if cv.out_h <= 0 or cv.out_w <= 0:
        raise BlockScheduleError("tiled convolution has no output pixels")
    if geometry.separate_accumulator_space is not True:
        raise BlockScheduleError("tiled convolution requires separate operand and accumulator stores")
    if knobs.placement == BANK_ALIGNED and (
        type(geometry.operand_bank_rows) is not int or geometry.operand_bank_rows <= 0
    ):
        raise BlockScheduleError("bank_aligned placement requires positive known operand_bank_rows")
    kt = cv.kh * cv.kw * _ceil_div(cv.ci, d)
    mt, nt = _ceil_div(cv.pixels, d), _ceil_div(cv.co, d)
    # Reject oversized proposals before materializing taps, gathers or operations.
    if 2 * mt * nt * kt + mt * nt > max_operations:
        raise BlockScheduleError("tiled convolution exceeds its finite operation budget")
    max_m = min(mt, geometry.accumulator_rows // d, geometry.operand_rows // d - 1)
    max_n = min(nt, geometry.accumulator_rows // d, geometry.operand_rows // d - 1)
    if max_m <= 0 or max_n <= 0:
        raise BlockScheduleError("one primitive block per operand and output does not fit")
    if max_m * max_n > candidate_budget:
        raise BlockScheduleError("tiled convolution search exceeds its explicit candidate budget")
    best, considered = None, 0
    source_k = cv.kh * cv.kw * cv.ci
    for bm in range(1, max_m + 1):
        for bn in range(1, max_n + 1):
            considered += 1
            bk = min(kt, geometry.operand_rows // ((bm + bn) * d))
            if bm * bn * d > geometry.accumulator_rows or bk == 0:
                continue
            regions, notes = _regions({LHS: bm * bk * d, WEIGHT: bn * bk * d}, geometry, knobs)
            lhs_repeats = _ceil_div(nt, bn) if knobs.load_on_index_change or knobs.loop_order[-1] == N else nt
            rhs_repeats = _ceil_div(mt, bm) if knobs.load_on_index_change or knobs.loop_order[-1] == M else mt
            loads = mt * kt * lhs_repeats + nt * kt * rhs_repeats
            commands = 2 * mt * nt * kt + mt * nt + loads
            if commands > max_operations:
                continue
            requested = cv.pixels * source_k * lhs_repeats + source_k * cv.co * rhs_repeats
            key = (requested, loads, _ceil_div(kt, bk), bm + bn, bm, bn)
            if best is None or key < best[0]:
                best = key, bm, bn, bk, regions, notes
    if best is None:
        raise BlockScheduleError("no output panel satisfies storage and finite operation budgets")
    key, bm, bn, bk, regions, notes = best
    taps = [(ky, kx, ci, min(d, cv.ci - ci)) for ky in range(cv.kh) for kx in range(cv.kw) for ci in range(0, cv.ci, d)]
    operations, panels = [], []
    for m0 in range(0, mt, bm):
        for n0 in range(0, nt, bn):
            for k0 in range(0, kt, bk):
                mr, nr, kr = min(bm, mt - m0), min(bn, nt - n0), min(bk, kt - k0)

                def lhs(mi, kk, m_base=m0, k_base=k0):
                    ky, kx, ci, depth = taps[kk]
                    gather = []
                    for pixel in range(mi * d, min(cv.pixels, (mi + 1) * d)):
                        image, spatial = divmod(pixel, cv.out_h * cv.out_w)
                        y, x = divmod(spatial, cv.out_w)
                        iy = y * cv.stride[0] - cv.padding[0] + ky * cv.dilation[0]
                        ix = x * cv.stride[1] - cv.padding[1] + kx * cv.dilation[1]
                        gather.append(
                            ((image * cv.in_h + iy) * cv.in_w + ix, ci)
                            if 0 <= iy < cv.in_h and 0 <= ix < cv.in_w
                            else None
                        )
                    return Load(
                        LHS,
                        (mi, kk),
                        -1,
                        -1,
                        len(gather),
                        depth,
                        regions[LHS][0] + ((kk - k_base) * bm + mi - m_base) * d,
                        kk,
                        tuple(gather),
                    )

                def rhs(kk, nj, n_base=n0, k_base=k0):
                    ky, kx, ci, depth = taps[kk]
                    return Load(
                        WEIGHT,
                        (kk, nj),
                        (ky * cv.kw + kx) * cv.ci + ci,
                        nj * d,
                        depth,
                        min(d, cv.co - nj * d),
                        regions[WEIGHT][0] + ((kk - k_base) * bn + nj - n_base) * d,
                        kk,
                    )

                start = len(operations)
                operations.extend(
                    _schedule_ops(
                        knobs,
                        d,
                        {K: range(k0, k0 + kr), M: range(m0, m0 + mr), N: range(n0, n0 + nr)},
                        lhs,
                        rhs,
                        lambda kk: taps[kk][3],
                        lambda mi: min(d, cv.pixels - mi * d),
                        lambda nj: min(d, cv.co - nj * d),
                        m0,
                        n0,
                        bn,
                        kt,
                    )
                )
                panels.append(ConvolutionPanel(m0, n0, k0, mr, nr, kr, start, len(operations)))
    if len(operations) > max_operations:
        raise BlockScheduleError("actual emitted schedule exceeds its finite operation budget")
    schedule = BlockSchedule(tuple(operations), cv, geometry, knobs, regions, tuple(notes))
    check_residency(schedule)
    return TiledConvolutionSchedule(schedule, tuple(panels), bm, bn, bk, considered, key[0])
