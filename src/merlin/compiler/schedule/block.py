"""Construct checked contraction nests, storage placements and ordered block commands."""

from __future__ import annotations

from dataclasses import (
    replace,
)
from typing import (
    Any,
)

from .model import (
    BANK_ALIGNED,
    CONTIGUOUS,
    LHS,
    NEST,
    OPPOSITE_END,
    WEIGHT,
    BlockSchedule,
    BlockScheduleError,
    ColumnBias,
    Compute,
    Contraction,
    ConvContraction,
    Geometry,
    InitializeAccumulator,
    K,
    Knobs,
    Load,
    M,
    N,
    Op,
    Preload,
    Store,
    _ceil_div,
)
from .residency import (
    check_residency,
)


def _regions(
    rows_needed: dict[str, int], geometry: Geometry, knobs: Knobs
) -> tuple[dict[str, tuple[int, int]], list[str]]:
    """``(role -> (base row, rows), notes)`` for the placement policy. The streamed operand starts at
    row 0; the resident operand's base is the policy's whole content."""
    notes: list[str] = []
    lhs_rows, weight_rows = rows_needed[LHS], rows_needed[WEIGHT]
    if lhs_rows > geometry.operand_rows or weight_rows > geometry.operand_rows:
        raise BlockScheduleError(
            f"one operand alone needs {max(lhs_rows, weight_rows)} rows of a {geometry.operand_rows}-row "
            "store: this contraction has to be tiled before it can be scheduled"
        )
    if knobs.placement == OPPOSITE_END:
        base = geometry.operand_rows - weight_rows
    else:
        if knobs.placement == BANK_ALIGNED and not geometry.operand_bank_rows:
            raise BlockScheduleError(
                "placement 'bank_aligned' needs the store's per-bank row count, which this target's "
                "facts do not give (operand_bank_rows is unknown)"
            )
        wanted = (
            lhs_rows
            if knobs.placement == CONTIGUOUS
            else _ceil_div(lhs_rows, geometry.operand_bank_rows) * geometry.operand_bank_rows
        )
        # A policy that does not fit degrades to the end of the store rather than failing: the two
        # regions then overlap, which is legal exactly when every overwrite lands after the
        # overlapping block's last read -- and check_residency, not this function, decides that.
        if wanted + weight_rows <= geometry.operand_rows:
            base = wanted
        else:
            base = geometry.operand_rows - weight_rows
            notes.append(
                f"placement {knobs.placement!r} wanted row {wanted} but {wanted} + "
                f"{weight_rows} rows exceeds the {geometry.operand_rows}-row store; fell back "
                "to the store's end"
            )
    if base + weight_rows > geometry.operand_rows or base < 0:
        raise BlockScheduleError(
            f"the resident operand's region [{base}, {base + weight_rows}) leaves the {geometry.operand_rows}-row store"
        )
    if base < lhs_rows:
        notes.append(
            f"the two operand regions OVERLAP by {lhs_rows - base} rows; the schedule is only "
            "correct where every overwrite lands after the overlapping block's last read, "
            "which check_residency verifies"
        )
    return {LHS: (0, lhs_rows), WEIGHT: (base, weight_rows)}, notes


def with_column_bias(
    schedule: BlockSchedule, source_name: str, *, bits: int = 32, max_operations: int = 65536
) -> BlockSchedule:
    """Add an explicit modular column seed before each first reduction preload.

    The already checked zero-seed schedule retains all operand movement, source
    reduction order and stores. Actual effect checks prove seed ownership and
    lifetime after insertion. No hardware load/completion semantics are inferred.
    """
    check_residency(schedule)
    if schedule.column_bias is not None:
        raise BlockScheduleError("schedule already has an explicit accumulator seed")
    n = schedule.contraction.co if isinstance(schedule.contraction, ConvContraction) else schedule.contraction.n
    result = []
    pending = None
    weight = None
    for op in schedule.ops:
        if isinstance(op, Preload):
            if op.weight_row is not None:
                weight = op.weight_block
            pending = (len(result), op, weight)
            result.append(replace(op, accumulate=True))
        else:
            if isinstance(op, Compute) and op.input_block[1] == 0:
                index, preload, weight = pending
                output = (op.input_block[0], weight[1])
                result.insert(
                    index,
                    InitializeAccumulator(
                        output, output[1] * schedule.geometry.block, preload.rows, preload.cols, preload.accumulator_row
                    ),
                )
            result.append(op)
            if isinstance(op, Compute):
                pending = None
    if type(max_operations) is not int or max_operations <= 0 or len(result) > max_operations:
        raise BlockScheduleError(
            f"bias schedule exceeds explicit finite operation budget: {len(result)} > {max_operations}"
        )
    changed = replace(schedule, ops=tuple(result), column_bias=ColumnBias(source_name, n, bits))
    check_residency(changed)
    return changed


def schedule_contraction(contraction: Contraction, geometry: Geometry, knobs: Knobs | None = None) -> BlockSchedule:
    """Schedule ``contraction`` on ``geometry`` under ``knobs``, or refuse.

    The nest is ``knobs.loop_order`` -- reduction-major by default (step, then resident-operand block,
    then streamed-operand block), which is what keeps one weight block resident across the streamed
    blocks that read it. ``knobs`` decides the nest, WHERE each load is emitted relative to the computes,
    and where the resident region sits; every order computes the same products, and
    :func:`check_residency` refuses any that would read an overwritten block.
    """
    knobs = knobs or Knobs()
    knobs.validate()
    c = contraction
    if min(c.m, c.k, c.n) <= 0:
        raise BlockScheduleError(f"a {c.m}x{c.k}x{c.n} contraction is empty")
    d = geometry.block
    i_blocks, j_blocks, k_blocks = _ceil_div(c.m, d), _ceil_div(c.n, d), _ceil_div(c.k, d)

    def _lhs_load(mi: int, kk: int, base: int) -> Load:
        return Load(
            LHS,
            (mi, kk),
            mi * d,
            kk * d,
            min(d, c.m - mi * d),
            min(d, c.k - kk * d),
            base + (mi * k_blocks + kk) * d,
            kk,
        )

    def _weight_load(kk: int, nj: int, base: int) -> Load:
        return Load(
            WEIGHT,
            (kk, nj),
            kk * d,
            nj * d,
            min(d, c.k - kk * d),
            min(d, c.n - nj * d),
            base + (kk * j_blocks + nj) * d,
            kk,
        )

    return _schedule_nest(
        c,
        geometry,
        knobs,
        blocks=(k_blocks, j_blocks, i_blocks),
        rows_needed={LHS: i_blocks * k_blocks * d, WEIGHT: k_blocks * j_blocks * d},
        lhs_load=_lhs_load,
        weight_load=_weight_load,
        k_depth=lambda kk: min(d, c.k - kk * d),
        m_rows=lambda mi: min(d, c.m - mi * d),
        n_cols=lambda nj: min(d, c.n - nj * d),
    )


def schedule_convolution(conv: ConvContraction, geometry: Geometry, knobs: Knobs | None = None) -> BlockSchedule:
    """Schedule an NHWC convolution as a gathered contraction, or refuse.

    Reduction blocks are ``(tap row, tap col, channel slice)`` in kernel order, each ``min(block,
    ci - c0)`` channels deep; a streamed block is a block of output pixels whose rows are GATHERED, one
    per pixel, from the input position that tap reads -- or from the target's zero path where the tap
    falls in the padding. The gather depends on the tap and the pixel block only, never on the output
    channel block, which is why ``load_on_index_change`` can load it once per ``(tap, pixel block)``
    instead of once per output-channel block: the same bytes into the same rows, which
    :func:`check_residency` verifies rather than assumes.
    """
    knobs = knobs or Knobs()
    knobs.validate()
    cv = conv
    if min(cv.batch, cv.in_h, cv.in_w, cv.ci, cv.kh, cv.kw, cv.co) <= 0 or cv.out_h <= 0 or cv.out_w <= 0:
        raise BlockScheduleError(f"an empty convolution: {cv}")
    d = geometry.block
    taps = [(kr, kc, c0, min(d, cv.ci - c0)) for kr in range(cv.kh) for kc in range(cv.kw) for c0 in range(0, cv.ci, d)]
    m = cv.pixels
    i_blocks, j_blocks, k_blocks = _ceil_div(m, d), _ceil_div(cv.co, d), len(taps)
    sh, sw = cv.stride
    pt, pl, _, _ = cv.padding
    dh, dw = cv.dilation
    plane = cv.out_h * cv.out_w

    def _gather(mi: int, gi: int) -> tuple[tuple[int, int] | None, ...]:
        kr, kc, c0, _ = taps[gi]
        entries: list[tuple[int, int] | None] = []
        for pixel in range(mi * d, min(m, mi * d + d)):
            batch_index, spatial = divmod(pixel, plane)
            oh, ow = divmod(spatial, cv.out_w)
            ih, iw = oh * sh - pt + kr * dh, ow * sw - pl + kc * dw
            inside = 0 <= ih < cv.in_h and 0 <= iw < cv.in_w
            entries.append(((batch_index * cv.in_h + ih) * cv.in_w + iw, c0) if inside else None)
        return tuple(entries)

    def _lhs_load(mi: int, gi: int, base: int) -> Load:
        gather = _gather(mi, gi)
        return Load(LHS, (mi, gi), -1, -1, len(gather), taps[gi][3], base + (gi * i_blocks + mi) * d, gi, gather)

    def _weight_load(gi: int, nj: int, base: int) -> Load:
        kr, kc, c0, depth = taps[gi]
        return Load(
            WEIGHT,
            (gi, nj),
            (kr * cv.kw + kc) * cv.ci + c0,
            nj * d,
            depth,
            min(d, cv.co - nj * d),
            base + (gi * j_blocks + nj) * d,
            gi,
        )

    return _schedule_nest(
        cv,
        geometry,
        knobs,
        blocks=(k_blocks, j_blocks, i_blocks),
        rows_needed={LHS: k_blocks * i_blocks * d, WEIGHT: k_blocks * j_blocks * d},
        lhs_load=_lhs_load,
        weight_load=_weight_load,
        k_depth=lambda gi: taps[gi][3],
        m_rows=lambda mi: min(d, m - mi * d),
        n_cols=lambda nj: min(d, cv.co - nj * d),
    )


def _schedule_nest(
    contraction: Contraction | ConvContraction,
    geometry: Geometry,
    knobs: Knobs,
    *,
    blocks: tuple[int, int, int],
    rows_needed: dict[str, int],
    lhs_load: Any,
    weight_load: Any,
    k_depth: Any,
    m_rows: Any,
    n_cols: Any,
) -> BlockSchedule:
    """The one block nest both contractions share: regions, the load-due rule, the computes, the drains.

    ``lhs_load(mi, kk, base)`` / ``weight_load(kk, nj, base)`` build the block moves; the three extents
    are ``(K, N, M)`` block counts; ``k_depth`` / ``m_rows`` / ``n_cols`` give a block's edge along each
    axis, so partial edge blocks need no special case here.
    """
    d = geometry.block
    k_blocks, j_blocks, i_blocks = blocks
    regions, notes = _regions(rows_needed, geometry, knobs)
    lhs_base, weight_base = regions[LHS][0], regions[WEIGHT][0]
    ops = _schedule_ops(
        knobs,
        d,
        {K: range(k_blocks), N: range(j_blocks), M: range(i_blocks)},
        lambda mi, kk: lhs_load(mi, kk, lhs_base),
        lambda kk, nj: weight_load(kk, nj, weight_base),
        k_depth,
        m_rows,
        n_cols,
        0,
        0,
        j_blocks,
        k_blocks,
    )
    schedule = BlockSchedule(tuple(ops), contraction, geometry, knobs, regions, tuple(notes))
    check_residency(schedule)
    return schedule


def _schedule_ops(
    knobs, d, indices, lhs_load, weight_load, k_depth, m_rows, n_cols, m_start, n_start, n_span, k_blocks
):
    """Build one panel, continuing global K and storing only at the final reduction."""

    # One emission group per iteration of the two OUTER loops. Each compute carries the loads that
    # become due for it: a block is due the first time a compute needs it -- per group without
    # load_on_index_change, per nest with it. Nothing here names a particular loop as the one whose index
    # change triggers a load; under the default order this is exactly "a streamed block is loaded when its
    # (mi, kk) index first appears", i.e. at nj == 0.
    outer, middle, inner = knobs.loop_order
    groups: list[tuple[int, list[tuple[list[Load], list[Op]]]]] = []
    loaded_in_nest: set[tuple] = set()
    resident_weight: tuple[int, int] | None = None
    for a in indices[outer]:
        for b in indices[middle]:
            loaded_in_group: set[tuple] = set()
            computes: list[tuple[list[Load], list[Op]]] = []
            for c in indices[inner]:
                index = {outer: a, middle: b, inner: c}
                kk, nj, mi = index[K], index[N], index[M]
                weight, streamed = weight_load(kk, nj), lhs_load(mi, kk)
                due: list[Load] = []
                for load in (weight, streamed):
                    identity = (load.role, load.block)
                    seen = loaded_in_nest if knobs.load_on_index_change else loaded_in_group
                    if identity not in seen:
                        seen.add(identity)
                        due.append(load)
                rows, cols = m_rows(mi), n_cols(nj)
                acc_row = ((mi - m_start) * n_span + nj - n_start) * d
                fresh = resident_weight != (kk, nj)
                resident_weight = (kk, nj)
                ops_here: list[Op] = [
                    Preload(
                        weight.row if fresh else None,
                        (kk, nj) if fresh else None,
                        acc_row,
                        kk > 0,
                        rows,
                        cols,
                        k_depth(kk),
                    ),
                    Compute(streamed.row, (mi, kk), rows, k_depth(kk), fresh, streamed.gather),
                ]
                if kk == k_blocks - 1:
                    ops_here.append(Store(mi * d, nj * d, rows, cols, acc_row))
                computes.append((due, ops_here))
            groups.append((a, computes))

    return _emit(groups, knobs)


def _ordered(loads: list[Load], knobs: Knobs) -> list[Load]:
    """The loads of one emission group, ordered by ``load_grouping`` (stable within a role)."""
    if knobs.load_grouping == NEST:
        return list(loads)
    ordered: list[Load] = []
    for role in knobs.operand_order:
        ordered.extend(load for load in loads if load.role == role)
    return ordered


def _emit(groups: list[tuple[int, list[tuple[list[Load], list[Op]]]]], knobs: Knobs) -> list[Op]:
    """Place each group's loads relative to the computes, per ``lookahead_steps``.

    At depth 0 the emission group is one iteration of the two outer loops, and a load sits in its nest
    position -- immediately before the compute that first needs it -- or, under ``ROLE`` grouping, with
    the group's other loads ahead of its computes. At depth ``d`` the group is a whole outer step, issued
    ``d`` steps early. At ``None`` every load is hoisted ahead of every compute, and the whole prologue is
    one group.
    """
    if knobs.lookahead_steps == 0:
        ops: list[Op] = []
        for _, computes in groups:
            if knobs.load_grouping == NEST:
                for due, body in computes:
                    ops.extend(due)
                    ops.extend(body)
            else:
                ops.extend(_ordered([load for due, _ in computes for load in due], knobs))
                ops.extend(op for _, body in computes for op in body)
        return ops

    by_step: dict[int, list[Load]] = {}
    bodies: dict[int, list[Op]] = {}
    for step, computes in groups:
        by_step.setdefault(step, []).extend(load for due, _ in computes for load in due)
        bodies.setdefault(step, []).extend(op for _, body in computes for op in body)
    steps = sorted(bodies)
    if knobs.lookahead_steps is None:
        prologue: list[Load] = []
        for step in steps:
            prologue.extend(by_step.get(step, ()))
        ops = list(_ordered(prologue, knobs))
        for step in steps:
            ops.extend(bodies[step])
        return ops

    depth = knobs.lookahead_steps
    ops = []
    for step in steps[:depth]:
        ops.extend(_ordered(by_step.get(step, []), knobs))
    for index, step in enumerate(steps):
        ahead = index + depth
        if ahead < len(steps):
            ops.extend(_ordered(by_step.get(steps[ahead], []), knobs))
        ops.extend(bodies[step])
    return ops
