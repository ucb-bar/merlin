"""Check synchronous operand residency and complete accumulator reduction lifetimes."""

from __future__ import annotations

from .model import (
    LHS,
    ROLES,
    WEIGHT,
    BlockSchedule,
    BlockScheduleError,
    ColumnBias,
    Compute,
    Contraction,
    ConvContraction,
    InitializeAccumulator,
    Load,
    Preload,
    Store,
    _ceil_div,
)


def check_residency(schedule: BlockSchedule) -> None:
    """Check synchronous operand residency, accumulator ownership and output coverage.

    This is the class of fault an in-order executor cannot see: it recomputes the same arithmetic on
    whatever bytes are resident, so a schedule that overwrote a block too early produces numbers, not a
    crash. Every read here names the block identity it expects, so an overwrite is caught structurally.
    Reduction prefixes and physical accumulator rows must agree before a store.
    Unknown effects or shared backings without an alias model are refused.
    """
    geometry = schedule.geometry
    for value in (geometry.block, geometry.operand_rows, geometry.accumulator_rows):
        if type(value) is not int or value <= 0:
            raise BlockScheduleError("schedule geometry requires positive integer extents")
    if geometry.separate_accumulator_space is not True:
        raise BlockScheduleError("shared operand/accumulator backing needs an explicit alias model")
    if not isinstance(schedule.contraction, (Contraction, ConvContraction)):
        raise BlockScheduleError("unsupported contraction has no schedule effect contract")
    if schedule.contraction.out in (schedule.contraction.lhs, schedule.contraction.weight):
        raise BlockScheduleError("output alias with an operand needs an explicit lifetime model")
    bias = schedule.column_bias
    n = schedule.contraction.co if isinstance(schedule.contraction, ConvContraction) else schedule.contraction.n
    if bias is not None and (
        type(bias) is not ColumnBias
        or type(bias.name) is not str
        or not bias.name
        or bias.name in (schedule.contraction.lhs, schedule.contraction.weight, schedule.contraction.out)
        or type(bias.columns) is not int
        or bias.columns != n
        or type(bias.bits) is not int
        or bias.bits not in (8, 16, 32, 64)
    ):
        raise BlockScheduleError("column bias requires a distinct complete signed accumulator-width source")
    for op in schedule.ops:
        if not isinstance(op, (Load, Preload, Compute, Store, InitializeAccumulator)):
            raise BlockScheduleError("unknown operation has no schedule effect contract")
        if any(type(value) is not int or not 0 < value <= geometry.block for value in (op.rows, op.cols)):
            raise BlockScheduleError("schedule operation requires positive primitive-bounded extents")
        if isinstance(op, Load):
            addresses = (op.row,)
            if op.role not in ROLES:
                raise BlockScheduleError("unknown load role has no schedule effect contract")
            if op.gather is not None and (op.role != LHS or len(op.gather) != op.rows):
                raise BlockScheduleError("gather must describe exactly the declared input rows")
        elif isinstance(op, InitializeAccumulator):
            if bias is None:
                raise BlockScheduleError("accumulator initialization has no declared source bias")
            addresses = (op.row, op.dram_col)
            if op.row + op.rows > geometry.accumulator_rows:
                raise BlockScheduleError("bias initialization exceeds accumulator rows")
        elif isinstance(op, Preload):
            addresses = (op.accumulator_row,) + (() if op.weight_row is None else (op.weight_row,))
            if op.weight_rows is not None and (
                type(op.weight_rows) is not int or not 0 < op.weight_rows <= geometry.block
            ):
                raise BlockScheduleError("resident weight depth must be a positive primitive-bounded extent")
        elif isinstance(op, Compute):
            addresses = (op.input_row,)
        else:
            addresses = (op.accumulator_row, op.dram_row, op.dram_col)
        if any(type(value) is not int or value < 0 for value in addresses):
            raise BlockScheduleError("schedule addresses require nonnegative integer rows")
    resident: dict[int, tuple] = {}

    def _expect(row: int, rows: int, cols: int, identity: tuple, what: str) -> None:
        for offset in range(rows):
            have = resident.get(row + offset)
            if have is None or have[:3] != identity or have[3] != offset or have[4] < cols:
                if have is None:
                    holds = "nothing loaded"
                elif have[:2] != identity[:2]:
                    holds = str(have[:2])
                else:
                    holds = "that block with a DIFFERENT gather, row offset or width"
                gathered = " with its gather" if identity[2] is not None else ""
                raise BlockScheduleError(
                    f"{what} reads on-chip row {row + offset} expecting block {identity[:2]}{gathered}, "
                    f"but it holds {holds}: the schedule overwrites a block "
                    "that is still live (or never loads it)"
                )

    for op in schedule.ops:
        if isinstance(op, Load):
            if op.row < 0 or op.row + op.rows > geometry.operand_rows:
                raise BlockScheduleError(
                    f"a {op.role} load addresses rows [{op.row}, {op.row + op.rows}) of a "
                    f"{geometry.operand_rows}-row store"
                )
            for offset in range(op.rows):
                resident[op.row + offset] = (op.role, op.block, op.gather, offset, op.cols)
        elif isinstance(op, Preload):
            if op.accumulator_row < 0 or op.accumulator_row + op.rows > geometry.accumulator_rows:
                raise BlockScheduleError(
                    f"a preload addresses accumulator rows [{op.accumulator_row}, "
                    f"{op.accumulator_row + op.rows}) of {geometry.accumulator_rows}"
                )
            if op.weight_row is not None:
                _expect(op.weight_row, op.weight_rows or op.rows, op.cols, (WEIGHT, op.weight_block, None), "a preload")
        elif isinstance(op, Compute):
            _expect(op.input_row, op.rows, op.cols, (LHS, op.input_block, op.input_gather), "a compute")
        elif isinstance(op, Store):
            if op.accumulator_row < 0 or op.accumulator_row + op.rows > geometry.accumulator_rows:
                raise BlockScheduleError(
                    f"a store reads accumulator rows [{op.accumulator_row}, "
                    f"{op.accumulator_row + op.rows}) of {geometry.accumulator_rows}"
                )
    _check_reduction_lifetimes(schedule)


def _check_reduction_lifetimes(schedule: BlockSchedule) -> None:
    """Verify ordered, synchronous reductions and physical accumulator ownership.

    This is the execution model of BlockSchedule, not a hardware completion
    model. An asynchronous adapter must establish its dependencies separately.
    State is proportional to represented rows and output tiles, not elements.
    """
    c, d = schedule.contraction, schedule.geometry.block
    if isinstance(c, ConvContraction):
        m, n = c.pixels, c.co
        channel_blocks = _ceil_div(c.ci, d)
        reduction_blocks = c.kh * c.kw * channel_blocks

        def depth(k):
            return min(d, c.ci - (k % channel_blocks) * d)
    else:
        m, n = c.m, c.n
        reduction_blocks = _ceil_div(c.k, d)

        def depth(k):
            return min(d, c.k - k * d)

    mt, nt = _ceil_div(m, d), _ceil_div(n, d)
    if min(m, n, reduction_blocks) <= 0:
        raise BlockScheduleError("empty contraction has no complete reduction schedule")
    progress: dict[tuple[int, int], int] = {}
    accumulator: dict[int, tuple] = {}
    stored: set[tuple[int, int]] = set()
    initialized: set[tuple[int, int]] = set()
    pending = None
    weight = None
    for op in schedule.ops:
        if isinstance(op, InitializeAccumulator):
            if pending is not None or schedule.column_bias is None:
                raise BlockScheduleError("bias initialization interrupts a preload or lacks source semantics")
            if type(op.block) is not tuple or len(op.block) != 2 or any(type(x) is not int for x in op.block):
                raise BlockScheduleError("bias initialization requires exact output tile identity")
            mi, nj = output = op.block
            if (
                not (0 <= mi < mt and 0 <= nj < nt)
                or output in initialized
                or output in progress
                or output in stored
                or op.dram_col != nj * d
                or type(op.source_row_stride) is not int
                or type(op.source_col_stride) is not int
                or op.source_row_stride != 0
                or op.source_col_stride != 1
                or (op.rows, op.cols) != (min(d, m - mi * d), min(d, n - nj * d))
            ):
                raise BlockScheduleError("bias initialization differs from source columns/extent/order")
            for offset in range(op.rows):
                previous = accumulator.get(op.row + offset)
                if previous is not None and previous[0] not in stored:
                    raise BlockScheduleError("bias initialization overwrites a live accumulator")
                accumulator[op.row + offset] = (output, offset, 0, op.cols)
            initialized.add(output)
        elif isinstance(op, Preload):
            if pending is not None:
                raise BlockScheduleError("a preload replaces an unconsumed preload")
            if type(op.accumulate) is not bool:
                raise BlockScheduleError("accumulation policy must be explicit Boolean")
            if op.weight_row is not None:
                weight = (op.weight_block, op.weight_rows if op.weight_rows is not None else op.rows, op.cols)
            elif op.weight_block is not None:
                raise BlockScheduleError("retained weights cannot declare a different weight block")
            pending = op
        elif isinstance(op, Compute):
            if pending is None or weight is None:
                raise BlockScheduleError("a compute needs a preload and resident weight block")
            mi, k = op.input_block
            block, weight_rows, cols = weight
            if block is None or len(block) != 2:
                raise BlockScheduleError("resident weight identity is missing")
            wk, nj = block
            if not (0 <= mi < mt and 0 <= nj < nt and 0 <= k < reduction_blocks) or wk != k:
                raise BlockScheduleError("compute operand identities disagree on the reduction")
            rows = min(d, m - mi * d)
            if (op.rows, op.cols, pending.rows, pending.cols, weight_rows, cols) != (
                rows,
                depth(k),
                rows,
                min(d, n - nj * d),
                depth(k),
                min(d, n - nj * d),
            ):
                raise BlockScheduleError("compute/preload extents differ from the contraction edges")
            if type(op.fresh_weights) is not bool or op.fresh_weights != (pending.weight_row is not None):
                raise BlockScheduleError("compute fresh-weight policy differs from its preload")
            output = (mi, nj)
            if output in stored or progress.get(output, 0) != k:
                raise BlockScheduleError("output reduction is repeated or not in source order")
            if pending.accumulate != (k != 0 or schedule.column_bias is not None):
                raise BlockScheduleError("accumulator initialization/accumulation boundary differs")
            for offset in range(rows):
                row = pending.accumulator_row + offset
                previous = accumulator.get(row)
                if k or schedule.column_bias is not None:
                    if previous != (output, offset, k, cols):
                        raise BlockScheduleError("accumulating compute reads a stale or uninitialized accumulator")
                elif previous is not None and previous[0] not in stored:
                    raise BlockScheduleError("compute overwrites an accumulator whose output is still live")
                accumulator[row] = (output, offset, k + 1, cols)
            progress[output] = k + 1
            pending = None
        elif isinstance(op, Store):
            if pending is not None:
                raise BlockScheduleError("store interrupts an unconsumed preload")
            if op.dram_row < 0 or op.dram_col < 0 or op.dram_row % d or op.dram_col % d:
                raise BlockScheduleError("store output coordinates must identify an aligned output tile")
            output = (op.dram_row // d, op.dram_col // d)
            mi, nj = output
            if output in stored:
                raise BlockScheduleError("output store is duplicated")
            if not (0 <= mi < mt and 0 <= nj < nt) or (op.rows, op.cols) != (min(d, m - mi * d), min(d, n - nj * d)):
                raise BlockScheduleError("store output extent differs from the contraction edges")
            if progress.get(output) != reduction_blocks:
                raise BlockScheduleError("store reads an output before its reduction is complete")
            for offset in range(op.rows):
                if accumulator.get(op.accumulator_row + offset) != (output, offset, reduction_blocks, op.cols):
                    raise BlockScheduleError("store reads the wrong or incomplete accumulator")
            stored.add(output)
    if pending is not None:
        raise BlockScheduleError("schedule leaves an unconsumed preload")
    if len(stored) != mt * nt:
        raise BlockScheduleError("schedule does not store every output tile exactly once")
    if schedule.column_bias is not None and initialized != stored:
        raise BlockScheduleError("bias initialization must cover every output exactly once")
