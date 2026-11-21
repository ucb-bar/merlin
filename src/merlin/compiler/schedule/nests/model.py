"""Typed contraction, geometry, command and knob declarations for shared block scheduling."""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
)

# ---- model -------------------------------------------------------------------------------------

#: Roles this pass schedules: the operand STREAMED through the array each compute, and the one made
#: RESIDENT in it. Named by role, never by a target's spelling of them.
LHS = "lhs"
WEIGHT = "weight"
ROLES = (LHS, WEIGHT)

# Conservative compiler verification policy, independent of hardware capacity.
# Adapters preflight dynamic operation counts before materializing a projection.
DEFAULT_PROJECTION_OPERATION_LIMIT = 4096

#: ``load_grouping`` values -- how the loads inside one emission group are ordered.
#: ``NEST``: exactly as the loop nest produces them (a weight load at its own position, each streamed
#: block immediately before the compute that reads it). ``ROLE``: all of one role's loads, then the
#: other's, in ``operand_order``, each role keeping its nest order.
NEST = "nest"
ROLE = "role"
GROUPINGS = (NEST, ROLE)

#: Loop axes, named by the ``Contraction`` dimension each iterates in blocks: ``K`` the reduction axis
#: (the width of the streamed operand, the depth of the resident one), ``N`` the resident operand's
#: output axis, ``M`` the streamed operand's rows. ``loop_order`` is a permutation of the three,
#: outermost first.
K = "k"
N = "n"
M = "m"
AXES = (K, N, M)

#: ``placement`` values for the resident operand's region. The streamed operand always starts at row 0.
#: ``OPPOSITE_END``: flush with the end of the store, growing back toward the streamed operand.
#: ``BANK_ALIGNED``: the first bank boundary at or after the streamed region (separating the two
#: operands into different banks), falling back to ``OPPOSITE_END`` when that does not fit.
#: ``CONTIGUOUS``: directly after the streamed region, sharing its bank.
OPPOSITE_END = "opposite_end"
BANK_ALIGNED = "bank_aligned"
CONTIGUOUS = "contiguous"
PLACEMENTS = (OPPOSITE_END, BANK_ALIGNED, CONTIGUOUS)


class BlockScheduleError(ValueError):
    """A schedule the target's derived facts, the program, or the knobs do not admit."""


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class Geometry:
    """The on-chip geometry a block schedule addresses, derived from one target's facts.

    ``block`` is the square block edge (the array edge a store row spans). ``operand_bank_rows`` is the
    PER-BANK row count -- the granularity a placement policy can separate operands at -- and is ``None``
    when the facts could not resolve it, which makes :data:`BANK_ALIGNED` a refusal rather than a guess.
    """

    block: int
    operand_rows: int
    operand_bank_rows: int | None
    accumulator_rows: int
    separate_accumulator_space: bool
    sources: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Knobs:
    """The scheduling choices, with their defaults.

    Defaults are the values that measured best over the corpus this pass was lifted from, stated so a
    reader knows what they are getting: load on index change, one reduction step of lookahead, the
    streamed operand's load first inside a step, and the resident operand at the far end of the store.
    They are a starting point for a target's own measurement, NOT a claim that they are optimal
    anywhere else -- on that corpus the best value differed per shape and per substrate.
    """

    #: Load an operand block only when a compute first needs it -- a block, once loaded into its own rows,
    #: stays resident -- instead of re-loading it in every emission group that reads it.
    load_on_index_change: bool = True
    #: Reduction steps of lookahead. ``0`` issues a load in its nest position; ``d`` issues step
    #: ``i + d``'s loads before step ``i``'s computes; ``None`` hoists every load ahead of every
    #: compute (measured as the worst of the three on a deep shape, and rejected outright by
    #: :func:`check_residency` whenever regions overlap).
    lookahead_steps: int | None = 1
    #: Order of the roles inside an emission group; only read when ``load_grouping`` is :data:`ROLE`.
    operand_order: tuple[str, ...] = (LHS, WEIGHT)
    #: How loads are ordered inside an emission group: :data:`NEST` or :data:`ROLE`.
    load_grouping: str = ROLE
    #: Where the resident operand's region starts; one of :data:`PLACEMENTS`.
    placement: str = OPPOSITE_END
    #: The block nest, outermost first -- a permutation of :data:`AXES`. The default is
    #: reduction-major, which keeps one resident block live across every streamed block that reads it.
    #: An emission group is one iteration of the two OUTER loops, and a lookahead "step" is one iteration
    #: of the outermost; so under the default a step is a reduction step, as the knobs above describe.
    loop_order: tuple[str, ...] = (K, N, M)

    def validate(self) -> None:
        if tuple(self.operand_order) not in ((LHS, WEIGHT), (WEIGHT, LHS)):
            raise BlockScheduleError(f"operand_order must order exactly {ROLES}, got {tuple(self.operand_order)!r}")
        if self.load_grouping not in GROUPINGS:
            raise BlockScheduleError(f"load_grouping must be one of {GROUPINGS}, got {self.load_grouping!r}")
        if self.placement not in PLACEMENTS:
            raise BlockScheduleError(f"placement must be one of {PLACEMENTS}, got {self.placement!r}")
        if sorted(self.loop_order) != sorted(AXES):
            raise BlockScheduleError(f"loop_order must be a permutation of {AXES}, got {tuple(self.loop_order)!r}")
        if self.lookahead_steps is not None and self.lookahead_steps < 0:
            raise BlockScheduleError(f"lookahead_steps must be None or >= 0, got {self.lookahead_steps!r}")


@dataclass(frozen=True)
class Contraction:
    """One contraction to schedule: ``[m, k] x [k, n] -> [m, n]``, with the operand names a backend
    resolves addresses against."""

    m: int
    k: int
    n: int
    lhs: str = LHS
    weight: str = WEIGHT
    out: str = "out"


@dataclass(frozen=True)
class ConvContraction:
    """One NHWC convolution scheduled as a contraction: each kernel tap x input-channel slice is a
    reduction block, each output-channel block a resident block, each block of output pixels a streamed
    block gathered tap by tap from the input feature map.

    Operands, in the element coordinates :func:`execute` and a renderer address: the input as a
    ``[batch*in_h*in_w, ci]`` matrix, the weight as ``[kh*kw*ci, co]`` (pre-im2col layout), the output as
    ``[batch*out_h*out_w, co]``.
    """

    batch: int
    in_h: int
    in_w: int
    ci: int
    kh: int
    kw: int
    co: int
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int, int, int] = (0, 0, 0, 0)  # top, left, bottom, right
    dilation: tuple[int, int] = (1, 1)
    lhs: str = "ifm"
    weight: str = WEIGHT
    out: str = "out"

    @property
    def out_h(self) -> int:
        pt, _, pb, _ = self.padding
        return (self.in_h + pt + pb - (self.dilation[0] * (self.kh - 1) + 1)) // self.stride[0] + 1

    @property
    def out_w(self) -> int:
        _, pl, _, pr = self.padding
        return (self.in_w + pl + pr - (self.dilation[1] * (self.kw - 1) + 1)) // self.stride[1] + 1

    @property
    def pixels(self) -> int:
        return self.batch * self.out_h * self.out_w


@dataclass(frozen=True)
class InitializeAccumulator:
    """Synchronous overwrite from a contiguous signed accumulator-width bias.

    The single source row is broadcast over destination rows. This is an
    explicit DRAM read plus accumulator write, not an implicit unary readout.
    Target encoders must preserve the source words without scaling/narrowing.
    """

    block: tuple[Any, Any]
    dram_col: Any
    rows: Any
    cols: Any
    row: Any
    source_row_stride: Any = 0
    source_col_stride: Any = 1


@dataclass(frozen=True)
class Load:
    """Move one block of ``role`` from its DRAM position (in ELEMENTS) to on-chip row ``row``.

    A GATHERED block (``gather`` set) is not a contiguous DRAM rectangle: on-chip row ``row + i`` is filled
    from ``gather[i]``, a ``(row, col)`` element position in the operand's DRAM matrix, or from the
    target's zero path when the entry is ``None`` (a convolution tap that falls outside the image). Then
    ``dram_row``/``dram_col`` are ``-1``, and the gather is part of the block's identity: two loads of one
    block position with different gathers are different bytes.
    """

    role: str
    block: tuple[int, int]
    dram_row: int
    dram_col: int
    rows: int
    cols: int
    row: int
    step: int
    gather: tuple[tuple[int, int] | None, ...] | None = None


@dataclass(frozen=True)
class GatherRun:
    """A maximal stretch of a gather that one multi-row load can move.

    ``offset`` is the first gathered row, ``rows`` how many. ``source`` is the first DRAM ``(row, col)``,
    and the run reads rows ``source[0] + i * row_step`` for ``i < rows`` at that column -- or ``source`` is
    ``None`` and every row comes from the target's zero path (``row_step`` 0). A stride-``s`` convolution
    gathers every ``s``-th pixel of an image row, so its runs have ``row_step == s``.
    """

    offset: int
    rows: int
    source: tuple[int, int] | None
    row_step: int


def gather_runs(gather: tuple[tuple[int, int] | None, ...]) -> tuple[GatherRun, ...]:
    """Split a gather into the fewest runs of rows that are equally spaced in DRAM (or all zero).

    A run's spacing is fixed by its first two entries and must be positive; a single entry is a run of
    one. Joining the runs back, row by row, reproduces the gather exactly -- which is what lets a target
    that can load several DRAM rows at a configured stride move a gather in a handful of loads instead of
    one per row, without changing a single byte that reaches the chip.
    """
    runs: list[GatherRun] = []
    index = 0
    while index < len(gather):
        first = gather[index]
        length, step = 1, 0
        if first is None:
            while index + length < len(gather) and gather[index + length] is None:
                length += 1
        elif index + 1 < len(gather) and gather[index + 1] is not None:
            nxt = gather[index + 1]
            if nxt[1] == first[1] and nxt[0] > first[0]:
                step = nxt[0] - first[0]
                while (
                    index + length < len(gather)
                    and gather[index + length] is not None
                    and gather[index + length][1] == first[1]
                    and gather[index + length][0] == first[0] + length * step
                ):
                    length += 1
        runs.append(GatherRun(index, length, first, step if first is not None and length > 1 else 0))
        index += length
    return tuple(runs)


@dataclass(frozen=True)
class Preload:
    """Name the accumulator block the next :class:`Compute` writes, and (when ``weight_row`` is not
    ``None``) make that weight block resident first; ``None`` keeps the resident block."""

    weight_row: int | None
    weight_block: tuple[int, int] | None
    accumulator_row: int
    accumulate: bool
    rows: int
    cols: int
    #: Rows the weight block spans (the reduction depth of this step) -- set whether or not this preload
    #: makes a block resident, because a target's preload word carries the depth either way. The
    #: residency check reads it only when ``weight_row`` is set; ``rows`` is the streamed row count the
    #: accumulator block takes.
    weight_rows: int | None = None


@dataclass(frozen=True)
class Compute:
    """Stream ``rows`` rows from ``input_row`` through the resident weight block."""

    input_row: int
    input_block: tuple[int, int]
    rows: int
    cols: int
    fresh_weights: bool
    #: The gather the streamed block must have been loaded with (``None`` for a contiguous block).
    input_gather: tuple[tuple[int, int] | None, ...] | None = None


@dataclass(frozen=True)
class Store:
    """Drain one accumulator block to its DRAM position (in ELEMENTS)."""

    dram_row: int
    dram_col: int
    rows: int
    cols: int
    accumulator_row: int
    # An independently qualified output transformation, distinct from the
    # accumulator's element format. None preserves the original full store.
    readout: str | None = None


@dataclass(frozen=True)
class ColumnBias:
    """Explicit signed accumulator-width modular seed, broadcast across rows."""

    name: str
    columns: int
    bits: int = 32


Op = Load | Preload | Compute | Store | InitializeAccumulator


@dataclass(frozen=True)
class BlockSchedule:
    """An ordered block schedule plus what produced it."""

    ops: tuple[Op, ...]
    contraction: Contraction | ConvContraction
    geometry: Geometry
    knobs: Knobs
    regions: dict[str, tuple[int, int]]  # role -> (first row, row count)
    notes: tuple[str, ...] = ()
    column_bias: ColumnBias | None = None

    def count(self, kind: type) -> int:
        return sum(1 for op in self.ops if isinstance(op, kind))


# ---- residency ---------------------------------------------------------------------------------
# Check synchronous operand residency and complete accumulator reduction lifetimes.


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
