"""Typed contraction, geometry, command and knob declarations for shared block scheduling."""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
)

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
