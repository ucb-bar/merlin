"""Order the block moves and computes of a contraction over a target's on-chip stores.

``capacity.py`` answers which tile FITS. This module answers the question after it: in what ORDER the
blocks of that tile move on chip, which of them stays resident, and where each operand's region sits --
the part a weight-stationary backend otherwise hand-writes, and the part that decides cycles.

Measured on one such backend (the capsule head-to-head experiment, whose ``STATUS.md`` carries the
Verilator and FPGA tables), four choices inside this ordering moved 1-24% of cycles on identical
instruction packing:

* loading an operand block only when its index changes, instead of once per inner step (-28% on a
  4x4-block matmul);
* issuing loads a BOUNDED number of reduction steps ahead: one step ahead beat both interleaved
  (-6%) and hoisting every load up front (which cost 7% on the same shape, because the loads crowd
  the queues ahead of the first compute);
* which operand's load is issued first inside a step;
* where the non-streamed operand's region starts: sharing a bank with the streamed operand cost
  3-15%, and the position INSIDE the other bank moved several cycles either way.

None of the four has a value that wins everywhere. The best setting depends on the shape AND on the
substrate -- one capsule reverses between a fast-memory simulator and the FPGA, where real DRAM
latency makes an earlier load pay. So they are KNOBS here, with documented defaults, never constants
compiled into a backend: a target picks values by measurement (keyed by kernel digest, the way
``perf/calibration`` keys a measured cycle count) and this pass produces the schedule for them.

Everything about the device is derived. Block edge, store rows, bank depth and the accumulator's size
come from :func:`merlin.targetgen.address_space.derive_address_space`, i.e. from that target's own RTL
facts; a quantity the facts cannot answer is a refusal (:class:`BlockScheduleError`), never a default.

This module imports nothing from merlin, so a generated backend that may not depend on merlin at run time
can VENDOR it byte for byte and run the pass itself (``tests/infra`` pins that). Deriving a
:class:`Geometry` from a target's facts is :func:`merlin.compile.scheduling.derive.geometry_from_address_space`.

Every schedule is checked before it is returned (:func:`check_residency`): a load may not overwrite a
block some later compute still expects to read. That check is what makes an aggressive knob value SAFE
to offer -- when two regions overlap, one order can be correct (the overwrite lands after the last
read) while another silently computes on the wrong bytes, and an in-order executor cannot see the
difference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Roles this pass schedules: the operand STREAMED through the array each compute, and the one made
#: RESIDENT in it. Named by role, never by a target's spelling of them.
LHS = "lhs"
WEIGHT = "weight"
ROLES = (LHS, WEIGHT)

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


Op = Load | Preload | Compute | Store


@dataclass(frozen=True)
class BlockSchedule:
    """An ordered block schedule plus what produced it."""

    ops: tuple[Op, ...]
    contraction: Contraction | ConvContraction
    geometry: Geometry
    knobs: Knobs
    regions: dict[str, tuple[int, int]]  # role -> (first row, row count)
    notes: tuple[str, ...] = ()

    def count(self, kind: type) -> int:
        return sum(1 for op in self.ops if isinstance(op, kind))


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


def check_residency(schedule: BlockSchedule) -> None:
    """Refuse a schedule whose loads overwrite a block a later compute still reads, or that addresses a
    row the target does not have.

    This is the class of fault an in-order executor cannot see: it recomputes the same arithmetic on
    whatever bytes are resident, so a schedule that overwrote a block too early produces numbers, not a
    crash. Every read here names the block identity it expects, so an overwrite is caught structurally.
    """
    geometry = schedule.geometry
    resident: dict[int, tuple] = {}

    def _expect(row: int, rows: int, identity: tuple, what: str) -> None:
        for offset in range(rows):
            have = resident.get(row + offset)
            if have != identity:
                if have is None:
                    holds = "nothing loaded"
                elif have[:2] != identity[:2]:
                    holds = str(have[:2])
                else:
                    holds = "that block with a DIFFERENT gather"
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
                resident[op.row + offset] = (op.role, op.block, op.gather)
        elif isinstance(op, Preload):
            if op.accumulator_row < 0 or op.accumulator_row + op.rows > geometry.accumulator_rows:
                raise BlockScheduleError(
                    f"a preload addresses accumulator rows [{op.accumulator_row}, "
                    f"{op.accumulator_row + op.rows}) of {geometry.accumulator_rows}"
                )
            if op.weight_row is not None:
                _expect(op.weight_row, op.weight_rows or op.rows, (WEIGHT, op.weight_block, None), "a preload")
        elif isinstance(op, Compute):
            _expect(op.input_row, op.rows, (LHS, op.input_block, op.input_gather), "a compute")
        elif isinstance(op, Store):
            if op.accumulator_row + op.rows > geometry.accumulator_rows:
                raise BlockScheduleError(
                    f"a store reads accumulator rows [{op.accumulator_row}, "
                    f"{op.accumulator_row + op.rows}) of {geometry.accumulator_rows}"
                )


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

    # One emission group per iteration of the two OUTER loops. Each compute carries the loads that
    # become due for it: a block is due the first time a compute needs it -- per group without
    # load_on_index_change, per nest with it. Nothing here names a particular loop as the one whose index
    # change triggers a load; under the default order this is exactly "a streamed block is loaded when its
    # (mi, kk) index first appears", i.e. at nj == 0.
    extent = {K: k_blocks, N: j_blocks, M: i_blocks}
    outer, middle, inner = knobs.loop_order
    groups: list[tuple[int, list[tuple[list[Load], list[Op]]]]] = []
    loaded_in_nest: set[tuple] = set()
    resident_weight: tuple[int, int] | None = None
    for a in range(extent[outer]):
        for b in range(extent[middle]):
            loaded_in_group: set[tuple] = set()
            computes: list[tuple[list[Load], list[Op]]] = []
            for c in range(extent[inner]):
                index = {outer: a, middle: b, inner: c}
                kk, nj, mi = index[K], index[N], index[M]
                weight, streamed = weight_load(kk, nj, weight_base), lhs_load(mi, kk, lhs_base)
                due: list[Load] = []
                for load in (weight, streamed):
                    identity = (load.role, load.block)
                    seen = loaded_in_nest if knobs.load_on_index_change else loaded_in_group
                    if identity not in seen:
                        seen.add(identity)
                        due.append(load)
                rows, cols = m_rows(mi), n_cols(nj)
                acc_row = (mi * j_blocks + nj) * d
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

    ops = _emit(groups, knobs)
    schedule = BlockSchedule(tuple(ops), contraction, geometry, knobs, regions, tuple(notes))
    check_residency(schedule)
    return schedule


def execute(schedule: BlockSchedule, lhs: Any, weight: Any) -> Any:
    """Run ``schedule`` on host arrays exactly as an in-order device would, and return the output.

    The schedule's own oracle: every row a load writes, every block a preload makes resident, every
    accumulate and every drain is simulated at the addresses the schedule names -- nothing is recomputed
    from the contraction. So a schedule that addresses the wrong rows, drains the wrong accumulator block
    or reuses a stale resident block produces a wrong ``out``, which a direct ``lhs @ weight`` exposes in
    milliseconds, before any backend or simulator is involved.
    """
    import numpy as np

    c = schedule.contraction
    lhs, weight = np.asarray(lhs), np.asarray(weight)
    if isinstance(c, ConvContraction):
        want_lhs, want_weight, out_shape = (
            (c.batch * c.in_h * c.in_w, c.ci),
            (c.kh * c.kw * c.ci, c.co),
            (c.pixels, c.co),
        )
    else:
        want_lhs, want_weight, out_shape = (c.m, c.k), (c.k, c.n), (c.m, c.n)
    if lhs.shape != want_lhs or weight.shape != want_weight:
        raise BlockScheduleError(
            f"operands {lhs.shape} x {weight.shape} are not the scheduled {want_lhs} x {want_weight}"
        )
    dram = {LHS: lhs, WEIGHT: weight}
    rows_on_chip: dict[int, Any] = {}
    accumulator: dict[int, Any] = {}
    out = np.zeros(out_shape, dtype=np.result_type(lhs, weight, np.int64))
    resident = None
    target_row, accumulate = None, False

    def _row(row: int, what: str) -> Any:
        if row not in rows_on_chip:
            raise BlockScheduleError(f"{what} reads on-chip row {row}, which no load ever wrote")
        return rows_on_chip[row]

    def _accumulated(row: int, what: str) -> Any:
        if row not in accumulator:
            raise BlockScheduleError(f"{what} reads accumulator block {row}, which no compute ever wrote")
        return accumulator[row]

    for op in schedule.ops:
        if isinstance(op, Load):
            source = dram[op.role]
            if op.gather is not None:
                for offset, entry in enumerate(op.gather):
                    rows_on_chip[op.row + offset] = (
                        np.zeros(op.cols, dtype=source.dtype)
                        if entry is None
                        else source[entry[0], entry[1] : entry[1] + op.cols].copy()
                    )
                continue
            block = source[op.dram_row : op.dram_row + op.rows, op.dram_col : op.dram_col + op.cols]
            for offset in range(op.rows):
                rows_on_chip[op.row + offset] = block[offset].copy()
        elif isinstance(op, Preload):
            if op.weight_row is not None:
                resident = np.stack(
                    [_row(op.weight_row + r, "a preload")[: op.cols] for r in range(op.weight_rows or op.rows)]
                )
            target_row, accumulate = op.accumulator_row, op.accumulate
        elif isinstance(op, Compute):
            if resident is None:
                raise BlockScheduleError("a compute ran with no resident block")
            streamed = np.stack([_row(op.input_row + r, "a compute")[: op.cols] for r in range(op.rows)])
            product = streamed.astype(out.dtype) @ resident.astype(out.dtype)
            if accumulate:
                accumulator[target_row] = _accumulated(target_row, "an accumulating compute") + product
            else:
                accumulator[target_row] = product
        elif isinstance(op, Store):
            out[op.dram_row : op.dram_row + op.rows, op.dram_col : op.dram_col + op.cols] = _accumulated(
                op.accumulator_row, "a store"
            )[: op.rows, : op.cols]
    return out


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


def schedule_interface_program(
    tensors: dict[str, Any], commands: list[dict[str, Any]], geometry: Geometry, knobs: Knobs | None = None
) -> list[BlockSchedule]:
    """Schedule every resident matmul an interface program commits, and every convolution, in program
    order.

    ``tensors`` maps a name to that tensor's spec (anything with ``shape``, or a mapping carrying
    ``"shape"``); ``commands`` is the program's command list. Any other command -- a convolution, a
    movement, an epilogue this pass does not model -- is a refusal, so a caller never receives a
    schedule that silently dropped work.
    """
    resident_of: dict[str, str] = {}
    pending: dict[str, tuple[str, str]] = {}
    schedules: list[BlockSchedule] = []
    for command in commands:
        opcode = command.get("opcode")
        operands = command.get("operands", {})
        if opcode == "RES_PACK":
            resident_of[operands["dst"]] = operands["src"]
        elif opcode == "MATMUL_RESIDENT":
            rhs = operands["rhs"]
            if rhs not in resident_of:
                raise BlockScheduleError(f"{rhs!r} is used as a resident operand before it is packed")
            pending[operands["dst"]] = (operands["lhs"], resident_of[rhs])
        elif opcode == "COMMIT":
            src = operands["src"]
            if src not in pending:
                raise BlockScheduleError(f"commit of {src!r}, which no matmul produced")
            lhs_name, weight_name = pending.pop(src)
            attributes = command.get("attributes", {}) or {}
            epilogue = tuple(attributes.get("epilogue", ()) or ())
            unsupported = [e for e in epilogue if e not in ("relu", "acc_scale")]
            if unsupported:
                raise BlockScheduleError(
                    f"epilogue {unsupported} changes which accumulator block a store drains; this pass "
                    "schedules block moves only and will not guess that mapping"
                )
            m, k = _shape(tensors, lhs_name)
            wk, n = _shape(tensors, weight_name)
            if wk != k:
                raise BlockScheduleError(f"contraction mismatch: {lhs_name} is {m}x{k}, {weight_name} is {wk}x{n}")
            schedules.append(
                schedule_contraction(Contraction(m, k, n, lhs_name, weight_name, operands["dst"]), geometry, knobs)
            )
        elif opcode == "CONV2D":
            schedules.append(_conv_command(tensors, command, resident_of, geometry, knobs))
        elif opcode != "EVICT":
            raise BlockScheduleError(
                f"{opcode!r} is neither a resident matmul nor a convolution; this pass "
                "schedules those only, and refuses rather than dropping the command"
            )
    return schedules


def _conv_command(
    tensors: dict[str, Any],
    command: dict[str, Any],
    resident_of: dict[str, str],
    geometry: Geometry,
    knobs: Knobs | None,
) -> BlockSchedule:
    """One ``CONV2D`` command as a :class:`ConvContraction`, refusing what the nest cannot express."""
    operands, attributes = command.get("operands", {}) or {}, command.get("attributes", {}) or {}
    missing = [key for key in ("ifm", "weight", "dst") if key not in operands]
    missing += [key for key in ("kernel",) if key not in attributes]
    if missing:
        raise BlockScheduleError(f"a CONV2D command missing {missing}; refusing rather than guessing them")
    epilogue = tuple(attributes.get("epilogue", ()) or ())
    unsupported = [e for e in epilogue if e not in ("relu", "acc_scale")]
    if unsupported:
        # Pooling drains the accumulator N-major and defers every drain past the nest: a different
        # accumulator LAYOUT, not an ordering choice, so it is refused rather than approximated.
        raise BlockScheduleError(
            f"epilogue {unsupported} changes which accumulator block a store drains; "
            "this pass schedules block moves only and will not guess that mapping"
        )
    if attributes.get("layout", "nhwc") != "nhwc":
        raise BlockScheduleError(f"conv layout {attributes.get('layout')!r}; this pass schedules NHWC")
    weight = operands["weight"]
    if weight not in resident_of:
        raise BlockScheduleError(f"{weight!r} is used as a resident operand before it is packed")
    ifm = _shape_nd(tensors, operands["ifm"], 4)
    kh, kw, kci, co = (int(v) for v in attributes["kernel"])
    batch, in_h, in_w, ci = ifm
    if kci != ci:
        raise BlockScheduleError(f"kernel expects {kci} input channels, the input has {ci}")
    packed = _shape_nd(tensors, resident_of[weight], 2)
    if packed != (kh * kw * ci, co):
        raise BlockScheduleError(
            f"conv weight {resident_of[weight]!r} is {packed}, not the pre-im2col {(kh * kw * ci, co)}"
        )
    conv = ConvContraction(
        batch,
        in_h,
        in_w,
        ci,
        kh,
        kw,
        co,
        tuple(int(v) for v in attributes.get("stride", (1, 1))),
        tuple(int(v) for v in attributes.get("padding", (0, 0, 0, 0))),
        tuple(int(v) for v in attributes.get("dilation", (1, 1))),
        operands["ifm"],
        resident_of[weight],
        operands["dst"],
    )
    return schedule_convolution(conv, geometry, knobs)


def _shape_nd(tensors: dict[str, Any], name: str, rank: int) -> tuple[int, ...]:
    spec = tensors.get(name)
    if spec is None:
        raise BlockScheduleError(f"the program names no tensor {name!r}")
    shape = spec["shape"] if isinstance(spec, dict) else getattr(spec, "shape", None)
    if shape is None or len(shape) != rank:
        raise BlockScheduleError(f"{name!r} has shape {shape!r}; expected rank {rank}")
    return tuple(int(v) for v in shape)


def _shape(tensors: dict[str, Any], name: str) -> tuple[int, int]:
    spec = tensors.get(name)
    if spec is None:
        raise BlockScheduleError(f"the program names no tensor {name!r}")
    shape = spec["shape"] if isinstance(spec, dict) else getattr(spec, "shape", None)
    if shape is None or len(shape) != 2:
        raise BlockScheduleError(f"{name!r} has shape {shape!r}; this pass schedules 2-D operands")
    return int(shape[0]), int(shape[1])
