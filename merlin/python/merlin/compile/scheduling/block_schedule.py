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

    @classmethod
    def from_address_space(cls, space: Any) -> "Geometry":
        """Derive from :class:`~merlin.targetgen.address_space.AddressSpace`, or refuse.

        Refuses rather than defaults on: no array geometry, a non-square array (this weight-stationary
        model has one square block edge, and which edge a rectangular array's block spans is a choice
        no fact here makes), a missing operand or accumulator store, and a store whose row count the
        facts could not derive. Stores are resolved to their roles by ROW WIDTH
        (:func:`~merlin.targetgen.address_space.operand_store`,
        :func:`~merlin.targetgen.address_space.accumulator_store`), never by the name an extractor
        happened to give them, and a refusal quotes the resolver's reason.
        """
        if space.array_rows is None or space.array_cols is None:
            raise BlockScheduleError(
                f"{space.target!r}: no array geometry in its facts, so there is no block edge to "
                f"schedule in (unknowns: {list(space.unknown_quantities())})")
        if space.array_rows != space.array_cols:
            raise BlockScheduleError(
                f"{space.target!r}: a {space.array_rows}x{space.array_cols} array is not square; this "
                "pass schedules one square block edge and will not choose an edge for you")
        from merlin.targetgen.address_space import accumulator_store, operand_store
        resolved = {"operand": operand_store(space), "accumulator": accumulator_store(space)}
        for role, resolution in resolved.items():
            if resolution.store is None:
                raise BlockScheduleError(
                    f"{space.target!r}: no {role} store to schedule into: {resolution.reason}")
            if resolution.store.total_rows is None:
                raise BlockScheduleError(
                    f"{space.target!r}: the row count of its {role} store "
                    f"{resolution.store.name!r} is UNKNOWN "
                    f"({[u.reason for u in space.unknowns if u.store == resolution.store.name]})")
        operand, accumulator = resolved["operand"].store, resolved["accumulator"].store
        if operand.row_elems is not None and operand.row_elems != space.array_rows:
            raise BlockScheduleError(
                f"{space.target!r}: its operand row spans {operand.row_elems} elements but its array "
                f"edge is {space.array_rows}; a block cannot be one row wide and another edge tall")
        return cls(block=space.array_rows,
                   operand_rows=operand.total_rows,
                   operand_bank_rows=operand.depth,
                   accumulator_rows=accumulator.total_rows,
                   separate_accumulator_space=bool(space.separate_accumulator_space),
                   sources={"facts": space.sources.get("facts", "derive_address_space"),
                            "block": f"arrays[{space.array_name!r}] edge",
                            "operand_rows": (f"{operand.name}.total_rows (operand store, "
                                             f"{resolved['operand'].basis})"),
                            "operand_bank_rows": f"{operand.name}.depth",
                            "accumulator_rows": (f"{accumulator.name}.total_rows (accumulator store, "
                                                 f"{resolved['accumulator'].basis})")})


@dataclass(frozen=True)
class Knobs:
    """The scheduling choices, with their defaults.

    Defaults are the values that measured best over the corpus this pass was lifted from, stated so a
    reader knows what they are getting: load on index change, one reduction step of lookahead, the
    streamed operand's load first inside a step, and the resident operand at the far end of the store.
    They are a starting point for a target's own measurement, NOT a claim that they are optimal
    anywhere else -- on that corpus the best value differed per shape and per substrate.
    """

    #: Load an operand block only when its block index changes, instead of once per inner iteration.
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

    def validate(self) -> None:
        if tuple(self.operand_order) not in ((LHS, WEIGHT), (WEIGHT, LHS)):
            raise BlockScheduleError(f"operand_order must order exactly {ROLES}, got "
                                     f"{tuple(self.operand_order)!r}")
        if self.load_grouping not in GROUPINGS:
            raise BlockScheduleError(f"load_grouping must be one of {GROUPINGS}, got "
                                     f"{self.load_grouping!r}")
        if self.placement not in PLACEMENTS:
            raise BlockScheduleError(f"placement must be one of {PLACEMENTS}, got {self.placement!r}")
        if self.lookahead_steps is not None and self.lookahead_steps < 0:
            raise BlockScheduleError(f"lookahead_steps must be None or >= 0, got "
                                     f"{self.lookahead_steps!r}")


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
class Load:
    """Move one block of ``role`` from its DRAM position (in ELEMENTS) to on-chip row ``row``."""

    role: str
    block: tuple[int, int]
    dram_row: int
    dram_col: int
    rows: int
    cols: int
    row: int
    step: int


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
    #: Rows the weight block itself spans (the contraction depth of this step), which is what the
    #: residency check reads -- not ``rows``, the streamed row count the accumulator block takes.
    weight_rows: int | None = None


@dataclass(frozen=True)
class Compute:
    """Stream ``rows`` rows from ``input_row`` through the resident weight block."""

    input_row: int
    input_block: tuple[int, int]
    rows: int
    cols: int
    fresh_weights: bool


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
    contraction: Contraction
    geometry: Geometry
    knobs: Knobs
    regions: dict[str, tuple[int, int]]      # role -> (first row, row count)
    notes: tuple[str, ...] = ()

    def count(self, kind: type) -> int:
        return sum(1 for op in self.ops if isinstance(op, kind))


def _regions(rows_needed: dict[str, int], geometry: Geometry,
             knobs: Knobs) -> tuple[dict[str, tuple[int, int]], list[str]]:
    """``(role -> (base row, rows), notes)`` for the placement policy. The streamed operand starts at
    row 0; the resident operand's base is the policy's whole content."""
    notes: list[str] = []
    lhs_rows, weight_rows = rows_needed[LHS], rows_needed[WEIGHT]
    if lhs_rows > geometry.operand_rows or weight_rows > geometry.operand_rows:
        raise BlockScheduleError(
            f"one operand alone needs {max(lhs_rows, weight_rows)} rows of a {geometry.operand_rows}-row "
            "store: this contraction has to be tiled before it can be scheduled")
    if knobs.placement == OPPOSITE_END:
        base = geometry.operand_rows - weight_rows
    else:
        if knobs.placement == BANK_ALIGNED and not geometry.operand_bank_rows:
            raise BlockScheduleError(
                "placement 'bank_aligned' needs the store's per-bank row count, which this target's "
                "facts do not give (operand_bank_rows is unknown)")
        wanted = (lhs_rows if knobs.placement == CONTIGUOUS
                  else _ceil_div(lhs_rows, geometry.operand_bank_rows) * geometry.operand_bank_rows)
        # A policy that does not fit degrades to the end of the store rather than failing: the two
        # regions then overlap, which is legal exactly when every overwrite lands after the
        # overlapping block's last read -- and check_residency, not this function, decides that.
        if wanted + weight_rows <= geometry.operand_rows:
            base = wanted
        else:
            base = geometry.operand_rows - weight_rows
            notes.append(f"placement {knobs.placement!r} wanted row {wanted} but {wanted} + "
                         f"{weight_rows} rows exceeds the {geometry.operand_rows}-row store; fell back "
                         "to the store's end")
    if base + weight_rows > geometry.operand_rows or base < 0:
        raise BlockScheduleError(f"the resident operand's region [{base}, {base + weight_rows}) leaves "
                                 f"the {geometry.operand_rows}-row store")
    if base < lhs_rows:
        notes.append(f"the two operand regions OVERLAP by {lhs_rows - base} rows; the schedule is only "
                     "correct where every overwrite lands after the overlapping block's last read, "
                     "which check_residency verifies")
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
                raise BlockScheduleError(
                    f"{what} reads on-chip row {row + offset} expecting block {identity}, but it holds "
                    f"{have if have is not None else 'nothing loaded'}: the schedule overwrites a block "
                    "that is still live (or never loads it)")

    for op in schedule.ops:
        if isinstance(op, Load):
            if op.row < 0 or op.row + op.rows > geometry.operand_rows:
                raise BlockScheduleError(
                    f"a {op.role} load addresses rows [{op.row}, {op.row + op.rows}) of a "
                    f"{geometry.operand_rows}-row store")
            for offset in range(op.rows):
                resident[op.row + offset] = (op.role, op.block)
        elif isinstance(op, Preload):
            if op.accumulator_row < 0 or op.accumulator_row + op.rows > geometry.accumulator_rows:
                raise BlockScheduleError(
                    f"a preload addresses accumulator rows [{op.accumulator_row}, "
                    f"{op.accumulator_row + op.rows}) of {geometry.accumulator_rows}")
            if op.weight_row is not None:
                _expect(op.weight_row, op.weight_rows or op.rows, (WEIGHT, op.weight_block),
                        "a preload")
        elif isinstance(op, Compute):
            _expect(op.input_row, op.rows, (LHS, op.input_block), "a compute")
        elif isinstance(op, Store):
            if op.accumulator_row + op.rows > geometry.accumulator_rows:
                raise BlockScheduleError(
                    f"a store reads accumulator rows [{op.accumulator_row}, "
                    f"{op.accumulator_row + op.rows}) of {geometry.accumulator_rows}")


def schedule_contraction(contraction: Contraction, geometry: Geometry,
                         knobs: Knobs | None = None) -> BlockSchedule:
    """Schedule ``contraction`` on ``geometry`` under ``knobs``, or refuse.

    The nest is reduction-major (step, then resident-operand block, then streamed-operand block), which
    is what keeps one weight block resident across the streamed blocks that read it. ``knobs`` decides
    only WHERE each load is emitted relative to the computes, and where the resident region sits.
    """
    knobs = knobs or Knobs()
    knobs.validate()
    if min(contraction.m, contraction.k, contraction.n) <= 0:
        raise BlockScheduleError(f"a {contraction.m}x{contraction.k}x{contraction.n} contraction is empty")
    d = geometry.block
    i_blocks = _ceil_div(contraction.m, d)
    j_blocks = _ceil_div(contraction.n, d)
    k_blocks = _ceil_div(contraction.k, d)
    regions, notes = _regions({LHS: i_blocks * k_blocks * d, WEIGHT: k_blocks * j_blocks * d},
                              geometry, knobs)
    lhs_base, weight_base = regions[LHS][0], regions[WEIGHT][0]

    def _lhs_load(mi: int, kk: int) -> Load:
        return Load(LHS, (mi, kk), mi * d, kk * d,
                    min(d, contraction.m - mi * d), min(d, contraction.k - kk * d),
                    lhs_base + (mi * k_blocks + kk) * d, kk)

    def _weight_load(kk: int, nj: int) -> Load:
        return Load(WEIGHT, (kk, nj), kk * d, nj * d,
                    min(d, contraction.k - kk * d), min(d, contraction.n - nj * d),
                    weight_base + (kk * j_blocks + nj) * d, kk)

    # Per (step, resident block) the loads that become due, and the computes that read them. A streamed
    # block is due once per (mi, kk) when load_on_index_change, else once per (mi, kk, nj).
    groups: list[tuple[int, int, list[Load], list[Op]]] = []
    for kk in range(k_blocks):
        for nj in range(j_blocks):
            due_lhs = [_lhs_load(mi, kk) for mi in range(i_blocks)
                       if nj == 0 or not knobs.load_on_index_change]
            body: list[Op] = []
            for mi in range(i_blocks):
                rows, cols = min(d, contraction.m - mi * d), min(d, contraction.n - nj * d)
                acc_row = (mi * j_blocks + nj) * d
                body.append(Preload(weight_base + (kk * j_blocks + nj) * d if mi == 0 else None,
                                    (kk, nj) if mi == 0 else None,
                                    acc_row, kk > 0, rows, cols,
                                    min(d, contraction.k - kk * d) if mi == 0 else None))
                body.append(Compute(lhs_base + (mi * k_blocks + kk) * d, (mi, kk), rows,
                                    min(d, contraction.k - kk * d), mi == 0))
                if kk == k_blocks - 1:
                    body.append(Store(mi * d, nj * d, rows, cols, acc_row))
            # Nest order inside a group: the resident block's load sits at the outer position, the
            # streamed blocks at the inner one. ROLE grouping reorders this; NEST keeps it.
            groups.append((kk, nj, [_weight_load(kk, nj)] + due_lhs, body))

    ops = _emit(groups, k_blocks, knobs)
    schedule = BlockSchedule(tuple(ops), contraction, geometry, knobs, regions, tuple(notes))
    check_residency(schedule)
    return schedule


def _ordered(loads: list[Load], knobs: Knobs) -> list[Load]:
    """The loads of one emission group, ordered by ``load_grouping`` (stable within a role)."""
    if knobs.load_grouping == NEST:
        return list(loads)
    ordered: list[Load] = []
    for role in knobs.operand_order:
        ordered.extend(load for load in loads if load.role == role)
    return ordered


def _emit(groups: list[tuple[int, int, list[Load], list[Op]]], k_blocks: int,
          knobs: Knobs) -> list[Op]:
    """Place each group's loads relative to the computes, per ``lookahead_steps``.

    At depth 0 the emission group is one (step, resident block) pair and a load sits in its nest
    position. At depth ``d`` the group is a whole reduction step, issued ``d`` steps early. At ``None``
    every load is hoisted ahead of every compute, and the whole prologue is one group.
    """
    if knobs.lookahead_steps == 0:
        ops: list[Op] = []
        for _, _, loads, body in groups:
            if knobs.load_grouping == NEST:
                # Nest position: the resident block's load, then each streamed block immediately
                # before the compute that reads it.
                streamed = [load for load in loads if load.role == LHS]
                ops.extend(load for load in loads if load.role == WEIGHT)
                pending = list(streamed)
                for op in body:
                    if isinstance(op, Preload) and pending:
                        ops.append(pending.pop(0))
                    ops.append(op)
                ops.extend(pending)
            else:
                ops.extend(_ordered(loads, knobs))
                ops.extend(body)
        return ops

    by_step: dict[int, list[Load]] = {}
    bodies: dict[int, list[Op]] = {}
    for kk, _, loads, body in groups:
        by_step.setdefault(kk, []).extend(loads)
        bodies.setdefault(kk, []).extend(body)
    steps = sorted(bodies)
    if knobs.lookahead_steps is None:
        prologue: list[Load] = []
        for kk in steps:
            prologue.extend(by_step.get(kk, ()))
        ops = list(_ordered(prologue, knobs))
        for kk in steps:
            ops.extend(bodies[kk])
        return ops

    depth = knobs.lookahead_steps
    ops = []
    for kk in steps[:depth]:
        ops.extend(_ordered(by_step.get(kk, []), knobs))
    for index, kk in enumerate(steps):
        ahead = index + depth
        if ahead < len(steps):
            ops.extend(_ordered(by_step.get(steps[ahead], []), knobs))
        ops.extend(bodies[kk])
    return ops


def schedule_interface_program(tensors: dict[str, Any], commands: list[dict[str, Any]],
                               geometry: Geometry,
                               knobs: Knobs | None = None) -> list[BlockSchedule]:
    """Schedule every resident matmul an interface program commits, in program order.

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
                    "schedules block moves only and will not guess that mapping")
            m, k = _shape(tensors, lhs_name)
            wk, n = _shape(tensors, weight_name)
            if wk != k:
                raise BlockScheduleError(f"contraction mismatch: {lhs_name} is {m}x{k}, "
                                         f"{weight_name} is {wk}x{n}")
            schedules.append(schedule_contraction(
                Contraction(m, k, n, lhs_name, weight_name, operands["dst"]), geometry, knobs))
        elif opcode != "EVICT":
            raise BlockScheduleError(f"{opcode!r} is not a resident matmul; this pass schedules those "
                                     "only, and refuses rather than dropping the command")
    return schedules


def _shape(tensors: dict[str, Any], name: str) -> tuple[int, int]:
    spec = tensors.get(name)
    if spec is None:
        raise BlockScheduleError(f"the program names no tensor {name!r}")
    shape = spec["shape"] if isinstance(spec, dict) else getattr(spec, "shape", None)
    if shape is None or len(shape) != 2:
        raise BlockScheduleError(f"{name!r} has shape {shape!r}; this pass schedules 2-D operands")
    return int(shape[0]), int(shape[1])
