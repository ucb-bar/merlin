"""Lower a replayed Voyager GEMM schedule onto a weight-stationary systolic array, op for op.

:func:`merlin.baselines.voyager_ir.replay` yields what Voyager's compiled program DOES: which tile is
loaded into which scratchpad slot at which step, which compute consumes it under which interstellar
mapping, and when each output tile is stored. This module turns that trace into an abstract schedule
for a weight-stationary array addressed in DIM x DIM blocks -- ``Mvin`` / ``Preload`` / ``Compute`` /
``Mvout`` -- without choosing anything Voyager chose:

* every Voyager load becomes the loads of exactly its tile's blocks, in the same order, into the rows
  Voyager's own byte address and slot select (``row = slot_address // row_bytes``);
* every compute walks the loop nest Voyager serialized (interstellar levels, innermost first), so the
  order in which weight blocks are made resident and activation rows streamed is Voyager's;
* every store writes exactly its tile.

The only translations are the ones the target forces, each named in the experiment's concession
register: the output tile lives in the accumulator rather than a scratchpad slot (the array accumulates
there), a K split accumulates in the integer accumulator rather than as a bf16 add (C2), and the fused
tail's arithmetic is not replayed here -- the readout (int32, or scaled int8 with an activation) is the
caller's decision (C1). Anything else refuses with :class:`UnsupportedConstruct`: a bias operand, a
tile that is not a whole number of blocks, an innermost loop that is not the row stream.

Geometry (block edge, scratchpad rows and row width, accumulator rows) is passed in by the caller from
the target's derived facts; nothing here is a hardware constant. Dependency-free apart from the optional
numpy executor :func:`execute`, which proves a schedule's arithmetic before any simulator runs it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .voyager_ir import Copy, FusedCompute, Ref, Trace, UnsupportedConstruct

__all__ = ["Compute", "Geometry", "Mvin", "Mvout", "Preload", "Schedule", "execute", "lower_gemm"]

#: Anchors a GEMM fusion may start with. A convolution is a different nest and is not lowered here.
_GEMM_ANCHORS = ("quantized_ops::linear", "aten::linear", "aten::matmul", "quantized_ops::matmul")
#: Tail op that combines a K split into the partial already in the destination (concession C2).
_SPLIT_K_COMBINE = "aten::add"


@dataclass(frozen=True)
class Geometry:
    """The array and its stores, in the units a backend addresses them. Derived by the caller."""

    dim: int                 # array edge = block edge
    spad_rows: int           # scratchpad rows (all banks)
    spad_row_bytes: int      # bytes per scratchpad row
    acc_rows: int            # accumulator rows (all banks)


@dataclass(frozen=True)
class Mvin:
    role: str                # "lhs" | "weight"
    dram_row: int
    dram_col: int
    rows: int
    cols: int
    spad_row: int


@dataclass(frozen=True)
class Preload:
    """Make ``weight_row``'s block resident (None keeps the resident block) and name the accumulator
    block the next :class:`Compute` writes, overwriting it or accumulating into it."""

    weight_row: int | None
    acc_row: int
    accumulate: bool
    rows: int
    cols: int


@dataclass(frozen=True)
class Compute:
    input_row: int
    rows: int
    fresh_weights: bool      # the first compute after a weight change


@dataclass(frozen=True)
class Mvout:
    role: str                # "out"
    dram_row: int
    dram_col: int
    rows: int
    cols: int
    acc_row: int


Op = Mvin | Preload | Compute | Mvout


@dataclass
class Schedule:
    """The lowered program plus what it was lowered from."""

    ops: list[Op]
    shapes: dict[str, tuple[int, int]]       # role -> logical [rows, cols]
    dram_nodes: dict[str, str]               # role -> Voyager DRAM tensor
    geometry: Geometry
    notes: list[str] = field(default_factory=list)

    def count(self, kind: type) -> int:
        return sum(1 for op in self.ops if isinstance(op, kind))


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def _two_d(shape: tuple[int, ...], what: str) -> tuple[int, int]:
    dims = [d for d in shape if d != 1] if len(shape) > 2 else list(shape)
    if len(dims) != 2:
        raise UnsupportedConstruct(f"{what} has shape {shape}; only 2-D GEMM operands are lowered")
    return dims[0], dims[1]


class _Lowerer:
    def __init__(self, trace: Trace, geometry: Geometry):
        self.trace, self.g = trace, geometry
        self.ops: list[Op] = []
        self.notes: list[str] = []
        self.roles: dict[str, str] = {}          # on-chip node -> role
        self.dram: dict[str, str] = {}           # role -> DRAM node
        self.shapes: dict[str, tuple[int, int]] = {}
        self.acc_slot_rows = 0

    # placement ------------------------------------------------------------------------------
    def spad_block_row(self, ref: Ref, bi: int, bj: int, rows: int) -> int:
        """Row of block (bi, bj) of the tile held in ``ref``'s slot: blocks column-major inside the
        byte region Voyager allotted, so the tile occupies exactly its allotted rows."""
        base, rem = divmod(ref.slot_address, self.g.spad_row_bytes)
        if rem:
            raise UnsupportedConstruct(f"{ref.box.node} slot address {ref.slot_address} is not "
                                       f"row aligned ({self.g.spad_row_bytes} B rows)")
        row = base + (bj * _ceil(rows, self.g.dim) + bi) * self.g.dim
        if row + self.g.dim > self.g.spad_rows:
            raise UnsupportedConstruct(f"{ref.box.node} block ({bi},{bj}) lands at scratchpad row "
                                       f"{row}, past the {self.g.spad_rows} rows the target has")
        return row

    def acc_block_row(self, ref: Ref, bi: int, bj: int, rows: int) -> int:
        row = ref.slot * self.acc_slot_rows + (bj * _ceil(rows, self.g.dim) + bi) * self.g.dim
        if row + self.g.dim > self.g.acc_rows:
            raise UnsupportedConstruct(f"output slot {ref.slot} block ({bi},{bj}) needs accumulator "
                                       f"row {row}; the target has {self.g.acc_rows}")
        return row

    # roles ------------------------------------------------------------------------------------
    def bind_roles(self, computes: list[FusedCompute]) -> None:
        for comp in computes:
            anchor = comp.anchor
            if anchor.target not in _GEMM_ANCHORS:
                raise UnsupportedConstruct(f"{comp.name}: anchor {anchor.target!r} is not a GEMM")
            if anchor.kwargs.get("bias") is not None:
                raise UnsupportedConstruct(f"{comp.name}: a bias operand is not lowered yet (it needs "
                                           "an accumulator preload of the bias rows)")
            extra = [t for t in comp.tail
                     if t not in ("quantized_ops::dequantize", "aten::relu", _SPLIT_K_COMBINE,
                                  "quantized_ops::quantize")]
            if extra:
                raise UnsupportedConstruct(f"{comp.name}: fused tail {extra} has no readout mapping")
            for kw, role in (("input", "lhs"), ("weight", "weight")):
                ref = anchor.kwargs.get(kw)
                if not isinstance(ref, Ref):
                    raise UnsupportedConstruct(f"{comp.name}: {kw} operand is not a tensor reference")
                self.roles.setdefault(ref.box.node, role)
                self.shapes.setdefault(role + "_tile", _two_d(ref.output_shape or ref.box.shape, kw))
            if len(comp.destinations) != 1:
                raise UnsupportedConstruct(f"{comp.name}: {len(comp.destinations)} destinations")
            dest = comp.destinations[0]
            self.roles.setdefault(dest.box.node, "out")
            self.shapes.setdefault("out_tile", _two_d(dest.output_shape or dest.box.shape, "output"))
        for copy in self.trace.of(Copy):
            if copy.is_load and copy.dst.box.node in self.roles:
                role = self.roles[copy.dst.box.node]
                self.dram.setdefault(role, copy.src.box.node)
                self.shapes.setdefault(role, _two_d(copy.src.box.shape, role))
            elif copy.is_store and copy.src.box.node in self.roles:
                self.dram.setdefault("out", copy.dst.box.node)
                self.shapes.setdefault("out", _two_d(copy.dst.box.shape, "out"))
        missing = {"lhs", "weight", "out"} - set(self.dram)
        if missing:
            raise UnsupportedConstruct(f"no DRAM transfer feeds or drains {sorted(missing)}")
        tm, tn = self.shapes["out_tile"]
        self.acc_slot_rows = _ceil(tm, self.g.dim) * _ceil(tn, self.g.dim) * self.g.dim

    # events -----------------------------------------------------------------------------------
    @staticmethod
    def tile_origin(copy: Copy) -> tuple[int, int, int, int]:
        """(row0, col0, rows, cols) of the DRAM tile a copy moves, by Voyager's own async_copy rule:
        ``offset[dim] = index * (strides or sizes)[dim]``, over ``dims`` when given, else positionally."""
        if copy.pad or copy.transposed or copy.count:
            raise UnsupportedConstruct(f"{copy.name}: padded/transposed/partial copies not lowered")
        sizes = tuple(copy.sizes)
        if len(sizes) != 2:
            raise UnsupportedConstruct(f"{copy.name}: a {len(sizes)}-D tile copy is not lowered")
        pitch = tuple(copy.strides) if copy.strides else sizes
        offsets = [0, 0]
        if copy.dims is None:
            for dim, index in enumerate(copy.indices):
                offsets[dim] = index * pitch[dim]
        else:
            for index, dim in zip(copy.indices, copy.dims):
                offsets[dim] = index * pitch[dim]
        return offsets[0], offsets[1], sizes[0], sizes[1]

    def load(self, copy: Copy) -> None:
        role = self.roles[copy.dst.box.node]
        r0, c0, rows, cols = self.tile_origin(copy)
        for bj in range(_ceil(cols, self.g.dim)):
            for bi in range(_ceil(rows, self.g.dim)):
                self.ops.append(Mvin(role, r0 + bi * self.g.dim, c0 + bj * self.g.dim,
                                     min(self.g.dim, rows - bi * self.g.dim),
                                     min(self.g.dim, cols - bj * self.g.dim),
                                     self.spad_block_row(copy.dst, bi, bj, rows)))

    def store(self, copy: Copy) -> None:
        r0, c0, rows, cols = self.tile_origin(copy)
        for bi in range(_ceil(rows, self.g.dim)):
            for bj in range(_ceil(cols, self.g.dim)):
                self.ops.append(Mvout("out", r0 + bi * self.g.dim, c0 + bj * self.g.dim,
                                      min(self.g.dim, rows - bi * self.g.dim),
                                      min(self.g.dim, cols - bj * self.g.dim),
                                      self.acc_block_row(copy.src, bi, bj, rows)))

    def compute(self, comp: FusedCompute) -> None:
        d = self.g.dim
        lhs, weight = comp.anchor.kwargs["input"], comp.anchor.kwargs["weight"]
        dest = comp.destinations[0]
        tm, tk = _two_d(lhs.output_shape or lhs.box.shape, "input tile")
        tk2, tn = _two_d(weight.output_shape or weight.box.shape, "weight tile")
        if tk != tk2:
            raise UnsupportedConstruct(f"{comp.name}: contraction {tk} vs {tk2}")
        if tm % d or tk % d or tn % d:
            raise UnsupportedConstruct(f"{comp.name}: tile {tm}x{tk}x{tn} is not whole {d}-blocks")
        # Interstellar levels come innermost first; a loop absent from every level has bound 1.
        levels = list(comp.tiling)
        order: list[tuple[str, int]] = []
        for level in reversed(levels):
            order.extend(reversed(level))
        extents = {"LOOP_OX": 1, "LOOP_IC": 1, "LOOP_OC": 1}
        for loop, bound in order:
            if loop not in extents:
                raise UnsupportedConstruct(f"{comp.name}: loop {loop} has no GEMM meaning")
            extents[loop] *= bound
        if (extents["LOOP_OX"], extents["LOOP_IC"] * d, extents["LOOP_OC"] * d) != (tm, tk, tn):
            raise UnsupportedConstruct(f"{comp.name}: mapping {order} does not cover the "
                                       f"{tm}x{tk}x{tn} tile")
        if not order or order[-1][0] != "LOOP_OX":
            raise UnsupportedConstruct(f"{comp.name}: innermost loop {order[-1:]} is not the row "
                                       "stream a weight-stationary array needs")
        stream = order[-1][1]
        outer = order[:-1]
        split_k_combine = _SPLIT_K_COMBINE in comp.tail
        if split_k_combine:
            self.notes.append(f"{comp.name}: K split combined in the integer accumulator (C2)")
        # Walk the outer loops as a mixed-radix counter in Voyager's order (the innermost outer loop
        # varies fastest). Each point fixes (first row of the stream, ic block, oc block) and streams
        # `stream` rows through one resident weight block. A loop that appears at several levels
        # composes: the inner level's extent is the outer level's place value.
        points = 1
        for _, bound in outer:
            points *= bound
        resident: tuple[int, int] | None = None
        for p in range(points):
            rem = p
            coord = {"LOOP_OX": 0, "LOOP_IC": 0, "LOOP_OC": 0}
            place = {"LOOP_OX": stream, "LOOP_IC": 1, "LOOP_OC": 1}
            for loop, bound in reversed(outer):
                digit = rem % bound
                rem //= bound
                coord[loop] += digit * place[loop]
                place[loop] *= bound
            row0, ic, oc = coord["LOOP_OX"], coord["LOOP_IC"], coord["LOOP_OC"]
            if row0 % d:
                raise UnsupportedConstruct(f"{comp.name}: row stream starts at {row0}, mid-block")
            first_contribution = ic == 0 and not split_k_combine
            new_weights = resident != (ic, oc)
            for chunk in range(_ceil(stream, d)):
                rows = min(d, stream - chunk * d)
                bi = (row0 + chunk * d) // d
                self.ops.append(Preload(
                    weight_row=self.spad_block_row(weight, ic, oc, tk) if new_weights and chunk == 0
                    else None,
                    acc_row=self.acc_block_row(dest, bi, oc, tm), accumulate=not first_contribution,
                    rows=rows, cols=d))
                self.ops.append(Compute(input_row=self.spad_block_row(lhs, bi, ic, tm), rows=rows,
                                        fresh_weights=new_weights and chunk == 0))
            resident = (ic, oc)


def lower_gemm(trace: Trace, geometry: Geometry) -> Schedule:
    """Lower every GEMM in ``trace`` (in program order) to a :class:`Schedule` on ``geometry``."""
    computes = trace.of(FusedCompute)
    if not computes:
        raise UnsupportedConstruct("the trace has no compute to lower")
    lowerer = _Lowerer(trace, geometry)
    lowerer.bind_roles(computes)
    for event in trace.events:
        if isinstance(event, Copy):
            if event.is_load and event.dst.box.node in lowerer.roles:
                lowerer.load(event)
            elif event.is_store and event.src.box.node in lowerer.roles:
                lowerer.store(event)
            else:
                raise UnsupportedConstruct(f"{event.name}: a copy between {event.src.box.level} and "
                                           f"{event.dst.box.level} that no GEMM operand explains")
        elif isinstance(event, FusedCompute):
            lowerer.compute(event)
    lowerer.notes.append("async_wait/commit semaphores carry no instruction: the target orders "
                         "moves and computes by address dependence")
    return Schedule(ops=lowerer.ops, shapes={k: v for k, v in lowerer.shapes.items()
                                             if not k.endswith("_tile")},
                    dram_nodes=dict(lowerer.dram), geometry=geometry, notes=lowerer.notes)


def execute(schedule: Schedule, lhs: Any, weight: Any) -> Any:
    """Run ``schedule`` on integer operands with numpy and return the int32 output.

    An executable semantics for the abstract ops (a resident weight block, scratchpad and accumulator
    rows), used to prove a lowering's arithmetic before any simulator sees it. Not a timing model.
    """
    import numpy as np

    g = schedule.geometry
    d = g.dim
    spad = np.zeros((g.spad_rows, d), dtype=np.int64)
    acc = np.zeros((g.acc_rows, d), dtype=np.int64)
    out = np.zeros(schedule.shapes["out"], dtype=np.int64)
    sources = {"lhs": np.asarray(lhs, dtype=np.int64), "weight": np.asarray(weight, dtype=np.int64)}
    resident = np.zeros((d, d), dtype=np.int64)
    target: Preload | None = None
    for op in schedule.ops:
        if isinstance(op, Mvin):
            block = np.zeros((d, d), dtype=np.int64)
            block[:op.rows, :op.cols] = sources[op.role][op.dram_row:op.dram_row + op.rows,
                                                         op.dram_col:op.dram_col + op.cols]
            spad[op.spad_row:op.spad_row + d] = block
        elif isinstance(op, Preload):
            if op.weight_row is not None:
                resident = spad[op.weight_row:op.weight_row + d].copy()
            target = op
        elif isinstance(op, Compute):
            if target is None:
                raise UnsupportedConstruct("compute with no preceding preload")
            product = spad[op.input_row:op.input_row + d] @ resident
            rows = slice(target.acc_row, target.acc_row + d)
            acc[rows] = acc[rows] + product if target.accumulate else product
            target = None
        elif isinstance(op, Mvout):
            out[op.dram_row:op.dram_row + op.rows, op.dram_col:op.dram_col + op.cols] = \
                acc[op.acc_row:op.acc_row + op.rows, :op.cols]
    return out.astype(np.int32)
