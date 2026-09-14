"""Lower a replayed Voyager GEMM or convolution schedule onto a weight-stationary array, op for op.

:func:`merlin.baselines.voyager_ir.replay` yields what Voyager's compiled program DOES: which tile is
loaded into which scratchpad slot at which step, which compute consumes it under which interstellar
mapping, and when each output tile is stored. This module turns that trace into an abstract schedule
for a weight-stationary array addressed in DIM x DIM blocks -- ``Mvin`` / ``AccMvin`` / ``Preload`` /
``Compute`` / ``Mvout`` -- without choosing anything Voyager chose:

* every Voyager load becomes the loads of exactly its tile's blocks, in the same order, into the rows
  Voyager's own byte address and slot select (``row = slot_address // row_bytes``);
* every compute walks the loop nest Voyager serialized (interstellar levels, innermost first), so the
  order in which weight blocks are made resident and activation rows streamed is Voyager's;
* every store writes exactly its tile.

The only translations are the ones the target forces, each named in the experiment's concession
register: the output tile lives in the accumulator rather than a scratchpad slot (the array accumulates
there), a K split accumulates in the integer accumulator rather than as a bf16 add (C2), and the fused
tail's arithmetic is not replayed here -- the readout (int32, or scaled int8 with an activation) is the
caller's decision (C1). A convolution adds three more: a stride-s input tile is stored phase-split so a
tap's pixels are consecutive rows (C4); a bias moves DRAM -> accumulator at its first use, because the
target has no scratchpad -> accumulator path (C5); and because a K split keeps its partial in the
accumulator, each output tile's split-K group takes the next region of a ring as deep as Voyager's
output buffer, and the output slot the last part names is bound to that region (renaming, not a copy).
Anything else refuses with :class:`UnsupportedConstruct`: a residual add, a tile that is not a whole
number of blocks, an innermost loop that is not a pixel stream.

Geometry (block edge, scratchpad rows and row width, accumulator rows) is passed in by the caller from
the target's derived facts; nothing here is a hardware constant. Dependency-free apart from the optional
numpy executor :func:`execute`, which proves a schedule's arithmetic before any simulator runs it.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from .voyager_ir import Box, Copy, FusedCompute, Ref, TensorOp, Trace, UnsupportedConstruct

__all__ = ["AccMvin", "Compute", "Geometry", "Mvin", "Mvout", "Preload", "Schedule", "execute",
           "lower_conv", "lower_gemm"]

#: Anchors a GEMM fusion may start with.
_GEMM_ANCHORS = ("quantized_ops::linear", "aten::linear", "aten::matmul", "quantized_ops::matmul")
#: Anchors a convolution fusion may start with.
_CONV_ANCHORS = ("quantized_ops::conv2d", "aten::conv2d")
#: Tail op that combines a K split into the partial already in the destination (concession C2).
_SPLIT_K_COMBINE = "aten::add"
#: Tail ops whose arithmetic is the readout's (concession C1).
_READOUT_TAIL = ("quantized_ops::dequantize", "aten::relu", "aten::relu_", "quantized_ops::quantize")
#: Convolution loops and what one iteration of each advances.
_CONV_LOOPS = ("LOOP_OY", "LOOP_OX", "LOOP_FY", "LOOP_FX", "LOOP_IC", "LOOP_OC")


@dataclass(frozen=True)
class Geometry:
    """The array and its stores, in the units a backend addresses them. Derived by the caller."""

    dim: int                 # array edge = block edge
    spad_rows: int           # scratchpad rows (all banks)
    spad_row_bytes: int      # bytes per scratchpad row
    acc_rows: int            # accumulator rows (all banks)


@dataclass(frozen=True)
class Mvin:
    role: str                # "lhs" | "weight" | "zero" (a zero page: the halo of a padded tile)
    dram_row: int
    dram_col: int
    rows: int
    cols: int
    spad_row: int
    row_step: int = 1        # DRAM rows between consecutive block rows (a stride-s gather uses s)


@dataclass(frozen=True)
class AccMvin:
    """Load integer rows straight into the accumulator, overwriting it -- the bias a first K part
    accumulates onto. ``row_step`` 0 broadcasts one DRAM row to every block row."""

    role: str                # "bias"
    dram_row: int
    dram_col: int
    rows: int
    cols: int
    acc_row: int
    row_step: int = 0


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


Op = Mvin | AccMvin | Preload | Compute | Mvout


@dataclass
class Schedule:
    """The lowered program plus what it was lowered from.

    ``shapes`` gives each DRAM role as the 2-D matrix the ops address: a GEMM operand as itself, an
    NHWC activation as ``[N*H*W, C]``, an HWIO weight as ``[KH*KW*Cin, Cout]``, a bias as ``[1, C]``.
    """

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


def _copy_window(copy: Copy) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(first source element per dim, tile sizes) of a copy, by Voyager's own async_copy rule:
    ``offset[dim] = index * (strides or sizes)[dim]`` over ``dims`` when given, else positionally; a
    ``pad`` moves the window back by each dim's leading pad and fills what falls outside the source."""
    if copy.transposed or copy.count:
        raise UnsupportedConstruct(f"{copy.name}: transposed/partial copies are not lowered")
    if copy.pad and copy.pad_value not in (None, 0, 0.0):
        raise UnsupportedConstruct(f"{copy.name}: a pad value of {copy.pad_value} is not a zero page")
    sizes = tuple(copy.sizes)
    pitch = tuple(copy.strides) if copy.strides else sizes
    offsets = [0] * len(sizes)
    if copy.dims is None:
        for dim, index in enumerate(copy.indices):
            offsets[dim] = index * pitch[dim]
    else:
        for index, dim in zip(copy.indices, copy.dims):
            offsets[dim] = index * pitch[dim]
    pad = tuple(copy.pad) if copy.pad else (0,) * len(sizes)
    return tuple(o - p for o, p in zip(offsets, pad)), sizes


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
            extra = [t for t in comp.tail if t not in _READOUT_TAIL + (_SPLIT_K_COMBINE,)]
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
        (r0, c0), _ = _copy_window(copy)
        return r0, c0, sizes[0], sizes[1]

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


class _ConvLowerer:
    """Lower NHWC convolutions. Placement keeps each tile inside the byte region Voyager allotted it;
    the output tile lives in an accumulator region chosen by split-K group (see the module doc)."""

    def __init__(self, trace: Trace, geometry: Geometry):
        self.trace, self.g = trace, geometry
        self.ops: list[Op] = []
        self.notes: list[str] = []
        self.roles: dict[str, str] = {}              # on-chip node -> lhs|weight|bias|out|partial
        self.dram: dict[str, Box] = {}               # role -> DRAM tensor
        self.stride: dict[str, tuple[int, int]] = {}  # input node -> the stride its convs use
        self.bias_start: dict[tuple[str, int], int] = {}  # (bias node, slot) -> first channel held
        self.tile: tuple[int, ...] = ()
        self.region_rows = 0
        self.regions = 0
        self.bound: dict[tuple[str, int], int] = {}  # (out node, slot) -> the region it names
        self.unstored: set[int] = set()              # regions finished but not yet stored
        self.group: dict[str, Any] | None = None     # the split-K group being accumulated
        self.groups = 0

    def _note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    # placement ------------------------------------------------------------------------------
    def _region(self, ref: Ref) -> tuple[int, int]:
        """(first row, rows) of the scratchpad region Voyager allotted ``ref``'s slot."""
        base, rem = divmod(ref.slot_address, self.g.spad_row_bytes)
        if rem:
            raise UnsupportedConstruct(f"{ref.box.node} slot address {ref.slot_address} is not "
                                       f"row aligned ({self.g.spad_row_bytes} B rows)")
        elements = 1
        for extent in ref.box.shape:
            elements *= extent
        return base, _ceil(elements, self.g.spad_row_bytes)

    def _checked(self, ref: Ref, row: int, rows: int) -> int:
        base, allotted = self._region(ref)
        if row < base or row + rows > base + allotted or row + rows > self.g.spad_rows:
            raise UnsupportedConstruct(f"{ref.box.node} rows {row}..{row + rows} leave the "
                                       f"{allotted}-row region Voyager allotted at row {base}")
        return row

    def input_row(self, ref: Ref, cb: int, h: int, w: int) -> int:
        """Row of pixel (h, w), channel block ``cb``, of the input tile in ``ref``'s slot: one row per
        pixel, channel blocks outermost, and a stride-s tile phase-split along W (C4) -- columns
        p, p+s, ... are consecutive rows, so a tap's pixels are a contiguous stream."""
        _, th, tw, _ = ref.box.shape
        s = self.stride[ref.box.node][1]
        base, _ = self._region(ref)
        before = sum(_ceil(tw - q, s) for q in range(w % s))
        return base + (cb * th + h) * tw + before + w // s

    def weight_row(self, ref: Ref, fy: int, fx: int, icb: int, ocb: int) -> int:
        d = self.g.dim
        _, kw, ci, co = ref.box.shape
        base, _ = self._region(ref)
        return self._checked(ref, base + (((fy * kw + fx) * (ci // d) + icb) * (co // d) + ocb) * d, d)

    # roles ------------------------------------------------------------------------------------
    def bind_roles(self, computes: list[FusedCompute]) -> None:
        d = self.g.dim
        if self.g.spad_row_bytes != d:
            raise UnsupportedConstruct(f"a {self.g.spad_row_bytes} B scratchpad row does not hold one "
                                       f"{d}-channel row of 8-bit pixels")
        stored = {c.src.box.node for c in self.trace.of(Copy) if c.is_store}
        for comp in computes:
            anchor = comp.anchor
            if anchor.target not in _CONV_ANCHORS:
                raise UnsupportedConstruct(f"{comp.name}: anchor {anchor.target!r} is not a conv")
            if anchor.kwargs.get("groups", 1) != 1:
                raise UnsupportedConstruct(f"{comp.name}: grouped convolution is not lowered")
            if tuple(anchor.kwargs.get("padding", (0, 0))) != (0, 0):
                raise UnsupportedConstruct(f"{comp.name}: padding inside the compute is not lowered "
                                           "(Voyager pads in the copy)")
            extra = [t for t in comp.tail if t not in _READOUT_TAIL + (_SPLIT_K_COMBINE,)]
            if extra:
                raise UnsupportedConstruct(f"{comp.name}: fused tail {extra} has no readout mapping")
            for kw, role in (("input", "lhs"), ("weight", "weight"), ("bias", "bias")):
                ref = anchor.kwargs.get(kw)
                if ref is None and kw == "bias":
                    continue
                if not isinstance(ref, Ref):
                    raise UnsupportedConstruct(f"{comp.name}: {kw} operand is not a tensor reference")
                if self.roles.setdefault(ref.box.node, role) != role:
                    raise UnsupportedConstruct(f"{ref.box.node} is both {self.roles[ref.box.node]} "
                                               f"and {role}")
                if ref.box.dtype not in ("int8", "uint8") and role != "bias":
                    raise UnsupportedConstruct(f"{comp.name}: {kw} is {ref.box.dtype}; only 8-bit "
                                               "operands are lowered")
            stride = tuple(anchor.kwargs.get("stride", (1, 1)))
            if self.stride.setdefault(anchor.kwargs["input"].box.node, stride) != stride:
                raise UnsupportedConstruct(f"{comp.name}: one input buffer read at two strides")
            if len(comp.destinations) != 1:
                raise UnsupportedConstruct(f"{comp.name}: {len(comp.destinations)} destinations")
            dest = comp.destinations[0]
            self.roles.setdefault(dest.box.node, "out" if dest.box.node in stored else "partial")
            if self.tile and tuple(dest.box.shape) != self.tile:
                raise UnsupportedConstruct(f"{comp.name}: output tiles {dest.box.shape} and "
                                           f"{self.tile} in one layer")
            self.tile = tuple(dest.box.shape)
            if dest.box.node in stored:
                self.regions = max(self.regions, dest.box.bank_count)
        for copy in self.trace.of(Copy):
            if copy.is_load and copy.dst.box.node in self.roles:
                self.dram.setdefault(self.roles[copy.dst.box.node], copy.src.box)
            elif copy.is_store and self.roles.get(copy.src.box.node) == "out":
                self.dram.setdefault("out", copy.dst.box)
        missing = {"lhs", "weight", "out"} - set(self.dram)
        if missing:
            raise UnsupportedConstruct(f"no DRAM transfer feeds or drains {sorted(missing)}")
        for role in ("lhs", "weight", "out"):
            if len(self.dram[role].shape) != 4:
                raise UnsupportedConstruct(f"{role} {self.dram[role].shape} is not a 4-D tensor")
        n, oh, ow, oc = self.tile
        if n != 1 or oc % d:
            raise UnsupportedConstruct(f"output tile {self.tile} is not one image of whole "
                                       f"{d}-channel blocks")
        self.region_rows = oc // d * oh * ow
        if self.regions * self.region_rows > self.g.acc_rows:
            raise UnsupportedConstruct(f"{self.regions} output tiles of {self.region_rows} rows need "
                                       f"more accumulator than the target has ({self.g.acc_rows})")

    def views(self) -> dict[str, tuple[int, int]]:
        n, h, w, c = self.dram["lhs"].shape
        kh, kw, ci, co = self.dram["weight"].shape
        on, oh, ow, oc = self.dram["out"].shape
        shapes = {"lhs": (n * h * w, c), "weight": (kh * kw * ci, co), "out": (on * oh * ow, oc)}
        if "bias" in self.dram:
            shapes["bias"] = (1, self.dram["bias"].shape[-1])
        return shapes

    # loads and stores -------------------------------------------------------------------------
    def load(self, copy: Copy) -> None:
        role = self.roles[copy.dst.box.node]
        start, sizes = _copy_window(copy)
        if role == "bias":
            if len(sizes) != 1:
                raise UnsupportedConstruct(f"{copy.name}: a {len(sizes)}-D bias tile")
            self.bias_start[(copy.dst.box.node, copy.dst.slot)] = start[0]
            self._note("bias rows move DRAM -> accumulator at their first use: the target has no "
                       "scratchpad -> accumulator path (C5)")
        elif role == "lhs":
            self._load_input(copy, start, sizes)
        elif role == "weight":
            self._load_weight(copy, start, sizes)
        else:
            raise UnsupportedConstruct(f"{copy.name}: a load into the {role} buffer")

    def _load_input(self, copy: Copy, start: tuple[int, ...], sizes: tuple[int, ...]) -> None:
        d = self.g.dim
        n0, h0, w0, c0 = start
        tn, th, tw, tc = sizes
        _, height, width, channels = self.dram["lhs"].shape
        if tn != 1 or tc % d or c0 % d or c0 + tc > channels:
            raise UnsupportedConstruct(f"{copy.name}: input tile {sizes} at {start} is not one image "
                                       f"of whole {d}-channel blocks")
        s = self.stride[copy.dst.box.node][1]
        if s > 1:
            self._note(f"stride-{s} input tiles are loaded phase-split along W (C4)")
        for cb in range(tc // d):
            for h in range(th):
                inside = 0 <= h0 + h < height
                for phase in range(s):
                    run: list[int] = []
                    for w in range(phase, tw, s):
                        valid = inside and 0 <= w0 + w < width
                        if run and (len(run) == d or valid != run_valid):
                            self._emit_input(copy, cb, h, run, run_valid, start, s)
                            run = []
                        if not run:
                            run_valid = valid
                        run.append(w)
                    if run:
                        self._emit_input(copy, cb, h, run, run_valid, start, s)

    def _emit_input(self, copy: Copy, cb: int, h: int, run: list[int], valid: bool,
                    start: tuple[int, ...], s: int) -> None:
        d = self.g.dim
        n0, h0, w0, c0 = start
        _, height, width, _ = self.dram["lhs"].shape
        row = self._checked(copy.dst, self.input_row(copy.dst, cb, h, run[0]), len(run))
        if valid:
            pixel = (n0 * height + h0 + h) * width + w0 + run[0]
            self.ops.append(Mvin("lhs", pixel, c0 + cb * d, len(run), d, row, row_step=s))
        else:
            self._note("halo pixels outside the image are loaded from a zero page")
            self.ops.append(Mvin("zero", 0, 0, len(run), d, row, row_step=0))

    def _load_weight(self, copy: Copy, start: tuple[int, ...], sizes: tuple[int, ...]) -> None:
        d = self.g.dim
        fy0, fx0, ic0, oc0 = start
        tkh, tkw, tic, toc = sizes
        _, kw, ci, co = self.dram["weight"].shape
        if tic % d or toc % d or ic0 % d or oc0 % d or ic0 + tic > ci or oc0 + toc > co:
            raise UnsupportedConstruct(f"{copy.name}: weight tile {sizes} at {start} is not whole "
                                       f"{d}x{d} blocks")
        for fy in range(tkh):
            for fx in range(tkw):
                for icb in range(tic // d):
                    for ocb in range(toc // d):
                        self.ops.append(Mvin(
                            "weight", ((fy0 + fy) * kw + fx0 + fx) * ci + ic0 + icb * d,
                            oc0 + ocb * d, d, d, self.weight_row(copy.dst, fy, fx, icb, ocb)))

    def store(self, copy: Copy) -> None:
        d = self.g.dim
        key = (copy.src.box.node, copy.src.slot)
        region = self.bound.pop(key, None)
        if region is None:
            raise UnsupportedConstruct(f"{copy.name}: stores {key[0]} slot {key[1]}, which no "
                                       "finished compute names")
        self.unstored.discard(region)
        start, sizes = _copy_window(copy)
        n0, oh0, ow0, oc0 = start
        _, toh, tow, toc = sizes
        _, out_h, out_w, out_c = self.dram["out"].shape
        if tuple(sizes) != self.tile or oh0 + toh > out_h or ow0 + tow > out_w or oc0 + toc > out_c:
            raise UnsupportedConstruct(f"{copy.name}: a partial output tile {sizes} at {start}")
        base, pixels = region * self.region_rows, toh * tow
        for ocb in range(toc // d):
            run: list[int] | None = None                  # [dram row, acc row, rows]
            for oy in range(toh):
                for ox in range(tow):
                    acc = base + ocb * pixels + oy * tow + ox
                    dram = (n0 * out_h + oh0 + oy) * out_w + ow0 + ox
                    if run and acc == run[1] + run[2] and dram == run[0] + run[2] and run[2] < d:
                        run[2] += 1
                        continue
                    if run:
                        self.ops.append(Mvout("out", run[0], oc0 + ocb * d, run[2], d, run[1]))
                    run = [dram, acc, 1]
            if run:
                self.ops.append(Mvout("out", run[0], oc0 + ocb * d, run[2], d, run[1]))

    # computes ---------------------------------------------------------------------------------
    def _combines(self, comp: FusedCompute) -> bool:
        combine = False
        for call in comp.chain[1:]:
            if call.target != _SPLIT_K_COMBINE:
                continue
            other = call.kwargs.get("other")
            if (self.group is not None and isinstance(other, Ref)
                    and other.box.node == self.group["partial"]):
                combine = True
            else:
                what = other.box.node if isinstance(other, Ref) else other
                raise UnsupportedConstruct(f"{comp.name}: an add of {what} is a residual, not a K "
                                           "split; residual epilogues are not lowered yet (C5)")
        return combine

    def _open_group(self, comp: FusedCompute) -> None:
        if self.group is not None:
            raise UnsupportedConstruct(f"{comp.name} starts an output tile while the previous K "
                                       "split is still open")
        region = self.groups % self.regions
        if region in self.unstored:
            raise UnsupportedConstruct(f"{comp.name}: accumulator region {region} still waits for "
                                       f"its store; a {self.regions}-deep ring is too shallow")
        self.groups += 1
        self.group = {"region": region, "initialized": set(), "partial": None}

    def compute(self, comp: FusedCompute) -> None:
        d = self.g.dim
        anchor = comp.anchor
        lhs, weight, bias = anchor.kwargs["input"], anchor.kwargs["weight"], anchor.kwargs.get("bias")
        dest = comp.destinations[0]
        _, th, tw, tc = lhs.box.shape
        tkh, tkw, tic, toc = weight.box.shape
        _, toh, tow, _ = self.tile
        sh, sw = anchor.kwargs.get("stride", (1, 1))
        dh, dw = anchor.kwargs.get("dilation", (1, 1))
        if tic != tc or toc != self.tile[3] or tic % d:
            raise UnsupportedConstruct(f"{comp.name}: input {lhs.box.shape}, weight "
                                       f"{weight.box.shape}, output {self.tile} do not compose")
        if (toh - 1) * sh + (tkh - 1) * dh >= th or (tow - 1) * sw + (tkw - 1) * dw >= tw:
            raise UnsupportedConstruct(f"{comp.name}: the input tile {lhs.box.shape} does not hold "
                                       f"the {toh}x{tow} output window")
        combine = self._combines(comp)
        if combine:
            self._note("K splits combine in the integer accumulator (C2); each output tile's "
                       "split-K group renames onto the next accumulator region of the output ring")
        else:
            self._open_group(comp)
        group = self.group
        base, pixels = group["region"] * self.region_rows, toh * tow
        initialized: set[int] = group["initialized"]
        if bias is not None:
            if combine:
                raise UnsupportedConstruct(f"{comp.name}: a bias on a later K part")
            oc0 = self.bias_start.get((bias.box.node, bias.slot))
            if oc0 is None:
                raise UnsupportedConstruct(f"{comp.name}: its bias slot was never loaded")
            for ocb in range(toc // d):
                for p0 in range(0, pixels, d):
                    rows = min(d, pixels - p0)
                    row = base + ocb * pixels + p0
                    self.ops.append(AccMvin("bias", 0, oc0 + ocb * d, rows, d, row))
                    initialized.update(range(row, row + rows))
        # Voyager's nest, outermost first. A loop at several levels composes: the inner level's
        # extent is the outer level's place value.
        order = [(loop, bound) for level in reversed(comp.tiling) for loop, bound in reversed(level)]
        extents = dict.fromkeys(_CONV_LOOPS, 1)
        for loop, bound in order:
            if loop not in extents:
                raise UnsupportedConstruct(f"{comp.name}: loop {loop} has no convolution meaning")
            extents[loop] *= bound
        want = {"LOOP_OY": toh, "LOOP_OX": tow, "LOOP_FY": tkh, "LOOP_FX": tkw,
                "LOOP_IC": tic // d, "LOOP_OC": toc // d}
        if extents != want:
            raise UnsupportedConstruct(f"{comp.name}: mapping {order} does not cover {want}")
        if not order or order[-1][0] not in ("LOOP_OY", "LOOP_OX"):
            raise UnsupportedConstruct(f"{comp.name}: innermost loop {order[-1:]} is not a pixel "
                                       "stream")
        # Stream each weight residency's pixels in Voyager's order; a run of consecutive scratchpad
        # rows into consecutive accumulator rows (at most one block) becomes one compute.
        resident = run = None
        for digits in itertools.product(*(range(bound) for _, bound in order)):
            coord = dict.fromkeys(_CONV_LOOPS, 0)
            place = dict.fromkeys(_CONV_LOOPS, 1)
            for (loop, bound), digit in zip(reversed(order), reversed(digits)):
                coord[loop] += digit * place[loop]
                place[loop] *= bound
            oy, ox = coord["LOOP_OY"], coord["LOOP_OX"]
            key = (coord["LOOP_FY"], coord["LOOP_FX"], coord["LOOP_IC"], coord["LOOP_OC"])
            in_row = self.input_row(lhs, key[2], oy * sh + key[0] * dh, ox * sw + key[1] * dw)
            acc_row = base + key[3] * pixels + oy * tow + ox
            ready = acc_row in initialized
            if (run and run[0] == key and in_row == run[1] + run[3] and acc_row == run[2] + run[3]
                    and run[3] < d and ready == run[4]):
                run[3] += 1
                continue
            if run:
                resident = self._flush(run, weight, resident, initialized)
            run = [key, in_row, acc_row, 1, ready]
        if run:
            self._flush(run, weight, resident, initialized)
        if self.roles[dest.box.node] == "out":
            self.bound[(dest.box.node, dest.slot)] = group["region"]
            self.unstored.add(group["region"])
            self.group = None
        elif group["partial"] not in (None, dest.box.node):
            raise UnsupportedConstruct(f"{comp.name}: a K split moves between partial buffers")
        else:
            group["partial"] = dest.box.node

    def _flush(self, run: list, weight: Ref, resident: tuple | None, initialized: set[int]) -> tuple:
        key, in_row, acc_row, rows, ready = run
        fresh = resident != key
        self._checked(weight, self.weight_row(weight, *key), self.g.dim)
        self.ops.append(Preload(weight_row=self.weight_row(weight, *key) if fresh else None,
                                acc_row=acc_row, accumulate=ready, rows=rows, cols=self.g.dim))
        self.ops.append(Compute(input_row=in_row, rows=rows, fresh_weights=fresh))
        initialized.update(range(acc_row, acc_row + rows))
        return key


def lower_conv(trace: Trace, geometry: Geometry) -> Schedule:
    """Lower every NHWC convolution in ``trace`` (in program order) to a :class:`Schedule`."""
    computes = trace.of(FusedCompute)
    if not computes:
        raise UnsupportedConstruct("the trace has no compute to lower")
    lowerer = _ConvLowerer(trace, geometry)
    lowerer.bind_roles(computes)
    for event in trace.events:
        if isinstance(event, Copy):
            if event.is_load and event.dst.box.node in lowerer.roles:
                lowerer.load(event)
            elif event.is_store and lowerer.roles.get(event.src.box.node) == "out":
                lowerer.store(event)
            else:
                raise UnsupportedConstruct(f"{event.name}: a copy between {event.src.box.level} and "
                                           f"{event.dst.box.level} that no conv operand explains")
        elif isinstance(event, FusedCompute):
            lowerer.compute(event)
        elif isinstance(event, TensorOp):
            lowerer._note(f"{event.call.name} ({event.call.target}) is a host op outside the "
                          "accelerator schedule (C6)")
    if lowerer.group is not None or lowerer.bound:
        raise UnsupportedConstruct("the trace ends with an output tile that is never stored")
    lowerer.notes.append("async_wait/commit semaphores carry no instruction: the target orders "
                         "moves and computes by address dependence")
    return Schedule(ops=lowerer.ops, shapes=lowerer.views(),
                    dram_nodes={role: box.node for role, box in lowerer.dram.items()},
                    geometry=geometry, notes=lowerer.notes)


def execute(schedule: Schedule, lhs: Any, weight: Any, **operands: Any) -> Any:
    """Run ``schedule`` on integer operands with numpy and return the int32 output.

    An executable semantics for the abstract ops (a resident weight block, scratchpad and accumulator
    rows), used to prove a lowering's arithmetic before any simulator sees it. Not a timing model.
    Operands are the 2-D views :attr:`Schedule.shapes` names; ``operands`` supplies any further role a
    schedule reads (``bias``).
    """
    import numpy as np

    g = schedule.geometry
    d = g.dim
    spad = np.zeros((g.spad_rows, d), dtype=np.int64)
    acc = np.zeros((g.acc_rows, d), dtype=np.int64)
    out = np.zeros(schedule.shapes["out"], dtype=np.int64)
    sources = {"lhs": np.asarray(lhs, dtype=np.int64), "weight": np.asarray(weight, dtype=np.int64)}
    sources.update({role: np.asarray(v, dtype=np.int64) for role, v in operands.items()})
    resident = np.zeros((d, d), dtype=np.int64)
    target: Preload | None = None
    for op in schedule.ops:
        if isinstance(op, Mvin):
            block = np.zeros((op.rows, d), dtype=np.int64)
            if op.role != "zero":
                rows = op.dram_row + op.row_step * np.arange(op.rows)
                block[:, :op.cols] = sources[op.role][rows, op.dram_col:op.dram_col + op.cols]
            spad[op.spad_row:op.spad_row + op.rows] = block
        elif isinstance(op, AccMvin):
            rows = op.dram_row + op.row_step * np.arange(op.rows)
            acc[op.acc_row:op.acc_row + op.rows] = 0
            acc[op.acc_row:op.acc_row + op.rows, :op.cols] = \
                sources[op.role][rows, op.dram_col:op.dram_col + op.cols]
        elif isinstance(op, Preload):
            if op.weight_row is not None:
                resident = spad[op.weight_row:op.weight_row + d].copy()
            target = op
        elif isinstance(op, Compute):
            if target is None:
                raise UnsupportedConstruct("compute with no preceding preload")
            if target.rows != op.rows:
                raise UnsupportedConstruct("a preload and its compute disagree on the row count")
            product = spad[op.input_row:op.input_row + op.rows] @ resident
            rows = slice(target.acc_row, target.acc_row + op.rows)
            acc[rows] = acc[rows] + product if target.accumulate else product
            target = None
        elif isinstance(op, Mvout):
            out[op.dram_row:op.dram_row + op.rows, op.dram_col:op.dram_col + op.cols] = \
                acc[op.acc_row:op.acc_row + op.rows, :op.cols]
    return out.astype(np.int32)
