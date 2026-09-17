"""Lower a replayed Voyager program onto a weight-stationary array, layer by layer and op for op.

:func:`merlin.baselines.voyager_ir.replay` yields what Voyager's compiled program DOES: which tile is
loaded into which scratchpad slot at which step, which compute consumes it under which interstellar
mapping, and when each output tile is stored. This module turns that trace into an abstract schedule
for a weight-stationary array addressed in DIM x DIM blocks -- ``Mvin`` / ``AccMvin`` / ``Preload`` /
``Compute`` / ``Mvout`` -- plus explicit :class:`HostOp` entries for what the array cannot run, without
choosing anything Voyager chose:

* every Voyager load becomes the loads of exactly its tile's blocks, in the same order, into the
  scratchpad region Voyager's own byte address and slot select (``row = slot_address // row_bytes``);
* every compute walks the loop nest Voyager serialized (interstellar levels, innermost first), so the
  order in which weight blocks are made resident and activation rows streamed is Voyager's;
* every store writes exactly its tile.

A GEMM is lowered as the 1x1 convolution it is: its operands are read as rank-4 NHWC / HWIO tensors
with leading unit dimensions, so one lowering serves both anchors.

The only translations are the ones the target forces, each named in the experiment's concession
register: the output tile lives in the accumulator rather than a scratchpad slot (the array accumulates
there); a K split accumulates in the integer accumulator rather than as a bf16 add (C2); the fused
tail's dequantize -> [relu] -> quantize becomes the accumulator readout -- one scale, an activation,
saturation -- rather than Voyager's bf16 arithmetic (C1); a stride-s input tile is stored phase-split
so a tap's pixels are consecutive rows (C4); a bias moves DRAM -> accumulator at its first use and a
residual is ADDED into the accumulator in accumulator units before the readout (C5); and pooling, the
bf16 classifier and standalone (de)quantization run on the host (C6). Because a partial now lives in
the accumulator, each output tile's split-K group takes the next region of a ring as deep as
Voyager's output buffer, and the output slot the group finishes in is bound to that region (renaming,
not a copy). Anything else refuses with :class:`UnsupportedConstruct`.

Geometry (block edge, scratchpad rows and row width, accumulator rows, and whether an accumulator load
can scale) is passed in by the caller from the target's derived facts; nothing here is a hardware
constant. Dependency-free apart from the optional numpy executors (:func:`execute`,
:func:`execute_model`), which prove a schedule's arithmetic before any simulator runs it.
"""

from __future__ import annotations

import itertools
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .voyager_ir import Box, Copy, FusedCompute, Ref, TensorOp, Trace, UnsupportedConstruct

__all__ = [
    "AccMvin",
    "Compute",
    "Geometry",
    "HostOp",
    "Layer",
    "Mvin",
    "Mvout",
    "Preload",
    "Schedule",
    "execute",
    "execute_model",
    "load_scalars",
    "lower_conv",
    "lower_gemm",
    "lower_model",
]

#: Anchors a GEMM fusion may start with.
_GEMM_ANCHORS = ("quantized_ops::linear", "aten::linear", "aten::matmul", "quantized_ops::matmul")
#: Anchors a convolution fusion may start with.
_CONV_ANCHORS = ("quantized_ops::conv2d", "aten::conv2d")
_DEQUANTIZE = "quantized_ops::dequantize"
_QUANTIZE = "quantized_ops::quantize"
_ADDS = ("aten::add", "aten::add_")
_RELUS = ("aten::relu", "aten::relu_")
#: Convolution loops (a GEMM uses OX, IC and OC).
_CONV_LOOPS = ("LOOP_OY", "LOOP_OX", "LOOP_FY", "LOOP_FX", "LOOP_IC", "LOOP_OC")
#: Host op the bridge inserts to bring a residual into accumulator units, and the one that turns an
#: integer readout into the floating-point tensor Voyager's graph expects there.
REQUANTIZE = "merlin::requantize"
DEQUANTIZE_ACC = "merlin::dequantize"
_INT32 = (-(1 << 31), (1 << 31) - 1)
_INT8 = (-(1 << 7), (1 << 7) - 1)


@dataclass(frozen=True)
class Geometry:
    """The array and its stores, in the units a backend addresses them. Derived by the caller."""

    dim: int  # array edge = block edge
    spad_rows: int  # scratchpad rows (all banks)
    spad_row_bytes: int  # bytes per scratchpad row
    acc_rows: int  # accumulator rows (all banks)
    scaled_acc_loads: bool = False  # an accumulator load can multiply by a scale on the way in


@dataclass(frozen=True)
class Mvin:
    role: str  # "lhs" | "weight" | "zero" (a zero page: the halo of a padded tile)
    dram_row: int
    dram_col: int
    rows: int
    cols: int
    spad_row: int
    row_step: int = 1  # DRAM rows between consecutive block rows (a stride-s gather uses s)


@dataclass(frozen=True)
class AccMvin:
    """Load integer rows straight into the accumulator: ``bias`` (``row_step`` 0 broadcasts its one
    DRAM row) or a ``residual`` tile. ``accumulate`` adds onto the rows instead of overwriting them;
    ``scale`` multiplies each value on the way in (only when the geometry says the target can)."""

    role: str
    dram_row: int
    dram_col: int
    rows: int
    cols: int
    acc_row: int
    row_step: int = 0
    accumulate: bool = False
    scale: float = 1.0


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
    fresh_weights: bool  # the first compute after a weight change


@dataclass(frozen=True)
class Mvout:
    """Store accumulator rows. ``out_dtype`` "int8" applies the readout -- ``relu``, then ``scale``,
    then saturation; "int32" reads the raw accumulator and carries neither."""

    role: str  # "out"
    dram_row: int
    dram_col: int
    rows: int
    cols: int
    acc_row: int
    scale: float = 1.0
    relu: bool = False
    out_dtype: str = "int32"


@dataclass(frozen=True)
class HostOp:
    """An operation the host runs on whole DRAM tensors (concession C6). ``inputs``/``outputs`` map
    an operand name to a DRAM tensor; ``attrs`` carries its scalars with every scale resolved."""

    target: str
    inputs: dict
    outputs: dict
    attrs: dict


Op = Mvin | AccMvin | Preload | Compute | Mvout


@dataclass
class Schedule:
    """The lowered program plus what it was lowered from.

    ``shapes`` gives each DRAM role as the 2-D matrix the ops address: an NHWC activation as
    ``[N*H*W, C]``, an HWIO weight as ``[KH*KW*Cin, Cout]`` (a GEMM operand is its own matrix), a bias
    as ``[1, C]``.
    """

    ops: list[Op]
    shapes: dict[str, tuple[int, int]]  # role -> logical [rows, cols]
    dram_nodes: dict[str, str]  # role -> DRAM tensor
    geometry: Geometry
    notes: list[str] = field(default_factory=list)

    def count(self, kind: type) -> int:
        return sum(1 for op in self.ops if isinstance(op, kind))


@dataclass(frozen=True)
class Layer:
    """One entry of a lowered model, in program order."""

    name: str  # Voyager's layer name
    kind: str  # "conv" | "gemm" | "host"
    program: Schedule | HostOp
    reads: tuple[str, ...]  # DRAM tensors read
    writes: tuple[str, ...]  # DRAM tensors written
    readout: Mapping[str, Any] = field(default_factory=dict)


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def _f32(value: float) -> float:
    """``value`` rounded to IEEE single precision, the width of the target's scale registers."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _as4(shape: tuple[int, ...], what: str) -> tuple[int, ...]:
    """A rank <= 4 shape read as NHWC / HWIO, leading unit dimensions added."""
    shape = tuple(shape)
    if len(shape) > 4 or not shape:
        raise UnsupportedConstruct(f"{what} has shape {shape}; only rank 1-4 tensors are lowered")
    return (1,) * (4 - len(shape)) + shape


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


def _window4(copy: Copy) -> tuple[tuple[int, ...], tuple[int, ...]]:
    start, sizes = _copy_window(copy)
    if len(sizes) > 4:
        raise UnsupportedConstruct(f"{copy.name}: a {len(sizes)}-D tile copy is not lowered")
    lead = 4 - len(sizes)
    return (0,) * lead + start, (1,) * lead + sizes


def _scale_ref(call, what: str) -> str:
    """The node naming a per-tensor scale; anything richer refuses."""
    kw = call.kwargs
    for extra in ("zero_point", "axes", "block_size", "input_qmap", "output_qmap", "output_code"):
        if kw.get(extra) is not None:
            raise UnsupportedConstruct(f"{call.name}: a {what} with {extra} is not a per-tensor symmetric scale")
    scale = kw.get("scale")
    if not isinstance(scale, Ref) or scale.box.shape not in ((), (1,)):
        raise UnsupportedConstruct(f"{call.name}: {what} scale is not one scalar")
    return scale.box.node


@dataclass
class _Tail:
    """What a fused chain after its anchor asks of the readout."""

    acc_scale: str | None = None  # dequantize(acc) scale node
    residual: Ref | None = None  # on-chip residual tile
    residual_scale: str | None = None
    combine: Ref | None = None  # the partial a K split adds into
    relu: bool = False
    out_scale: str | None = None  # quantize scale node (int8 output) or None


def _value_kind(ref: Any, values: dict[str, str]) -> str | None:
    if not isinstance(ref, Ref):
        return None
    if ref.box.level is None:
        return values.get(ref.box.node)
    return "partial" if ref.box.on_chip else None


def _parse_tail(comp: FusedCompute, standalone: bool) -> _Tail:
    """Read the chain structurally. A GEMM/conv chain starts from the anchor's accumulator; a
    standalone epilogue (anchor ``dequantize``) starts from an on-chip residual and adds a partial."""
    tail = _Tail()
    values: dict[str, str] = {}
    calls = comp.chain if standalone else comp.chain[1:]
    if not standalone:
        values[comp.anchor.name] = "acc"
    finished = False  # a relu or quantize has run: no add may follow
    for call in calls:
        kw = call.kwargs
        if call.target == _DEQUANTIZE:
            src = kw.get("input")
            kind = _value_kind(src, values)
            if kind == "acc":
                tail.acc_scale = _scale_ref(call, "dequantize")
                values[call.name] = "accf"
            elif isinstance(src, Ref) and src.box.on_chip:
                if tail.residual is not None:
                    raise UnsupportedConstruct(f"{comp.name}: two residual operands")
                tail.residual, tail.residual_scale = src, _scale_ref(call, "dequantize")
                values[call.name] = "resf"
            else:
                raise UnsupportedConstruct(f"{comp.name}: dequantize of {src!r} is not modelled")
        elif call.target in _ADDS:
            if finished or kw.get("alpha", 1) not in (1, None):
                raise UnsupportedConstruct(f"{comp.name}: an add after the activation, or scaled")
            a, b = kw.get("input"), kw.get("other")
            kinds = sorted(str(_value_kind(v, values)) for v in (a, b))
            if kinds == ["accf", "partial"] and not standalone:
                tail.combine = a if _value_kind(a, values) == "partial" else b
            elif kinds == ["accf", "resf"] and not standalone:
                pass
            elif kinds == ["partial", "resf"] and standalone:
                tail.combine = a if _value_kind(a, values) == "partial" else b
            else:
                raise UnsupportedConstruct(f"{comp.name}: an add of {kinds} is not modelled")
            values[call.name] = "accf"
        elif call.target in _RELUS:
            if _value_kind(kw.get("input"), values) != "accf":
                raise UnsupportedConstruct(f"{comp.name}: relu of a value that is not the sum")
            tail.relu, finished = True, True
            values[call.name] = "accf"
        elif call.target == _QUANTIZE:
            if _value_kind(kw.get("input"), values) != "accf" or tail.out_scale is not None:
                raise UnsupportedConstruct(f"{comp.name}: quantize of a value that is not the sum")
            for extra in ("zero_point", "axes", "block_size", "output_code"):
                if kw.get(extra) is not None:
                    raise UnsupportedConstruct(f"{comp.name}: a quantize with {extra}")
            scale = kw.get("scale")
            if not isinstance(scale, Ref) or scale.box.shape not in ((), (1,)):
                raise UnsupportedConstruct(f"{comp.name}: quantize scale is not one scalar")
            tail.out_scale, finished = scale.box.node, True
            values[call.name] = "q"
        else:
            raise UnsupportedConstruct(f"{comp.name}: fused op {call.target} has no readout mapping")
    if standalone and (tail.residual is None or tail.combine is None):
        raise UnsupportedConstruct(f"{comp.name}: a standalone epilogue that is not residual + partial")
    if tail.residual is not None and tail.acc_scale is None and not standalone:
        raise UnsupportedConstruct(f"{comp.name}: a residual added to an undequantized accumulator")
    return tail


@dataclass(frozen=True)
class _Readout:
    scale: float = 1.0
    relu: bool = False
    out_dtype: str = "int32"


class _Lowerer:
    """Lower one layer. Placement keeps each tile inside the byte region Voyager allotted it; the
    output tile lives in an accumulator region chosen by split-K group (see the module doc)."""

    def __init__(
        self,
        trace: Trace,
        geometry: Geometry,
        anchors: tuple[str, ...],
        what: str,
        scalars: Mapping[str, float] | None = None,
    ):
        self.trace, self.g, self.anchors, self.what = trace, geometry, anchors, what
        self.scalars = scalars
        self.ops: list[Op] = []
        self.notes: list[str] = []
        self.roles: dict[str, str] = {}  # on-chip node -> lhs|weight|bias|residual|out|partial
        self.dram: dict[str, Box] = {}  # role -> Voyager DRAM tensor
        self.dram_names: dict[str, str] = {}  # role -> DRAM tensor the ops address
        self.stride: dict[str, tuple[int, int]] = {}  # input node -> the stride its computes use
        self.bias_start: dict[tuple[str, int], int] = {}  # (bias node, slot) -> first channel held
        self.residual_window: dict[tuple[str, int], tuple] = {}  # (node, slot) -> (start, sizes)
        self.tile: tuple[int, ...] = ()
        self.region_rows = 0
        self.regions = 0
        self.bound: dict[tuple[str, int], int] = {}  # (out node, slot) -> the region it names
        self.readouts: dict[tuple[str, int], _Readout] = {}
        self.unstored: set[int] = set()  # regions bound but not yet stored
        self.group: dict[str, Any] | None = None  # the split-K group being accumulated
        self.groups = 0
        self.host_before: dict[str, HostOp] = {}  # int32 buffer -> the requantization feeding it
        self.dequant_after: dict[str, tuple[float, bool, str]] = {}  # out node -> (scale, relu, dtype)
        self.readout_params: dict[str, Any] = {}
        self.pass_mode = False  # the output tile cannot live in the accumulator
        self.pass_count = 0

    def _note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    def _scalar(self, node: str) -> float:
        if self.scalars is None:
            raise UnsupportedConstruct(
                f"scale {node} is needed but no scale values were given (lower the program with lower_model)"
            )
        try:
            return float(self.scalars[node])
        except KeyError:
            raise UnsupportedConstruct(f"no value for scale {node}") from None

    # placement ------------------------------------------------------------------------------
    def _region(self, ref: Ref) -> tuple[int, int]:
        """(first row, rows) of the scratchpad region Voyager allotted ``ref``'s slot."""
        base, rem = divmod(ref.slot_address, self.g.spad_row_bytes)
        if rem:
            raise UnsupportedConstruct(
                f"{ref.box.node} slot address {ref.slot_address} is not row aligned ({self.g.spad_row_bytes} B rows)"
            )
        elements = 1
        for extent in ref.box.shape:
            elements *= extent
        return base, _ceil(elements, self.g.spad_row_bytes)

    def _checked(self, ref: Ref, row: int, rows: int) -> int:
        base, allotted = self._region(ref)
        if row < base or row + rows > base + allotted or row + rows > self.g.spad_rows:
            raise UnsupportedConstruct(
                f"{ref.box.node} rows {row}..{row + rows} leave the "
                f"{allotted}-row region Voyager allotted at row {base}"
            )
        return row

    def input_row(self, ref: Ref, cb: int, h: int, w: int) -> int:
        """Row of pixel (h, w), channel block ``cb``, of the input tile in ``ref``'s slot: one row per
        pixel, channel blocks outermost, and a stride-s tile phase-split along W (C4) -- columns
        p, p+s, ... are consecutive rows, so a tap's pixels are a contiguous stream."""
        _, th, tw, _ = _as4(ref.box.shape, ref.box.node)
        s = self.stride[ref.box.node][1]
        base, _ = self._region(ref)
        before = sum(_ceil(tw - q, s) for q in range(w % s))
        return base + (cb * th + h) * tw + before + w // s

    def weight_row(self, ref: Ref, fy: int, fx: int, icb: int, ocb: int) -> int:
        d = self.g.dim
        _, kw, ci, co = _as4(ref.box.shape, ref.box.node)
        base, _ = self._region(ref)
        return self._checked(ref, base + (((fy * kw + fx) * (ci // d) + icb) * (co // d) + ocb) * d, d)

    # roles ------------------------------------------------------------------------------------
    def _standalone(self, comp: FusedCompute) -> bool:
        return comp.anchor.target == _DEQUANTIZE

    def _set_role(self, node: str, role: str) -> None:
        if self.roles.setdefault(node, role) != role:
            raise UnsupportedConstruct(f"{node} is both {self.roles[node]} and {role}")

    def bind_roles(self, computes: list[FusedCompute]) -> None:
        d = self.g.dim
        if self.g.spad_row_bytes != d:
            raise UnsupportedConstruct(
                f"a {self.g.spad_row_bytes} B scratchpad row does not hold one {d}-channel row of 8-bit pixels"
            )
        stored = {c.src.box.node for c in self.trace.of(Copy) if c.is_store}
        for comp in computes:
            standalone = self._standalone(comp)
            anchor = comp.anchor
            if not standalone and anchor.target not in self.anchors:
                raise UnsupportedConstruct(f"{comp.name}: anchor {anchor.target!r} is not a {self.what}")
            tail = _parse_tail(comp, standalone)
            if tail.residual is not None:
                self._set_role(tail.residual.box.node, "residual")
            if len(comp.destinations) != 1:
                raise UnsupportedConstruct(f"{comp.name}: {len(comp.destinations)} destinations")
            dest = comp.destinations[0]
            self._set_role(dest.box.node, "out" if dest.box.node in stored else "partial")
            shape = _as4(dest.box.shape, dest.box.node)
            if self.tile and shape != self.tile:
                raise UnsupportedConstruct(f"{comp.name}: output tiles {shape} and {self.tile} in one layer")
            self.tile = shape
            if dest.box.node in stored:
                self.regions = max(self.regions, dest.box.bank_count)
            if standalone:
                continue
            if anchor.kwargs.get("groups", 1) != 1:
                raise UnsupportedConstruct(f"{comp.name}: grouped convolution is not lowered")
            if tuple(anchor.kwargs.get("padding", (0, 0))) != (0, 0):
                raise UnsupportedConstruct(
                    f"{comp.name}: padding inside the compute is not lowered (Voyager pads in the copy)"
                )
            for kw, role in (("input", "lhs"), ("weight", "weight"), ("bias", "bias")):
                ref = anchor.kwargs.get(kw)
                if ref is None and kw == "bias":
                    continue
                if not isinstance(ref, Ref):
                    raise UnsupportedConstruct(f"{comp.name}: {kw} operand is not a tensor reference")
                if ref.output_shape and tuple(ref.output_shape) != tuple(ref.box.shape):
                    raise UnsupportedConstruct(f"{comp.name}: {kw} reads a sub-window of its buffer")
                self._set_role(ref.box.node, role)
                if ref.box.dtype not in ("int8", "uint8") and role != "bias":
                    raise UnsupportedConstruct(f"{comp.name}: {kw} is {ref.box.dtype}; only 8-bit operands are lowered")
            stride = tuple(anchor.kwargs.get("stride", (1, 1)))
            if self.stride.setdefault(anchor.kwargs["input"].box.node, stride) != stride:
                raise UnsupportedConstruct(f"{comp.name}: one input buffer read at two strides")
        for copy in self.trace.of(Copy):
            if copy.is_load and copy.dst.box.node in self.roles:
                self.dram.setdefault(self.roles[copy.dst.box.node], copy.src.box)
            elif copy.is_store and self.roles.get(copy.src.box.node) == "out":
                self.dram.setdefault("out", copy.dst.box)
        missing = {"lhs", "weight", "out"} - set(self.dram)
        if missing:
            raise UnsupportedConstruct(f"no DRAM transfer feeds or drains {sorted(missing)}")
        self.dram_names = {role: box.node for role, box in self.dram.items() if role != "residual"}
        n, oh, ow, oc = self.tile
        if n != 1 or oc % d:
            raise UnsupportedConstruct(f"output tile {self.tile} is not one image of whole {d}-channel blocks")
        self.region_rows = oc // d * oh * ow
        fit = self.g.acc_rows // self.region_rows
        if fit == 0:
            # The tile cannot live in the accumulator whole: each compute runs in passes along its
            # outermost output loops (_compute_in_passes), or refuses there.
            self.pass_mode = True
            self._note(
                f"Voyager's output tile ({self.region_rows} rows) is larger than the "
                f"accumulator ({self.g.acc_rows} rows): each compute runs in passes along its "
                "outermost output loops, each pass stored as soon as it completes (C7)"
            )
            return
        if fit < self.regions:
            self._note(
                f"Voyager's {self.regions}-slot output buffer is deeper than the {fit} output "
                "tile(s) the accumulator holds: a tile's store is issued before the next tile "
                "first writes its region (C7)"
            )
            self.regions = fit

    def views(self) -> dict[str, tuple[int, int]]:
        shapes = {}
        for role, box in self.dram.items():
            if role == "bias":
                shapes["bias"] = (1, box.shape[-1])
            else:
                n, h, w, c = _as4(box.shape, box.node)
                shapes[role] = (n * h * w, c)
        return shapes

    # loads and stores -------------------------------------------------------------------------
    def load(self, copy: Copy) -> None:
        role = self.roles[copy.dst.box.node]
        if role == "bias":
            start, sizes = _copy_window(copy)
            if len(sizes) != 1:
                raise UnsupportedConstruct(f"{copy.name}: a {len(sizes)}-D bias tile")
            self.bias_start[(copy.dst.box.node, copy.dst.slot)] = start[0]
            self._note(
                "bias rows move DRAM -> accumulator at their first use: the target has no "
                "scratchpad -> accumulator path (C5)"
            )
            return
        start, sizes = _window4(copy)
        if role == "lhs":
            self._load_input(copy, start, sizes)
        elif role == "weight":
            self._load_weight(copy, start, sizes)
        elif role == "residual":
            self.residual_window[(copy.dst.box.node, copy.dst.slot)] = (start, sizes)
            self._note("a residual tile moves DRAM -> accumulator, added in accumulator units before the readout (C5)")
        else:
            raise UnsupportedConstruct(f"{copy.name}: a load into the {role} buffer")

    def _load_input(self, copy: Copy, start: tuple[int, ...], sizes: tuple[int, ...]) -> None:
        d = self.g.dim
        n0, h0, w0, c0 = start
        tn, th, tw, tc = sizes
        _, height, width, channels = _as4(self.dram["lhs"].shape, "lhs")
        if tn != 1 or tc % d or c0 % d or c0 + tc > channels:
            raise UnsupportedConstruct(
                f"{copy.name}: input tile {sizes} at {start} is not one image of whole {d}-channel blocks"
            )
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

    def _emit_input(
        self, copy: Copy, cb: int, h: int, run: list[int], valid: bool, start: tuple[int, ...], s: int
    ) -> None:
        d = self.g.dim
        n0, h0, w0, c0 = start
        _, height, width, _ = _as4(self.dram["lhs"].shape, "lhs")
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
        _, kw, ci, co = _as4(self.dram["weight"].shape, "weight")
        if tic % d or toc % d or ic0 % d or oc0 % d or ic0 + tic > ci or oc0 + toc > co:
            raise UnsupportedConstruct(f"{copy.name}: weight tile {sizes} at {start} is not whole {d}x{d} blocks")
        for fy in range(tkh):
            for fx in range(tkw):
                for icb in range(tic // d):
                    for ocb in range(toc // d):
                        self.ops.append(
                            Mvin(
                                "weight",
                                ((fy0 + fy) * kw + fx0 + fx) * ci + ic0 + icb * d,
                                oc0 + ocb * d,
                                d,
                                d,
                                self.weight_row(copy.dst, fy, fx, icb, ocb),
                            )
                        )

    def _pixel_runs(self, start: tuple[int, ...], shape: tuple[int, ...], base: int):
        """Runs (dram row, acc row, rows) over an output-tile-shaped window: consecutive pixels that
        are consecutive both in the DRAM matrix ``shape`` views and in the accumulator region."""
        d = self.g.dim
        n0, oh0, ow0, _ = start
        _, height, width, _ = shape
        _, toh, tow, toc = self.tile
        pixels = toh * tow
        for ocb in range(toc // d):
            run: list[int] | None = None
            for oy in range(toh):
                for ox in range(tow):
                    acc = base + ocb * pixels + oy * tow + ox
                    dram = (n0 * height + oh0 + oy) * width + ow0 + ox
                    if run and acc == run[1] + run[2] and dram == run[0] + run[2] and run[2] < d:
                        run[2] += 1
                        continue
                    if run:
                        yield ocb, run[0], run[1], run[2]
                    run = [dram, acc, 1]
            if run:
                yield ocb, run[0], run[1], run[2]

    def store(self, copy: Copy) -> None:
        d = self.g.dim
        key = (copy.src.box.node, copy.src.slot)
        region = self.bound.pop(key, None)
        if region is None:
            raise UnsupportedConstruct(f"{copy.name}: stores {key[0]} slot {key[1]}, which no finished compute names")
        readout = self.readouts.pop(key)
        self.unstored.discard(region)
        if self.group is not None and self.group["bound"] == key:
            self.group = None
        start, sizes = _window4(copy)
        out = _as4(self.dram["out"].shape, "out")
        if tuple(sizes) != self.tile or any(s + t > e for s, t, e in zip(start, sizes, out)):
            raise UnsupportedConstruct(f"{copy.name}: a partial output tile {sizes} at {start}")
        for ocb, dram, acc, rows in self._pixel_runs(start, out, region * self.region_rows):
            self.ops.append(
                Mvout(
                    "out",
                    dram,
                    start[3] + ocb * d,
                    rows,
                    d,
                    acc,
                    scale=readout.scale,
                    relu=readout.relu,
                    out_dtype=readout.out_dtype,
                )
            )

    # computes ---------------------------------------------------------------------------------
    def _open_group(self, comp: FusedCompute) -> None:
        if self.group is not None:
            if self.group["bound"] is None:
                raise UnsupportedConstruct(
                    f"{comp.name} starts an output tile while the previous K split never reached an output slot"
                )
            self.group = None
        region = self.groups % self.regions
        if region in self.unstored:
            self._store_early(comp, region)
        self.groups += 1
        self.group = {"region": region, "initialized": set(), "partial": None, "bound": None, "acc_scale": None}

    def _store_early(self, comp: FusedCompute, region: int) -> None:
        """Issue, now, the stores Voyager issues later from ``region``: the ring is shallower than
        Voyager's output buffer, and the data in the region is final once its group is bound."""
        for key in [k for k, r in self.bound.items() if r == region]:
            for index in range(self._at + 1, len(self._events)):
                event = self._events[index]
                if isinstance(event, Copy) and event.is_store and (event.src.box.node, event.src.slot) == key:
                    self.store(event)
                    self._done.add(index)
                    break
            else:
                raise UnsupportedConstruct(
                    f"{comp.name}: accumulator region {region} holds {key[0]} slot {key[1]}, which is never stored"
                )

    def _continue_group(self, comp: FusedCompute, partial: Ref) -> None:
        if self.group is None or self.group["partial"] != (partial.box.node, partial.slot):
            raise UnsupportedConstruct(
                f"{comp.name}: adds into {partial.box.node} slot {partial.slot}, which is not the open K split"
            )
        self._note(
            "K splits combine in the integer accumulator (C2); each output tile's split-K "
            "group renames onto the next accumulator region of the output ring"
        )

    def _readout(self, comp: FusedCompute, tail: _Tail, acc_scale: str | None, out: Box) -> _Readout:
        if tail.out_scale is None:
            if out.dtype not in ("int32",):
                if self.scalars is not None:
                    if acc_scale is None:
                        raise UnsupportedConstruct(f"{comp.name}: a {out.dtype} output with no dequantize scale")
                    self.dequant_after[out.node] = (_f32(self._scalar(acc_scale)), tail.relu, out.dtype)
                    self._note("an unquantized output is read out as int32 and dequantized on the host (C6)")
                else:
                    self._note("the readout of a floating-point output is the caller's (C1)")
            return _Readout()
        if self.scalars is None:
            self._note("the int8 readout scale is the caller's (C1): no scale values were given")
            return _Readout()
        if acc_scale is None:
            raise UnsupportedConstruct(f"{comp.name}: quantize with no dequantize scale")
        self._note(
            "dequantize -> [relu] -> quantize becomes the accumulator readout: relu, one fp32 "
            "scale (acc scale / output scale), saturation to int8 (C1)"
        )
        return _Readout(_f32(self._scalar(acc_scale) / self._scalar(tail.out_scale)), tail.relu, "int8")

    def _residual_source(
        self, comp: FusedCompute, tail: _Tail, acc_scale: str | None
    ) -> tuple[str, float, tuple, tuple]:
        """(DRAM tensor the accumulator load reads, its load scale, the residual tile's window start,
        the DRAM tensor's rank-4 shape); registers the host requantization when loads cannot scale."""
        key = (tail.residual.box.node, tail.residual.slot)
        window = self.residual_window.get(key)
        if window is None:
            raise UnsupportedConstruct(f"{comp.name}: its residual slot was never loaded")
        start, sizes = window
        if tuple(sizes) != self.tile:
            raise UnsupportedConstruct(f"{comp.name}: residual tile {sizes} is not the output tile")
        source = self.dram["residual"]
        if acc_scale is None:
            raise UnsupportedConstruct(f"{comp.name}: a residual with no accumulator scale")
        ratio = _f32(self._scalar(tail.residual_scale) / self._scalar(acc_scale))
        if self.g.scaled_acc_loads:
            name, scale = source.node, ratio
        else:
            name, scale = f"{source.node}__acc_i32", 1.0
            op = HostOp(
                REQUANTIZE,
                {"input": source.node},
                {"output": name},
                {"scale": ratio, "rounding": "nearest_even", "dtype": "int32"},
            )
            if self.host_before.setdefault(name, op) != op:
                raise UnsupportedConstruct(f"{comp.name}: one residual requantized at two scales")
            self._note(
                "the target's accumulator loads cannot scale, so the host brings the residual "
                "into accumulator units first (C6)"
            )
        if self.dram_names.setdefault("residual", name) != name:
            raise UnsupportedConstruct(f"{comp.name}: two residual tensors in one layer")
        return name, scale, start, _as4(source.shape, source.node)

    def _emit_residual(self, comp: FusedCompute, tail: _Tail, acc_scale: str | None) -> None:
        d = self.g.dim
        _, scale, start, shape = self._residual_source(comp, tail, acc_scale)
        group = self.group
        base = group["region"] * self.region_rows
        for ocb, dram, acc, rows in self._pixel_runs(start, shape, base):
            ready = {r in group["initialized"] for r in range(acc, acc + rows)}
            if len(ready) != 1:
                raise UnsupportedConstruct(f"{comp.name}: a residual run over partly written rows")
            self.ops.append(
                AccMvin(
                    "residual", dram, start[3] + ocb * d, rows, d, acc, row_step=1, accumulate=ready.pop(), scale=scale
                )
            )
            group["initialized"].update(range(acc, acc + rows))

    def _bind(self, comp: FusedCompute, dest: Ref, tail: _Tail, acc_scale: str | None) -> None:
        group = self.group
        group["partial"] = (dest.box.node, dest.slot)
        if self.roles[dest.box.node] != "out":
            return
        key = (dest.box.node, dest.slot)
        if group["bound"] not in (None, key):
            raise UnsupportedConstruct(f"{comp.name}: one K split finishes in two output slots")
        group["bound"] = key
        self.bound[key] = group["region"]
        self.unstored.add(group["region"])
        readout = self._readout(comp, tail, acc_scale, self.dram["out"])
        self.readouts[key] = readout
        params = {"scale": readout.scale, "relu": readout.relu, "out_dtype": readout.out_dtype}
        if self.readout_params.setdefault("store", params) != params:
            raise UnsupportedConstruct(f"{comp.name}: two readouts in one layer")

    def epilogue(self, comp: FusedCompute) -> None:
        """A standalone residual add: rename it onto the open group's accumulator region."""
        if self.pass_mode:
            raise UnsupportedConstruct(
                f"{comp.name}: the output tile is larger than the accumulator "
                "and a standalone epilogue cannot run in passes"
            )
        tail = _parse_tail(comp, standalone=True)
        self._continue_group(comp, tail.combine)
        acc_scale = self.group["acc_scale"]
        self._emit_residual(comp, tail, acc_scale)
        self._bind(comp, comp.destinations[0], tail, acc_scale)
        self._note(
            f"{comp.name}: the standalone residual add is renamed onto the accumulator region its K split finished in"
        )

    def compute(self, comp: FusedCompute) -> None:
        if self._standalone(comp):
            self.epilogue(comp)
            return
        d = self.g.dim
        anchor = comp.anchor
        lhs, weight, bias = anchor.kwargs["input"], anchor.kwargs["weight"], anchor.kwargs.get("bias")
        dest = comp.destinations[0]
        _, th, tw, tc = _as4(lhs.box.shape, lhs.box.node)
        tkh, tkw, tic, toc = _as4(weight.box.shape, weight.box.node)
        _, toh, tow, _ = self.tile
        sh, sw = anchor.kwargs.get("stride", (1, 1))
        dh, dw = anchor.kwargs.get("dilation", (1, 1))
        if tic != tc or toc != self.tile[3] or tic % d:
            raise UnsupportedConstruct(
                f"{comp.name}: input {lhs.box.shape}, weight {weight.box.shape}, output {self.tile} do not compose"
            )
        if (toh - 1) * sh + (tkh - 1) * dh >= th or (tow - 1) * sw + (tkw - 1) * dw >= tw:
            raise UnsupportedConstruct(
                f"{comp.name}: the input tile {lhs.box.shape} does not hold the {toh}x{tow} output window"
            )
        tail = _parse_tail(comp, standalone=False)
        want = {
            "LOOP_OY": toh,
            "LOOP_OX": tow,
            "LOOP_FY": tkh,
            "LOOP_FX": tkw,
            "LOOP_IC": tic // d,
            "LOOP_OC": toc // d,
        }
        if self.pass_mode:
            self._compute_in_passes(comp, tail, self._nest(comp, want), (sh, sw), (dh, dw))
            return
        if tail.combine is not None:
            self._continue_group(comp, tail.combine)
        else:
            self._open_group(comp)
        group = self.group
        if tail.acc_scale is not None:
            if group["acc_scale"] not in (None, tail.acc_scale):
                raise UnsupportedConstruct(f"{comp.name}: K parts dequantize at two scales")
            group["acc_scale"] = tail.acc_scale
        base, pixels = group["region"] * self.region_rows, toh * tow
        initialized: set[int] = group["initialized"]
        if bias is not None:
            if tail.combine is not None:
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
        if tail.residual is not None:
            self._emit_residual(comp, tail, group["acc_scale"])
        order = self._nest(comp, want)
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
            if (
                run
                and run[0] == key
                and in_row == run[1] + run[3]
                and acc_row == run[2] + run[3]
                and run[3] < d
                and (acc_row in initialized) == run[4]
            ):
                run[3] += 1
                continue
            if run:
                resident = self._flush(run, weight, resident, initialized)
            # Read "already written" AFTER the flush, never before it: the run just flushed may be
            # what initialised these rows, and a run starting on them must accumulate, not overwrite.
            run = [key, in_row, acc_row, 1, acc_row in initialized]
        if run:
            self._flush(run, weight, resident, initialized)
        self._bind(comp, dest, tail, group["acc_scale"])

    def _nest(self, comp: FusedCompute, want: dict[str, int]) -> list[tuple[str, int]]:
        """Voyager's nest, outermost first, checked to cover the tile exactly. A loop at several
        levels composes: the inner level's extent is the outer level's place value. The batch loop
        of a one-image tile carries nothing and is dropped."""
        order = [
            (loop, bound)
            for level in reversed(comp.tiling)
            for loop, bound in reversed(level)
            if not (loop == "LOOP_ON" and bound == 1)
        ]
        extents = dict.fromkeys(_CONV_LOOPS, 1)
        for loop, bound in order:
            if loop not in extents:
                raise UnsupportedConstruct(f"{comp.name}: loop {loop} has no {self.what} meaning")
            extents[loop] *= bound
        if extents != want:
            raise UnsupportedConstruct(f"{comp.name}: mapping {order} does not cover {want}")
        if not order or order[-1][0] not in ("LOOP_OY", "LOOP_OX"):
            raise UnsupportedConstruct(f"{comp.name}: innermost loop {order[-1:]} is not a pixel stream")
        return order

    def _take_store(self, comp: FusedCompute, dest: Ref) -> Copy:
        """The store that drains ``dest``'s slot, taken now: it will not be replayed later."""
        key = (dest.box.node, dest.slot)
        for index in range(self._at + 1, len(self._events)):
            event = self._events[index]
            if isinstance(event, Copy) and event.is_store and (event.src.box.node, event.src.slot) == key:
                self._done.add(index)
                return event
        raise UnsupportedConstruct(f"{comp.name}: its output tile is never stored")

    def _runs(self, cells):
        """Merge (dram row, acc row) cells, in order, into runs of consecutive rows of <= one block."""
        run = None
        for dram, acc in cells:
            if run and dram == run[0] + run[2] and acc == run[1] + run[2] and run[2] < self.g.dim:
                run[2] += 1
                continue
            if run:
                yield tuple(run)
            run = [dram, acc, 1]
        if run:
            yield tuple(run)

    def _compute_in_passes(
        self,
        comp: FusedCompute,
        tail: _Tail,
        order: list[tuple[str, int]],
        stride: tuple[int, int],
        dilation: tuple[int, int],
    ) -> None:
        """Run a compute whose output tile the accumulator cannot hold.

        Voyager's nest is split at its outermost OUTPUT loops: iterations that share those digits
        write a disjoint set of output rows and are consecutive in Voyager's order, so each such pass
        accumulates completely in one accumulator region -- bias and residual first -- and is stored
        as soon as it completes. The only change is that the tile's store is issued in pieces,
        earlier (C7). A K split, or a reduction loop outermost, cannot run this way and refuses.
        """
        d = self.g.dim
        anchor = comp.anchor
        lhs, weight, bias = anchor.kwargs["input"], anchor.kwargs["weight"], anchor.kwargs.get("bias")
        dest = comp.destinations[0]
        sh, sw = stride
        dh, dw = dilation
        if tail.combine is not None or self.roles[dest.box.node] != "out":
            raise UnsupportedConstruct(
                f"{comp.name}: the output tile is larger than the accumulator "
                "and the compute is a K split, which passes cannot run"
            )
        prefix = 0
        while prefix < len(order) and order[prefix][0] in ("LOOP_OC", "LOOP_OY", "LOOP_OX"):
            prefix += 1
        if prefix == 0:
            raise UnsupportedConstruct(
                f"{comp.name}: the output tile is larger than the accumulator "
                f"and its outermost loop {order[0][0]} is a reduction"
            )
        iterations: list[tuple] = []
        passes: dict[tuple, dict] = {}
        for digits in itertools.product(*(range(bound) for _, bound in order)):
            coord = dict.fromkeys(_CONV_LOOPS, 0)
            place = dict.fromkeys(_CONV_LOOPS, 1)
            for (loop, bound), digit in zip(reversed(order), reversed(digits)):
                coord[loop] += digit * place[loop]
                place[loop] *= bound
            pid = digits[:prefix]
            key = (coord["LOOP_FY"], coord["LOOP_FX"], coord["LOOP_IC"], coord["LOOP_OC"])
            iterations.append((pid, key, coord["LOOP_OY"], coord["LOOP_OX"]))
            cells = passes.setdefault(pid, {"ocbs": set(), "pixels": set()})
            cells["ocbs"].add(key[3])
            cells["pixels"].add((coord["LOOP_OY"], coord["LOOP_OX"]))
        rows = 0
        for cells in passes.values():
            cells["ocbs"] = sorted(cells["ocbs"])
            cells["pixels"] = sorted(cells["pixels"])
            rows = max(rows, len(cells["ocbs"]) * len(cells["pixels"]))
        fit = self.g.acc_rows // rows
        if fit == 0:
            raise UnsupportedConstruct(
                f"{comp.name}: even one pass ({rows} rows) is larger than the accumulator ({self.g.acc_rows} rows)"
            )
        store = self._take_store(comp, dest)
        out_start, out_sizes = _window4(store)
        out_shape = _as4(self.dram["out"].shape, "out")
        if tuple(out_sizes) != self.tile or any(s + z > e for s, z, e in zip(out_start, out_sizes, out_shape)):
            raise UnsupportedConstruct(f"{store.name}: a partial output tile {out_sizes}")
        readout = self._readout(comp, tail, tail.acc_scale, self.dram["out"])
        params = {"scale": readout.scale, "relu": readout.relu, "out_dtype": readout.out_dtype}
        if self.readout_params.setdefault("store", params) != params:
            raise UnsupportedConstruct(f"{comp.name}: two readouts in one layer")
        residual = self._residual_source(comp, tail, tail.acc_scale) if tail.residual else None
        oc0 = None
        if bias is not None:
            oc0 = self.bias_start.get((bias.box.node, bias.slot))
            if oc0 is None:
                raise UnsupportedConstruct(f"{comp.name}: its bias slot was never loaded")
        resident = None
        for pid, group in itertools.groupby(iterations, key=lambda it: it[0]):
            cells = passes[pid]
            base = (self.pass_count % fit) * rows
            self.pass_count += 1
            npix = len(cells["pixels"])
            local = {px: i for i, px in enumerate(cells["pixels"])}
            slot = {ocb: base + i * npix for i, ocb in enumerate(cells["ocbs"])}
            initialized: set[int] = set()
            if oc0 is not None:
                for ocb, first in slot.items():
                    for p0 in range(0, npix, d):
                        n = min(d, npix - p0)
                        self.ops.append(AccMvin("bias", 0, oc0 + ocb * d, n, d, first + p0))
                        initialized.update(range(first + p0, first + p0 + n))
            if residual is not None:
                _, scale, (rn, rh, rw, rc), (_, height, width, _) = residual
                for ocb, first in slot.items():
                    cells_rw = [
                        ((rn * height + rh + oy) * width + rw + ox, first + local[(oy, ox)])
                        for oy, ox in cells["pixels"]
                    ]
                    for dram, acc, n in self._runs(cells_rw):
                        ready = {r in initialized for r in range(acc, acc + n)}
                        if len(ready) != 1:
                            raise UnsupportedConstruct(f"{comp.name}: a residual run over partly written rows")
                        self.ops.append(
                            AccMvin(
                                "residual",
                                dram,
                                rc + ocb * d,
                                n,
                                d,
                                acc,
                                row_step=1,
                                accumulate=ready.pop(),
                                scale=scale,
                            )
                        )
                        initialized.update(range(acc, acc + n))
            run = None
            for _, key, oy, ox in group:
                in_row = self.input_row(lhs, key[2], oy * sh + key[0] * dh, ox * sw + key[1] * dw)
                acc_row = slot[key[3]] + local[(oy, ox)]
                if (
                    run
                    and run[0] == key
                    and in_row == run[1] + run[3]
                    and acc_row == run[2] + run[3]
                    and run[3] < d
                    and (acc_row in initialized) == run[4]
                ):
                    run[3] += 1
                    continue
                if run:
                    resident = self._flush(run, weight, resident, initialized)
                # After the flush, never before it (see the pass above): consecutive reduction steps
                # revisit the same accumulator rows, and the second must accumulate onto the first.
                run = [key, in_row, acc_row, 1, acc_row in initialized]
            if run:
                resident = self._flush(run, weight, resident, initialized)
            on, oh0, ow0, oc_start = out_start
            _, height, width, _ = out_shape
            for ocb, first in slot.items():
                cells_out = [
                    ((on * height + oh0 + oy) * width + ow0 + ox, first + local[(oy, ox)]) for oy, ox in cells["pixels"]
                ]
                for dram, acc, n in self._runs(cells_out):
                    self.ops.append(
                        Mvout(
                            "out",
                            dram,
                            oc_start + ocb * d,
                            n,
                            d,
                            acc,
                            scale=readout.scale,
                            relu=readout.relu,
                            out_dtype=readout.out_dtype,
                        )
                    )

    def _flush(self, run: list, weight: Ref, resident: tuple | None, initialized: set[int]) -> tuple:
        key, in_row, acc_row, rows, ready = run
        fresh = resident != key
        self.ops.append(
            Preload(
                weight_row=self.weight_row(weight, *key) if fresh else None,
                acc_row=acc_row,
                accumulate=ready,
                rows=rows,
                cols=self.g.dim,
            )
        )
        self.ops.append(Compute(input_row=in_row, rows=rows, fresh_weights=fresh))
        initialized.update(range(acc_row, acc_row + rows))
        return key

    def run(self) -> Schedule:
        computes = self.trace.of(FusedCompute)
        if not computes:
            raise UnsupportedConstruct("the trace has no compute to lower")
        self.bind_roles(computes)
        self._events, self._done = self.trace.events, set()
        for index, event in enumerate(self._events):
            if index in self._done:
                continue
            self._at = index
            if isinstance(event, Copy):
                if event.is_load and event.dst.box.node in self.roles:
                    self.load(event)
                elif event.is_store and self.roles.get(event.src.box.node) == "out":
                    self.store(event)
                else:
                    raise UnsupportedConstruct(
                        f"{event.name}: a copy between {event.src.box.level} "
                        f"and {event.dst.box.level} that no {self.what} "
                        "operand explains"
                    )
            elif isinstance(event, FusedCompute):
                self.compute(event)
            elif isinstance(event, TensorOp):
                self._note(
                    f"{event.call.name} ({event.call.target}) is a host op outside the accelerator schedule (C6)"
                )
        if (self.group is not None and self.group["bound"] is None) or self.bound:
            raise UnsupportedConstruct("the trace ends with an output tile that is never stored")
        self.notes.append(
            "async_wait/commit semaphores carry no instruction: the target orders "
            "moves and computes by address dependence"
        )
        names = dict(self.dram_names)
        if self.dram["out"].node in self.dequant_after:
            names["out"] = f"{self.dram['out'].node}__acc_i32"
        return Schedule(ops=self.ops, shapes=self.views(), dram_nodes=names, geometry=self.g, notes=self.notes)


def lower_gemm(trace: Trace, geometry: Geometry, scalars: Mapping[str, float] | None = None) -> Schedule:
    """Lower every GEMM in ``trace`` (one layer, in program order) to a :class:`Schedule`."""
    return _Lowerer(trace, geometry, _GEMM_ANCHORS, "GEMM", scalars).run()


def lower_conv(trace: Trace, geometry: Geometry, scalars: Mapping[str, float] | None = None) -> Schedule:
    """Lower every NHWC convolution in ``trace`` (one layer, in program order) to a
    :class:`Schedule`."""
    return _Lowerer(trace, geometry, _CONV_ANCHORS, "conv", scalars).run()


# --- whole programs ------------------------------------------------------------------------------


class _TensorDirScalars(dict):
    def __init__(self, directory: Path):
        super().__init__()
        self.directory = directory

    def __missing__(self, node: str) -> float:
        data = (self.directory / f"{node}.bin").read_bytes()
        if len(data) != 4:
            raise KeyError(node)
        value = struct.unpack("<f", data)[0]
        self[node] = value
        return value


def load_scalars(tensor_dir: str | Path) -> Mapping[str, float]:
    """Scale values from Voyager's ``compile(dump_tensors=True)`` directory, read on demand.

    The dump widens every tensor to little-endian float32 (a scalar scale is 4 bytes, an int8 weight 4
    bytes per element), so a bf16 scale reads back exactly.
    """
    return _TensorDirScalars(Path(tensor_dir))


def _sub_trace(trace: Trace, first: int, end: int) -> Trace:
    return Trace(
        inputs=trace.inputs,
        parameters=trace.parameters,
        outputs=trace.outputs,
        allocations=trace.allocations,
        events=trace.events[first:end],
        semaphores=trace.semaphores,
    )


def _layer_name(events: list) -> str:
    for event in events:
        if isinstance(event, FusedCompute):
            return event.name.split("_fused")[0]
    for event in events:
        if isinstance(event, TensorOp):
            return event.call.name
    return "layer"


def _host_layer(name: str, events: list, trace: Trace, scalars: Mapping[str, float] | None) -> list[Layer]:
    """A layer of host ops: one :class:`HostOp` per distinct tensor op, on whole DRAM tensors."""
    feeds: dict[str, Copy] = {}
    writes: list[str] = []
    for event in events:
        if not isinstance(event, Copy):
            continue
        if event.is_load:
            feeds.setdefault(event.dst.box.node, event)
        elif event.is_store:
            if event.dst.box.node not in writes:
                writes.append(event.dst.box.node)
        else:
            raise UnsupportedConstruct(f"{event.name}: an on-chip copy in a host layer")
    layers, seen = [], set()
    for event in events:
        if not isinstance(event, TensorOp) or event.call.name in seen:
            continue
        seen.add(event.call.name)
        call = event.call
        inputs, attrs = {}, {}
        for key, value in call.kwargs.items():
            if isinstance(value, Ref) and value.box.level == "IMMEDIATE":
                attrs[key] = _f32(float(scalars[value.box.node])) if scalars is not None else value.box.node
            elif isinstance(value, Ref) and value.box.on_chip:
                copy = feeds.get(value.box.node)
                if copy is None:
                    raise UnsupportedConstruct(f"{call.name}: {key} is on chip but no load feeds it")
                inputs[key] = copy.src.box.node
                view = tuple(copy.src.output_shape or ())
                if view and view != tuple(copy.src.box.shape):
                    attrs[f"{key}_view"] = view  # the shape the op reads its DRAM tensor as
                if copy.pad:
                    attrs[f"{key}_pad_before"] = tuple(copy.pad)
                    attrs[f"{key}_pad_value"] = copy.pad_value
            elif isinstance(value, Ref):
                inputs[key] = value.box.node
            elif value is not None:
                attrs[key] = value
        outputs = {"output": writes[0]} if len(writes) == 1 else {f"output{i}": node for i, node in enumerate(writes)}
        if not writes and call.name in trace.allocations:
            outputs = {"output": call.name}
        op = HostOp(call.target, inputs, outputs, attrs)
        layers.append(Layer(name, "host", op, tuple(inputs.values()), tuple(outputs.values())))
    return layers


def lower_model(trace: Trace, geometry: Geometry, scalars: Mapping[str, float] | None) -> list[Layer]:
    """Lower a whole replayed program into an ordered list of :class:`Layer` entries.

    Layers come from ``trace.layers`` (one Voyager top-level loop each; see ``voyager_ir``). A layer
    whose computes are GEMMs or convolutions becomes one accelerator :class:`Schedule`, preceded by
    any host requantization of a residual it adds and followed by any host dequantization of an
    unquantized output; a layer of tensor ops only becomes host ops. ``scalars`` maps scale nodes to
    values (see :func:`load_scalars`).
    """
    if not trace.layers:
        raise UnsupportedConstruct("the trace carries no layer segmentation")
    layers: list[Layer] = []
    for _, first, end in trace.layers:
        events = trace.events[first:end]
        name = _layer_name(events)
        computes = [e for e in events if isinstance(e, FusedCompute)]
        if not computes:
            layers.extend(_host_layer(name, events, trace, scalars))
            continue
        anchors = {c.anchor.target for c in computes if c.anchor.target != _DEQUANTIZE}
        if anchors and anchors <= set(_CONV_ANCHORS):
            kind, lowerer = "conv", _Lowerer(_sub_trace(trace, first, end), geometry, _CONV_ANCHORS, "conv", scalars)
        elif anchors and anchors <= set(_GEMM_ANCHORS):
            kind, lowerer = "gemm", _Lowerer(_sub_trace(trace, first, end), geometry, _GEMM_ANCHORS, "GEMM", scalars)
        else:
            raise UnsupportedConstruct(f"{name}: computes anchored on {sorted(anchors)}")
        schedule = lowerer.run()
        for op in lowerer.host_before.values():
            layers.append(Layer(name, "host", op, tuple(op.inputs.values()), tuple(op.outputs.values())))
        reads = tuple(schedule.dram_nodes[r] for r in ("lhs", "weight", "bias", "residual") if r in schedule.dram_nodes)
        readout = dict(lowerer.readout_params.get("store", {}))
        layers.append(Layer(name, kind, schedule, reads, (schedule.dram_nodes["out"],), readout))
        for node, (scale, relu, dtype) in lowerer.dequant_after.items():
            op = HostOp(
                DEQUANTIZE_ACC,
                {"input": f"{node}__acc_i32"},
                {"output": node},
                {"scale": scale, "relu": relu, "dtype": dtype},
            )
            layers.append(Layer(name, "host", op, (f"{node}__acc_i32",), (node,)))
        tensor_ops = [e for e in events if isinstance(e, TensorOp)]
        if tensor_ops:
            layers.extend(_host_layer(name, tensor_ops, trace, scalars))
    return layers


# --- executable semantics ------------------------------------------------------------------------


def _scale_unit(values: Any, scale: float, lo: int, hi: int) -> Any:
    """The target's scale unit: integer -> fp32 (round to nearest even), fp32 multiply (nearest
    even), fp32 -> integer (nearest even), saturated to [lo, hi].

    Derived from the pinned chipyard checkout of the systolic target, not assumed:
    ``src/main/scala/*/Configs.scala`` ``mvin_scale_args`` / ``acc_scale_args`` (INToRecFN,
    MulAddRecFN and RecFNToIN all ``round_near_even``, saturating on overflow), and its generated C
    header ``gemmini_params.h`` ``ROUND_NEAR_EVEN`` / ``ACC_SCALE`` (float multiply, round to
    nearest even, clamp to int8). The same header's ``MVIN_SCALE_ACC(x, scale) (x)`` and the config's
    ``mvin_scale_acc_args = None`` are why a default build cannot scale an accumulator load.
    """
    import numpy as np

    y = np.rint(np.asarray(values, dtype=np.int64).astype(np.float32) * np.float32(scale))
    return np.clip(y, lo, hi).astype(np.int64)


def execute(schedule: Schedule, lhs: Any, weight: Any, **operands: Any) -> Any:
    """Run ``schedule`` on integer operands with numpy and return its output (int32 array).

    An executable semantics for the abstract ops (a resident weight block, scratchpad and accumulator
    rows), used to prove a lowering's arithmetic before any simulator sees it. Not a timing model.
    Operands are the 2-D views :attr:`Schedule.shapes` names; ``operands`` supplies any further role a
    schedule reads (``bias``, ``residual``). The readout follows the target's store path (activation,
    then the scale unit, then saturation: ``AccumulatorScale.scala`` applies the activation before
    ``scale_func`` and clips to the output width after it).
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
                block[:, : op.cols] = sources[op.role][rows, op.dram_col : op.dram_col + op.cols]
            spad[op.spad_row : op.spad_row + op.rows] = block
        elif isinstance(op, AccMvin):
            rows = op.dram_row + op.row_step * np.arange(op.rows)
            block = np.zeros((op.rows, d), dtype=np.int64)
            block[:, : op.cols] = sources[op.role][rows, op.dram_col : op.dram_col + op.cols]
            if op.scale != 1.0:
                block = _scale_unit(block, op.scale, *_INT32)
            span = slice(op.acc_row, op.acc_row + op.rows)
            acc[span] = acc[span] + block if op.accumulate else block
        elif isinstance(op, Preload):
            if op.weight_row is not None:
                resident = spad[op.weight_row : op.weight_row + d].copy()
            target = op
        elif isinstance(op, Compute):
            if target is None:
                raise UnsupportedConstruct("compute with no preceding preload")
            if target.rows != op.rows:
                raise UnsupportedConstruct("a preload and its compute disagree on the row count")
            product = spad[op.input_row : op.input_row + op.rows] @ resident
            rows = slice(target.acc_row, target.acc_row + op.rows)
            acc[rows] = acc[rows] + product if target.accumulate else product
            target = None
        elif isinstance(op, Mvout):
            values = acc[op.acc_row : op.acc_row + op.rows, : op.cols]
            if op.out_dtype == "int8":
                if op.relu:
                    values = np.maximum(values, 0)
                values = _scale_unit(values, op.scale, *_INT8)
            elif op.out_dtype != "int32" or op.relu or op.scale != 1.0:
                raise UnsupportedConstruct(
                    f"a {op.out_dtype} readout with scale {op.scale} and relu={op.relu} is not a store the target has"
                )
            out[op.dram_row : op.dram_row + op.rows, op.dram_col : op.dram_col + op.cols] = values
    return out.astype(np.int32)


def _bf16(values: Any) -> Any:
    """Round float32 values to bfloat16 (nearest even), kept in float32."""
    import numpy as np

    bits = np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.astype(np.uint32).view(np.float32)


def _int_range(dtype: str) -> tuple[int, int]:
    """The value range of an integer dtype name (``int8``, ``uint8``, ``int32``, ...)."""
    signed = not dtype.startswith("uint")
    digits = dtype[4:] if not signed else dtype[3:]
    if not dtype.startswith(("int", "uint")) or not digits.isdigit():
        raise UnsupportedConstruct(f"{dtype} is not an integer dtype")
    bits = int(digits)
    return (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)


def _as_dtype(values: Any, dtype: str) -> Any:
    """A host result in the executor's representation of ``dtype``: an integer tensor as int64
    (it must already hold integers), bfloat16 as its values in float32, float32 as itself."""
    import numpy as np

    if dtype == "bfloat16":
        return _bf16(values)
    if dtype == "float32":
        return np.asarray(values, dtype=np.float32)
    values = np.asarray(values)
    if values.dtype.kind == "f" and not np.array_equal(values, np.rint(values)):
        raise UnsupportedConstruct(f"a non-integer result stored as {dtype}")
    return values.astype(np.int64)


def _operand(op: HostOp, tensors: Mapping[str, Any], key: str) -> Any:
    """``op``'s DRAM operand ``key``, read in the shape the op reads it (the load's view)."""
    import numpy as np

    values = np.asarray(tensors[op.inputs[key]])
    view = op.attrs.get(f"{key}_view")
    return values.reshape(view) if view else values


def _host_max_pool2d(op: HostOp, tensors: Mapping[str, Any], out_shape: tuple, out_dtype: str):
    """Voyager's ``quantized_ops::max_pool2d``, the NHWC twin of ``aten.max_pool2d``
    (``ops/layout.py``), read from the tile its load builds: ``async_copy`` fills the tile with the
    pad value and copies the source in after each dim's leading pad (``bufferize/ops.py``
    ``async_copy``). Only the load pads; a kernel padding of its own refuses."""
    import numpy as np

    x = np.asarray(_operand(op, tensors, "input"), dtype=np.float32)
    a = op.attrs
    kh, kw = a["kernel_size"]
    sh, sw = tuple(a.get("stride") or ()) or (kh, kw)
    dh, dw = a.get("dilation", (1, 1))
    if a.get("ceil_mode") or tuple(a.get("padding", (0, 0))) != (0, 0) or x.ndim != 4:
        raise UnsupportedConstruct("max_pool2d with ceil_mode, its own padding, or a non-NHWC input")
    pad = tuple(a.get("input_pad_before", (0, 0, 0, 0)))
    fill = a.get("input_pad_value")
    fill = np.float32(0.0 if fill is None else fill)
    if pad[0] or pad[3]:
        raise UnsupportedConstruct("max_pool2d padded along batch or channels")
    n, oh, ow, c = out_shape
    th, tw = (oh - 1) * sh + (kh - 1) * dh + 1, (ow - 1) * sw + (kw - 1) * dw + 1
    tile = np.full((x.shape[0], th, tw, x.shape[3]), fill, dtype=np.float32)
    rows, cols = min(th, pad[1] + x.shape[1]) - pad[1], min(tw, pad[2] + x.shape[2]) - pad[2]
    tile[:, pad[1] : pad[1] + rows, pad[2] : pad[2] + cols] = x[:, :rows, :cols]
    out = np.full((x.shape[0], oh, ow, x.shape[3]), -np.inf, dtype=np.float32)
    for fy in range(kh):
        for fx in range(kw):
            out = np.maximum(
                out, tile[:, fy * dh : fy * dh + sh * (oh - 1) + 1 : sh, fx * dw : fx * dw + sw * (ow - 1) + 1 : sw]
            )
    return _as_dtype(out.reshape(out_shape), out_dtype)


def _host_adaptive_avg_pool2d(op: HostOp, tensors: Mapping[str, Any], out_shape: tuple, out_dtype: str):
    """Voyager's ``quantized_ops::adaptive_avg_pool2d``, the NHWC twin of the aten op
    (``ops/layout.py``): per output cell, the window [floor(i*H/OH), ceil((i+1)*H/OH)), summed in
    float32 in row-major order and divided by its size in float32, then stored in the output dtype."""
    import numpy as np

    x = np.asarray(_operand(op, tensors, "input"), dtype=np.float32)
    if x.ndim != 4:
        raise UnsupportedConstruct("adaptive_avg_pool2d on a non-NHWC input")
    oh, ow = op.attrs["output_size"]
    n, h, w, c = x.shape
    out = np.empty((n, oh, ow, c), dtype=np.float32)
    for i in range(oh):
        h0, h1 = (i * h) // oh, -(-((i + 1) * h) // oh)
        for j in range(ow):
            w0, w1 = (j * w) // ow, -(-((j + 1) * w) // ow)
            acc = np.zeros((n, c), dtype=np.float32)
            for y in range(h0, h1):
                for z in range(w0, w1):
                    acc = acc + x[:, y, z, :]
            out[:, i, j, :] = acc / np.float32((h1 - h0) * (w1 - w0))
    return _as_dtype(out.reshape(out_shape), out_dtype)


def _host_quantize(op: HostOp, tensors: Mapping[str, Any], out_shape: tuple, out_dtype: str):
    """Voyager's ``quantized_ops::quantize`` for a per-tensor scale: ``input / scale`` in the
    input's bfloat16 (computed in float32, rounded once; ``ops/quantized.py`` ``quantize``), then the
    integer table ``get_quantization_map`` builds -- ``clamp(round(v))``, round half to even
    (``quantization/fake_quantize.py`` ``get_quantization_map``) -- looked up by bit pattern
    (``ops/quantized.py`` ``vmap``)."""
    import numpy as np

    if op.attrs.get("zero_point") is not None or op.attrs.get("block_size") is not None:
        raise UnsupportedConstruct("a zero point or block-wise quantize is not a per-tensor scale")
    x = np.asarray(_operand(op, tensors, "input"), dtype=np.float32)
    quotient = _bf16(x / np.float32(op.attrs["scale"]))
    lo, hi = _int_range(out_dtype)
    return np.clip(np.rint(quotient), lo, hi).astype(np.int64).reshape(out_shape)


def _host_dequantize(op: HostOp, tensors: Mapping[str, Any], out_shape: tuple, out_dtype: str):
    """Voyager's ``quantized_ops::dequantize`` for a per-tensor scale: ``input * scale``
    (``ops/quantized.py`` ``dequantize``). Type promotion casts the integer input to the scale's
    floating dtype FIRST -- for a bfloat16 scale the integer is rounded to bfloat16 -- then the
    product is computed in float32 and rounded to the output dtype."""
    import numpy as np

    if op.attrs.get("zero_point") is not None or op.attrs.get("block_size") is not None:
        raise UnsupportedConstruct("a zero point or block-wise dequantize is not a per-tensor scale")
    x = np.asarray(_operand(op, tensors, "input")).astype(np.float32)
    if out_dtype == "bfloat16":
        x = _bf16(x)
    return _as_dtype((x * np.float32(op.attrs["scale"])).reshape(out_shape), out_dtype)


def _host_linear(op: HostOp, tensors: Mapping[str, Any], out_shape: tuple, out_dtype: str):
    """The classifier's ``aten::linear`` as Voyager's graph runs it: bfloat16 operands (int8
    activations and weights, exact in bfloat16), a float32 accumulation with the bias, one rounding
    to bfloat16. The products are summed exactly here, which is what float32 accumulation gives
    while partial sums stay below 2**24; beyond that the order of a library's blocked sum would
    matter, and a result is refused rather than guessed. The bias is the IR's integer tile, added
    exactly."""
    import numpy as np

    x = np.asarray(_operand(op, tensors, "input"), dtype=np.float64)
    w = np.asarray(_operand(op, tensors, "weight"), dtype=np.float64)
    total = x @ w.T
    exact_below = float(1 << (np.finfo(np.float32).nmant + 1))  # float32 holds every integer below
    if (np.abs(x) @ np.abs(w).T).max(initial=0) >= exact_below:
        raise UnsupportedConstruct("linear partial sums past float32's exact-integer range")
    if "bias" in op.inputs:
        total = total + np.asarray(_operand(op, tensors, "bias"), dtype=np.float64)
    return _as_dtype(_bf16(total.astype(np.float32)).reshape(out_shape), out_dtype)


def host_op_reference(op: HostOp, tensors: Mapping[str, Any], out_shape: tuple, out_dtype: str) -> Any:
    """The numpy reference of one host op, as the executor's value of its (single) output.

    ``tensors`` maps DRAM names to arrays (integers as int64, bfloat16 values as float32);
    ``out_shape``/``out_dtype`` are the output tensor's. Voyager's own ops follow Voyager's reference
    semantics (checked against its op library by ``merlin/tests/ir/test_voyager_host_ops.py``); the
    bridge's ops follow the target: ``merlin::requantize`` is the scale unit into int32,
    ``merlin::dequantize`` a float32 scale, then relu, then the output dtype.
    """
    import numpy as np

    target = op.target
    if target == REQUANTIZE:
        return _scale_unit(tensors[op.inputs["input"]], op.attrs["scale"], *_INT32)
    if target == DEQUANTIZE_ACC:
        values = np.asarray(tensors[op.inputs["input"]]).astype(np.float32)
        values = values * np.float32(op.attrs["scale"])
        if op.attrs["relu"]:
            values = np.maximum(values, np.float32(0))
        return _bf16(values) if op.attrs["dtype"] == "bfloat16" else values
    if target == "aten::permute":
        return np.transpose(tensors[op.inputs["input"]], op.attrs["dims"])
    handler = _HOST_OPS.get(target)
    if handler is None:
        raise UnsupportedConstruct(f"host op {target} has no reference semantics here")
    return handler(op, tensors, tuple(out_shape), out_dtype)


_HOST_OPS = {
    "quantized_ops::max_pool2d": _host_max_pool2d,
    "quantized_ops::adaptive_avg_pool2d": _host_adaptive_avg_pool2d,
    "quantized_ops::quantize": _host_quantize,
    "quantized_ops::dequantize": _host_dequantize,
    "aten::linear": _host_linear,
}


def execute_model(layers: list[Layer], tensors: dict[str, Any], trace: Trace) -> dict[str, Any]:
    """Run a lowered model with numpy. ``tensors`` maps DRAM tensor names to arrays in their Voyager
    shapes (inputs and parameters; integers as int64, bfloat16 values as float32); results are added
    under their names and the dict is returned.

    Accelerator layers run through :func:`execute`; host ops through :func:`host_op_reference`.
    """
    import numpy as np

    def shape_of(name: str) -> tuple[int, ...]:
        base = name.removesuffix("__acc_i32")
        return tuple(trace.allocations[base].shape)

    for layer in layers:
        program = layer.program
        if isinstance(program, Schedule):
            view = {
                role: np.asarray(tensors[node]).reshape(program.shapes[role])
                for role, node in program.dram_nodes.items()
                if role != "out"
            }
            lhs, weight = view.pop("lhs"), view.pop("weight")
            out = execute(program, lhs, weight, **view)
            tensors[program.dram_nodes["out"]] = out.reshape(shape_of(program.dram_nodes["out"]))
            continue
        if len(program.outputs) != 1:
            raise UnsupportedConstruct(f"host op {program.target} writes {len(program.outputs)} tensors")
        (out,) = program.outputs.values()
        box = trace.allocations.get(out.removesuffix("__acc_i32"))
        dtype = "int32" if out.endswith("__acc_i32") or box is None else box.dtype
        tensors[out] = host_op_reference(program, tensors, shape_of(out), dtype)
    return tensors
