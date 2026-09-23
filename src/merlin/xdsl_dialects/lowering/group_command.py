"""A closed compute group stated as the device program it asks for.

A capture writes a layer the way the framework lowered it, which is not the way a unit runs it. A
convolution arrives as a gather into patches, a reshape, and ``W[Cout, K] @ patches[K, positions]``
with the bias along the ROWS of the result. A unit holds the weight stationary and streams the
activation past it, ``A[M, K] @ W[K, N]``, with its bias and scale along the columns, and it forms
the patches itself. Restating the first as the second is what turns "this group is admitted" into
"this is the command the backend has to emit", and it is pure structure:

* which operand is the STORED tensor (it comes from a model argument through views and movement);
* which result axis the bias runs along, followed through the reshapes after the contraction, which
  has to be the stored tensor's own output axis or the readout cannot apply it;
* the convolution geometry, when the activation is a windowed gather: taps, strides and output
  extent from the gather's index expressions, padding from the slice insertion before it.

The result is an entry in the vocabulary the capsule generator builds from (``corpus_spec``), with
the group's real multiplier as ``acc_scale``. So one statement serves three consumers: the capsule
that DEMANDS the pattern of a backend, the emission that runs it, and the prepack that lays the
stored tensor out the way the statement says. A group that cannot be restated is refused with the
reason (:class:`~.compute_groups.NoCapsuleForm`); nothing here knows a target.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any

from merlin.common import mlir_query as mq
from merlin.kernels import shapes as KS

from . import compute_groups as CG
from . import group_numerics as GN

SCHEMA = "group_program_v1"
#: The capture's patch-column order when it gathers ``[channel, tap_h, tap_w]``; the command-buffer
#: ABI's convolution packs ``[tap_h, tap_w, channel]``. A prepack step permutes the stored tensor.
CAPTURE_COLUMN_ORDER = ("channel", "tap_h", "tap_w")


@dataclass(frozen=True)
class GroupProgram:
    """One group as a generator entry, with what a prepack step needs to honour it."""

    entry: dict[str, Any]
    #: Root operand index of the stored tensor; ``None`` for a group that reads no stored tensor.
    stored_operand: int | None
    #: The capture computes the transpose of the device form (stored tensor on the left).
    transposed: bool
    #: Model-argument index of the stored tensor and of the bias, for the prepack.
    stored_arg: int | None = None
    bias_arg: int | None = None
    column_order: tuple[str, ...] | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "entry": dict(self.entry),
            "stored_operand": self.stored_operand,
            "transposed": self.transposed,
            "stored_arg": self.stored_arg,
            "bias_arg": self.bias_arg,
            "column_order": list(self.column_order) if self.column_order else None,
            "notes": list(self.notes),
        }


def _source_argument(value) -> int | None:
    """The model-argument index ``value`` is, looking back through views and movement only."""
    from xdsl.ir import BlockArgument

    for _ in range(12):
        if isinstance(value, BlockArgument):
            return int(value.index)
        owner = getattr(value, "owner", None)
        if owner is None or not getattr(owner, "operands", None):
            return None
        stage = CG.classify(owner)
        if stage is None or stage.kind not in (CG.VIEW, CG.MOVEMENT):
            return None
        value = owner.operands[0]
    return None


def stored_operand(group: CG.Group, weight_args: Collection[int] | None = None) -> tuple[int, int]:
    """``(root operand index, model-argument index)`` of the group's stored tensor.

    With a weights manifest the stored tensor is the operand a weight argument feeds. Without one it
    is the only operand an argument feeds at all; a first layer, where the activation is an argument
    too, is then refused rather than guessed.
    """
    sources: list[int | None] = []
    for operand in list(group.root.operands)[:2]:
        _adapters, dequantize, _dtype = CG._input_chain(operand)
        sources.append(_source_argument(dequantize.operands[0] if dequantize is not None else operand))
    stored = [
        index for index, arg in enumerate(sources) if arg is not None and (weight_args is None or arg in weight_args)
    ]
    if len(stored) != 1:
        raise CG.NoCapsuleForm(
            "both operands are model arguments and no weights manifest says which one is stored"
            if len(stored) == 2
            else "neither operand of the contraction is a stored tensor"
        )
    return stored[0], int(sources[stored[0]])


def _result_axis_now(group: CG.Group, until, axis: int) -> int | None:
    """Where the root's result axis ``axis`` sits at the input of member ``until``."""
    position = {id(member): index for index, member in enumerate(group.members)}
    between = [
        member
        for member in group.members
        if position[id(group.root)] < position[id(member)] < position[id(until)]
        and CG.classify(member).kind in (CG.VIEW, CG.MOVEMENT)
    ]
    shape, _ = mq.type_shape_dtype(group.root.results[0].type)
    # `_axis_through` applies its adapters last-first; these run forward from the root.
    return CG._axis_through(list(reversed(between)), axis, shape)


def _bias_side(group: CG.Group) -> tuple[str | None, int | None]:
    """``("row" | "column", bias argument)`` for the group's bias stage, or ``(None, None)``."""
    member = next((m for m in group.members if CG.classify(m).kind == CG.BIAS_ADD), None)
    if member is None:
        return None, None
    varies = CG.classify(member).varies_over
    if varies is None or len(varies) != 1:
        raise CG.NoCapsuleForm("the bias does not run along exactly one axis of the result")
    shape, _ = mq.type_shape_dtype(group.root.results[0].type)
    if len(shape) != 2:
        raise CG.NoCapsuleForm("the bias axis is read against a two-axis contraction result only")
    sides = {"row": _result_axis_now(group, member, 0), "column": _result_axis_now(group, member, 1)}
    found = [side for side, axis in sides.items() if axis is not None and axis == varies[0]]
    if len(found) != 1:
        raise CG.NoCapsuleForm("the bias axis could not be followed back to the contraction's result")
    argument = next((a for a in (_source_argument(o) for o in member.operands) if a is not None), None)
    return found[0], argument


def _attr_ints(op, key: str) -> list[int] | None:
    for table in mq._attr_tables(op):
        raw = table.get(key)
        data = getattr(raw, "get_values", None)
        if callable(data):
            return [int(v) for v in data()]
    return None


def _window(group: CG.Group, activation_index: int, reduced: int) -> dict[str, Any] | None:
    """Convolution geometry when the activation operand is a windowed gather, else ``None``.

    ``reduced`` is the extent of the contraction's reduction: the gather's leading output axes whose
    product is that extent are the channel and the taps, the rest are the output positions. That is
    what tells a tap dim from a position dim where the stride is one and the index expression is a
    plain sum.
    """
    adapters, _dequantize, _dtype = CG._input_chain(list(group.root.operands)[activation_index])
    gathers = [op for op in adapters if mq.op_name(op) == "linalg.generic" and CG._is_windowed(op)]
    if not gathers:
        return None
    if len(gathers) != 1:
        raise CG.NoCapsuleForm("more than one windowed gather feeds the contraction")
    gather = gathers[0]
    maps = KS.indexing_maps(gather)
    in_shape, _ = mq.type_shape_dtype(gather.operands[0].type)
    out_shape, _ = mq.type_shape_dtype(gather.results[0].type)
    if not maps or len(maps[0]) != len(in_shape) or any(KS._dim_position(e) is None for e in maps[-1]):
        raise CG.NoCapsuleForm("the gather's index maps are not a window over a plain output")
    windows = [sorted(CG._dims_of(expr)) for expr in maps[0] if KS._dim_position(expr) is None]
    # Every place the leading axes multiply to the reduction's extent. Axes of extent one make
    # that ambiguous (a one-tap window), so the split is the first one that leaves each window
    # expression with one dim on either side.
    candidates, running = [], 1
    for index, extent in enumerate(out_shape):
        running *= int(extent)
        if running == reduced:
            candidates.append(index + 1)
        elif running > reduced:
            break
    split = next(
        (c for c in candidates if all(len(w) == 2 and sum(d < c for d in w) == 1 for w in windows)),
        None,
    )
    if split is None:
        raise CG.NoCapsuleForm("the gather's output does not factor into taps and positions")
    taps: list[int] = []
    strides: list[int] = []
    outs: list[int] = []
    padded: list[int] = []
    window_axes: list[int] = []
    channels = batch = 1
    for axis, expr in enumerate(maps[0]):
        plain = KS._dim_position(expr)
        if plain is not None:
            if plain < split:
                channels *= int(out_shape[plain])
            else:
                batch *= int(out_shape[plain])
            continue
        dims = sorted(CG._dims_of(expr))
        stride = CG._stride_of(expr)
        tap = [d for d in dims if d < split]
        place = [d for d in dims if d >= split]
        if len(dims) != 2 or len(tap) != 1 or len(place) != 1 or stride is None:
            raise CG.NoCapsuleForm("a gather index is not `position * stride + tap`")
        taps.append(int(out_shape[tap[0]]))
        outs.append(int(out_shape[place[0]]))
        strides.append(int(stride))
        padded.append(int(in_shape[axis]))
        window_axes.append(axis)
    if len(taps) != 2:
        raise CG.NoCapsuleForm("the window is not two-dimensional")
    if batch != 1:
        raise CG.NoCapsuleForm(f"a batch of {batch} is not one convolution command")
    before = [0, 0]
    sizes = list(padded)
    pads = [op for op in adapters if mq.op_name(op) == "tensor.insert_slice"]
    if len(pads) > 1:
        raise CG.NoCapsuleForm("more than one slice insertion pads the activation")
    if pads:
        offsets, extents = _attr_ints(pads[0], "static_offsets"), _attr_ints(pads[0], "static_sizes")
        if offsets is None or extents is None:
            raise CG.NoCapsuleForm("the padding is not static")
        before = [offsets[axis] for axis in window_axes]
        sizes = [extents[axis] for axis in window_axes]
    after = [padded[i] - before[i] - sizes[i] for i in range(2)]
    if min(after) < 0:
        raise CG.NoCapsuleForm("the padded extent is smaller than the image it holds")
    for i in range(2):
        if (padded[i] - taps[i]) // strides[i] + 1 != outs[i]:
            raise CG.NoCapsuleForm("the gather's output extent does not follow from its window")
    return {
        "ci": channels,
        "Himg": sizes[0],
        "Wimg": sizes[1],
        "kh": taps[0],
        "kw": taps[1],
        "stride": strides,
        "padding": [before[0], before[1], after[0], after[1]],
        "positions": outs[0] * outs[1],
    }


def _pool(group: CG.Group) -> dict[str, Any]:
    """The pooling stage's window, as the generator's ``pool_*`` fields. Read, never defaulted."""
    from . import contraction_coverage as CC

    member = next(m for m in group.members if CG.classify(m).kind == CG.POOL)
    maps, kinds, extents = KS.indexing_maps(member), KS._iterator_types(member), CC.loop_extents(member)
    if not maps or not kinds or not extents:
        raise CG.NoCapsuleForm("the pooling window could not be read from its index maps")
    size: list[int] = []
    stride: list[int] = []
    axes: list[int] = []
    for axis, expr in enumerate(maps[0]):
        if KS._dim_position(expr) is not None:
            continue
        taps = [d for d in sorted(CG._dims_of(expr)) if kinds[d] == "reduction"]
        step = CG._stride_of(expr)
        if len(taps) != 1 or step is None or taps[0] not in extents:
            raise CG.NoCapsuleForm("a pooling index is not `position * stride + tap`")
        size.append(int(extents[taps[0]]))
        stride.append(int(step))
        axes.append(axis)
    if len(size) != 2:
        raise CG.NoCapsuleForm("the pooling window is not two-dimensional")
    padded, _ = mq.type_shape_dtype(member.operands[0].type)
    before, inner = [0, 0], [int(padded[a]) for a in axes]
    pad = getattr(member.operands[0], "owner", None)
    if pad is not None and any(pad is m for m in group.members) and mq.op_name(pad) == "tensor.insert_slice":
        offsets, sizes = _attr_ints(pad, "static_offsets"), _attr_ints(pad, "static_sizes")
        if offsets is None or sizes is None:
            raise CG.NoCapsuleForm("the pooling padding is not static")
        before, inner = [offsets[a] for a in axes], [sizes[a] for a in axes]
    after = [int(padded[a]) - before[i] - inner[i] for i, a in enumerate(axes)]
    window = {"pool_size": size, "pool_stride": stride, "pool_padding": [before[0], before[1], after[0], after[1]]}
    if any(window["pool_padding"]):
        # WHAT A PADDED CELL CONTRIBUTES. The capture says it: the padded tensor is a fill, and a
        # max pool's fill is the identity of max. In the unit's integer output domain that identity
        # is the readout's own lower clamp, which the group's closing quantize declares; a fill of
        # zero is zero. Anything else is a pool this statement cannot make, and is refused: a pad
        # value nobody read would be a number every engine agreed on and no model contained.
        import math

        fill = GN._constant_of(pad.operands[1]) if pad is not None and len(pad.operands) > 1 else None
        if fill is None:
            raise CG.NoCapsuleForm("the pooling pad's fill value could not be read")
        if fill == 0:
            window["pool_pad_value"] = 0
        elif isinstance(fill, float) and (math.isinf(fill) and fill < 0):
            window["pool_pad_value"] = int(GN.numerics_of(group).clamp[0])
        else:
            raise CG.NoCapsuleForm(f"a pooling pad of {fill!r} is neither zero nor the identity of max")
    return window


def program(group: CG.Group, *, weight_args: Collection[int] | None = None, name: str | None = None) -> GroupProgram:
    """``group`` as a generator entry in device form. Raises ``NoCapsuleForm`` with the reason."""
    from merlin.runtime.commandbuffer import EPILOGUE_STAGES

    base = CG.capsule_entry(group, name=name)  # the stage names, and the refusals that go with them
    if group.window_mean is not None:
        return GroupProgram(
            entry=base,
            stored_operand=None,
            transposed=False,
            notes=(
                "a mean over a trailing window: the stored operand is a constant one, and the readout's "
                "scale carries both scales and the reciprocal of the count",
            ),
        )
    if base["op"] == "residual_add":
        # Two activations and no stored tensor: nothing to prepack, nothing to transpose.
        factor = float(group.operand_sum.get("readout_factor") or 1.0)
        notes = (
            (f"a multiplier exceeds one: the loads carry the multipliers over {factor!r} and the readout scales by it",)
            if factor != 1.0
            else ()
        )
        return GroupProgram(entry=base, stored_operand=None, transposed=False, notes=notes)
    if base["op"] == "conv2d":
        return GroupProgram(entry=base, stored_operand=1, transposed=False)  # already a windowed root
    shape, _ = mq.type_shape_dtype(group.root.results[0].type)
    if len(shape) != 2:
        raise CG.NoCapsuleForm("a batched contraction is not restated as one device command yet")
    stored, stored_arg = stored_operand(group, weight_args)
    side, bias_arg = _bias_side(group)
    # The stored tensor's own output axis: rows when it is the left operand, columns when the right.
    wanted = "row" if stored == 0 else "column"
    if side is not None and side != wanted:
        raise CG.NoCapsuleForm(
            f"the bias runs along the result's {side}s and the stored tensor's outputs are its "
            f"{wanted}s: a per-output bias readout cannot apply it"
        )
    rows, reduced, columns = int(base["M"]), int(base["K"]), int(base["N"])
    positions, features = (columns, rows) if stored == 0 else (rows, columns)
    entry: dict[str, Any] = {key: base[key] for key in ("name", "kind", "scale_granularity", "operand_dtype")}
    # In the order a readout applies them (bias, scale, activation, pool), which is the ABI's declared
    # stage order. The capture applies the activation BEFORE it quantizes; with a positive multiplier
    # and a zero output zero point the two orders are the same function (see group_numerics).
    entry["epilogue"] = sorted(base["epilogue"], key=EPILOGUE_STAGES.index)
    notes: list[str] = []
    window = _window(group, 1 - stored, reduced)
    column_order = None
    if window is not None and (window["kh"], window["kw"], window["stride"], window["padding"]) != (
        1,
        1,
        [1, 1],
        [0, 0, 0, 0],
    ):
        if window["positions"] != positions or window["ci"] * window["kh"] * window["kw"] != reduced:
            raise CG.NoCapsuleForm("the gather's geometry does not match the contraction's extents")
        entry.update({"op": "conv2d", "N": features, **{k: v for k, v in window.items() if k != "positions"}})
        column_order = CAPTURE_COLUMN_ORDER
        notes.append("the capture gathers the patches on the host; the unit's convolution forms them itself")
    else:
        entry.update({"op": "matmul", "M": positions, "K": reduced, "N": features})
        if window is not None:
            notes.append("a one-tap unit-stride window is a contraction over positions")
    if "maxpool" in entry["epilogue"]:
        entry.update(_pool(group))
        if entry["op"] != "conv2d":
            # A contraction has no spatial extent of its own; the rows unflatten to the pool's input.
            member = next(m for m in group.members if CG.classify(m).kind == CG.POOL)
            spatial, _ = mq.type_shape_dtype(member.results[0].type)
            raise CG.NoCapsuleForm(f"a pooled contraction over {list(spatial)} needs its input plane stated")
    if "acc_scale" in entry["epilogue"]:
        numerics = GN.numerics_of(group)
        if numerics.multiplier is None:
            raise CG.NoCapsuleForm("the group's requantization multiplier is not a compile-time number")
        entry["acc_scale"] = float(numerics.multiplier)
    if stored == 0:
        notes.append("the capture computes the transpose: the stored tensor is its left operand")
    return GroupProgram(
        entry=entry,
        stored_operand=stored,
        transposed=stored == 0,
        stored_arg=stored_arg,
        bias_arg=bias_arg,
        column_order=column_order,
        notes=tuple(notes),
    )
