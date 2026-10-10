"""The host side of a group call under the logical kernel ABI: what the call passes and reads back.

A routed group's kernel interface names every pointer it takes. This module supplies each from the
host program, the way a runner-owned capsule harness does: the activation (or, for a convolution the
capture gathered on the host, the image before the gather, in the interface's layout), the stored
weight laid out as the device program holds it, the folded bias as a constant, and destinations in the
device's ``[positions, features]`` layout read back to the capture's own. Whatever fed only the routed
group is erased afterwards. A value with no host source is refused by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _prepack_arrays(groups, capture) -> tuple[dict[int, dict[str, Any]], dict[str, Any], str]:
    """``(rows by group, arrays, "")`` of the groups' prepack -- the folded biases and the weights laid
    out as each device program holds them -- or ``({}, {}, reason)``.

    The host sources a logical-ABI call passes for a group's constant interface values. One
    definition: the same :func:`merlin.xdsl_dialects.lowering.group_prepack.prepack` the sidecar
    records."""
    if capture is None:
        return {}, {}, "no capture was given, so no weights manifest could be read and no bias folded"
    beside = Path(capture)
    beside = beside if beside.is_dir() else beside.parent
    manifest = next(iter(sorted(beside.glob("*.manifest.json"))), None)
    weights = next(iter(sorted(beside.glob("*.safetensors"))), None)
    if manifest is None or weights is None:
        return {}, {}, f"no weights manifest and safetensors pair beside {beside}"
    from merlin.xdsl_dialects.lowering import group_prepack as GP

    try:
        result = GP.prepack(groups, manifest, weights, device_layout=True)
    except Exception as error:  # noqa: BLE001 -- named, never a silently absent prepack
        return {}, {}, f"prepack: {type(error).__name__}: {error}"
    rows = {int(row["group"]): row for row in result["record"]["groups"]}
    return rows, result["arrays"], ""


def _constant_tensor(array, element_type):
    """``arith.constant`` holding ``array`` as a dense tensor of ``element_type``."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType

    shape = [int(v) for v in array.shape]
    values = [int(v) for v in array.reshape(-1).tolist()]
    return arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(element_type, shape), values))


class _NotAGather(Exception):
    """The activation is not a host-side window gather; the group keeps its stated contraction."""


def _owner(value):
    from xdsl.ir import Operation

    owner = getattr(value, "owner", None)
    return owner if isinstance(owner, Operation) else None


def _ints(op, key: str) -> list[int] | None:
    from merlin.xdsl_dialects.lowering.group_command import _attr_ints

    return _attr_ints(op, key)


def _int_attr(op, key: str) -> int | None:
    from merlin.common import mlir_query as mq

    for table in mq._attr_tables(op):  # noqa: SLF001 -- the shared property/attribute reader
        raw = getattr(table.get(key), "value", None)
        if raw is not None and hasattr(raw, "data"):
            return int(raw.data)
    return None


def _shape(value) -> list[int]:
    from merlin.common import mlir_query as mq

    return [int(v) for v in mq.type_shape_dtype(value.type)[0]]


def _gathered_piece(value, gather_ops: list):
    """One ``[positions, channels]`` piece of a host-side window gather, read back to its image.

    The capture's spelling: strided ``tensor.extract_slice``s of an NCHW image (taking every channel of
    its single batch), an NCHW->NHWC ``linalg.transpose``, a ``collapse_shape`` to one axis and an
    ``expand_shape`` to ``[positions, channels]``. Returns ``(image, (offset_h, offset_w),
    (stride_h, stride_w), (rows, cols), channels)``; raises :class:`_NotAGather` for anything else."""
    expand = _owner(value)
    if expand is None or expand.name != "tensor.expand_shape" or len(_shape(value)) != 2:
        raise _NotAGather("not an expand to [positions, channels]")
    collapse = _owner(expand.operands[0])
    if collapse is None or collapse.name != "tensor.collapse_shape" or len(_shape(collapse.results[0])) != 1:
        raise _NotAGather("not a flattened view")
    transpose = _owner(collapse.operands[0])
    if transpose is None or transpose.name != "linalg.transpose" or _ints(transpose, "permutation") != [0, 2, 3, 1]:
        raise _NotAGather("not an NCHW->NHWC transpose")
    gather_ops += [expand, collapse, transpose]
    init = _owner(transpose.operands[1])
    if init is not None and init.name == "tensor.empty":
        gather_ops.append(init)
    slices, current = [], transpose.operands[0]
    while (op := _owner(current)) is not None and op.name == "tensor.extract_slice":
        offsets, sizes, strides = _ints(op, "static_offsets"), _ints(op, "static_sizes"), _ints(op, "static_strides")
        if offsets is None or sizes is None or strides is None or len(offsets) != 4 or len(op.operands) != 1:
            raise _NotAGather("a slice is not static")
        slices.append((offsets, strides))
        gather_ops.append(op)
        current = op.operands[0]
    image_shape = _shape(current)
    viewed = _shape(transpose.operands[0])
    if len(image_shape) != 4 or image_shape[0] != 1 or len(viewed) != 4:
        raise _NotAGather("the gathered tensor is not one NCHW image")
    offset, stride = [0, 0, 0, 0], [1, 1, 1, 1]
    for offsets, strides in reversed(slices):  # outermost first: each inner slice reads the outer one
        offset = [offset[i] + offsets[i] * stride[i] for i in range(4)]
        stride = [stride[i] * strides[i] for i in range(4)]
    if offset[:2] != [0, 0] or stride[1] != 1 or viewed[:2] != [1, image_shape[1]]:
        raise _NotAGather("a piece does not take every channel of the single image")
    return current, (offset[2], offset[3]), (stride[2], stride[3]), (viewed[2], viewed[3]), image_shape[1]


def _host_window_gather(activation, rows: int, reduced: int):
    """The convolution a host-side window gather feeds, read from the module, or ``None``.

    ``None`` when the activation is not a gather (a plain contraction keeps its statement). A gather
    whose pre-gather image cannot be identified exactly raises :class:`ValueError` naming why: routing
    it as a contraction would hand the device the host's patch matrix. Returns the image BEFORE the
    gather and its padding, the window geometry, the reduction's order as the gather lays it out
    (``(tap_h, tap_w)`` per piece, channels innermost), and every op the gather consists of."""
    import numpy as np

    owner = _owner(activation)
    if owner is None:
        return None
    gather_ops: list = []
    pieces = list(owner.operands) if owner.name == "tensor.concat" else [activation]
    if owner.name == "tensor.concat":
        if _int_attr(owner, "dim") != 1:
            raise ValueError("the patch matrix concatenates its pieces along an axis other than the reduction")
        gather_ops.append(owner)
    try:
        read = [_gathered_piece(piece, gather_ops) for piece in pieces]
    except _NotAGather as why:
        if owner.name == "tensor.concat":
            raise ValueError(f"the activation is a concatenated gather whose image cannot be read back: {why}") from why
        return None
    images = {id(image) for image, *_rest in read}
    if len(images) != 1 or len({(s, size, ch) for _i, _o, s, size, ch in read}) != 1:
        raise ValueError("the gather's pieces do not read one image with one stride and one extent")
    source, _o, (sh, sw), (ho, wo), channels = read[0]
    taps = [offset for _i, offset, *_rest in read]
    tap_h, tap_w = sorted({h for h, _w in taps}), sorted({w for _h, w in taps})
    kh, kw = len(tap_h), len(tap_w)
    if tap_h != list(range(kh)) or tap_w != list(range(kw)) or len(set(taps)) != len(taps) or len(taps) != kh * kw:
        raise ValueError(f"the gather's taps {taps} are not a dense window starting at the padded origin")
    padded = _shape(source)
    if (padded[2] - kh) // sh + 1 != ho or (padded[3] - kw) // sw + 1 != wo:
        raise ValueError("the gather's extents do not follow from its window and stride")
    if ho * wo != rows or kh * kw * channels != reduced:
        raise ValueError("the gather's positions or reduction disagree with the contraction")
    image, pads = source, [0, 0, 0, 0]
    pad = _owner(source)
    if pad is not None and pad.name == "tensor.insert_slice":
        offsets, sizes, strides = _ints(pad, "static_offsets"), _ints(pad, "static_sizes"), _ints(pad, "static_strides")
        fill = _owner(pad.operands[1])
        value = _owner(fill.operands[0]) if fill is not None and fill.name == "tensor.splat" else None
        constant = getattr(value.properties.get("value"), "value", None) if value is not None else None
        if (
            offsets is None
            or sizes is None
            or strides != [1, 1, 1, 1]
            or offsets[:2] != [0, 0]
            or constant is None
            or int(constant.data) != 0
        ):
            raise ValueError("the gathered image is padded by something other than a static zero insertion")
        image = pad.operands[0]
        top, left = offsets[2], offsets[3]
        pads = [top, left, padded[2] - top - sizes[2], padded[3] - left - sizes[3]]
        gather_ops += [pad, fill, value]
    height, width = _shape(image)[2], _shape(image)[3]
    # The reduction's order as the gather lays it out, against the entry's [tap_h, tap_w, channel].
    order = np.array([((h * kw + w) * channels + ch) for h, w in taps for ch in range(channels)], dtype=np.int64)
    return {
        "image": image,
        "window": {
            "ci": channels,
            "Himg": height,
            "Wimg": width,
            "kh": kh,
            "kw": kw,
            "stride": [sh, sw],
            "padding": pads,
        },
        "rows": order,
    }


#: The host-gather recognition, shared by the group statement (``group_command.program``) and the route.
host_window_gather = _host_window_gather


def _nhwc(image) -> list:
    """``[empty, transpose]`` giving ``image`` (NCHW) in the convolution interface's NHWC layout."""
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import DenseArrayBase, TensorType, i64
    from xdsl.dialects.linalg import TransposeOp

    n, c, h, w = _shape(image)
    element = image.type.get_element_type()
    out = TensorType(element, [n, h, w, c])
    empty = tensor.EmptyOp((), out)
    return [empty, TransposeOp(image, empty.results[0], DenseArrayBase.from_list(i64, [0, 2, 3, 1]), out)]


def _nhwc_matrix(value) -> list:
    """``[empty, transpose, collapse]`` giving an NCHW ``value`` as the ``[positions, channels]`` matrix
    an elementwise interface declares (positions in row-major ``(h, w)`` order)."""
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, TensorType, i64

    n, c, h, w = _shape(value)
    empty, transpose = _nhwc(value)
    groups = ArrayAttr([ArrayAttr([IntegerAttr(i, i64) for i in (0, 1, 2)]), ArrayAttr([IntegerAttr(3, i64)])])
    collapse = tensor.CollapseShapeOp(
        operands=[transpose.results[0]],
        result_types=[TensorType(value.type.get_element_type(), [n * h * w, c])],
        properties={"reassociation": groups},
    )
    return [empty, transpose, collapse]


def _device_result(group, members, entry, result_type, *, elementwise: bool = False, batch: tuple = ()):
    """``(relayout, device_type, "")``: the type the device writes for ``entry`` and, when the group
    hands the capture that result in another layout, how the host reads it back -- or a reason.

    The device commits ``[positions, features]`` (:func:`group_command.device_output_shape`). A group
    whose own views and transpose turn that into NCHW (``[1, features, rows, cols]``; a fused pool
    after the transpose only shrinks rows and cols, which the device's pooled rows already are) gets
    the same views and transpose on the host, after the call; any other layout is refused by name."""
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import ArrayAttr, DenseArrayBase, IntegerAttr, TensorType, i64
    from xdsl.dialects.linalg import TransposeOp

    from merlin.xdsl_dialects.lowering import group_command as GC

    try:
        device = [int(v) for v in GC.device_output_shape(entry)]
    except GC.NoDeviceShape as why:
        return None, result_type, str(why)
    have = _shape(members[-1].results[0])
    element = result_type.get_element_type()
    device_type = TensorType(element, device)
    if have == device or (batch and have == [*map(int, batch), *device]):
        return None, result_type, ""
    transposes = [m for m in members if m.name == "linalg.transpose"]
    positions, features = device
    # A convolution group's own views and transpose take the device's rows to NCHW; an elementwise
    # group whose NCHW inputs were handed over as [positions, channels] is read back the same way.
    views_to_nchw = elementwise or (
        len(transposes) == 1
        and _ints(transposes[0], "permutation") == [0, 3, 1, 2]
        and len(have) == 4
        and len(_shape(transposes[0].operands[0])) == 4
        and _shape(transposes[0].operands[0])[0] == 1
        and _shape(transposes[0].operands[0])[3] == features
    )
    if len(have) != 4 or have[0] != 1 or have[1] != features or have[2] * have[3] != positions or not views_to_nchw:
        return None, result_type, f"the group hands the capture {have}, not the device's {device} or its NCHW view"
    nhwc = TensorType(element, [1, have[2], have[3], features])

    def relayout(value) -> list:
        groups = ArrayAttr([ArrayAttr([IntegerAttr(i, i64) for i in (0, 1, 2)]), ArrayAttr([IntegerAttr(3, i64)])])
        expand = tensor.ExpandShapeOp(value, (), groups, [1, have[2], have[3], features], nhwc)
        empty = tensor.EmptyOp((), result_type)
        transpose = TransposeOp(
            expand.results[0], empty.results[0], DenseArrayBase.from_list(i64, [0, 3, 1, 2]), result_type
        )
        return [expand, empty, transpose]

    return relayout, device_type, ""


#: Value-semantic ops that compute nothing but their result: erased once nothing reads them.
_PURE = frozenset(
    {
        "arith.constant",
        "linalg.fill",
        "linalg.generic",
        "linalg.transpose",
        "tensor.collapse_shape",
        "tensor.concat",
        "tensor.empty",
        "tensor.expand_shape",
        "tensor.extract_slice",
        "tensor.insert_slice",
        "tensor.splat",
    }
)


def _erase_dead(ops) -> None:
    """Erase the pure ops among ``ops`` -- and, transitively, their producers -- that nothing reads."""
    pending = [op for op in ops if op is not None]
    while pending:
        op = pending.pop()
        if op.parent is None or op.name not in _PURE or any(result.uses for result in op.results):
            continue
        producers = [_owner(value) for value in op.operands]
        op.detach()
        op.erase()
        pending.extend(producer for producer in producers if producer is not None)


def _logical_group_call(group, stated, entry, pair, row, arrays, why_no_prepack, *, weight_rows=None):
    """``(operands, constants, roles, "")`` -- what a logical-ABI group call passes besides its
    destination, in call order -- or ``(None, (), (), reason)``.

    The activation and the stored weight come from the module (the weight as the device program holds
    it: the prepack's laid-out copy when the program transposes or reorders the stored tensor); a fused
    bias is the prepack's folded accumulator-domain constant. An interface value with no host source is
    named in the reason rather than dropped from the call."""
    from xdsl.dialects.builtin import IntegerType

    if stated.stored_operand is None:
        return [pair[0], pair[1]], [], ["input_0", "input_1"], ""
    operands, constants, roles = [pair[0]], [], ["input_0", "weight_0"]
    weight_row = (row or {}).get("weight") or {}
    laid_out = arrays.get(weight_row.get("array")) if isinstance(weight_row.get("array"), str) else None
    relaid = bool(stated.transposed or stated.column_order or weight_rows is not None)
    if laid_out is not None and weight_rows is not None:
        # The rows a host gather laid out as (its taps, channels) go to the convolution interface's
        # [tap_h, tap_w, channel] order.
        reordered = laid_out.copy()
        reordered[weight_rows] = laid_out
        laid_out = reordered
    from xdsl.ir import BlockArgument

    # A WEIGHT THE HOST WOULD RE-LAY OUT ON EVERY INFERENCE (a transpose or view of the stored argument)
    # is read from the prepack's copy: the same bytes, computed once from the same weights file.
    computed_on_host = not isinstance(pair[1], BlockArgument)
    if laid_out is not None and (relaid or computed_on_host or list(laid_out.shape) != list(pair[1].type.get_shape())):
        constant = _constant_tensor(laid_out, pair[1].type.get_element_type())
        constants.append(constant)
        operands.append(constant.results[0])
    elif relaid:
        why = weight_row.get("refused") or why_no_prepack or "the prepack laid out no weight for this group"
        return None, (), (), f"interface value weight_0 has no host source in the program's layout: {why}"
    else:
        operands.append(pair[1])
    if "bias_add" in (entry.get("epilogue") or ()):
        bias_row = (row or {}).get("bias") or {}
        folded = arrays.get(bias_row.get("array")) if isinstance(bias_row.get("array"), str) else None
        if folded is None:
            why = why_no_prepack or "the prepack folded no bias for this group"
            return None, (), (), f"interface value bias_0 has no host source: {why}"
        constant = _constant_tensor(folded.reshape(-1), IntegerType(32))
        constants.append(constant)
        operands.append(constant.results[0])
        roles.append("bias_0")
    return operands, constants, roles, ""


def _reassociation(op) -> str:
    from merlin.common import mlir_query as mq

    for table in mq._attr_tables(op):  # noqa: SLF001 -- the shared property/attribute reader
        if table.get("reassociation") is not None:
            return str(table["reassociation"])
    return ""


def cancel_relayouts(views) -> dict[str, int]:
    """Carry the device's layout between device groups: cancel the relayouts the route inserted.

    The route reads each device result back to the capture's NCHW and hands each device input over
    in NHWC (or as ``[positions, channels]``). Where one device group's result feeds the next, that is
    a transpose followed by its inverse. Every such pair among ``views`` (the ops the route inserted)
    is replaced by the value it started from, and a ``collapse_shape`` of the ``expand_shape`` it undoes
    by the original value -- both exact identities, so the program computes the same elements. A
    relayout whose value also reaches a host op keeps that consumer; only what nothing reads any more
    is removed, so a single relayout survives at each true host boundary."""
    cancelled = folded = 0
    for op in list(views):
        if op.parent is None or op.name != "linalg.transpose":
            continue
        before = _owner(op.operands[0])
        if before is None or before.name != "linalg.transpose" or before.parent is None:
            continue
        outer, inner = _ints(op, "permutation"), _ints(before, "permutation")
        if outer is None or inner is None or len(outer) != len(inner):
            continue
        if any(inner[outer[axis]] != axis for axis in range(len(outer))):
            continue
        op.results[0].replace_all_uses_with(before.operands[0])
        _erase_dead([op])
        cancelled += 1
    # One relayout per host value: two device groups reading the same host tensor share its transpose.
    seen: dict[tuple, Any] = {}
    for op in list(views):
        if op.parent is None or op.name != "linalg.transpose":
            continue
        key = (id(op.operands[0]), tuple(_ints(op, "permutation") or ()), str(op.results[0].type))
        first = seen.get(key)
        if first is None or first.parent is not op.parent:
            seen[key] = op
            continue
        position = {id(o): i for i, o in enumerate(op.parent.ops)}
        keep, drop = (first, op) if position[id(first)] < position[id(op)] else (op, first)
        drop.results[0].replace_all_uses_with(keep.results[0])
        _erase_dead([drop])
        seen[key] = keep
        cancelled += 1
    for op in list(views):
        if op.parent is None or op.name != "tensor.collapse_shape":
            continue
        source = _owner(op.operands[0])
        if source is None or source.name != "tensor.expand_shape" or source.parent is None:
            continue
        if source.operands[0].type != op.results[0].type or _reassociation(source) != _reassociation(op):
            continue
        op.results[0].replace_all_uses_with(source.operands[0])
        _erase_dead([op])
        folded += 1
    return {"transposes_cancelled": cancelled, "views_folded": folded}
