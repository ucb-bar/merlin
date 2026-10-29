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


def _prepack_arrays(groups, capture) -> tuple[dict | None, dict[int, dict[str, Any]], dict[str, Any], str]:
    """``(record, rows by group, arrays, "")`` of the prepack, or ``(None, {}, {}, reason)``.

    The host sources a logical-ABI call passes for a group's constant interface values. One
    definition: the same :func:`merlin.xdsl_dialects.lowering.group_prepack.prepack` the sidecar
    records."""
    if capture is None:
        return None, {}, {}, "no capture was given, so no weights manifest could be read and no bias folded"
    beside = Path(capture)
    beside = beside if beside.is_dir() else beside.parent
    from .weights_pack import weight_files_beside

    manifest, weights = weight_files_beside(capture)
    if manifest is None or weights is None:
        return None, {}, {}, f"no weights manifest and safetensors pair beside {beside}"
    from merlin.xdsl_dialects.lowering import group_prepack as GP

    try:
        result = GP.prepack(groups, manifest, weights, device_layout=True)
    except Exception as error:  # noqa: BLE001 -- named, never a silently absent prepack
        return None, {}, {}, f"prepack: {type(error).__name__}: {error}"
    rows = {int(row["group"]): row for row in result["record"]["groups"]}
    return result["record"], rows, result["arrays"], ""


def _routed_prepack(record, groups):
    """Select the actual calls from the immutable pre-mutation preparation record."""
    if record is None or not groups:
        return None
    selected = {group.index for group in groups}
    rows = [row for row in record["groups"] if row["group"] in selected]
    skipped = [row for row in record["skipped"] if row["group"] in selected]
    return {
        **record,
        "groups": rows,
        "skipped": skipped,
        "folded_bias_elements": sum(row.get("bias", {}).get("elements", 0) for row in rows),
        "device_weights": sum("array" in (row.get("weight") or {}) for row in rows),
    }


def _constant_tensor(array, element_type):
    """``arith.constant`` holding ``array`` as a dense tensor of ``element_type``."""
    import numpy as np
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import BytesAttr, DenseIntOrFPElementsAttr, IntegerType, Signedness, TensorType

    shape = [int(v) for v in array.shape]
    tensor_type = TensorType(element_type, shape)
    if (
        type(array) is np.ndarray
        and isinstance(element_type, IntegerType)
        and element_type.width.data in (8, 16, 32, 64)
        and array.dtype.kind in "iu"
        and array.dtype.itemsize * 8 == element_type.width.data
        and (
            element_type.signedness.data == Signedness.SIGNLESS
            or array.dtype.kind == ("u" if element_type.signedness.data == Signedness.UNSIGNED else "i")
        )
    ):
        # xDSL's integer storage is little-endian. Matching-width integer
        # arrays already carry its normalized bits, including unsigned values
        # normalized to the signed representation of signless integers.
        payload = np.ascontiguousarray(array, dtype=np.dtype(element_type.format)).tobytes()
        return arith.ConstantOp(DenseIntOrFPElementsAttr(tensor_type, BytesAttr(payload)))
    values = [int(v) for v in array.reshape(-1).tolist()]
    return arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(tensor_type, values))


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


def _plain_column_concat(owner, rows: int, reduced: int) -> bool:
    """A complete rank-two argument/slice assembly is not a spatial gather.

    Do not infer this for arbitrary computed/view pieces: ambiguous or mixed
    window gathers retain the existing refusal. Every slice preserves the full
    row axis and reads a static, in-bounds column interval of a rank-two input.
    """
    from xdsl.ir import BlockArgument

    if owner.name != "tensor.concat" or _int_attr(owner, "dim") != 1 or len(owner.results) != 1:
        return False
    if _shape(owner.results[0]) != [rows, reduced] or not owner.operands:
        return False
    total = 0
    for piece in owner.operands:
        shape = _shape(piece)
        if len(shape) != 2 or shape[0] != rows or shape[1] < 1:
            return False
        total += shape[1]
        seen = set()
        while (op := _owner(piece)) is not None and op.name == "tensor.extract_slice":
            if id(op) in seen or len(op.operands) != 1:
                return False
            seen.add(id(op))
            offsets, sizes, strides = (
                _ints(op, "static_offsets"),
                _ints(op, "static_sizes"),
                _ints(op, "static_strides"),
            )
            source = _shape(op.operands[0])
            if (
                len(source) != 2
                or source[0] != rows
                or offsets is None
                or sizes != _shape(piece)
                or len(offsets) != 2
                or offsets[0] != 0
                or strides != [1, 1]
                or offsets[1] < 0
                or offsets[1] + sizes[1] > source[1]
            ):
                return False
            piece = op.operands[0]
        if not isinstance(piece, BlockArgument) or len(_shape(piece)) != 2:
            return False
    return total == reduced


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
            if _plain_column_concat(owner, rows, reduced):
                return None
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


def _contiguous_result_views(group, members, device, *, allow_no_views=False):
    """Prove the root's complete ordered result path, retaining only its static views.

    Arithmetic admission remains the group's ordinary source/readout proof. Here each intervening
    pointwise stage must additionally preserve coordinates; equal element counts alone prove nothing.
    """
    from math import prod

    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import NoneAttr, TensorType
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.dialects.linalg.ops import TransposeOp
    from xdsl.ir.affine import AffineMap
    from xdsl.traits import is_side_effect_free

    from merlin.common import mlir_query as mq
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    if group.root not in members or len(group.root.results) != 1 or _shape(group.root.results[0]) != device:
        return None
    value, views, pending_transpose = group.root.results[0], [], None
    for op in members[members.index(group.root) + 1:]:
        if (
            op.parent is not group.root.parent
            or len(op.results) != 1
            or sum(operand is value for operand in op.operands) != 1
            or any(use.operation is not op for use in value.uses)
        ):
            return None
        if any(not isinstance(v.type, TensorType) or v.type.encoding != NoneAttr() for v in (value, op.results[0])):
            return None
        shape, result = _shape(value), _shape(op.results[0])
        if not shape or not result or any(n < 1 for n in (*shape, *result)):
            return None
        if isinstance(op, TransposeOp):
            permutation = _ints(op, "permutation")
            if (len(op.operands) != 2 or op.operands[0] is not value
                    or not isinstance(_owner(op.operands[1]), tensor.EmptyOp)
                    or op.operands[1].type != op.results[0].type
                    or value.type.element_type != op.results[0].type.element_type
                    or set(op.properties) != {"permutation"}
                    or any(not key.startswith("prov.") for key in op.attributes)
                    or permutation is None or sorted(permutation) != list(range(len(shape)))
                    or result != [shape[axis] for axis in permutation]):
                return None
            if pending_transpose is None:
                pending_transpose = op
            else:
                first = _ints(pending_transpose, "permutation")
                if (first is None or len(first) != len(permutation)
                        or any(first[permutation[axis]] != axis for axis in range(len(first)))
                        or pending_transpose.operands[0].type != op.results[0].type):
                    return None
                pending_transpose = None
            value = op.results[0]
            continue
        if pending_transpose is not None:
            return None
        if isinstance(op, tensor.CollapseShapeOp | tensor.ExpandShapeOp):
            if len(op.operands) != 1 or value.type.element_type != op.results[0].type.element_type:
                return None
            expanded, collapsed = (shape, result) if isinstance(op, tensor.CollapseShapeOp) else (result, shape)
            groups = [[int(axis.value.data) for axis in group] for group in op.reassociation]
            allowed = {"reassociation"}
            if isinstance(op, tensor.ExpandShapeOp):
                allowed.add("static_output_shape")
                if list(op.static_output_shape.get_values()) != result:
                    return None
            if (
                set(op.properties) != allowed
                or any(not key.startswith("prov.") for key in op.attributes)
                or len(groups) != len(collapsed)
                or any(not group for group in groups)
                or [axis for group in groups for axis in group] != list(range(len(expanded)))
                or [prod(expanded[axis] for axis in group) for group in groups] != collapsed
            ):
                return None
            views.append(op)
        else:
            stage = CG.classify(op)
            if shape != result or stage is None or stage.kind not in (
                CG.CAST, CG.SCALE, CG.BIAS_ADD, CG.RELU, CG.CLAMP, CG.ROUND, CG.QUANTIZE,
            ):
                return None
            index = next(i for i, operand in enumerate(op.operands) if operand is value)
            if mq.op_name(op) == "quant_ext.quantize_per_tensor":
                if index != 0 or len(op.operands) != 3 or op.regions:
                    return None
            elif mq.op_name(op) == "linalg.generic":
                identity = AffineMap.identity(len(shape))
                maps = op.indexing_maps.data
                body = list(op.body.block.ops)
                if (
                    index >= len(op.inputs)
                    or len(maps) != len(op.operands)
                    or maps[index].data != identity
                    or maps[-1].data != identity
                    or len(op.iterator_types) != len(shape)
                    or any(it.data != IteratorType.PARALLEL for it in op.iterator_types)
                    or not body or body[-1].name != "linalg.yield"
                    or any(not is_side_effect_free(inner) for inner in body[:-1])
                ):
                    return None
            else:
                return None
        value = op.results[0]
    return views if pending_transpose is None and (views or allow_no_views) else None


def _contiguous_residual_matrix_view(group, members, entry):
    """Prove a source-owned residual's dense leading-axis matrix view.

    The independently admitted residual still owns its arithmetic and quantization. This
    This proof only collapses consecutive leading axes in its pointer ABI. The original add
    must use identity coordinates, and its complete nonescaping readout must retain them.
    """
    from math import prod

    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, NoneAttr, TensorType, i64
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.ir.affine import AffineMap
    from xdsl.traits import is_side_effect_free

    from merlin.xdsl_dialects.lowering import group_command as GC

    root = getattr(group, "root", None)
    if entry.get("op") != "residual_add" or getattr(group, "operand_sum", None) is None or root is None:
        return None
    if root.name != "linalg.generic" or len(root.inputs) != 2 or len(root.outputs) != 1 or len(root.results) != 1:
        return None
    shape = _shape(root.results[0])
    device = GC.device_output_shape(entry)
    if (len(shape) <= 2 or any(n < 1 for n in (*shape, *device))
            or [prod(shape[:-1]), shape[-1]] != device):
        return None
    values = [*root.operands, root.results[0]]
    if any(not isinstance(v.type, TensorType) or v.type.encoding != NoneAttr() or _shape(v) != shape for v in values):
        return None
    if any(v.type.element_type != root.results[0].type.element_type for v in root.operands):
        return None
    body = list(root.body.block.ops)
    if (
        any(m.data != AffineMap.identity(len(shape)) for m in root.indexing_maps)
        or len(root.indexing_maps) != len(root.operands)
        or len(root.iterator_types) != len(shape)
        or any(it.data != IteratorType.PARALLEL for it in root.iterator_types)
        or not body or body[-1].name != "linalg.yield"
        or any(not is_side_effect_free(op) for op in body[:-1])
        or _contiguous_result_views(group, members, shape, allow_no_views=True) != []
    ):
        return None
    # The row axis is exactly the ordered product of the original leading axes.
    return ArrayAttr([
        ArrayAttr([IntegerAttr(i, i64) for i in range(len(shape) - 1)]),
        ArrayAttr([IntegerAttr(len(shape) - 1, i64)]),
    ])


def _device_result(
    group, members, entry, result_type, *, elementwise: bool = False, batch: tuple = (), transposed: bool = False,
):
    """``(relayout, device_type, "")``: the type the device writes for ``entry`` and, when the group
    hands the capture that result in another layout, how the host reads it back -- or a reason.

    The device commits ``[positions, features]`` (:func:`group_command.device_output_shape`). A group
    whose own views and transpose turn that into NCHW (``[1, features, rows, cols]``; a fused pool
    after the transpose only shrinks rows and cols, which the device's pooled rows already are) gets
    the same views and transpose on the host, after the call. A non-transposed matrix may also replay
    its explicitly proved complete contiguous views through coordinate-preserving readout stages;
    any other layout is refused by name."""
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
    residual_view = _contiguous_residual_matrix_view(group, members, entry) if not batch and not transposed else None
    if residual_view is not None:
        def relayout(value) -> list:
            return [tensor.ExpandShapeOp(value, (), residual_view, have, result_type)]

        return relayout, device_type, ""
    views = (
        _contiguous_result_views(group, members, device)
        if entry.get("op") in ("matmul", "conv2d") and not batch and not transposed else None
    )
    if views:
        from math import prod

        if len(have) > 2 and have[-1] == device[1] and prod(have[:-1]) == device[0]:
            groups = ArrayAttr([
                ArrayAttr([IntegerAttr(i, i64) for i in range(len(have) - 1)]),
                ArrayAttr([IntegerAttr(len(have) - 1, i64)]),
            ])

            def relayout(value) -> list:
                return [tensor.ExpandShapeOp(value, (), groups, have, result_type)]

            return relayout, device_type, ""

        def relayout(value) -> list:
            result = []
            for view in views:
                out = TensorType(element, _shape(view.results[0]))
                if isinstance(view, tensor.CollapseShapeOp):
                    op = tensor.CollapseShapeOp(
                        operands=[value], result_types=[out], properties={"reassociation": view.reassociation},
                    )
                else:
                    op = tensor.ExpandShapeOp(value, (), view.reassociation, list(out.get_shape()), out)
                result.append(op)
                value = op.results[0]
            return result

        return relayout, device_type, ""
    if group.window_mean is not None and device[1] == 1 and have in ([1, device[0]], [device[0]]):
        from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, i64

        def relayout(value) -> list:
            groups = ArrayAttr([ArrayAttr([IntegerAttr(i, i64) for i in (0, 1)])])
            collapsed = tensor.CollapseShapeOp(
                operands=[value],
                result_types=[TensorType(element, [device[0]])],
                properties={"reassociation": groups},
            )
            if len(have) == 1:
                return [collapsed]
            expanded = tensor.ExpandShapeOp(collapsed.results[0], (), groups, [1, device[0]], result_type)
            return [collapsed, expanded]

        return relayout, device_type, ""
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


def _logical_group_call(group, stated, entry, pair, row, arrays, why_no_prepack, *, weight_rows=None, bind_weight=None,
                        constant_panel_columns=None, constant_panel_minimum_input_channels=1):
    """``(operands, constants, roles, "")`` -- what a logical-ABI group call passes besides its
    destination, in call order -- or ``(None, (), (), reason)``.

    The activation and the stored weight come from the module (the weight as the device program holds
    it: the prepack's laid-out copy when the program transposes or reorders the stored tensor); a fused
    bias is the prepack's folded accumulator-domain constant. An interface value with no host source is
    named in the reason rather than dropped from the call. ``bind_weight(array, element_type)``, when
    given, supplies the laid-out weight as a value (:class:`merlin.llvmlower.blob_weights.WeightArgs`)
    instead of an ``arith.constant``."""
    if group.window_pool is not None:
        return [pair[0]], [], ["input_0"], ""
    if group.window_mean is not None:
        from xdsl.dialects import tensor
        from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, TensorType, i64

        rows, window = int(group.window_mean["rows"]), int(group.window_mean["window"])
        shape = _shape(pair[0])
        if shape == [rows, window]:
            activation, views = pair[0], []
        elif len(shape) == 4 and shape[0] == 1 and shape[1] == rows and shape[2] * shape[3] == window:
            groups = ArrayAttr(
                [
                    ArrayAttr([IntegerAttr(i, i64) for i in (0, 1)]),
                    ArrayAttr([IntegerAttr(i, i64) for i in (2, 3)]),
                ]
            )
            collapse = tensor.CollapseShapeOp(
                operands=[pair[0]],
                result_types=[TensorType(pair[0].type.get_element_type(), [rows, window])],
                properties={"reassociation": groups},
            )
            activation, views = collapse.results[0], [collapse]
        else:
            return None, (), (), f"window mean activation shape {shape} is not [{rows}, {window}]"
        weight_row = (row or {}).get("weight") or {}
        laid_out = arrays.get(weight_row.get("array")) if isinstance(weight_row.get("array"), str) else None
        if laid_out is None or list(laid_out.shape) != [window, 1] or not bool((laid_out == 1).all()):
            return None, (), (), "window mean prepack has no exact constant-one operand"
        ones = _constant_tensor(laid_out, activation.type.get_element_type())
        return [activation, ones.results[0]], [*views, ones], ["input_0", "weight_0"], ""
    if stated.stored_operand == 0 and not stated.transposed:
        from merlin.xdsl_dialects.lowering import group_command as GC

        if not GC.stationary_is_activation(entry):
            return None, (), (), "an immutable left operand needs an explicitly activation-stationary statement"
        # The prepared immutable bytes belong to the left operand, while the
        # runtime right operand still occupies the interface's weight slot.
        # Do not reinterpret either as the other's preparation.
        weight_row = (row or {}).get("weight") or {}
        laid_out = arrays.get(weight_row.get("array")) if isinstance(weight_row.get("array"), str) else None
        constants = []
        left = pair[0]
        if laid_out is not None:
            if list(laid_out.shape) != list(left.type.get_shape()):
                return None, (), (), "the immutable left preparation differs from its source-oriented matrix shape"
            if bind_weight is not None:
                left = bind_weight(laid_out, left.type.get_element_type())
            else:
                constant = _constant_tensor(laid_out, left.type.get_element_type())
                constants.append(constant)
                left = constant.results[0]
        return _with_logical_bias(entry, [left, pair[1]], constants,
                                  ["input_0", "weight_0"], row, arrays, why_no_prepack)
    if stated.stored_operand is None:
        # AN ACTIVATION IN THE STATIONARY SLOT binds where a weight would (as the whole-program route binds
        # it): a contraction stated with a stationary activation holds its right operand in the weight
        # role, packed and evicted within the call, so the call hands this per-inference tensor over as
        # that operand. Any other two-input group (a residual add) states both operands as inputs.
        from merlin.xdsl_dialects.lowering import group_command as GC

        second = "weight_0" if GC.stationary_is_activation(entry) else "input_1"
        residual_view = _contiguous_residual_matrix_view(group, group.members, entry)
        views = []
        if residual_view is not None:
            from xdsl.dialects import tensor
            from xdsl.dialects.builtin import NoneAttr, TensorType

            shape = _shape(group.root.results[0])
            if any(
                not isinstance(value.type, TensorType) or value.type.encoding != NoneAttr() or _shape(value) != shape
                for value in pair
            ) or pair[0].type != pair[1].type:
                return None, (), (), "residual inputs have no identical dense source-owned matrix view"
            for value in pair:
                views.append(tensor.CollapseShapeOp(
                    operands=[value], result_types=[TensorType(value.type.element_type, GC.device_output_shape(entry))],
                    properties={"reassociation": residual_view},
                ))
            pair = tuple(view.results[0] for view in views)
        return _with_logical_bias(entry, [pair[0], pair[1]], views, ["input_0", second], row, arrays, why_no_prepack)
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

    if constant_panel_columns is not None:
        from .constant_panel_layout import prepare_constant_panels

        laid_out, storage = prepare_constant_panels(entry, laid_out, constant_panel_columns,
                                                   pair[1].type.get_element_type(),
                                                   minimum_input_channels=constant_panel_minimum_input_channels)
        if storage is not None:
            entry["weight_storage_encoding"] = storage.to_dict()
            relaid = True

    # A WEIGHT THE HOST WOULD RE-LAY OUT ON EVERY INFERENCE (a transpose or view of the stored argument)
    # is read from the prepack's copy: the same bytes, computed once from the same weights file.
    computed_on_host = not isinstance(pair[1], BlockArgument)
    if laid_out is not None and (relaid or computed_on_host or list(laid_out.shape) != list(pair[1].type.get_shape())):
        if bind_weight is not None:
            operands.append(bind_weight(laid_out, pair[1].type.get_element_type()))
        else:
            constant = _constant_tensor(laid_out, pair[1].type.get_element_type())
            constants.append(constant)
            operands.append(constant.results[0])
    elif relaid:
        why = weight_row.get("refused") or why_no_prepack or "the prepack laid out no weight for this group"
        return None, (), (), f"interface value weight_0 has no host source in the program's layout: {why}"
    else:
        operands.append(pair[1])
    return _with_logical_bias(entry, operands, constants, roles, row, arrays, why_no_prepack)


def _with_logical_bias(entry, operands, constants, roles, row, arrays, why_no_prepack):
    """Append the prepack's declared signed integer bias for either operand ownership form."""
    from xdsl.dialects.builtin import IntegerType

    if "bias_add" in (entry.get("epilogue") or ()):
        bias_row = (row or {}).get("bias") or {}
        folded = arrays.get(bias_row.get("array")) if isinstance(bias_row.get("array"), str) else None
        if folded is None:
            why = why_no_prepack or "the prepack folded no bias for this group"
            return None, (), (), f"interface value bias_0 has no host source: {why}"
        if folded.dtype.kind != "i":
            return None, (), (), "interface value bias_0 is not a signed accumulator-domain integer"
        constant = _constant_tensor(folded.reshape(-1), IntegerType(folded.dtype.itemsize * 8))
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
    transposes = list(views)
    # An original input-layout transpose can directly undo a generated result
    # transpose. Observe only that use relation, then apply the same exact
    # inverse proof below; arbitrary surrounding operations remain untouched.
    for op in views:
        if op.parent is not None and op.name == "linalg.transpose":
            transposes.extend(use.operation for use in op.results[0].uses
                              if use.operation.name == "linalg.transpose" and use.operation.parent is op.parent)
    for op in transposes:
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
