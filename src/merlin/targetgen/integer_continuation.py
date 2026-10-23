"""Bound exact integer host work between a device producer and its rational readout."""

from __future__ import annotations

import math


def readout_continuation(contraction):
    """Find one readout through closed integer copies and bounded parallel arithmetic.

    Every intermediate is bounded independently of the sample values. Unknown
    operations, reductions, dynamic slices, integer overflow and floating casts
    fail closed. The rational boundary itself is screened by its ordinary owner.
    """
    from xdsl.utils.exceptions import VerifyException

    from merlin.common import mlir_query as mq
    from merlin.kernels.shapes import observe_contractions, zero_initialised
    from merlin.targetgen.core_aten_spec_int8 import closed_integer_mac

    bounds, visiting, checked = {}, set(), {}
    copies = {"tensor.collapse_shape", "tensor.expand_shape", "tensor.extract_slice", "linalg.transpose"}

    def shape(value):
        result = tuple(value.type.get_shape())
        if any(size <= 0 for size in result):
            raise ValueError("integer continuation requires positive static shapes")
        return result

    def bounded(value, interval):
        bits = getattr(value.type, "element_type", value.type)
        if str(bits) not in {"i32", "i64"}:
            raise ValueError("integer continuation requires signed full-width values")
        width = int(str(bits)[1:])
        low, high = interval
        if not -(2 ** (width - 1)) <= low <= high < 2 ** (width - 1):
            raise ValueError("integer continuation may overflow before its exact readout")
        return interval

    def slice_geometry(op, source, destination):
        offsets, sizes, strides = [
            list(op.properties[key].iter_values()) for key in ("static_offsets", "static_sizes", "static_strides")
        ]
        extents = shape(destination if op.name == "tensor.insert_slice" else source)
        if len(offsets) != len(extents) or not len(offsets) == len(sizes) == len(strides):
            raise ValueError("integer continuation slice rank mismatch")
        if any(
            o < 0 or s <= 0 or t <= 0 or o + (s - 1) * t >= e
            for o, s, t, e in zip(offsets, sizes, strides, extents, strict=True)
        ):
            raise ValueError("integer continuation needs a static in-bounds slice")

    def constant(op):
        value = op.properties.get("value", op.attributes.get("value"))
        if hasattr(value, "get_values"):
            values = list(value.get_values())
        else:
            values = [value.value.data]
        if not values or any(type(v) is not int for v in values):
            raise ValueError("integer continuation needs literal integer constants")
        return min(values), max(values)

    def interval(value):
        if value in bounds:
            return bounds[value]
        if value in visiting:
            raise ValueError("cyclic integer continuation")
        visiting.add(value)
        op = value.owner
        name = getattr(op, "name", None)
        if name == "arith.constant":
            result = constant(op)
        elif name == "tensor.splat":
            result = interval(op.operands[0])
        elif closed_integer_mac(op):
            if not zero_initialised(op):
                raise ValueError("integer continuation MAC requires zero initialization")
            (match,) = observe_contractions(op)
            reduction = math.prod(match[1].reduction)
            width = op.operands[0].type.element_type.width.data
            magnitude = (1 << (width - 1)) ** 2
            result = (-magnitude * reduction, magnitude * reduction)
        elif name in copies:
            shape(value)
            if name == "tensor.extract_slice":
                slice_geometry(op, op.operands[0], value)
            elif name in {"tensor.collapse_shape", "tensor.expand_shape"}:
                if math.prod(shape(op.operands[0])) != math.prod(shape(value)):
                    raise ValueError("integer continuation reshape changes element count")
            op.verify()
            result = interval(op.operands[0])
        elif name == "tensor.insert_slice":
            slice_geometry(op, op.operands[0], op.operands[1])
            left, right = map(interval, op.operands[:2])
            result = min(left[0], right[0]), max(left[1], right[1])
        elif name == "linalg.generic":
            from merlin.kernels.shapes import _iterator_types

            if any(kind != "parallel" for kind in (_iterator_types(op) or [None])):
                raise ValueError("integer continuation requires parallel scalar arithmetic")
            block = op.body.block
            local = {arg: interval(source) for arg, source in zip(block.args[: len(op.inputs)], op.inputs, strict=True)}
            result = None
            for inner in block.ops:
                if inner.name == "linalg.yield":
                    result = local[inner.operands[0]]
                    continue
                if inner.name == "arith.constant":
                    pair = constant(inner)
                else:
                    args = [local[v] for v in inner.operands]
                    if inner.name == "arith.extsi":
                        pair = args[0]
                    elif inner.name == "arith.addi":
                        pair = args[0][0] + args[1][0], args[0][1] + args[1][1]
                    elif inner.name == "arith.subi":
                        pair = args[0][0] - args[1][1], args[0][1] - args[1][0]
                    elif inner.name == "arith.muli":
                        products = [a * b for a in args[0] for b in args[1]]
                        pair = min(products), max(products)
                    else:
                        raise ValueError("unsupported integer continuation scalar operation")
                    flags = getattr(inner, "overflow_flags", None)
                    if flags is not None and flags.data:
                        raise ValueError("integer continuation has overflow assumptions")
                local[inner.results[0]] = bounded(inner.results[0], pair)
            if result is None:
                raise ValueError("integer continuation has no scalar result")
        else:
            raise ValueError(f"unproved integer continuation operation {name}")
        result = bounded(value, result)
        bounds[value] = result
        visiting.remove(value)
        if hasattr(value.type, "get_shape"):
            checked[op] = result
        return result

    pending, seen, readers = [contraction.results[0]], set(), set()
    try:
        while pending:
            value = pending.pop()
            if value in seen:
                continue
            seen.add(value)
            interval(value)
            for use in value.uses:
                op = use.operation
                if mq.attr_str(op, "prov.role") == "readout":
                    readers.add(op)
                elif op.name in copies | {"tensor.insert_slice", "linalg.generic"}:
                    if len(op.results) != 1:
                        raise ValueError("integer continuation requires one result")
                    pending.append(op.results[0])
                else:
                    raise ValueError(f"unsupported integer continuation user {op.name}")
        if len(readers) != 1:
            return None
    except (AttributeError, KeyError, TypeError, ValueError, VerifyException):
        return None
    (reader,) = readers
    root = contraction
    while root.parent_op() is not None:
        root = root.parent_op()
    ordinals = {op: i for i, op in enumerate(root.walk())}
    return reader, dict(
        status="proven",
        method="static_integer_interval_propagation",
        operations=[dict(operation=op.name, minimum=lo, maximum=hi) for op, (lo, hi) in checked.items()],
        retained_operation_ordinals=sorted(
            {ordinals[value.owner] for value in seen if value.owner is not contraction} | {ordinals[reader]}
        ),
    )
