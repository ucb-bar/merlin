"""Reconstruct destination writes and complete unchanged source/effect wiring.

This reader does not call the rewrite's slab generator or consume its report.
In-bounds disjoint rectangles with the full volume prove complete coverage
without enumerating elements. Physical aliasing remains upstream's obligation.
"""

from __future__ import annotations

from math import prod

from xdsl.dialects import arith, builtin, func, tensor
from xdsl.dialects.linalg.ops import FillOp, GenericOp
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap

from .quant_round_check import pure_scalar_body


def _shape(value):
    typ = value.type
    if (not isinstance(typ, builtin.TensorType) or not isinstance(typ.encoding, builtin.NoneAttr)
            or any(n <= 0 for n in typ.get_shape())):
        raise ValueError("destination check requires positive static unencoded tensors")
    return tuple(typ.get_shape())


def _rectangle(op, base, shape):
    if set(op.properties) - {"static_offsets", "static_sizes", "static_strides", "operandSegmentSizes"}:
        raise ValueError("destination region has unknown properties")
    if op.operands[1 if isinstance(op, tensor.InsertSliceOp) else 0] is not base:
        raise ValueError("destination chain changed its backing tensor")
    offsets, sizes, strides = (tuple(field.get_values()) for field in (
        op.static_offsets, op.static_sizes, op.static_strides))
    if (op.offsets or op.sizes or op.strides or len(offsets) != len(shape)
            or len(sizes) != len(shape) or strides != (1,) * len(shape)
            or any(o < 0 or n <= 0 or o + n > d for o, n, d in zip(offsets, sizes, shape, strict=True))):
        raise ValueError("destination chain has an unsupported or out-of-bounds region")
    return offsets, sizes


def _disjoint(a, b):
    return any(x + nx <= y or y + ny <= x for x, nx, y, ny in zip(*a, *b, strict=True))


def _header(left, right, values):
    return (
        left.name == right.name and _metadata(left) == _metadata(right)
        and not left.successors and not right.successors
        and tuple(values.get(v) for v in left.operands) == tuple(right.operands)
        and tuple(v.type for v in left.results) == tuple(v.type for v in right.results)
        and len(left.regions) == len(right.regions)
    )


def _metadata(operation):
    # Float attribute equality can equate -0 and +0. Canonical typed printing
    # retains their distinct words and all nested attribute/property fields.
    return tuple(tuple(sorted((key, str(value)) for key, value in fields.items()))
                 for fields in (operation.attributes, operation.properties))


def _same_body(left, right):
    return left.is_structurally_equivalent(right) and tuple(_metadata(op) for op in left.walk()) == tuple(
        _metadata(op) for op in right.walk())


def _realization(original, producer, actual, start, values):
    """Read one actual empty/exterior/interior write sequence from operation IR."""
    source_shape, shape = _shape(original.source), _shape(original.dest)
    pad = original.dest.owner
    if (not isinstance(original.parent_op(), func.FuncOp)
            or not isinstance(pad, tensor.SplatOp) or pad.dynamicSizes or len(producer.outputs) != 1
            or producer.outputs[0].owner.name != "tensor.empty"
            or producer.indexing_maps.data[-1].data != AffineMap.identity(len(source_shape))
            or any(it.data.value != "parallel" for it in producer.iterator_types)
            or producer.body.block.args[-1].uses or not pure_scalar_body(producer.body.block)):
        raise ValueError("destination check refuses an unproved source producer")
    if set(producer.properties) - {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
        raise ValueError("destination source has unknown producer properties")
    for value, attr in zip(producer.inputs, producer.indexing_maps, strict=False):
        input_shape = _shape(value)
        mapping = attr.data
        if mapping.num_symbols or mapping.num_dims != len(source_shape) or len(mapping.results) != len(input_shape):
            raise ValueError("destination source has an unproved input indexing domain")
        axes = []
        for expression, size in zip(mapping.results, input_shape, strict=True):
            if isinstance(expression, AffineDimExpr):
                if not 0 <= expression.position < len(source_shape) or size != source_shape[expression.position]:
                    raise ValueError("destination input shapes do not establish the complete writer domain")
                axes.append(expression.position)
            elif not isinstance(expression, AffineConstantExpr) or expression.value != 0 or size != 1:
                raise ValueError("destination source has an unproved input coordinate")
        if len(set(axes)) != len(axes):
            raise ValueError("destination source has a repeated input axis")
    ancestor = original.parent_op()
    while ancestor is not None:
        allowed = {"llvm.emit_c_interface"} if isinstance(ancestor, func.FuncOp) else (
            {"llvm.data_layout", "llvm.target_triple"} if isinstance(ancestor, builtin.ModuleOp) else set()
        )
        if any(k not in allowed and not k.startswith("prov.") for k in ancestor.attributes):
            raise ValueError("destination source has an unknown enclosing contract")
        ancestor = ancestor.parent_op()
    for op in (producer, original, pad, producer.outputs[0].owner):
        if any(not key.startswith("prov.") for key in op.attributes):
            raise ValueError("destination source has unknown semantic attributes")
    original_region = _rectangle(original, original.dest, shape)
    if original_region[1] != source_shape:
        raise ValueError("original insertion does not cover the complete producer")
    # Arithmetic cannot move across unknown effects or other computations.
    cursor = producer.next_op
    while cursor is not original:
        if cursor is None or not isinstance(cursor, (arith.ConstantOp, tensor.SplatOp, tensor.EmptyOp)):
            raise ValueError("destination rewrite crossed an execution barrier")
        if any(not key.startswith("prov.") for key in cursor.attributes):
            raise ValueError("destination rewrite crossed unknown source attributes")
        cursor = cursor.next_op
    empty = actual[start]
    if (not isinstance(empty, tensor.EmptyOp) or empty.attributes or empty.dynamic_sizes
            or set(empty.properties) - {"operandSegmentSizes"}
            or empty.tensor.type != original.dest.type):
        raise ValueError("destination realization has no fresh complete parent")
    generated, regions = [empty], []
    current, i = empty.tensor, start + 1
    while i + 2 < len(actual):
        view, writer, insert = actual[i:i + 3]
        if not isinstance(view, tensor.ExtractSliceOp):
            raise ValueError("destination realization lacks an exact view")
        region = _rectangle(view, current, shape)
        if view.attributes or _shape(view.result) != region[1]:
            raise ValueError("destination view changed shape or semantics")
        if not isinstance(insert, tensor.InsertSliceOp) or _rectangle(insert, current, shape) != region:
            raise ValueError("destination writer does not return to its exact view")
        if isinstance(writer, FillOp):
            body = writer.regions[0].block
            if (tuple(writer.inputs) != (values[pad.input],) or tuple(writer.outputs) != (view.result,)
                    or writer.attributes or set(writer.properties) - {"operandSegmentSizes"}
                    or len(writer.results) != 1 or writer.results[0].type != view.result.type
                    or len(body.args) != 2 or len(tuple(body.ops)) != 1
                    or body.last_op.name != "linalg.yield" or tuple(body.last_op.operands) != (body.args[0],)
                    or insert.attributes
                    or insert.source is not writer.results[0]):
                raise ValueError("destination exterior changed its fill value or scalar semantics")
            regions.append(region)
            generated.extend((view, writer, insert))
            current, i = insert.result, i + 3
            continue
        if (not isinstance(writer, GenericOp) or _metadata(writer) != _metadata(producer)
                or tuple(writer.inputs) != tuple(values[v] for v in producer.inputs)
                or tuple(writer.outputs) != (view.result,)
                or tuple(v.type for v in writer.results) != tuple(v.type for v in producer.results)
                or not _same_body(writer.body, producer.body)
                or region != original_region or insert.attributes != original.attributes
                or insert.source is not writer.results[0] or insert.result.type != original.result.type):
            raise ValueError("destination interior changed arithmetic, inputs, layout or insertion")
        regions.append(region)
        generated.extend((view, writer, insert))
        if sum(prod(sizes) for _, sizes in regions) != prod(shape) or any(
            not _disjoint(a, b) for j, a in enumerate(regions) for b in regions[:j]
        ):
            raise ValueError("destination writes overlap or leave uninitialized elements")
        members = set(generated)
        if any(use.operation not in members for op in generated[:-1] for value in op.results for use in value.uses):
            raise ValueError("private destination intermediates escape their closed realization")
        return i + 3, insert, {
            "source_shape": list(source_shape), "destination_shape": list(shape),
            "exterior_regions": len(regions) - 1, "interior_elements": prod(source_shape),
            "exterior_elements": prod(shape) - prod(source_shape),
        }
    raise ValueError("destination realization has no complete interior writer")


def verify_destination_rewrite(before, after):
    """Check every original operation, interface, use and actual replacement."""
    before.verify()
    after.verify()
    values, observed = {}, []

    def bind(a, b):
        if a.type != b.type:
            raise ValueError("destination rewrite changed a complete value type")
        values[a] = b

    def operation(left, right):
        if not _header(left, right, values):
            raise ValueError("destination rewrite changed unmodeled operations, effects or dataflow")
        for a, b in zip(left.results, right.results, strict=True):
            bind(a, b)
        for a, b in zip(left.regions, right.regions, strict=True):
            if len(a.blocks) != len(b.blocks):
                raise ValueError("destination rewrite changed control flow")
            for x, y in zip(a.blocks, b.blocks, strict=True):
                block(x, y)

    def block(left, right):
        if len(left.args) != len(right.args):
            raise ValueError("destination rewrite changed block arguments")
        for a, b in zip(left.args, right.args, strict=True):
            bind(a, b)
        actual, index, pending = tuple(right.ops), 0, {}
        for original in left.ops:
            if isinstance(original, tensor.InsertSliceOp) and original.source in pending:
                if index >= len(actual):
                    raise ValueError("destination replacement is absent")
                index, final, record = _realization(original, pending.pop(original.source), actual, index, values)
                bind(original.result, final.result)
                observed.append(record)
            elif index < len(actual) and _header(original, actual[index], values):
                operation(original, actual[index])
                index += 1
            elif isinstance(original, GenericOp) and len(original.results) == 1:
                uses = tuple(original.results[0].uses)
                if (len(uses) != 1 or not isinstance(uses[0].operation, tensor.InsertSliceOp)
                        or uses[0].operation.parent_block() is not left):
                    raise ValueError("destination rewrite removed a live or unmodeled producer")
                pending[original.results[0]] = original
            else:
                raise ValueError("destination rewrite changed unmodeled operations or effects")
        if pending or index != len(actual):
            raise ValueError("destination realization contains unaccounted operations")

    operation(before, after)
    return observed
