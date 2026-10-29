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
    if (
        not isinstance(typ, builtin.TensorType)
        or not isinstance(typ.encoding, builtin.NoneAttr)
        or any(n <= 0 for n in typ.get_shape())
    ):
        raise ValueError("destination check requires positive static unencoded tensors")
    return tuple(typ.get_shape())


def _rectangle(op, base, shape):
    if set(op.properties) - {"static_offsets", "static_sizes", "static_strides", "operandSegmentSizes"}:
        raise ValueError("destination region has unknown properties")
    if op.operands[1 if isinstance(op, tensor.InsertSliceOp) else 0] is not base:
        raise ValueError("destination chain changed its backing tensor")
    offsets, sizes, strides = (
        tuple(field.get_values()) for field in (op.static_offsets, op.static_sizes, op.static_strides)
    )
    if (
        op.offsets
        or op.sizes
        or op.strides
        or len(offsets) != len(shape)
        or len(sizes) != len(shape)
        or strides != (1,) * len(shape)
        or any(o < 0 or n <= 0 or o + n > d for o, n, d in zip(offsets, sizes, shape, strict=True))
    ):
        raise ValueError("destination chain has an unsupported or out-of-bounds region")
    return offsets, sizes


def _disjoint(a, b):
    return any(x + nx <= y or y + ny <= x for x, nx, y, ny in zip(*a, *b, strict=True))


def _header(left, right, values):
    return (
        left.name == right.name
        and _metadata(left) == _metadata(right)
        and not left.successors
        and not right.successors
        and tuple(values.get(v) for v in left.operands) == tuple(right.operands)
        and tuple(v.type for v in left.results) == tuple(v.type for v in right.results)
        and len(left.regions) == len(right.regions)
    )


def _metadata(operation):
    # Float attribute equality can equate -0 and +0. Canonical typed printing
    # retains their distinct words and all nested attribute/property fields.
    return tuple(
        tuple(sorted((key, str(value)) for key, value in fields.items()))
        for fields in (operation.attributes, operation.properties)
    )


def _same_body(left, right, values=None):
    return left.is_structurally_equivalent(right, None if values is None else dict(values)) and tuple(
        _metadata(op) for op in left.walk()
    ) == tuple(_metadata(op) for op in right.walk())


def _realization(original, producer, actual, start, values):
    """Read one actual empty/exterior/interior write sequence from operation IR."""
    source_shape, shape = _shape(original.source), _shape(original.dest)
    pad = original.dest.owner
    if (
        not isinstance(original.parent_op(), func.FuncOp)
        or not isinstance(pad, tensor.SplatOp)
        or pad.dynamicSizes
        or len(producer.outputs) != 1
        or producer.outputs[0].owner.name != "tensor.empty"
        or producer.indexing_maps.data[-1].data != AffineMap.identity(len(source_shape))
        or any(it.data.value != "parallel" for it in producer.iterator_types)
        or producer.body.block.args[-1].uses
        or not pure_scalar_body(producer.body.block)
    ):
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
        allowed = (
            {"llvm.emit_c_interface"}
            if isinstance(ancestor, func.FuncOp)
            else ({"llvm.data_layout", "llvm.target_triple"} if isinstance(ancestor, builtin.ModuleOp) else set())
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
    if (
        not isinstance(empty, tensor.EmptyOp)
        or empty.attributes
        or empty.dynamic_sizes
        or set(empty.properties) - {"operandSegmentSizes"}
        or empty.tensor.type != original.dest.type
    ):
        raise ValueError("destination realization has no fresh complete parent")
    generated, regions = [empty], []
    current, i = empty.tensor, start + 1
    while i + 2 < len(actual):
        view, writer, insert = actual[i : i + 3]
        if not isinstance(view, tensor.ExtractSliceOp):
            raise ValueError("destination realization lacks an exact view")
        region = _rectangle(view, current, shape)
        if view.attributes or _shape(view.result) != region[1]:
            raise ValueError("destination view changed shape or semantics")
        if not isinstance(insert, tensor.InsertSliceOp) or _rectangle(insert, current, shape) != region:
            raise ValueError("destination writer does not return to its exact view")
        if isinstance(writer, FillOp):
            body = writer.regions[0].block
            if (
                tuple(writer.inputs) != (values[pad.input],)
                or tuple(writer.outputs) != (view.result,)
                or writer.attributes
                or set(writer.properties) - {"operandSegmentSizes"}
                or len(writer.results) != 1
                or writer.results[0].type != view.result.type
                or len(body.args) != 2
                or len(tuple(body.ops)) != 1
                or body.last_op.name != "linalg.yield"
                or tuple(body.last_op.operands) != (body.args[0],)
                or insert.attributes
                or insert.source is not writer.results[0]
            ):
                raise ValueError("destination exterior changed its fill value or scalar semantics")
            regions.append(region)
            generated.extend((view, writer, insert))
            current, i = insert.result, i + 3
            continue
        if (
            not isinstance(writer, GenericOp)
            or _metadata(writer) != _metadata(producer)
            or tuple(writer.inputs) != tuple(values[v] for v in producer.inputs)
            or tuple(writer.outputs) != (view.result,)
            or tuple(v.type for v in writer.results) != tuple(v.type for v in producer.results)
            or not _same_body(writer.body, producer.body)
            or region != original_region
            or insert.attributes != original.attributes
            or insert.source is not writer.results[0]
            or insert.result.type != original.result.type
        ):
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
        return (
            i + 3,
            insert,
            {
                "source_shape": list(source_shape),
                "destination_shape": list(shape),
                "exterior_regions": len(regions) - 1,
                "interior_elements": prod(source_shape),
                "exterior_elements": prod(shape) - prod(source_shape),
            },
        )
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
                if (
                    len(uses) != 1
                    or not isinstance(uses[0].operation, tensor.InsertSliceOp)
                    or uses[0].operation.parent_block() is not left
                ):
                    raise ValueError("destination rewrite removed a live or unmodeled producer")
                pending[original.results[0]] = original
            else:
                raise ValueError("destination rewrite changed unmodeled operations or effects")
        if pending or index != len(actual):
            raise ValueError("destination realization contains unaccounted operations")

    operation(before, after)
    return observed


def _permuted_realization(producer, transpose, empty, writer, values, index_bits, expansion=None):
    """Reconstruct a direct output permutation without using emitter decisions."""
    source_shape = _shape(producer.results[0])
    physical_shape = _shape(transpose.operands[0])
    shape = _shape(transpose.results[0])
    source_rank, rank = len(source_shape), len(physical_shape)
    coordinates = [AffineDimExpr(i) for i in range(source_rank)]
    if expansion is not None:
        groups = tuple(tuple(item.value.data for item in group) for group in expansion.reassociation)
        if (
            expansion.parent is not producer.parent
            or tuple(expansion.operands) != (producer.results[0],)
            or len(tuple(expansion.results[0].uses)) != 1
            or next(iter(expansion.results[0].uses)).operation is not transpose
            or set(expansion.properties) != {"reassociation", "static_output_shape"}
            or any(not key.startswith("prov.") for key in expansion.attributes)
            or tuple(expansion.static_output_shape.get_values()) != physical_shape
            or len(groups) != source_rank
            or tuple(axis for group in groups for axis in group) != tuple(range(rank))
            or any(not group or prod(physical_shape[i] for i in group) != n for group, n in zip(groups, source_shape))
            or any(
                not isinstance(value.type, builtin.IntegerType)
                for scalar in producer.body.block.ops
                for value in (*scalar.operands, *scalar.results)
            )
        ):
            raise ValueError("packed producer has an unsupported dimension split")
        coordinates = []
        for group in groups:
            # Reconstruct row-major linear coordinates independently of emitter.
            coordinate = AffineConstantExpr(0)
            for position, axis in enumerate(group):
                coordinate += AffineDimExpr(axis) * prod(physical_shape[i] for i in group[position + 1 :])
            coordinates.append(coordinate)
    permutation = tuple(transpose.permutation.get_values())
    if (
        not rank
        or sorted(permutation) != list(range(rank))
        or shape != tuple(physical_shape[p] for p in permutation)
        or len(producer.outputs) != 1
        or len(producer.iterator_types) != source_rank
        or any(it.data.value != "parallel" for it in producer.iterator_types)
        or len(producer.body.blocks) != 1
        or producer.body.block.args[-1].uses
        or not pure_scalar_body(producer.body.block, bounded_intrinsics=True)
        or producer.indexing_maps.data[-1].data != AffineMap.identity(source_rank)
        or not isinstance(producer.outputs[0].owner, tensor.EmptyOp)
        or not isinstance(transpose.operands[1].owner, tensor.EmptyOp)
        or set(producer.properties) - {"indexing_maps", "iterator_types", "operandSegmentSizes"}
        or set(transpose.properties) - {"permutation", "operandSegmentSizes"}
    ):
        raise ValueError("permuted destination has unsupported source semantics")
    parent = producer.parent_op()
    if not isinstance(parent, func.FuncOp) or len(parent.body.blocks) != 1:
        raise ValueError("permuted destination requires a single-block function")
    ancestor = parent
    while ancestor is not None:
        allowed = (
            {"llvm.emit_c_interface"}
            if isinstance(ancestor, func.FuncOp)
            else ({"llvm.data_layout", "llvm.target_triple"} if isinstance(ancestor, builtin.ModuleOp) else set())
        )
        if any(k not in allowed and not k.startswith("prov.") for k in ancestor.attributes):
            raise ValueError("permuted destination has an unknown enclosing contract")
        ancestor = ancestor.parent_op()
    for op in (producer, transpose, producer.outputs[0].owner, transpose.operands[1].owner):
        if any(not k.startswith("prov.") for k in op.attributes):
            raise ValueError("permuted destination has unknown source attributes")
    for value in (*producer.operands, *producer.results, *transpose.operands, *transpose.results):
        extents = _shape(value)
        element = value.type.get_element_type()
        width = element.width.data if isinstance(element, builtin.IntegerType) else getattr(element, "bitwidth", None)
        if (
            not isinstance(element, (builtin.IntegerType, builtin.Float32Type, builtin.Float64Type))
            or width not in (8, 16, 32, 64)
            or prod(extents) * (width // 8) > (1 << (index_bits - 1)) - 1
        ):
            raise ValueError("permuted destination exceeds the declared index domain")
    for value, attr in zip(producer.inputs, producer.indexing_maps, strict=False):
        mapping = attr.data
        if mapping.num_symbols or mapping.num_dims != source_rank or len(mapping.results) != len(_shape(value)):
            raise ValueError("permuted destination has an unknown input map")
        axes = []
        for extent, expr in zip(_shape(value), mapping.results, strict=True):
            if (
                isinstance(expr, AffineDimExpr)
                and 0 <= expr.position < source_rank
                and extent == source_shape[expr.position]
            ):
                axes.append(expr.position)
            elif not isinstance(expr, AffineConstantExpr) or expr.value != 0 or extent != 1:
                raise ValueError("permuted destination has an out-of-domain input access")
        if len(set(axes)) != len(axes):
            raise ValueError("permuted destination has a repeated input dimension")
    for scalar in producer.body.block.ops:
        for value in scalar.operands:
            owner = value.owner
            if owner is producer.body.block or getattr(owner, "parent", None) is producer.body.block:
                continue
            if (
                not isinstance(owner, arith.ConstantOp)
                or owner.parent is not producer.parent
                or not isinstance(owner.value, (builtin.IntegerAttr, builtin.FloatAttr))
                or set(owner.properties) != {"value"}
                or any(not k.startswith("prov.") for k in owner.attributes)
                or value not in values
            ):
                raise ValueError("permuted destination scalar body captures an external computation")
    cursor = producer.next_op
    while cursor is not transpose:
        if (
            cursor is None
            or (cursor is not expansion and not isinstance(cursor, (arith.ConstantOp, tensor.SplatOp, tensor.EmptyOp)))
            or any(not k.startswith("prov.") for k in cursor.attributes)
        ):
            raise ValueError("permuted destination crossed an execution barrier")
        cursor = cursor.next_op
    if (
        not isinstance(empty, tensor.EmptyOp)
        or empty.attributes
        or empty.dynamic_sizes
        or set(empty.properties) - {"operandSegmentSizes"}
        or empty.tensor.type != transpose.results[0].type
        or any(set(v.owner.properties) - {"operandSegmentSizes"} for v in (producer.outputs[0], transpose.operands[1]))
    ):
        raise ValueError("permuted destination lacks a fresh complete tensor")
    if (
        not isinstance(writer, GenericOp)
        or writer.attributes != producer.attributes
        or {k: str(v) for k, v in writer.properties.items() if k not in ("indexing_maps", "iterator_types")}
        != {k: str(v) for k, v in producer.properties.items() if k not in ("indexing_maps", "iterator_types")}
        or tuple(writer.inputs) != tuple(values.get(v) for v in producer.inputs)
        or tuple(writer.outputs) != (empty.tensor,)
        or len(writer.results) != 1
        or writer.results[0].type != transpose.results[0].type
        or not _same_body(producer.body, writer.body, values)
        or len(writer.indexing_maps) != len(producer.indexing_maps)
        or tuple(mapping.data for mapping in writer.indexing_maps.data[:-1])
        != tuple(
            mapping.data.replace_dims_and_symbols(coordinates, [], rank, 0)
            for mapping in producer.indexing_maps.data[:-1]
        )
        or len(writer.iterator_types) != rank
        or any(it.data.value != "parallel" for it in writer.iterator_types)
        or writer.indexing_maps.data[-1].data != AffineMap(rank, 0, tuple(AffineDimExpr(i) for i in permutation))
        or any(use.operation is not writer for use in empty.tensor.uses)
    ):
        raise ValueError("permuted destination changed arithmetic, loop order, inputs or output coordinates")
    element = transpose.results[0].type.get_element_type()
    width = element.width.data if isinstance(element, builtin.IntegerType) else element.bitwidth
    return {
        "source_shape": list(source_shape),
        **({"expanded_shape": list(physical_shape)} if expansion is not None else {}),
        "destination_shape": list(shape),
        "permutation": list(permutation),
        "eliminated_tensor_bytes": prod(shape) * (width // 8),
    }


def verify_permutation_rewrite(before, after, *, index_bits):
    """Check the complete actual program, including all unchanged opaque effects."""
    from xdsl.dialects.linalg.ops import TransposeOp

    if type(index_bits) is not int or not 2 <= index_bits <= 128:
        raise ValueError("permutation checker requires the selected index width")
    before.verify()
    after.verify()
    values, observed = {}, []

    def bind(a, b):
        if a.type != b.type:
            raise ValueError("permuted destination changed a complete value type")
        values[a] = b

    def operation(left, right):
        if not _header(left, right, values):
            raise ValueError("permuted destination changed unmodeled operations, effects or dataflow")
        for a, b in zip(left.results, right.results, strict=True):
            bind(a, b)
        for a, b in zip(left.regions, right.regions, strict=True):
            if len(a.blocks) != len(b.blocks):
                raise ValueError("permuted destination changed control flow")
            for x, y in zip(a.blocks, b.blocks, strict=True):
                block(x, y)

    def block(left, right):
        if len(left.args) != len(right.args):
            raise ValueError("permuted destination changed block arguments")
        for a, b in zip(left.args, right.args, strict=True):
            bind(a, b)
        actual, index, removed = tuple(right.ops), 0, set()
        for original in left.ops:
            if original in removed:
                continue
            if index < len(actual) and _header(original, actual[index], values):
                operation(original, actual[index])
                index += 1
                continue
            if not isinstance(original, GenericOp) or len(original.results) != 1 or index + 1 >= len(actual):
                raise ValueError("permuted destination changed unmodeled operations")
            uses = tuple(original.results[0].uses)
            if len(uses) != 1 or uses[0].index != 0:
                raise ValueError("permuted destination removed a live producer")
            expansion = uses[0].operation if isinstance(uses[0].operation, tensor.ExpandShapeOp) else None
            if expansion is not None:
                uses = tuple(expansion.results[0].uses)
            if (
                len(uses) != 1
                or uses[0].index != 0
                or not isinstance(uses[0].operation, TransposeOp)
                or uses[0].operation.parent is not left
            ):
                raise ValueError("permuted destination removed a live producer")
            transpose = uses[0].operation
            record = _permuted_realization(
                original, transpose, actual[index], actual[index + 1], values, index_bits, expansion
            )
            if expansion is not None:
                removed.add(expansion)
            bind(transpose.results[0], actual[index + 1].results[0])
            removed.add(transpose)
            observed.append(record)
            index += 2
        if index != len(actual):
            raise ValueError("permuted destination contains unaccounted operations")

    operation(before, after)
    return observed
