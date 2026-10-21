"""Reconstruct row-batched integer contractions from actual source IR.

This checks tensor values and complete unchanged SSA/effect order. It grants no
physical view, retained-load, target execution, allocation or cost authority.
"""

from __future__ import annotations

from dataclasses import asdict
from math import prod

from xdsl.dialects import arith, builtin, func, tensor
from xdsl.dialects.linalg.ops import GenericOp, TransposeOp, YieldOp
from xdsl.ir import Block
from xdsl.utils.exceptions import VerifyException

from merlin.compile.scheduling.pointwise_source import read_pointwise_source
from merlin.llvmlower.layout_conformance import _metadata
from merlin.llvmlower.segmented_matrix_view import SegmentedRows, prove_segmented_matrix
from merlin.transforms.dense_ir import (
    _trace_attributes,
    checked_static_linalg_shell,
    static_zero_integer_matmul_geometry,
)


class RowBatchingUnavailable(ValueError):
    """An original source is outside the bounded mathematical batching domain."""


def shape(value):
    typ = value.type
    if (
        type(typ) is not builtin.TensorType
        or type(typ.encoding) is not builtin.NoneAttr
        or not typ.get_shape()
        or any(type(n) is not int or n <= 0 for n in typ.get_shape())
    ):
        raise RowBatchingUnavailable("row batching requires positive static unencoded tensors")
    return tuple(typ.get_shape())


def boundary_attributes(operation):
    """Dispatch attribution survives unchanged; it never selects a batch."""
    attrs = dict(operation.attributes)
    dispatch = attrs.pop("prov.dispatch_id", None)
    return (dispatch is None or type(dispatch) is builtin.StringAttr) and _trace_attributes(attrs)


def geometry(operation, *, index_bits):
    if type(index_bits) is not int or not 2 <= index_bits <= 128:
        raise ValueError("row batching requires a bounded original index width")
    if type(operation) is not GenericOp:
        raise RowBatchingUnavailable("row batching member is not a typed generic contraction")
    dtypes = tuple(str(v.type.element_type) for v in operation.operands)
    try:
        dimensions = static_zero_integer_matmul_geometry(operation, dtypes)
    except (ValueError, AttributeError) as error:
        raise RowBatchingUnavailable(str(error)) from error
    limit = (1 << (index_bits - 1)) - 1
    if any(prod(shape(v)) * (v.type.element_type.width.data // 8) > limit for v in operation.operands):
        raise RowBatchingUnavailable("row batching tensor bytes exceed the selected index range")
    if any(v.type.element_type.width.data % 8 for v in operation.operands):
        raise RowBatchingUnavailable("row batching requires byte-aligned integer elements")
    return dimensions, dtypes


def _view_chain(chain):
    for operation in chain:
        if (
            not boundary_attributes(operation)
            or operation.regions
            or operation.successors
            or len(operation.operands) != 1
            or len(operation.results) != 1
        ):
            raise RowBatchingUnavailable("matrix view has an unknown source contract")
        definition = operation.get_irdl_definition()
        if any(key not in definition.properties for key in operation.properties):
            raise RowBatchingUnavailable("matrix view has an unknown property")
        operation.verify()


def matrix_view(value):
    """Read dense matrices or the existing checked rank-three view chain."""
    logical = shape(value)
    if len(logical) != 2:
        raise RowBatchingUnavailable("row batching operand is not rank two")
    if type(value.owner) is tensor.ExtractSliceOp and len(shape(value.owner.source)) == 2:
        operation = value.owner
        _view_chain((operation,))
        if operation.offsets or operation.sizes or operation.strides or len(operation.operands) != 1:
            raise RowBatchingUnavailable("row batching slice has dynamic coordinates")
        offsets, sizes, strides = (
            tuple(a.get_values()) for a in (operation.static_offsets, operation.static_sizes, operation.static_strides)
        )
        before = shape(operation.source)
        if (
            sizes != logical
            or strides != (1, 1)
            or offsets[1] != 0
            or logical[1] != before[1]
            or offsets[0] < 0
            or offsets[0] + logical[0] > before[0]
        ):
            raise RowBatchingUnavailable("row batching slice is not a complete contiguous row range")
        width = value.type.element_type.width.data
        return operation.source, SegmentedRows(
            logical[0],
            logical[1],
            logical[0],
            before[1],
            logical[0] * before[1],
            prod(before),
            width // 8,
            str(value.type.element_type),
            offsets[0] * before[1],
        )
    if type(value.owner) in (tensor.ExpandShapeOp, tensor.CollapseShapeOp):
        try:
            proof = prove_segmented_matrix(value)
        except ValueError as error:
            raise RowBatchingUnavailable(str(error)) from error
        _view_chain(proof.chain)
        return proof.owner, proof.address
    width = value.type.element_type.width.data
    return value, SegmentedRows(
        logical[0],
        logical[1],
        logical[0],
        logical[1],
        prod(logical),
        prod(logical),
        width // 8,
        str(value.type.element_type),
    )


def rhs_identity(value, mapper=None):
    """Exact root, coordinates, permutation, element type and logical shape."""
    permutation = None
    if type(value.owner) is TransposeOp:
        operation = value.owner
        if not transpose_value(operation):
            raise RowBatchingUnavailable("RHS transpose has unproved value semantics")
        permutation = tuple(operation.permutation.get_values())
        base = operation.inputs[0]
    else:
        base = value
    owner, view = matrix_view(base)
    return (
        (owner if mapper is None else mapper(owner)),
        tuple(asdict(view).items()),
        permutation,
        shape(value),
        str(value.type.element_type),
    )


def transpose_value(operation):
    """Read a registered transpose, retaining all its boundary attribution."""
    if (
        type(operation) is not TransposeOp
        or not boundary_attributes(operation)
        or set(operation.properties) != {"permutation"}
        or len(operation.operands) != 2
        or len(operation.results) != 1
        or len(operation.body.blocks) != 1
    ):
        return False
    block = operation.body.block
    source, result = shape(operation.operands[0]), shape(operation.results[0])
    permutation = tuple(operation.permutation.get_values())
    body = tuple(block.ops)
    return (
        len(block.args) == 2
        and len(body) == 1
        and type(body[0]) is YieldOp
        and tuple(body[0].operands) == (block.args[0],)
        and not block.args[1].uses
        and not body[0].properties
        and _trace_attributes(body[0].attributes)
        and sorted(permutation) == list(range(len(source)))
        and tuple(source[i] for i in permutation) == result
        and operation.operands[1].type == operation.results[0].type
        and operation.operands[0].type.element_type == operation.results[0].type.element_type
    )


def dominating(value, anchor):
    owner = value.owner
    return owner is anchor.parent or (
        not isinstance(owner, Block) and owner.parent is anchor.parent and owner.is_before_in_block(anchor)
    )


def uses_after(value, anchor):
    """Only direct uses strictly after the same-block insertion are admitted."""
    return all(use.operation.parent is anchor.parent and anchor.is_before_in_block(use.operation) for use in value.uses)


def _positive_division(operation):
    """An unchanged integer floor division by a positive literal is total."""
    try:
        shell = checked_static_linalg_shell(operation)
    except ValueError:
        return False
    body = shell.body
    if [type(op) for op in body] != [arith.ConstantOp, arith.FloorDivSIOp, YieldOp] or len(shell.args) != 2:
        return False
    constant, divide, yielded = body
    return (
        type(constant.value) is builtin.IntegerAttr
        and constant.value.value.data > 0
        and set(constant.properties) == {"value"}
        and not divide.properties
        and not yielded.properties
        and tuple(divide.operands) == (shell.args[0], constant.result)
        and tuple(yielded.operands) == (divide.result,)
        and constant.result.type == shell.args[0].type == divide.result.type
        and all(_trace_attributes(op.attributes) for op in body)
    )


def _constant_broadcast_source(operation):
    """Check scalarized literal broadcasts on an analysis clone only."""
    from xdsl.ir.affine import AffineMap

    normalized = None
    created = []
    try:
        if operation.parent is None:
            return False
        shell = checked_static_linalg_shell(operation, singleton_projection_inputs=True)
        if any(type(value.type.element_type) is not builtin.IntegerType for value in operation.operands):
            return False
        if any(not boundary_attributes(inner) for inner in operation.walk()) or any(
            not _trace_attributes(inner.attributes) for inner in shell.body
        ):
            return False
        identity = AffineMap.identity(len(shell.shape))
        scalar_map = AffineMap(len(shell.shape), 0, ())
        normalized = operation.clone()
        for inner in normalized.body.block.ops:
            inner.attributes.clear()
        operands, maps = list(normalized.operands), list(normalized.indexing_maps)
        for index, (value, mapping) in enumerate(zip(operation.inputs, shell.input_maps, strict=True)):
            if tuple(value.type.get_shape()) == shell.shape and mapping == identity:
                continue
            splat = value.owner
            if (
                tuple(value.type.get_shape()) != ()
                or mapping != scalar_map
                or type(splat) is not tensor.SplatOp
                or splat.result is not value
                or len(splat.operands) != 1
                or splat.dynamicSizes
                or splat.properties
                or splat.regions
                or splat.successors
                or not boundary_attributes(splat)
                or splat.parent is not operation.parent
                or not dominating(value, operation)
            ):
                return False
            constant = splat.input.owner
            if (
                type(constant) is not arith.ConstantOp
                or constant.operands
                or len(constant.results) != 1
                or constant.result is not splat.input
                or constant.regions
                or constant.successors
                or not boundary_attributes(constant)
                or set(constant.properties) != {"value"}
                or type(constant.value) is not builtin.IntegerAttr
                or type(constant.value.value.data) is not int
                or constant.value.type != value.type.element_type
                or constant.result.type != value.type.element_type
                or constant.parent is not splat.parent
                or not dominating(constant.result, splat)
            ):
                return False
            try:
                constant.verify()
                splat.verify()
            except Exception:
                return False
            literal = constant.clone()
            full = tensor.SplatOp(literal.result, [], builtin.TensorType(literal.result.type, shell.shape))
            created.extend((literal, full))
            operands[index], maps[index] = full.result, builtin.AffineMapAttr(identity)
        if not created:
            return False
        normalized.operands = operands
        normalized.properties["indexing_maps"] = builtin.ArrayAttr(maps)
        read_pointwise_source(
            normalized,
            max_elements=64 << 20,
            max_operations=128,
            max_ir_bytes=65536,
            allow_constant_captures=True,
        )
        return True
    except (ValueError, AttributeError, VerifyException):
        return False
    finally:
        if normalized is not None:
            normalized.drop_all_references()
        for auxiliary in reversed(created):
            auxiliary.drop_all_references()


def safe_span(operations):
    """Moving total tensor dot arithmetic must not cross unknown effects."""
    tensor_types = (
        tensor.EmptyOp,
        tensor.SplatOp,
        tensor.ExtractSliceOp,
        tensor.CollapseShapeOp,
        tensor.ExpandShapeOp,
        tensor.ConcatOp,
    )
    for operation in operations:
        if not boundary_attributes(operation) or operation.successors:
            return False
        if type(operation) is GenericOp:
            try:
                geometry(operation, index_bits=128)
                continue
            except RowBatchingUnavailable:
                pass
            # The unchanged scalar reader is stricter than the established
            # typed trace taxonomy. Normalize validated scalar attribution on
            # an analysis clone only; actual source/metadata is
            # retained and compared byte-for-byte by the whole-module check.
            if any(not boundary_attributes(inner) for inner in operation.walk()):
                return False
            if any(not _trace_attributes(inner.attributes) for inner in operation.body.block.ops):
                return False
            analyzed = operation.clone()
            for inner in analyzed.body.block.ops:
                inner.attributes.clear()
            try:
                try:
                    read_pointwise_source(
                        analyzed,
                        max_elements=64 << 20,
                        max_operations=128,
                        max_ir_bytes=65536,
                        allow_constant_captures=True,
                    )
                except ValueError:
                    if not _constant_broadcast_source(operation) and not _positive_division(operation):
                        return False
            finally:
                analyzed.drop_all_references()
        elif type(operation) is TransposeOp:
            if not transpose_value(operation):
                return False
        elif type(operation) is arith.ConstantOp:
            if type(operation.value) is not builtin.IntegerAttr:
                return False
        elif type(operation) in tensor_types:
            definition = operation.get_irdl_definition()
            if operation.regions or any(key not in definition.properties for key in operation.properties):
                return False
            if any(type(v.type) is not builtin.TensorType for v in operation.results):
                return False
        else:
            return False
        try:
            operation.verify()
        except Exception:
            return False
    return True


def supported_scope(operation):
    """The first slice has no enclosing loop, lifecycle or unknown ABI contract."""
    function = operation.parent_op()
    module = None if function is None else function.parent_op()
    if type(function) is not func.FuncOp or type(module) is not builtin.ModuleOp:
        return False
    for op, attributes, properties in (
        (function, {"llvm.emit_c_interface"}, {"sym_name", "function_type", "sym_visibility"}),
        (module, {"llvm.data_layout", "llvm.target_triple", "prov.weights_file", "prov.level"}, set()),
    ):
        trace = {key: value for key, value in op.attributes.items() if key not in attributes}
        if any(
            type(value) is not builtin.StringAttr
            for key, value in op.attributes.items()
            if key in ("prov.weights_file", "prov.level")
        ):
            return False
        if not _trace_attributes(trace) or set(op.properties) - properties:
            return False
    return len(function.body.blocks) == 1


def _groups(operation):
    return tuple(tuple(v.value.data for v in group) for group in operation.reassociation)


def _batch(operation, index_bits, *, original=False):
    """Recognize actual emitted prefix and exact output partition, no plan."""
    init = operation.prev_op
    lhs = None if init is None else init.prev_op
    flat = None if lhs is None else lhs.prev_op
    packed = type(lhs) is tensor.ConcatOp
    viewed = type(lhs) is tensor.ExpandShapeOp and type(flat) is tensor.CollapseShapeOp
    if not (packed or viewed) or type(init) is not tensor.SplatOp:
        return None
    # Syntax alone is not historical provenance. Original-only probing may
    # signal unsupported geometry; emitted candidates remain strict by default.
    views = tuple(use.operation for use in operation.results[0].uses)
    if not any(type(v) is tensor.ExtractSliceOp for v in views):
        return None
    try:
        (m, k, n), dtypes = geometry(operation, index_bits=index_bits)
    except RowBatchingUnavailable as error:
        if original:
            raise
        raise ValueError("existing batch has unproved scalar semantics") from error
    if (
        len(views) < 2
        or len(set(views)) != len(views)
        or any(type(v) is not tensor.ExtractSliceOp for v in views)
        or tuple(operation.operands) != (lhs.result, operation.inputs[1], init.result)
        or any(op.attributes for op in (lhs, init))
        or len(tuple(lhs.result.uses)) != 1
        or len(tuple(init.result.uses)) != 1
    ):
        raise ValueError("existing batch has an incomplete or malformed prelude/partition")
    if packed:
        if (
            set(lhs.properties) != {"dim"}
            or lhs.dim.value.data != 0
            or len(lhs.operands) < 2
            or lhs.regions
            or lhs.successors
            or shape(lhs.result) != (m, k)
            or any(
                len(shape(v)) != 2 or shape(v)[1] != k or v.type.element_type != lhs.result.type.element_type
                for v in lhs.operands
            )
            or sum(shape(v)[0] for v in lhs.operands) != m
        ):
            raise ValueError("actual materialized batch changed row concatenation")
        lhs.verify()
        prefix, root = (lhs, init, operation), None
    else:
        if (
            lhs.operands[0] is not flat.result
            or flat.attributes
            or len(tuple(flat.result.uses)) != 1
            or _groups(flat) != (tuple(range(len(shape(flat.operands[0])))),)
            or _groups(lhs) != ((0, 1),)
            or shape(flat.result) != (m * k,)
            or shape(lhs.result) != (m, k)
            or prod(shape(flat.operands[0])) != m * k
        ):
            raise ValueError("existing batch changed contiguous source coverage")
        _view_chain((flat, lhs))
        prefix, root = (flat, lhs, init, operation), flat.operands[0]
    intervals = []
    for view in views:
        if (
            not boundary_attributes(view)
            or view.offsets
            or view.sizes
            or view.strides
            or view.source is not operation.results[0]
        ):
            raise ValueError("actual batched result has unknown slice semantics")
        offsets, sizes, strides = (
            tuple(a.get_values()) for a in (view.static_offsets, view.static_sizes, view.static_strides)
        )
        if (
            set(view.properties) - {"static_offsets", "static_sizes", "static_strides", "operandSegmentSizes"}
            or offsets[1:] != (0,)
            or sizes[1:] != (n,)
            or strides != (1, 1)
            or shape(view.result) != sizes
            or offsets[0] < 0
            or sizes[0] <= 0
            or offsets[0] + sizes[0] > m
        ):
            raise ValueError("actual batched result changed row coordinates or element coverage")
        intervals.append((offsets[0], sizes[0], view))
    cursor = 0
    for offset, length, _ in sorted(intervals, key=lambda item: item[0]):
        if offset != cursor:
            raise ValueError("actual batched result slices overlap or leave a gap")
        cursor += length
    if cursor != m:
        raise ValueError("actual batched result slices do not cover all rows")
    return {
        "operation": operation,
        "prefix": prefix,
        "root": root,
        "concat": lhs if packed else None,
        "shape": (m, k, n),
        "dtypes": dtypes,
        "views": {v: (offset, length) for offset, length, v in intervals},
        "members": [],
    }


def verify_dense_row_batching(before, after, *, index_bits, max_materialized_bytes=0):
    """Inspect actual grouped domains and every untouched operation/SSA use."""
    before.verify()
    after.verify()
    if type(max_materialized_bytes) is not int or max_materialized_bytes < 0:
        raise ValueError("materialized row checking requires an explicit nonnegative byte budget")
    # Validate pre-existing recognizable batches even on the unchanged path.
    for op in before.walk():
        if type(op) is GenericOp:
            _batch(op, index_bits, original=False)
    if before.is_structurally_equivalent(after) and tuple(_metadata(op) for op in before.walk()) == tuple(
        _metadata(op) for op in after.walk()
    ):
        return ()
    batches = []
    for operation in after.walk():
        if type(operation) is GenericOp:
            try:
                item = _batch(operation, index_bits, original=False)
            except RowBatchingUnavailable:
                continue
            if item is not None:
                batches.append(item)
    if not batches:
        raise ValueError("changed source contains no independently reconstructed batched contraction")
    prefixes = {op for batch in batches for op in batch["prefix"]}
    view_batches = {view: batch for batch in batches for view in batch["views"]}
    operation_batches = {batch["operation"]: batch for batch in batches}
    existing = set()
    deferred_views = set()
    values = {}
    operations = {}

    def same_shell(left, right):
        return (
            left.name == right.name
            and _metadata(left) == _metadata(right)
            and tuple(v.type for v in left.results) == tuple(v.type for v in right.results)
            and tuple(values.get(v) for v in left.operands) == tuple(right.operands)
            and len(left.regions) == len(right.regions)
            and not left.successors
            and not right.successors
        )

    def compare(left, right):
        if right in view_batches and type(left) is GenericOp:
            batch = view_batches[right]
            dimensions, dtypes = geometry(left, index_bits=index_bits)
            if (
                dimensions != (batch["views"][right][1], *batch["shape"][1:])
                or dtypes != batch["dtypes"]
                or left.attributes != right.attributes
            ):
                raise ValueError("batched slice changed an original reduction domain or dtype")
            values[left.results[0]] = right.result
            batch["members"].append((left, right))
            return
        if not same_shell(left, right):
            raise ValueError("row batching changed unrelated IR, metadata, outputs or opaque ordering")
        operations[left] = right
        if right in operation_batches:
            existing.add(right)
        values.update(zip(left.results, right.results, strict=True))
        for a, b in zip(left.regions, right.regions, strict=True):
            if len(a.blocks) != len(b.blocks):
                raise ValueError("row batching changed region structure")
            for old, new in zip(a.blocks, b.blocks, strict=True):
                if tuple(v.type for v in old.args) != tuple(v.type for v in new.args):
                    raise ValueError("row batching changed block argument types")
                values.update(zip(old.args, new.args, strict=True))
                new_ops = tuple(new.ops)
                cursor = 0
                for x in old.ops:
                    if type(x) is GenericOp and (cursor == len(new_ops) or not same_shell(x, new_ops[cursor])):
                        matches = []
                        for view, batch in view_batches.items():
                            concat = batch["concat"]
                            if concat is None or view in deferred_views or x.attributes != view.attributes:
                                continue
                            try:
                                dimensions, _ = geometry(x, index_bits=index_bits)
                                rhs = rhs_identity(x.inputs[1], values.get)
                            except RowBatchingUnavailable:
                                continue
                            offset, length = batch["views"][view]
                            row = 0
                            for operand in concat.operands:
                                if (
                                    values.get(x.inputs[0]) is operand
                                    and offset == row
                                    and dimensions == (length, *batch["shape"][1:])
                                    and rhs == rhs_identity(batch["operation"].inputs[1])
                                ):
                                    matches.append(view)
                                row += shape(operand)[0]
                        if matches:
                            if len({view_batches[v]["operation"] for v in matches}) != 1:
                                raise ValueError("materialized batch has ambiguous original membership")
                            view = min(matches, key=lambda v: view_batches[v]["views"][v][0])
                            compare(x, view)
                            deferred_views.add(view)
                            continue
                    while cursor < len(new_ops) and (
                        new_ops[cursor] in deferred_views
                        or (new_ops[cursor] in prefixes and not same_shell(x, new_ops[cursor]))
                    ):
                        cursor += 1
                    if cursor == len(new_ops):
                        raise ValueError("row batching dropped an original operation")
                    compare(x, new_ops[cursor])
                    cursor += 1
                if cursor != len(new_ops):
                    raise ValueError("row batching added unrelated operations")

    compare(before, after)
    records = []
    materialized_bytes = 0
    for batch in batches:
        if batch["operation"] in existing:
            continue
        members = batch["members"]
        if len(members) != len(batch["views"]):
            raise ValueError("batched output lacks an original observed result")
        first = members[0][0]
        if not supported_scope(first) or any(op.parent is not first.parent for op, _ in members):
            raise ValueError("row batching crosses an unproved enclosing scope")
        body = tuple(first.parent.ops)
        lo, hi = min(body.index(op) for op, _ in members), max(body.index(op) for op, _ in members)
        if not safe_span(body[lo : hi + 1]):
            raise ValueError("row batching moved arithmetic across an unknown effect or observer")
        root, concat = batch["root"], batch["concat"]
        if (root is not None and not dominating(root, batch["operation"])) or not dominating(
            batch["operation"].inputs[1], batch["operation"]
        ):
            raise ValueError("batched operand does not dominate its insertion point")
        if concat is not None:
            ordered = sorted(members, key=lambda pair: body.index(pair[0]))
            anchor = ordered[-1][0]
            if (
                len(concat.operands) != len(members)
                or not all(dominating(v, anchor) for op, _ in members for v in op.inputs)
                or not all(uses_after(op.results[0], anchor) for op, _ in members)
                or tuple(values.get(op.inputs[0]) for op, _ in ordered) != tuple(concat.operands)
            ):
                raise ValueError("materialized rows changed operand order or crossed an early result use")
            # The concat must appear immediately at the last replaced member,
            # not merely somewhere with superficially equal operands.
            if any(op in operations and not operations[op].is_before_in_block(concat) for op in body[:hi]):
                raise ValueError("materialized batch moved before its last-member insertion point")
            if any(
                op in operations and not batch["operation"].is_before_in_block(operations[op]) for op in body[hi + 1 :]
            ):
                raise ValueError("materialized batch moved after its last-member insertion point")
            byte_count = prod(shape(concat.result)) * concat.result.type.element_type.width.data // 8
            materialized_bytes += byte_count
            if materialized_bytes > max_materialized_bytes:
                raise ValueError("actual LHS materialization exceeds the selected byte budget")
        for original, view in members:
            offset, length = batch["views"][view]
            if concat is None:
                owner, address = matrix_view(original.inputs[0])
                lhs_matches = not (
                    values.get(owner) is not root
                    or address.origin != offset * batch["shape"][1]
                    or address.rows != length
                    or address.cols != batch["shape"][1]
                    or address.row_stride != address.cols
                    or address.segment_stride != address.segment_rows * address.cols
                    or address.source_elements != batch["shape"][0] * batch["shape"][1]
                )
            else:
                position = next(i for i, (op, _) in enumerate(ordered) if op is original)
                lhs_matches = (
                    offset == sum(shape(v)[0] for v in concat.operands[:position])
                    and length == shape(concat.operands[position])[0]
                )
            if not lhs_matches or rhs_identity(original.inputs[1], values.get) != rhs_identity(
                batch["operation"].inputs[1]
            ):
                raise ValueError("batched operands changed original root, row order, RHS coordinates or dtype")
        records.append(
            {
                "members": len(members),
                "shape": list(batch["shape"]),
                "dtypes": list(batch["dtypes"]),
                "ordered_reductions": "unchanged",
                "outputs": [list(shape(view.result)) for _, view in members],
                "physical_execution": "not_established",
                "lhs_materialization_bytes": 0 if concat is None else byte_count,
            }
        )
    return tuple(records)
