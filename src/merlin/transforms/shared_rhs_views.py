"""Expose source-proven repeated matrix inputs to ordinary row batching.

The existing repeated-pointwise preparation may leave an identity broadcast to
preserve its public tensor interface. This pass reads that broadcast and its
static views, without inspecting runtime values or changing scalar arithmetic.
It creates complete per-shared-input row views; dense_row_batching remains the
consumer that actually combines contractions. No physical alias, target, cost
or immutable-argument authority is inferred here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

from xdsl.dialects import builtin, tensor
from xdsl.dialects.linalg.ops import GenericOp, TransposeOp, YieldOp
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap

from merlin.llvmlower.repeated_pointwise import _reshape
from merlin.transforms import dense_row_batching_check as views
from merlin.transforms.dense_row_batching import _state
from merlin.transforms.head_batching import _groups


@dataclass(frozen=True)
class _RepeatedMatrices:
    base: object
    factor: int
    chunk: int
    steps: tuple


def _axis(shape, factor, chunk):
    return next((i for i, n in enumerate(shape) if n % factor == 0 and prod(shape[i + 1 :]) == chunk), None)


def _repeated_matrices(value, *, max_views):
    """Read an actual identity broadcast, never a claimed repetition hint."""
    shape = views.shape(value)
    if len(shape) != 3:
        raise ValueError("shared RHS views require complete rank-three matrix inputs")
    path, current = [], value
    while type(current.owner) in (tensor.CollapseShapeOp, tensor.ExpandShapeOp, TransposeOp):
        op = current.owner
        if len(path) >= max_views:
            raise ValueError("shared RHS view depth exceeds the selected bound")
        if type(op) is TransposeOp:
            if not views.transpose_value(op):
                raise ValueError("shared RHS transpose lacks exact value semantics")
        else:
            views._view_chain((op,))
            if (
                prod(views.shape(op.operands[0])) != prod(views.shape(current))
                or op.operands[0].type.element_type != current.type.element_type
            ):
                raise ValueError("shared RHS reshape changes complete element coverage")
        path.append(op)
        current = op.operands[0]
    op = current.owner
    if (
        type(op) is not GenericOp
        or not views.boundary_attributes(op)
        or len(op.inputs) != 1
        or len(op.outputs) != 1
        or len(op.results) != 1
        or len(op.body.blocks) != 1
        or op.library_call is not None
        or set(op.properties) != {"indexing_maps", "iterator_types", "operandSegmentSizes"}
    ):
        raise ValueError("shared RHS root is not an explicit identity broadcast")
    full, small = views.shape(current), views.shape(op.inputs[0])
    maps = tuple(a.data for a in op.indexing_maps)
    block = op.body.block
    body = tuple(block.ops)
    if (
        len(full) != len(small)
        or current.type.element_type != op.inputs[0].type.element_type
        or op.outputs[0].type != current.type
        or len(block.args) != 2
        or block.args[0].type != current.type.element_type
        or block.args[1].type != current.type.element_type
        or len(body) != 1
        or type(body[0]) is not YieldOp
        or tuple(body[0].operands) != (block.args[0],)
        or block.args[1].uses
        or body[0].properties
        or not views.boundary_attributes(body[0])
        or len(maps) != 2
        or maps[-1] != AffineMap.identity(len(full))
        or len(op.iterator_types) != len(full)
        or any(it.data.value != "parallel" for it in op.iterator_types)
    ):
        raise ValueError("shared RHS broadcast changes scalar values or coordinates")
    axes = [i for i, (a, b) in enumerate(zip(small, full, strict=True)) if a != b]
    if len(axes) != 1:
        raise ValueError("shared RHS requires one explicit expanded unit dimension")
    axis = axes[0]
    expected = tuple(AffineConstantExpr(0) if i == axis else AffineDimExpr(i) for i in range(len(full)))
    if small[axis] != 1 or full[axis] <= 1 or maps[0] != AffineMap(len(full), 0, expected):
        raise ValueError("shared RHS broadcast does not prove complete repeated entries")
    factor, chunk = full[axis], prod(full[axis + 1 :])
    steps = []
    for operation in reversed(path):
        after = views.shape(operation.results[0])
        if type(operation) is TransposeOp:
            before = views.shape(operation.inputs[0])
            axis = _axis(before, factor, chunk)
            if axis is None:
                raise ValueError("shared RHS transpose splits an unrepresented repeated dimension")
            reduced = list(before)
            reduced[axis] //= factor
            permutation = tuple(operation.permutation.get_values())
            output = tuple(reduced[i] for i in permutation)
            chunk = prod(output[permutation.index(axis) + 1 :])
            steps.append((operation, tuple(reduced), output))
        else:
            axis = _axis(after, factor, chunk)
            reduced = list(after) if axis is not None else [prod(after) // factor]
            if axis is not None:
                reduced[axis] //= factor
            steps.append((operation, None, tuple(reduced)))
    if chunk != prod(shape[1:]) or shape[0] % factor:
        raise ValueError("shared RHS repetition does not cover complete contiguous matrices")
    return _RepeatedMatrices(op.inputs[0], factor, chunk, tuple(steps))


def _insert_reshape(value, shape, anchor, inserted):
    previous = anchor.prev_op
    result = _reshape(value, shape, anchor)
    current = anchor.parent.first_op if previous is None else previous.next_op
    while current is not anchor:
        inserted.append(current)
        current = current.next_op
    return result


def _slice(value, offsets, sizes, anchor, inserted):
    operation = tensor.ExtractSliceOp.from_static_parameters(value, offsets, sizes)
    anchor.parent.insert_op_before(operation, anchor)
    inserted.append(operation)
    return operation.result


def _emit(candidate, groups, value_map, operation_map):
    inserted, edits, records = [], [], []
    for group, relation in groups:
        members = tuple(operation_map[op] for op in group.members)
        body = tuple(members[0].parent.ops)
        anchor = min(members, key=body.index)
        h, m, k = views.shape(group.lhs)
        n = views.shape(group.rhs)[-1]
        count, factor = h // relation.factor, relation.factor
        rhs = value_map[relation.base]
        for operation, before, after in relation.steps:
            if type(operation) is TransposeOp:
                rhs = _insert_reshape(rhs, before, anchor, inserted)
                empty = tensor.EmptyOp([], builtin.TensorType(rhs.type.element_type, after))
                transpose = TransposeOp(rhs, empty.tensor, operation.permutation)
                anchor.parent.insert_ops_before((empty, transpose), anchor)
                inserted.extend((empty, transpose))
                rhs = transpose.results[0]
            else:
                rhs = _insert_reshape(rhs, after, anchor, inserted)
        rhs = _insert_reshape(rhs, (count, k, n), anchor, inserted)
        lhs_root = _insert_reshape(value_map[group.lhs], (count, factor * m, k), anchor, inserted)
        for g in range(count):
            lhs = _slice(lhs_root, (g, 0, 0), (1, factor * m, k), anchor, inserted)
            lhs = _insert_reshape(lhs, (factor * m, k), anchor, inserted)
            shared = _slice(rhs, (g, 0, 0), (1, k, n), anchor, inserted)
            shared = _insert_reshape(shared, (k, n), anchor, inserted)
            for r in range(factor):
                original, operation = group.members[g * factor + r], members[g * factor + r]
                row = _slice(lhs, (r * m, 0), (m, k), anchor, inserted)
                edits.append((operation, tuple(operation.operands), original))
                operation.operands = (row, shared, *operation.operands[2:])
        records.append((h, count, factor, (factor * m, k, n)))
    return tuple(inserted), tuple(edits), tuple(records)


def _verify(source, candidate, groups, value_map, operation_map, inserted, edits):
    """Check actual view construction, then restore all original operands/IR."""
    candidate.verify()

    def unshape(value, stop=None):
        while value is not stop and type(value.owner) in (tensor.CollapseShapeOp, tensor.ExpandShapeOp):
            operation = value.owner
            views._view_chain((operation,))
            if (
                operation.attributes
                or prod(views.shape(value)) != prod(views.shape(operation.operands[0]))
                or value.type.element_type != operation.operands[0].type.element_type
            ):
                raise ValueError("actual shared RHS reshape changed scalar values or attribution")
            value = operation.operands[0]
        return value

    def sliced(value, offsets, sizes):
        operation = value.owner
        if (
            type(operation) is not tensor.ExtractSliceOp
            or operation.attributes
            or operation.offsets
            or operation.sizes
            or operation.strides
            or tuple(operation.static_offsets.get_values()) != offsets
            or tuple(operation.static_sizes.get_values()) != sizes
            or tuple(operation.static_strides.get_values()) != (1,) * len(sizes)
            or views.shape(value) != sizes
        ):
            raise ValueError("actual shared RHS slice changed original element coordinates")
        views._view_chain((operation,))
        return operation.source

    for operation in inserted:
        if (
            type(operation)
            not in (tensor.EmptyOp, tensor.ExtractSliceOp, tensor.CollapseShapeOp, tensor.ExpandShapeOp, TransposeOp)
            or operation.attributes
        ):
            raise ValueError("shared RHS preparation added an unproved executable operation")
    for group, relation in groups:
        h, m, k = views.shape(group.lhs)
        n = views.shape(group.rhs)[-1]
        factor, count = relation.factor, h // relation.factor
        base = value_map[relation.base]
        # The source identity broadcast was checked before cloning. Read the
        # actual new views independently of the emitter and every saved edit.
        for g in range(count):
            rhs = None
            for r in range(factor):
                operation = operation_map[group.members[g * factor + r]]
                lhs_rows = sliced(operation.inputs[0], (r * m, 0), (m, k))
                if views.shape(lhs_rows) != (factor * m, k):
                    raise ValueError("shared RHS row root changes the complete group coverage")
                lhs_root = sliced(unshape(lhs_rows), (g, 0, 0), (1, factor * m, k))
                if (views.shape(lhs_root) != (count, factor * m, k)
                        or unshape(lhs_root, value_map[group.lhs]) is not value_map[group.lhs]):
                    raise ValueError("shared RHS rows changed the original LHS source")
                if rhs is not None and operation.inputs[1] is not rhs:
                    raise ValueError("shared RHS members do not use the same source-proven matrix")
                rhs = operation.inputs[1]
            reduced = sliced(unshape(rhs), (g, 0, 0), (1, k, n))
            if views.shape(reduced) != (count, k, n):
                raise ValueError("shared RHS reduced root changed original matrix coverage")
            for original, before, after in reversed(relation.steps):
                if type(original) is not TransposeOp:
                    continue
                reduced = unshape(reduced, base)
                transpose = reduced.owner
                if (
                    type(transpose) is not TransposeOp
                    or not views.transpose_value(transpose)
                    or transpose.permutation != original.permutation
                    or views.shape(transpose.inputs[0]) != before
                    or views.shape(reduced) != after
                ):
                    raise ValueError("shared RHS reduced transpose changed original value order")
                reduced = transpose.inputs[0]
            if unshape(reduced, base) is not base:
                raise ValueError("shared RHS views changed the source-proven repeated value")
    restored_values = {}
    restored = candidate.clone(value_mapper=restored_values)
    restored_ops = dict(zip(candidate.walk(), restored.walk(), strict=True))
    for operation, operands, original in edits:
        if operation is not operation_map[original] or tuple(operation.results) != (value_map[original.results[0]],):
            raise ValueError("shared RHS edit lost its original contraction identity")
        restored_ops[operation].operands = tuple(restored_values[value] for value in operands)
    for operation in reversed(inserted):
        rewritten = restored_ops[operation]
        rewritten.parent.erase_op(rewritten)
    if not source.is_structurally_equivalent(restored):
        raise ValueError("shared RHS inverse changed original SSA, effects or observed outputs")


@dataclass(frozen=True)
class SharedRhsRowViewsPlan:
    source: object
    candidate: object
    _original: tuple
    _candidate: tuple
    _records: tuple
    _owners: tuple = field(repr=False)

    def revalidate(self):
        if (id(self.source), id(self.candidate)) != self._owners or (
            _state(self.source) != self._original or _state(self.candidate) != self._candidate
        ):
            raise ValueError("shared RHS source, candidate or SSA changed; replan")
        self.candidate.verify()

    def record(self):
        self.revalidate()
        return {
            "schema": "merlin.shared_rhs_row_views.v1",
            "source_sha256": self._original[0],
            "candidate_sha256": self._candidate[0],
            "groups": [
                {
                    "original_products": h,
                    "shared_rhs_groups": count,
                    "members_per_group": factor,
                    "row_problem": list(problem),
                    "outputs": h,
                }
                for h, count, factor, problem in self._records
            ],
            "scope": "source tensor coordinates only; no target, alias, timing or immutable-input grant",
        }


def plan_shared_rhs_row_views(module, *, max_heads=256, max_groups=128, max_operations=16384, max_views=32):
    """Expose complete shared-RHS groups for existing dense row batching.

    Run the existing repeated-pointwise pass first when scalar work separates
    the source broadcast from the integer input. This reader never supplies an
    immutable coefficient binding and never infers equality from runtime data.
    Unsupported source returns None and the caller retains the original.
    """
    if type(module) is not builtin.ModuleOp or any(
        type(n) is not int or n <= 0 for n in (max_heads, max_groups, max_operations, max_views)
    ):
        raise ValueError("shared RHS views need an original module and positive bounded budgets")
    module.verify()
    if len(tuple(module.walk())) > max_operations:
        return None
    selected = []
    for group in _groups(module, max_heads=max_heads, max_groups=max_groups, max_candidates=max_operations):
        if not all(views.supported_scope(op) for op in group.members):
            continue
        try:
            relation = _repeated_matrices(group.rhs, max_views=max_views)
        except ValueError:
            continue
        selected.append((group, relation))
    if not selected:
        return None
    values = {}
    candidate = module.clone(value_mapper=values)
    operations = dict(zip(module.walk(), candidate.walk(), strict=True))
    inserted, edits, records = _emit(candidate, tuple(selected), values, operations)
    if len(tuple(candidate.walk())) > max_operations:
        return None
    _verify(module, candidate, tuple(selected), values, operations, inserted, edits)
    return SharedRhsRowViewsPlan(
        module, candidate, _state(module), _state(candidate), records, (id(module), id(candidate))
    )
