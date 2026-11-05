"""Refold complete independent integer matrix slices from actual source views.

This recognizes coordinates, never model/provenance names. Original readouts,
plane sums and observers keep their exact matrix results. The independent
inverse check restores every original operation and compares the entire IR.
Target implementation, effects, ABI and profitability remain separate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import prod

from xdsl.dialects import builtin, tensor
from xdsl.dialects.linalg.attrs import IteratorType, IteratorTypeAttr
from xdsl.dialects.linalg.ops import GenericOp
from xdsl.ir import Block
from xdsl.ir.affine import AffineDimExpr, AffineMap

from merlin.compile.scheduling.batched_dense import BatchedDenseProblem, BatchMatrixView
from merlin.transforms import dense_row_batching_check as views
from merlin.transforms.certificates import ir_digest
from merlin.transforms.dense_ir import recognize_signed_integer_matmul, static_zero_integer_matmul_geometry


def _slice(value):
    """Whole rank-two matrix obtained from one complete rank-three source slice."""
    matrix = views.shape(value)
    if len(matrix) != 2:
        raise ValueError("a head product must retain its original matrix geometry")
    seen = set()
    while type(value.owner) in (tensor.CollapseShapeOp, tensor.ExpandShapeOp):
        op = value.owner
        if value in seen:
            raise ValueError("unknown or cyclic matrix view")
        views._view_chain((op,))
        seen.add(value)
        if prod(views.shape(op.operands[0])) != prod(views.shape(value)):
            raise ValueError("matrix reshape changes element coverage")
        value = op.operands[0]
    op = value.owner
    if type(op) is not tensor.ExtractSliceOp or len(op.operands) != 1 or not views.boundary_attributes(op):
        raise ValueError("matrix does not have one static original head slice")
    root = op.source
    shape = views.shape(root)
    offsets = tuple(op.static_offsets.get_values())
    sizes = tuple(op.static_sizes.get_values())
    strides = tuple(op.static_strides.get_values())
    if (len(shape) != 3 or shape[1:] != matrix or sizes != (1, *matrix)
            or strides != (1, 1, 1) or len(offsets) != 3
            or offsets[1:] != (0, 0) or not 0 <= offsets[0] < shape[0]
            or views.shape(value) not in (sizes, matrix)):
        raise ValueError("matrix view is not a complete bounded source head")
    return root, offsets[0]


def read_batched_integer_product(op):
    """Independently check all batch maps and the original signed scalar body."""
    if type(op) is not GenericOp or len(op.inputs) != 2 or len(op.outputs) != 1 or len(op.results) != 1:
        raise ValueError("one explicit two-input batch product required")
    shapes = tuple(views.shape(v) for v in (*op.inputs, op.outputs[0]))
    if any(len(s) != 3 for s in shapes):
        raise ValueError("batch reader requires original rank-three tensors")
    (h, m, k), rhs, out = shapes
    n = rhs[-1]
    if rhs != (h, k, n) or out != (h, m, n) or op.results[0].type != op.outputs[0].type:
        raise ValueError("batch product changes original matrix coverage")
    dims = tuple(AffineDimExpr(i) for i in range(4))
    expected = ((dims[0], dims[1], dims[3]), (dims[0], dims[3], dims[2]), dims[:3])
    if tuple(a.data for a in op.indexing_maps) != tuple(AffineMap(4, 0, x) for x in expected):
        raise ValueError("batch product changes head or reduction coordinates")
    if tuple(a.data for a in op.iterator_types) != (
        IteratorType.PARALLEL, IteratorType.PARALLEL, IteratorType.PARALLEL, IteratorType.REDUCTION,
    ):
        raise ValueError("batch product changes independent/reduction axes")
    if set(op.properties) - {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
        raise ValueError("batch product has unknown executable properties")
    # Only the already checked tensor maps/ranks are normalized on an analysis
    # clone. The original scalar body goes through the existing strict reader.
    types = tuple(builtin.TensorType(v.type.element_type, s) for v, s in zip(
        (*op.inputs, op.outputs[0]), ((m, k), (k, n), (m, n)), strict=True,
    ))
    arguments = Block(arg_types=types)
    d0, d1, d2 = (AffineDimExpr(i) for i in range(3))
    normalized = GenericOp.create(
        operands=arguments.args, result_types=[types[-1]],
        properties={"operandSegmentSizes": builtin.DenseArrayBase.from_list(builtin.i32, [2, 1]),
                    "indexing_maps": builtin.ArrayAttr(builtin.AffineMapAttr(AffineMap(3, 0, x))
                    for x in ((d0, d2), (d2, d1), (d0, d1))),
                    "iterator_types": builtin.ArrayAttr(IteratorTypeAttr(x) for x in (
                        IteratorType.PARALLEL, IteratorType.PARALLEL, IteratorType.REDUCTION))},
        attributes=dict(op.attributes), regions=[op.body.clone()],
    )
    try:
        recognize_signed_integer_matmul(normalized)
    finally:
        normalized.drop_all_references()
    return BatchedDenseProblem((h,), BatchMatrixView(shapes[0], (0,)), BatchMatrixView(rhs, (0,)))


@dataclass(frozen=True)
class _Group:
    members: tuple
    lhs: object
    rhs: object
    problem: BatchedDenseProblem


def _groups(module, *, max_heads, max_groups, max_candidates):
    partitions = {}
    count = 0
    for op in module.walk():
        if type(op) is not GenericOp:
            continue
        try:
            m, k, n = static_zero_integer_matmul_geometry(op)
            a, ah = _slice(op.inputs[0])
            b, bh = _slice(op.inputs[1])
            h = views.shape(a)[0]
            if ah != bh or views.shape(b)[0] != h or not 2 <= h <= max_heads:
                continue
        except (ValueError, AttributeError):
            continue
        count += 1
        if count > max_candidates:
            return ()
        partitions.setdefault((op.parent, a, b, m, k, n), []).append((ah, op))
    result = []
    for (block, a, b, m, k, n), rows in partitions.items():
        rows.sort(key=lambda x: x[0])
        h = views.shape(a)[0]
        if tuple(i for i, _ in rows) != tuple(range(h)):
            continue
        members = tuple(op for _, op in rows)
        body = tuple(block.ops)
        first, last = min(map(body.index, members)), max(map(body.index, members))
        if not views.dominating(a, body[first]) or not views.dominating(b, body[first]):
            continue
        if not views.safe_span(body[first:last+1]):
            continue
        result.append(_Group(members, a, b, BatchedDenseProblem(
            (h,), BatchMatrixView((h, m, k), (0,)), BatchMatrixView((h, k, n), (0,)),
        )))
        if len(result) >= max_groups:
            break
    return tuple(result)


def _batch_type(shape):
    return builtin.TensorType(builtin.i32, shape)


def _emit(candidate, groups, value_map, operation_map):
    replacements = []
    for group in groups:
        mapped = tuple(operation_map[op] for op in group.members)
        body = tuple(mapped[0].parent.ops)
        anchor = min(mapped, key=body.index)
        init = anchor.outputs[0].owner
        zero = init.input if type(init) is tensor.SplatOp else init.inputs[0]
        result_type = _batch_type(group.problem.output_shape)
        initialized = tensor.SplatOp(zero, [], result_type)
        dims = tuple(AffineDimExpr(i) for i in range(4))
        batch = GenericOp.create(
            operands=[value_map[group.lhs], value_map[group.rhs], initialized.result],
            result_types=[result_type], regions=[anchor.body.clone()],
            properties={"operandSegmentSizes": builtin.DenseArrayBase.from_list(builtin.i32, [2, 1]),
                    "indexing_maps": builtin.ArrayAttr(builtin.AffineMapAttr(AffineMap(4, 0, x))
                        for x in ((dims[0], dims[1], dims[3]), (dims[0], dims[3], dims[2]), dims[:3])),
                        "iterator_types": builtin.ArrayAttr(IteratorTypeAttr(x) for x in (
                            IteratorType.PARALLEL, IteratorType.PARALLEL,
                            IteratorType.PARALLEL, IteratorType.REDUCTION))},
        )
        anchor.parent.insert_ops_before((initialized, batch), anchor)
        sliced = []
        for h, (source, op) in enumerate(zip(group.members, mapped, strict=True)):
            view = tensor.ExtractSliceOp.from_static_parameters(
                batch.results[0], (h, 0, 0), (1, group.problem.m, group.problem.n),
            )
            collapsed = tensor.CollapseShapeOp.create(
                operands=[view.result], result_types=[op.results[0].type],
                properties={"reassociation": builtin.ArrayAttr([
                    builtin.ArrayAttr(builtin.IntegerAttr(i, builtin.i64) for i in (0, 1)),
                    builtin.ArrayAttr([builtin.IntegerAttr(2, builtin.i64)]),
                ])},
            )
            op.parent.insert_ops_before((view, collapsed), op)
            op.results[0].replace_all_uses_with(collapsed.results[0])
            value_map[source.results[0]] = collapsed.results[0]
            op.parent.erase_op(op)
            sliced.append((view, collapsed))
        replacements.append((group, initialized, batch, tuple(sliced)))
    return tuple(replacements)


def _verify(source, candidate, value_map, replacements):
    candidate.verify()
    restored_values = {}
    restored = candidate.clone(value_mapper=restored_values)
    operations = dict(zip(candidate.walk(), restored.walk(), strict=True))
    original_values = {v: restored_values[w] for v, w in value_map.items() if w in restored_values}
    records = []
    for group, initialized, batch, sliced in replacements:
        if read_batched_integer_product(batch) != group.problem:
            raise ValueError("actual batch differs from original head geometry")
        if (batch.inputs[0] is not value_map[group.lhs] or batch.inputs[1] is not value_map[group.rhs]
                or batch.outputs[0] is not initialized.result):
            raise ValueError("actual batch changed original roots or initialization")
        from merlin.compile.analysis.contraction_shapes import zero_initialised

        if not zero_initialised(batch):
            raise ValueError("actual batch changed the original zero seed")
        for h, (original, (view, collapsed)) in enumerate(zip(group.members, sliced, strict=True)):
            if (view.source is not batch.results[0] or tuple(view.static_offsets.get_values()) != (h, 0, 0)
                    or tuple(view.static_sizes.get_values()) != (1, group.problem.m, group.problem.n)
                    or tuple(view.static_strides.get_values()) != (1, 1, 1)
                    or tuple(collapsed.operands) != (view.result,)
                    or tuple(tuple(x.value.data for x in a) for a in collapsed.reassociation) != ((0, 1), (2,))
                    or collapsed.results[0].type != original.results[0].type):
                raise ValueError("actual result view changed original head coverage")
            checked_view, checked_collapse = operations[view], operations[collapsed]
            old = original.clone(value_mapper=dict(original_values))
            checked_view.parent.insert_op_before(old, checked_view)
            checked_collapse.results[0].replace_all_uses_with(old.results[0])
            original_values[original.results[0]] = old.results[0]
            checked_collapse.parent.erase_op(checked_collapse)
            checked_view.parent.erase_op(checked_view)
        operations[batch].parent.erase_op(operations[batch])
        operations[initialized].parent.erase_op(operations[initialized])
        records.append(group.problem.record())
    if not source.is_structurally_equivalent(restored):
        raise ValueError("batch inverse did not restore complete original SSA, effects and metadata")
    return tuple(records)


@dataclass(frozen=True)
class HeadBatchingPlan:
    source: object
    candidate: object
    _source_digest: str
    _candidate_digest: str
    _value_map: tuple
    _replacements: tuple
    _verified_identity: tuple = field(init=False, repr=False, compare=False)

    def _identity(self):
        # The printed IR binds semantics; these identities also bind the SSA
        # objects used by the inverse witness. Replacing a graph with an equal
        # clone must not leave the witness pointing into the detached graph.
        def graph(root):
            return tuple((id(op), id(op.parent), tuple(map(id, op.operands)), tuple(map(id, op.results)),
                          tuple((id(region), id(region.parent), tuple(
                              (id(block), id(block.parent), tuple(map(id, block.args)))
                              for block in region.blocks)) for region in op.regions))
                         for op in root.walk())

        return (id(self.source), id(self.candidate), self._source_digest, self._candidate_digest,
                graph(self.source), graph(self.candidate),
                tuple((id(source), id(candidate)) for source, candidate in self._value_map),
                tuple((tuple(map(id, group.members)), id(group.lhs), id(group.rhs),
                       json.dumps(group.problem.record(), sort_keys=True), id(initialized), id(batch),
                       tuple((id(view), id(collapsed)) for view, collapsed in sliced))
                      for group, initialized, batch, sliced in self._replacements))

    def _check_digests(self):
        if ir_digest(self.source) != self._source_digest or ir_digest(self.candidate) != self._candidate_digest:
            raise ValueError("head batching source/candidate changed; replan")

    def __post_init__(self):
        # Every construction, including dataclasses.replace(), runs the full
        # inverse check. Callers cannot supply a prior proof or a trusted flag.
        self._check_digests()
        _verify(self.source, self.candidate, dict(self._value_map), self._replacements)
        object.__setattr__(self, "_verified_identity", self._identity())

    def revalidate(self):
        self._check_digests()
        if self._identity() != self._verified_identity:
            raise ValueError("head batching SSA or inverse witness changed; replan")
        self.candidate.verify()
        # Fresh records prevent a consumer from changing later reports. Only
        # inverse reconstruction is reused; graph/custody checks remain fresh.
        return tuple(group.problem.record() for group, *_ in self._replacements)

    def record(self):
        groups = self.revalidate()
        return {"schema": "merlin.independent_head_batching.v1", "source_sha256": self._source_digest,
                "candidate_sha256": self._candidate_digest, "batches": groups,
                "original_products": sum(g["products"] for g in groups),
                "scope": "exact tensor algebra; independent target/ABI/ownership/cost obligations"}


def plan_independent_head_batching(module, *, max_heads=256, max_groups=128, max_candidates=16384):
    """Return a checked source clone or the unchanged module on unsupported source.

    All original scalar readouts and plane-combination arithmetic remain in
    place. No source identifier selects applicability and no target is inferred.
    """
    if any(type(n) is not int or n <= 0 for n in (max_heads, max_groups, max_candidates)):
        raise ValueError("head batching needs explicit positive construction budgets")
    module.verify()
    groups = _groups(module, max_heads=max_heads, max_groups=max_groups, max_candidates=max_candidates)
    if not groups:
        digest = ir_digest(module)
        return HeadBatchingPlan(module, module, digest, digest, (), ())
    values = {}
    candidate = module.clone(value_mapper=values)
    operations = dict(zip(module.walk(), candidate.walk(), strict=True))
    replacements = _emit(candidate, groups, values, operations)
    plan = HeadBatchingPlan(module, candidate, ir_digest(module), ir_digest(candidate),
                            tuple(values.items()), replacements)
    return plan
