"""Bounded tensor-level row batching of exact common-RHS integer dots.

Choices use current scalar semantics and immutable tensor coordinates. No
instruction, retention, physical-layout, execution or profitability law is
inferred. Ordinary callers must independently qualify actual emitted products.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from math import prod
from pathlib import Path

from xdsl.dialects import builtin, tensor
from xdsl.dialects.linalg.ops import GenericOp, TransposeOp
from xdsl.ir import Operation

from merlin.compiler.transforms.certificates import ir_digest
from merlin.compiler.transforms.dense import row_batching_check as check
from merlin.compiler.transforms.dense.ir import recognize_signed_integer_matmul
from merlin.compiler.transforms.pipeline_state import AnalysisStamp, PassSpec, PipelineContext, PipelineSession

# ---- dense_row_batching ------------------------------------------------------------------------


@dataclass(frozen=True)
class DenseRowBatchingBudgets:
    max_candidates: int = 256
    max_members: int = 8
    max_batches: int = 8
    max_source_elements: int = 64 << 20
    max_constructed_operations: int = 16384
    index_bits: int = 64

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ValueError("row batching budgets require positive original integers")
        if not 2 <= self.max_members <= 64 or not 2 <= self.index_bits <= 128:
            raise ValueError("row batching member/index budgets exceed their bounded domain")


def implementation_paths():
    from merlin.compiler.lower.host import propagation as layout_conformance
    from merlin.compiler.lower.source import dense_group as segmented_matrix_view
    from merlin.compiler.schedule.elementwise import pointwise_source
    from merlin.compiler.transforms import certificates, pipeline_state
    from merlin.compiler.transforms.dense import ir as dense_ir

    return tuple(
        Path(module.__file__)
        for module in (
            __import__(__name__, fromlist=["_"]),
            check,
            dense_ir,
            segmented_matrix_view,
            layout_conformance,
            pointwise_source,
            certificates,
            pipeline_state,
        )
    )


def _state(module):
    return ir_digest(module), tuple(
        (
            operation,
            operation.parent,
            tuple(operation.operands),
            tuple(operation.results),
            tuple(value.type for value in (*operation.operands, *operation.results)),
        )
        + (tuple(frozenset((use.operation, use.index) for use in value.uses) for value in operation.results),)
        for operation in module.walk()
    )


def _functions():
    from merlin.compiler.transforms.dense import ir as dense_ir

    return (
        _functions,
        implementation_paths,
        _state,
        _groups,
        _emit,
        check.geometry,
        check.matrix_view,
        check.rhs_identity,
        check.safe_span,
        check.supported_scope,
        check.verify_dense_row_batching,
        check.shape,
        check.boundary_attributes,
        check._view_chain,
        check.transpose_value,
        check.dominating,
        check._positive_division,
        check._groups,
        check._batch,
        check._trace_attributes,
        check.static_zero_integer_matmul_geometry,
        check.checked_static_linalg_shell,
        check.read_pointwise_source,
        check.prove_segmented_matrix,
        check._metadata,
        dense_ir.recognize_signed_integer_matmul,
        dense_ir._trace_attributes,
    )


def _groups(module, budgets):
    """Read the original source only; no references or candidate costs are used."""
    candidates, count = [], 0
    for operation in module.walk():
        if type(operation) is not GenericOp or not check.supported_scope(operation):
            continue
        try:
            dimensions, dtypes = check.geometry(operation, index_bits=budgets.index_bits)
            # Existing realized batches are preserved rather than regrouped.
            if check._batch(operation, budgets.index_bits) is not None:
                return ()
            root, view = check.matrix_view(operation.inputs[0])
            rhs = check.rhs_identity(operation.inputs[1])
        except check.RowBatchingUnavailable:
            continue
        count += 1
        if count > budgets.max_candidates:
            return ()
        m, k, n = dimensions
        if view.row_stride != k or view.segment_stride != view.segment_rows * k:
            continue
        candidates.append((operation, dimensions, dtypes, root, view, rhs))
    partitions = {}
    for row in candidates:
        operation, (_, k, n), dtypes, root, _, rhs = row
        partitions.setdefault((operation.parent, k, n, dtypes, root, rhs), []).append(row)
    groups = []
    # This discovery is read-only and invokes no external callbacks. Several
    # independent partitions may share a large entry block; take its current
    # order once, without keeping any snapshot across later source edits.
    block_orders = {}
    for rows in partitions.values():
        if not 2 <= len(rows) <= budgets.max_members:
            continue
        rows = sorted(rows, key=lambda row: row[4].origin)
        root, k, n = rows[0][3], rows[0][1][1], rows[0][1][2]
        total = cursor = 0
        for _, (m, _, _), _, _, view, _ in rows:
            if view.origin != cursor:
                break
            total += m
            cursor += m * k
        else:
            if cursor != prod(check.shape(root)):
                continue
            source_elements = total * k + k * n + total * n
            if source_elements > budgets.max_source_elements:
                continue
            limit = (1 << (budgets.index_bits - 1)) - 1
            if source_elements * max(v.type.element_type.width.data // 8 for v in rows[0][0].operands) > limit:
                continue
            block = rows[0][0].parent
            if block not in block_orders:
                body = tuple(block.ops)
                block_orders[block] = body, {op: index for index, op in enumerate(body)}
            body, positions = block_orders[block]
            members = tuple(row[0] for row in rows)
            first = min(members, key=positions.__getitem__)
            last = max(members, key=positions.__getitem__)
            if not check.dominating(root, first) or not check.safe_span(body[positions[first] : positions[last] + 1]):
                continue
            groups.append((members, first, root, total, k, n))
            if len(groups) == budgets.max_batches:
                break
    return tuple(groups)


def _emit(candidate, original_groups, ordinals):
    """Emit actual matmul inputs and exact slices; no copy/concat is introduced."""
    cloned = tuple(candidate.walk())
    for members, first, root, total, k, n in original_groups:
        mapped = tuple(cloned[ordinals[member]] for member in members)
        anchor = cloned[ordinals[first]]
        original_root = root
        if hasattr(original_root.owner, "results"):
            root = cloned[ordinals[original_root.owner]].results[original_root.index]
        else:
            owner = cloned[ordinals[original_root.owner.parent_op()]]
            root = owner.regions[0].blocks[0].args[original_root.index]
        block = anchor.parent
        elem = root.type.element_type
        reassociation = builtin.ArrayAttr(
            [builtin.ArrayAttr([builtin.IntegerAttr(axis, builtin.i64) for axis in range(len(check.shape(root)))])]
        )
        flat = tensor.CollapseShapeOp(
            operands=[root],
            result_types=[builtin.TensorType(elem, (total * k,))],
            properties={"reassociation": reassociation},
        )
        expanded = tensor.ExpandShapeOp(
            flat.result,
            [],
            builtin.ArrayAttr(
                [builtin.ArrayAttr([builtin.IntegerAttr(0, builtin.i64), builtin.IntegerAttr(1, builtin.i64)])]
            ),
            (total, k),
            builtin.TensorType(elem, (total, k)),
        )
        init = anchor.outputs[0].owner
        zero = init.input if type(init) is tensor.SplatOp else init.inputs[0]
        result_type = builtin.TensorType(anchor.results[0].type.element_type, (total, n))
        initialized = tensor.SplatOp(zero, [], result_type)
        batch = GenericOp.create(
            operands=[expanded.result, anchor.inputs[1], initialized.result],
            result_types=[result_type],
            properties=dict(anchor.properties),
            regions=[anchor.body.clone()],
        )
        block.insert_ops_before((flat, expanded, initialized, batch), anchor)
        row = 0
        for operation in mapped:
            m = check.shape(operation.results[0])[0]
            view = tensor.ExtractSliceOp.from_static_parameters(batch.results[0], (row, 0), (m, n))
            view.attributes.update(operation.attributes)
            block.insert_op_before(view, operation)
            operation.results[0].replace_all_uses_with(view.result)
            block.erase_op(operation)
            row += m
    return {}


@dataclass(frozen=True)
class DenseRowBatchingPlan:
    source: builtin.ModuleOp
    candidate: builtin.ModuleOp
    stamp: AnalysisStamp
    budgets: DenseRowBatchingBudgets
    _original: tuple
    _candidate: tuple
    _sources: tuple
    _code: tuple
    _checks: tuple
    _selection: tuple
    _reader: object

    def revalidate(self):
        if (
            _state(self.source) != self._original
            or _state(self.candidate) != self._candidate
            or (asdict(self.stamp), asdict(self.budgets)) != self._selection
            or _functions is not self._reader
            or tuple(function for function, _ in self._code) != _functions()
            or any(function.__code__ is not code for function, code in self._code)
            or any(hashlib.sha256(path.read_bytes()).hexdigest() != digest for path, digest in self._sources)
        ):
            raise ValueError("row batching source, candidate or original checker changed; replan")
        actual = check.verify_dense_row_batching(self.source, self.candidate, index_bits=self.budgets.index_bits)
        if actual != self._checks:
            raise ValueError("row batching actual candidate correspondence changed")

    def record(self):
        self.revalidate()
        return {
            "schema": "merlin.source_dense_row_batching.v1",
            "source_sha256": self._original[0],
            "candidate_sha256": self._candidate[0],
            "context": asdict(self.stamp.context),
            "budgets": asdict(self.budgets),
            "batches": list(self._checks),
            "scope": "ordered tensor algebra only; no physical execution, retention or cost qualification",
        }

    def implementation_paths(self):
        self.revalidate()
        return tuple(path for path, _ in self._sources)


def plan_source_dense_row_batching(module, context, budgets):
    """Clone a bounded source candidate, retaining the original on no match."""
    if (
        type(module) is not builtin.ModuleOp
        or type(context) is not PipelineContext
        or type(budgets) is not DenseRowBatchingBudgets
    ):
        raise ValueError("row batching requires original typed module, context and budgets")
    context.__post_init__()
    budgets.__post_init__()
    module.verify()
    operations = tuple(module.walk())
    if len(operations) > budgets.max_constructed_operations:
        return None
    original = _state(module)
    code = tuple((function, function.__code__) for function in _functions())
    sources = tuple((path, hashlib.sha256(path.read_bytes()).hexdigest()) for path in implementation_paths())
    groups = _groups(module, budgets)
    if not groups:
        return None
    added = sum(4 + len(tuple(first.body.walk())) + len(members) for members, first, *_ in groups)
    if len(operations) + added > budgets.max_constructed_operations:
        return None
    candidate = module.clone()
    session = PipelineSession(candidate, context=context, ir_level="linalg", max_repairs=0)
    ordinals = {operation: i for i, operation in enumerate(operations)}
    session.run(
        PassSpec("dense-row-source", 0, mutates_ir=False, ir_level="linalg", produces=("source",)),
        lambda: {"source": ir_digest(module)},
    )
    session.run(
        PassSpec("dense-row-batching", 1, ir_level="linalg", requires=("source",), invalidates=("*",)),
        lambda: _emit(candidate, groups, ordinals),
    )
    candidate.verify()
    checks = check.verify_dense_row_batching(module, candidate, index_bits=budgets.index_bits)
    stamp = AnalysisStamp(original[0], context, "linalg")
    plan = DenseRowBatchingPlan(
        module,
        candidate,
        stamp,
        budgets,
        original,
        _state(candidate),
        sources,
        code,
        checks,
        (asdict(stamp), asdict(budgets)),
        _functions,
    )
    plan.revalidate()
    return plan


# ---- dense_reduction_layout --------------------------------------------------------------------
# Keep channel-last storage for exact single-column integer contractions.
#
# Recognize ``reshape(transpose(X)) @ w`` when the transpose merely places
# the original trailing channel axis first. Replace it by ``w.T @ reshape(X)``
# and a unit-axis output view. The signed modular scalar recurrence and reduction
# order are unchanged. This is a source transformation, not a target schedule or
# a claim about physical aliasing, execution costs or hardware support.


@dataclass
class ReductionLayoutReport:
    rewritten: int = 0
    eliminated_transpose_bytes: int = 0
    preferred_layouts: int = 0

    def to_dict(self):
        return asdict(self)


def _static(value):
    ty = value.type
    if (
        not isinstance(ty, builtin.TensorType)
        or ty.encoding != builtin.NoneAttr()
        or not ty.get_shape()
        or any(d <= 0 for d in ty.get_shape())
    ):
        return None
    return ty


def _pure(op):
    """Do not move reads through opaque effects or coordinate-observing bodies."""
    from merlin.compiler.lower.host.propagation import per_tensor_quantization

    if per_tensor_quantization(op):
        return True
    if op.name in {
        "tensor.empty",
        "tensor.splat",
        "tensor.collapse_shape",
        "tensor.expand_shape",
        "arith.constant",
        "linalg.transpose",
    }:
        return all(inner.name == "linalg.yield" for region in op.regions for inner in region.walk())
    if op.name != "linalg.generic":
        return False
    if set(op.properties) - {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
        return False
    return all(
        inner.name.startswith(("arith.", "math.")) or inner.name == "linalg.yield"
        for region in op.regions
        for inner in region.walk()
    )


def _reshape(value, shape, anchor):
    old = value.type
    if old.get_shape() == tuple(shape):
        return value
    rank = len(old.get_shape())
    flat_type = builtin.TensorType(old.element_type, [prod(shape)])
    collapse = tensor.CollapseShapeOp.create(
        operands=[value],
        result_types=[flat_type],
        properties={
            "reassociation": builtin.ArrayAttr(
                [builtin.ArrayAttr([builtin.IntegerAttr(i, builtin.i64) for i in range(rank)])]
            )
        },
    )
    anchor.parent.insert_op_before(collapse, anchor)
    reassociation = builtin.ArrayAttr(
        [builtin.ArrayAttr([builtin.IntegerAttr(i, builtin.i64) for i in range(len(shape))])]
    )
    result_type = builtin.TensorType(old.element_type, shape)
    expand = tensor.ExpandShapeOp(collapse.results[0], [], reassociation, shape, result_type)
    anchor.parent.insert_op_before(expand, anchor)
    return expand.results[0]


def rewrite_module(module) -> ReductionLayoutReport:
    """Apply only exact, sole-use, static channel-last reduction alternatives.

    Other consumers retain the original transpose. Unknown scalar recurrences,
    nonunit output columns, changed reduction ordering and opaque intervening
    effects remain unchanged. No model, operation provenance or target name
    participates in selection.
    """
    module.verify()
    report = ReductionLayoutReport()
    for op in tuple(module.walk()):
        try:
            lhs, rhs, init = recognize_signed_integer_matmul(op)
        except ValueError:
            continue
        m, k = lhs.type.get_shape()
        if rhs.type.get_shape() != (k, 1) or m <= 1 or k <= 1:
            continue
        chain, value = [], lhs
        while isinstance(value.owner, tensor.CollapseShapeOp | tensor.ExpandShapeOp):
            view = value.owner
            if (
                view.parent is not op.parent
                or not value.has_one_use()
                or len(view.operands) != 1
                or _static(view.operands[0]) is None
                or not _pure(view)
            ):
                break
            chain.append(view)
            value = view.operands[0]
        trans = value.owner
        direct = type(trans) is TransposeOp
        if getattr(trans, "parent", None) is not op.parent or not value.has_one_use() or not _pure(trans):
            continue
        if direct:
            source = trans.inputs[0]
            ty = _static(source)
            if ty is None:
                continue
            shape = ty.get_shape()
            permutation = tuple(trans.permutation.get_values())
            axes = tuple(i for i, extent in enumerate(shape) if extent != 1)
            expected = (len(shape) - 1, *axes[:-1])
            actual = tuple(i for i in permutation if shape[i] != 1)
            if len(shape) < 2 or shape[-1] != m or prod(shape[:-1]) != k or actual != expected:
                continue
        else:
            from merlin.compiler.lower.host.propagation import _input_layout

            ty = _static(value)
            permutation = _input_layout(value, op.parent)
            if ty is None or permutation is None:
                continue
            shape = ty.get_shape()
            axes = tuple(i for i, extent in enumerate(shape) if extent != 1)
            channel = permutation[-1]
            if (
                not axes
                or channel != axes[0]
                or shape[channel] != m
                or prod(shape[i] for i in range(len(shape)) if i != channel) != k
                or tuple(i for i in permutation if shape[i] != 1) != (*axes[1:], channel)
            ):
                continue
        current = trans.next_op
        while current is not None and current is not op and _pure(current):
            current = current.next_op
        if current is not op:
            continue
        if not direct:
            from merlin.compiler.lower.host.propagation import realize_input_layout

            source, _ = realize_input_layout(module, value, permutation, chain[-1] if chain else op)
            report.preferred_layouts += 1
        left = _reshape(rhs, (1, k), op)
        right = _reshape(source, (k, m), op)
        seed = _reshape(init, (1, m), op)
        replacement = GenericOp(
            [left, right],
            [seed],
            op.body.clone(),
            list(op.indexing_maps),
            list(op.iterator_types),
            [builtin.TensorType(init.type.element_type, [1, m])],
        )
        replacement.attributes.update(op.attributes)
        # Preserve the alternative selected by this source transformation when
        # stating a device command. Immutable ownership does not prescribe
        # which operand occupies the stationary execution slot.
        replacement.attributes["merlin.preserve_contraction_orientation"] = builtin.UnitAttr()
        op.parent.insert_op_before(replacement, op)
        result = _reshape(replacement.results[0], (m, 1), op)
        op.results[0].replace_all_uses_with(result)
        op.parent.erase_op(op)
        pending = [*chain, trans]
        while pending:
            dead = pending.pop(0)
            if (
                not isinstance(dead, Operation)
                or dead.parent is None
                or not _pure(dead)
                or any(result.uses for result in dead.results)
            ):
                continue
            pending.extend(operand.owner for operand in dead.operands)
            dead.parent.erase_op(dead)
        report.rewritten += 1
        if direct:
            report.eliminated_transpose_bytes += prod(shape)  # signed i8 input
    module.verify()
    return report
