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
from xdsl.dialects.linalg.ops import GenericOp

from merlin.transforms.certificates import ir_digest
from merlin.transforms.pipeline_state import AnalysisStamp, PassSpec, PipelineContext, PipelineSession

from . import dense_row_batching_check as check


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
    from merlin.compile.scheduling import pointwise_source
    from merlin.llvmlower import layout_conformance, segmented_matrix_view
    from merlin.transforms import certificates, dense_ir, pipeline_state

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
    from merlin.transforms import dense_ir

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
            body = tuple(rows[0][0].parent.ops)
            members = tuple(row[0] for row in rows)
            first = min(members, key=body.index)
            last = max(members, key=body.index)
            if not check.dominating(root, first) or not check.safe_span(body[body.index(first) : body.index(last) + 1]):
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
