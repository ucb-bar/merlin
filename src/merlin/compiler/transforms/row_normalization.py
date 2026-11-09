"""Materialize and check exact source normalization islands one row at a time.

The original maximum, subtraction, exponential, sum and division regions retain
their scalar order, attributes and initializers. Independent row fibers may be
interchanged only after source admission and the existing independent checker.
Unknown effects, layouts, observers and costs retain the original route.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import prod
from pathlib import Path

from xdsl.dialects import arith, builtin, func, scf, tensor
from xdsl.dialects.linalg import ops as L
from xdsl.ir import Block, BlockArgument, Operation, Region, SSAValue
from xdsl.ir.affine import AffineDimExpr, AffineMap
from xdsl.rewriter import Rewriter

from merlin.transforms.certificates import ir_digest
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession

from . import row_normalization_source
from .row_normalization_source import (
    RowNormalizationGroup as RowNormalizationGroup,
)
from .row_normalization_source import (
    _body as _body,
)
from .row_normalization_source import (
    _check_operation as _check_operation,
)
from .row_normalization_source import (
    _fiber_map as _fiber_map,
)
from .row_normalization_source import (
    _generic as _generic,
)
from .row_normalization_source import (
    _identity_copy as _identity_copy,
)
from .row_normalization_source import (
    _linear as _linear,
)
from .row_normalization_source import (
    _metadata as _metadata,
)
from .row_normalization_source import (
    _reduce as _reduce,
)
from .row_normalization_source import (
    _shape as _shape,
)
from .row_normalization_source import (
    _view as _view,
)
from .row_normalization_source import (
    local_type as local_type,
)
from .row_normalization_source import (
    match_row_normalization as match_row_normalization,
)
from .row_normalization_source import (
    ordered_contraction as ordered_contraction,
)
from .row_normalization_source import (
    row_input_projection as row_input_projection,
)
from .row_normalization_source import (
    row_input_specs as row_input_specs,
)
from .row_normalization_source import (
    row_input_values as row_input_values,
)


@dataclass(frozen=True)
class RowNormalizationRewrite:
    source: RowNormalizationGroup
    emitted: tuple[Operation, ...]
    result: SSAValue


def implementation_sources():
    """Reviewed current implementation/checker paths, including the old alias."""
    from merlin.transforms import pipeline, pipeline_state, row_normalization_check

    return (
        (__name__, Path(__file__)),
        (row_normalization_source.__name__, Path(row_normalization_source.__file__)),
        ("merlin.transforms.row_normalization", Path(__file__).parents[2] / "transforms/row_normalization.py"),
        (row_normalization_check.__name__, Path(row_normalization_check.__file__)),
        (pipeline.__name__, Path(pipeline.__file__)),
        (pipeline_state.__name__, Path(pipeline_state.__file__)),
    )


def emit_row_normalization(group):
    """Emit ordinary tensor/SCF/linalg; upstream still owns CPU bufferization."""
    source = group.source
    zero = arith.ConstantOp.from_int_and_width(0, builtin.IndexType())
    end = arith.ConstantOp.from_int_and_width(group.rows, builtin.IndexType())
    one = arith.ConstantOp.from_int_and_width(1, builtin.IndexType())
    output = tensor.EmptyOp((), source.type)
    prefix = [zero, end, one, output]
    block = Block(arg_types=[builtin.IndexType(), source.type])
    coordinates = []
    row_axes = {}
    static_offsets = []
    # Keep the public output's original rank. A collapse/expand around a
    # returned loop buffer hides it from ordinary out-parameter forwarding.
    for axis, extent in enumerate(group.shape[:-1]):
        if extent == 1:
            static_offsets.append(0)
            continue
        stride = arith.ConstantOp.from_int_and_width(prod(group.shape[axis + 1 : -1]), builtin.IndexType())
        quotient = arith.DivUIOp(block.args[0], stride.result)
        dimension = arith.ConstantOp.from_int_and_width(extent, builtin.IndexType())
        coordinate = arith.RemUIOp(quotient.result, dimension.result)
        block.add_ops((stride, quotient, dimension, coordinate))
        coordinates.append(coordinate.result)
        row_axes[axis] = coordinate.result
        static_offsets.append(builtin.DYNAMIC_INDEX)
    static_offsets.append(0)
    sizes = [1] * (len(group.shape) - 1) + [group.columns]
    mapped = {}
    for value, projection in row_input_specs(group):
        shape = _shape(value)
        if not shape:
            mapped[value] = value
            continue
        offsets = [row_axes[axis] for axis in projection if axis in row_axes]
        input_offsets = [builtin.DYNAMIC_INDEX if axis in row_axes else 0 for axis in projection]
        input_sizes = [group.columns if axis == len(group.shape) - 1 else 1 for axis in projection]
        local_shape = (group.columns,) if len(group.shape) - 1 in projection else ()
        sliced = tensor.ExtractSliceOp.build(
            operands=[value, offsets, [], []],
            result_types=[builtin.TensorType(value.type.get_element_type(), local_shape)],
            properties={
                "static_offsets": builtin.DenseArrayBase.from_list(builtin.i64, input_offsets),
                "static_sizes": builtin.DenseArrayBase.from_list(builtin.i64, input_sizes),
                "static_strides": builtin.DenseArrayBase.from_list(builtin.i64, [1] * len(shape)),
            },
        )
        block.add_op(sliced)
        mapped[value] = sliced.result
    for old in group.operations:
        if isinstance(old, arith.ConstantOp):
            op = old.clone()
            block.add_op(op)
            mapped[old.results[0]] = op.results[0]
            continue
        typ = local_type(old.results[0], group)
        if isinstance(old, (tensor.CollapseShapeOp, tensor.ExpandShapeOp)):
            mapped[old.results[0]] = mapped[old.operands[0]]
            continue
        operands = [mapped.get(value, value) for value in old.operands]
        if isinstance(old, L.GenericOp):
            maps = builtin.ArrayAttr(
                [
                    builtin.AffineMapAttr(AffineMap(1, 0, (AffineDimExpr(0),) if value.type.get_num_dims() else ()))
                    for value in operands
                ]
            )
            op = L.GenericOp(
                operands[:-1],
                (operands[-1],),
                old.regions[0].clone(),
                maps,
                builtin.ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)]),
                (typ,),
            )
        elif isinstance(old, L.ReduceOp):
            op = L.ReduceOp(
                operands[0], operands[1], builtin.DenseArrayBase.from_list(builtin.i64, [0]), old.regions[0].clone()
            )
        else:
            op = type(old).create(operands=operands, result_types=[typ], properties=old.properties.copy())
        op.attributes.update(old.attributes)
        block.add_op(op)
        mapped[old.results[0]] = op.results[0]
    inserted = tensor.InsertSliceOp.get(
        mapped[group.operations[-1].results[0]],
        block.args[1],
        static_sizes=sizes,
        static_offsets=static_offsets,
        offsets=coordinates,
    )
    block.add_ops((inserted, scf.YieldOp(inserted.result)))
    loop = scf.ForOp(zero.result, end.result, one.result, [output.tensor], Region(block))
    prefix.append(loop)
    return tuple(prefix), loop.results[0]


def row_state_cost_reason(group):
    """Conservative unknown-cost policy, including existing fusion boundaries."""
    if group.rows <= group.columns + 2:
        return "row_state_not_smaller"
    try:
        for value in row_input_values(group):
            source = _view(value)
            if prod(_shape(value)) < prod(group.shape):
                # A projected broadcast input remains outside the island and
                # is reused without duplicating its producer for each row.
                continue
            if (
                isinstance(source.owner, arith.ConstantOp)
                or prod(_shape(value)) == 1
                and isinstance(source.owner, tensor.SplatOp)
            ):
                continue
            reduction = ordered_contraction(source.owner)
            if (
                not reduction
                and not isinstance(source, BlockArgument)
                and not isinstance(source.owner, (L.MatmulOp, func.CallOp))
            ):
                return "producer_fusion_boundary_or_unknown"
    except ValueError:
        return "unknown_source_view"
    pending = [group.operations[-1].results[0]]
    visited = set()
    while pending:
        value = pending.pop()
        if value in visited:
            continue
        visited.add(value)
        if len(visited) > 64:
            return "consumer_view_budget"
        for use in value.uses:
            if isinstance(use.operation, (L.GenericOp, L.ReduceOp)) and not ordered_contraction(use.operation):
                return "consumer_fusion_boundary"
            if isinstance(use.operation, (tensor.CollapseShapeOp, tensor.ExpandShapeOp)):
                pending.extend(use.operation.results)
    return None


def fuse_row_normalizations(
    module,
    *,
    numerical_policy,
    max_added_operations=4096,
    max_groups=256,
    require_smaller_row_state=True,
    index_bits=64,
    hardware_identity=None,
):
    """Automatically row-schedule admitted closed source islands on a private IR.

    Standard pure tensor/scalar semantics retain their values, initializers and
    per-row reduction order. No approximation, finite-domain assumption, target
    capability or runtime persistence is inferred. Unmatched/observed/effectful
    islands retain their original route. The returned facts describe source IR;
    actual physical allocation and execution need independent observation.
    """
    if not isinstance(numerical_policy, str) or not numerical_policy:
        raise ValueError("row scheduling requires its selected original numerical policy")
    if any(type(n) is not int or n < 0 for n in (max_added_operations, max_groups)):
        raise ValueError("row scheduling requires explicit bounded code and group budgets")
    if type(require_smaller_row_state) is not bool:
        raise ValueError("row scheduling needs an explicit conservative storage selection")
    if type(index_bits) is not int or index_bits != 64:
        raise ValueError("source row scheduling currently requires selected 64-bit index lowering")
    from merlin.transforms import row_normalization_check as checker

    current = module.clone()
    sources = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in implementation_sources()}
    context = PipelineContext(
        hardware=hardware_identity,
        numerical_policy=numerical_policy,
        resources=(
            f"max_added_operations={max_added_operations};max_groups={max_groups};"
            f"smaller_row_state={require_smaller_row_state};index_bits={index_bits}"
        ),
        library=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
    )
    session = PipelineSession(current, context=context, ir_level="host_tensor", max_repairs=0)
    records = []
    counts = {"rewrites": 0, "refused_code_budget": 0, "refused_group_budget": 0, "retained_row_state_cost": 0}
    islands = []

    def rewrite():
        growth = 0
        for function in tuple(current.body.block.ops):
            if type(function) is not func.FuncOp or len(function.body.blocks) != 1:
                continue
            for operation in tuple(function.body.block.ops):
                if operation.parent is None:
                    continue
                group = match_row_normalization(operation)
                if group is None:
                    continue
                # Ordinary scalar lowering can already keep exponentials in
                # its public output. Row scheduling must then trade complete
                # row-scalar arrays for an exponential row plus scalar state.
                # Unknown timing retains the original when this logical state
                # comparison does not improve. Actual allocation is measured
                # separately; this does not predict bus traffic or cycles.
                if require_smaller_row_state and row_state_cost_reason(group) is not None:
                    # A row-loop boundary can prevent ordinary upstream
                    # pointwise fusion and create a new whole-tensor buffer.
                    # Unknown complete-graph costs retain that existing route.
                    counts["retained_row_state_cost"] += 1
                    continue
                if len(records) >= max_groups:
                    counts["refused_group_budget"] += 1
                    continue
                emitted, result = emit_row_normalization(group)
                added = sum(1 for op in emitted for _ in op.walk()) - sum(
                    1 for op in group.operations for _ in op.walk()
                )
                if growth + max(0, added) > max_added_operations:
                    for op in emitted:
                        op.drop_all_references()
                    counts["refused_code_budget"] += 1
                    continue
                mapped = {}
                snapshot = tuple(op.clone(value_mapper=mapped) for op in group.operations)
                record = RowNormalizationRewrite(
                    RowNormalizationGroup(mapped.get(group.source, group.source), snapshot, group.shape),
                    emitted,
                    result,
                )
                checker.verify_row_normalization_record(record)
                Rewriter.replace_op(group.operations[-1], emitted, new_results=[result])
                for op in reversed(group.operations[:-1]):
                    Rewriter.erase_op(op)
                records.append(record)
                growth += max(0, added)
                counts["rewrites"] += 1
                islands.append(
                    {
                        "shape": group.shape,
                        "rows": group.rows,
                        "columns": group.columns,
                        "source_operations": len(group.operations),
                        "added_operations": added,
                        "intermediate_full_row_elements_before": group.rows * group.columns,
                        "intermediate_full_row_elements_after": group.columns,
                        "logical_row_state_before": group.rows,
                        "logical_row_state_after": group.columns + 2,
                        "row_input_frontier": [str(value.type) for value in row_input_values(group)],
                        "pointwise_source_prefix": group.source.owner in set(group.operations),
                    }
                )
        return {"row_candidates": len(records)}

    try:
        session.run(
            PassSpec("row_source_admission", 0, produces=("row_source",), mutates_ir=False, ir_level="host_tensor"),
            lambda: {"row_source": ir_digest(module)},
        )
        session.run(
            PassSpec(
                "row_fiber_schedule",
                2,
                requires=("row_source",),
                produces=("row_candidates",),
                invalidates=("*",),
                ir_level="host_tensor",
            ),
            rewrite,
        )
        session.run(
            PassSpec(
                "row_fiber_conformance",
                4,
                requires=("row_candidates",),
                produces=("row_realization",),
                mutates_ir=False,
                ir_level="host_tensor",
            ),
            lambda: (checker.verify_row_normalizations(module, current, records), {"row_realization": len(records)})[1],
        )
    finally:
        # Detached source witnesses must not remain as hidden SSA users on the
        # returned IR. Later ordinary passes see only actual program users.
        for record in records:
            for op in record.source.operations:
                op.drop_all_references()
    return current, {
        "schema": "merlin.row_normalization.v1",
        "source_sha256": ir_digest(module),
        "output_sha256": ir_digest(current),
        "context": asdict(context),
        **counts,
        "islands": islands,
        "library_sources_sha256": sources,
        "checks": [asdict(row) for row in session.records],
        "physical_allocation": "not_established_by_source_transform",
        "machine_conformance": "not_established",
        "timing": None,
        "formal_coverage": "outside_existing_integer_transform_domains",
    }


# Preserve issued/public record pickle names across the owner migration.
RowNormalizationRewrite.__module__ = "merlin.transforms.row_normalization"
