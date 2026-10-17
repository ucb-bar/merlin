"""Schedule closed source normalization islands one complete row at a time.

The original maximum, subtraction, exponential, sum and division regions are
retained, including their scalar operation order, attributes and initializers.
Only independent row fibers are interchanged. Source masks and nonfinite values
remain ordinary source values; no finite-domain approximation is introduced.
Unknown effects, layouts, observers, indexing and source algorithms retain the
original route. Ordinary serial host lowering consumes this transformation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from math import prod
from pathlib import Path

from xdsl.dialects import arith, builtin, func, math, scf, tensor
from xdsl.dialects.linalg import ops as L
from xdsl.ir import Block, BlockArgument, Operation, Region, SSAValue
from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr, AffineDimExpr, AffineMap
from xdsl.rewriter import Rewriter
from xdsl.traits import get_effects

from .certificates import ir_digest
from .pipeline import PassSpec, PipelineContext, PipelineSession


@dataclass(frozen=True)
class RowNormalizationGroup:
    source: SSAValue
    operations: tuple[Operation, ...]
    shape: tuple[int, ...]

    @property
    def rows(self):
        return prod(self.shape[:-1])

    @property
    def columns(self):
        return self.shape[-1]


@dataclass(frozen=True)
class RowNormalizationRewrite:
    source: RowNormalizationGroup
    emitted: tuple[Operation, ...]
    result: SSAValue


def _shape(value):
    typ = value.type
    if not isinstance(typ, builtin.TensorType) or not isinstance(typ.encoding, builtin.NoneAttr):
        raise ValueError("row normalization requires unencoded tensor semantics")
    shape = typ.get_shape()
    if any(n <= 0 for n in shape) or typ.get_element_type() not in (builtin.f32, builtin.f64):
        raise ValueError("row normalization requires positive static floating tensor shapes")
    return shape


def _metadata(operation):
    if any(not key.startswith("prov.") for key in operation.attributes):
        raise ValueError("unknown normalization operation attributes are conservative barriers")


def _view(value):
    seen = 0
    while isinstance(value.owner, (tensor.CollapseShapeOp, tensor.ExpandShapeOp)):
        op = value.owner
        _metadata(op)
        if len(op.operands) != 1 or prod(_shape(op.operands[0])) != prod(_shape(value)):
            raise ValueError("row normalization cannot interpret dynamic/noncontiguous reshape effects")
        value = op.operands[0]
        seen += 1
        if seen > 64:
            raise ValueError("row normalization reshape chain exceeds its source budget")
    return value


def _body(operation, scalar_class, inputs):
    if len(operation.regions) != 1 or len(operation.regions[0].blocks) != 1:
        return False
    block = operation.regions[0].block
    ops = tuple(block.ops)
    if len(ops) != 2 or type(ops[0]) is not scalar_class or type(ops[1]) is not L.YieldOp:
        return False
    scalar = ops[0]
    if get_effects(scalar) != set() or scalar.regions or scalar.successors:
        return False
    if tuple(scalar.operands) != tuple(block.args[:inputs]) or tuple(ops[1].operands) != tuple(scalar.results):
        return False
    return len(block.args) == inputs + (1 if isinstance(operation, L.GenericOp) else 0)


def _generic(operation, scalar_class, inputs):
    return isinstance(operation, L.GenericOp) and len(operation.results) == 1 and _body(operation, scalar_class, inputs)


def _reduce(operation, scalar_class):
    return isinstance(operation, L.ReduceOp) and len(operation.operands) == 2 and _body(operation, scalar_class, 2)


def _linear(expression, rank):
    if isinstance(expression, AffineConstantExpr):
        return (expression.value, *([0] * rank))
    if isinstance(expression, AffineDimExpr):
        return (0, *(int(i == expression.position) for i in range(rank)))
    if isinstance(expression, AffineBinaryOpExpr):
        left, right = _linear(expression.lhs, rank), _linear(expression.rhs, rank)
        if expression.kind == AffineBinaryOpKind.Add:
            return tuple(a + b for a, b in zip(left, right))
        if expression.kind == AffineBinaryOpKind.Mul:
            if not any(left[1:]):
                return tuple(left[0] * x for x in right)
            if not any(right[1:]):
                return tuple(right[0] * x for x in left)
    raise ValueError("row normalization requires checked affine row-fiber indexing")


def _fiber_map(mapping, input_shape, loop_shape, kind):
    rank = len(loop_shape)
    if mapping.num_dims != rank or mapping.num_symbols or len(mapping.results) != len(input_shape):
        raise ValueError("row normalization has incompatible indexing ranks")
    actual = [0] * (rank + 1)
    for axis, expression in enumerate(mapping.results):
        coefficients = _linear(expression, rank)
        low = coefficients[0] + sum(min(0, c * (n - 1)) for c, n in zip(coefficients[1:], loop_shape, strict=True))
        high = coefficients[0] + sum(max(0, c * (n - 1)) for c, n in zip(coefficients[1:], loop_shape, strict=True))
        if low < 0 or high >= input_shape[axis]:
            raise ValueError("normalization indexing leaves a source tensor dimension")
        stride = prod(input_shape[axis + 1 :])
        for i, coefficient in enumerate(coefficients):
            actual[i] += coefficient * stride
    expected = [0] + [prod(loop_shape[i + 1 :]) for i in range(rank)]
    if kind == "scalar":
        expected = [0] * (rank + 1)
    elif kind == "row":
        expected = [0] + [prod(loop_shape[i + 1 : -1]) for i in range(rank - 1)] + [0]
    for i, extent in enumerate(loop_shape, 1):
        if extent == 1:
            actual[i] = expected[i] = 0
    if actual != expected:
        raise ValueError("normalization indexing crosses or permutes source row fibers")


def local_type(value, group):
    shape = _shape(value)
    count = prod(shape)
    if count == group.rows * group.columns:
        shape = (group.columns,)
    elif count in (1, group.rows):
        shape = ()
    else:
        raise ValueError("normalization tensor cannot be restricted to one complete row")
    return builtin.TensorType(value.type.get_element_type(), shape)


def _check_operation(operation, group):
    _metadata(operation)
    if len(operation.results) != 1:
        raise ValueError("normalization island requires single-result source operations")
    if isinstance(operation, arith.ConstantOp) and not isinstance(operation.results[0].type, builtin.TensorType):
        return
    local_type(operation.results[0], group)
    if isinstance(operation, L.GenericOp):
        shape = _shape(operation.results[0])
        if shape[-1:] != (group.columns,) or prod(shape[:-1]) != group.rows:
            raise ValueError("normalization pointwise iteration is not over complete source rows")
        if operation.properties.keys() - {"indexing_maps", "iterator_types", "operandSegmentSizes"}:
            raise ValueError("unknown normalization generic lowering properties")
        if len(operation.iterator_types) != len(shape) or any(
            "parallel" not in str(x) for x in operation.iterator_types
        ):
            raise ValueError("normalization pointwise body has an unknown iteration algorithm")
        for value, attr in zip(operation.operands, operation.indexing_maps, strict=True):
            count = prod(_shape(value))
            kind = "full" if count == group.rows * group.columns else "row" if count == group.rows else "scalar"
            _fiber_map(attr.data, _shape(value), shape, kind)
    elif isinstance(operation, L.ReduceOp):
        shape = _shape(operation.operands[0])
        if shape[-1:] != (group.columns,) or prod(shape[:-1]) != group.rows:
            raise ValueError("normalization reduction does not cover a complete row")
        if tuple(operation.dimensions.get_values()) != (len(shape) - 1,) or operation.properties.keys() != {
            "dimensions"
        }:
            raise ValueError("normalization reduction has a different domain or unknown properties")
        if prod(_shape(operation.operands[1])) != group.rows or prod(_shape(operation.results[0])) != group.rows:
            raise ValueError("normalization reduction initializer/result has incorrect row coverage")
    elif isinstance(operation, (tensor.CollapseShapeOp, tensor.ExpandShapeOp)):
        _view(operation.results[0])
    elif isinstance(operation, tensor.EmptyOp):
        if operation.operands:
            raise ValueError("dynamic empty tensors are normalization barriers")
    elif isinstance(operation, tensor.SplatOp):
        if len(operation.operands) != 1 or not isinstance(operation.operands[0].owner, arith.ConstantOp):
            raise ValueError("normalization initialization requires its original constant scalar splat")
    else:
        raise ValueError("unknown normalization tensor semantics are conservative barriers")


def match_row_normalization(final):
    """Recognize from current typed source dataflow; provenance never selects it."""
    if not _generic(final, arith.DivfOp, 2):
        return None
    try:
        exponential = _view(final.operands[0]).owner
        summation = _view(final.operands[1]).owner
        if not _generic(exponential, math.ExpOp, 1) or not _reduce(summation, arith.AddfOp):
            return None
        if summation.operands[0] is not exponential.results[0]:
            return None
        shift = _view(exponential.operands[0]).owner
        if not _generic(shift, arith.SubfOp, 2):
            return None
        maximum = _view(shift.operands[1]).owner
        if not _reduce(maximum, arith.MaximumfOp) or maximum.operands[0] is not shift.operands[0]:
            return None
        source = maximum.operands[0]
        shape = _shape(source)
        if final.results[0].type != source.type:
            return None
        if len(shape) < 2 or prod(shape[:-1]) <= 1 or shape[-1] <= 1:
            return None
        if prod(shape) * source.type.get_element_type().bitwidth // 8 >= 1 << 63:
            return None
        selected = set()

        def visit(value):
            if value is source:
                return
            if not isinstance(value.type, builtin.TensorType):
                if not isinstance(value.owner, arith.ConstantOp):
                    raise ValueError("unknown normalization scalar initialization")
                return
            op = value.owner
            if not isinstance(op, Operation) or op.parent is not final.parent:
                raise ValueError("normalization tensor has an external observation or producer")
            if op in selected:
                return
            if len(selected) >= 64:
                raise ValueError("normalization source island exceeds its structural budget")
            selected.add(op)
            for operand in op.operands:
                visit(operand)

        visit(final.results[0])

        def include_prefix(value):
            # Keep an existing same-shape pointwise producer inside the row
            # island rather than introducing a materialization boundary.
            op = value.owner
            if not isinstance(op, Operation) or op.parent is not final.parent or value.type != source.type:
                return
            if op in selected:
                return
            if not any(_generic(op, cls, 2) for cls in (arith.AddfOp, arith.SubfOp, arith.MulfOp, arith.DivfOp)):
                return
            if len(op.operands) != 3 or not isinstance(op.operands[-1].owner, tensor.EmptyOp):
                return
            if any(v.type != source.type and prod(_shape(v)) != 1 for v in op.operands[:-1]):
                return
            if len(selected) + 2 > 64:
                raise ValueError("normalization pointwise prefix exceeds its structural budget")
            selected.update((op, op.operands[-1].owner))
            for operand in op.operands[:-1]:
                include_prefix(operand)

        include_prefix(source)
        ordered = tuple(op for op in final.parent.ops if op in selected)
        if ordered[-1] is not final:
            return None
        active = False
        for op in final.parent.ops:
            active |= op is ordered[0]
            if active and isinstance(op, arith.ConstantOp) and not isinstance(op.results[0].type, builtin.TensorType):
                selected.add(op)
            if op is final:
                break
        ordered = tuple(op for op in final.parent.ops if op in selected)
        group = RowNormalizationGroup(source, ordered, shape)
        for op in ordered:
            _check_operation(op, group)
            if op is not final and any(use.operation not in selected for value in op.results for use in value.uses):
                return None
        first = False
        for op in final.parent.ops:
            first |= op is ordered[0]
            if op is final:
                break
            if first and op not in selected:
                return None
        return group
    except (ValueError, IndexError):
        return None


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
        static_offsets.append(builtin.DYNAMIC_INDEX)
    static_offsets.append(0)
    sizes = [1] * (len(group.shape) - 1) + [group.columns]
    mapped = {}
    for value in row_input_values(group):
        scalar = prod(_shape(value)) == 1
        shape = _shape(value)
        if not shape:
            mapped[value] = value
            continue
        sliced = tensor.ExtractSliceOp.build(
            operands=[value, [] if scalar else coordinates, [], []],
            result_types=[local_type(value, group)],
            properties={
                "static_offsets": builtin.DenseArrayBase.from_list(
                    builtin.i64, [0] * len(shape) if scalar else static_offsets
                ),
                "static_sizes": builtin.DenseArrayBase.from_list(builtin.i64, [1] * len(shape) if scalar else sizes),
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


def row_input_values(group):
    members = set(group.operations)
    values = {}
    for op in group.operations:
        for value in op.operands:
            if isinstance(value.type, builtin.TensorType) and value.owner not in members:
                if value.type != group.source.type and prod(_shape(value)) != 1:
                    raise ValueError("row input frontier has an unsupported shape or broadcast")
                values[value] = None
    return tuple(values)


def row_state_cost_reason(group):
    """Conservative unknown-cost policy, including existing fusion boundaries."""
    if group.rows <= group.columns + 2:
        return "row_state_not_smaller"
    try:
        for value in row_input_values(group):
            source = _view(value)
            if prod(_shape(value)) == 1 and isinstance(source.owner, (arith.ConstantOp, tensor.SplatOp)):
                continue
            if not isinstance(source, BlockArgument) and not isinstance(source.owner, (L.MatmulOp, func.CallOp)):
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
            if isinstance(use.operation, (L.GenericOp, L.ReduceOp)):
                return "consumer_fusion_boundary"
            if isinstance(use.operation, (tensor.CollapseShapeOp, tensor.ExpandShapeOp)):
                pending.extend(use.operation.results)
    return None


def fuse_row_normalizations(
    module, *, numerical_policy, max_added_operations=4096, max_groups=256, require_smaller_row_state=True
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
    from . import pipeline
    from . import row_normalization_check as checker

    current = module.clone()
    sources = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), Path(checker.__file__), Path(pipeline.__file__))
    }
    context = PipelineContext(
        numerical_policy=numerical_policy,
        resources=(
            f"max_added_operations={max_added_operations};max_groups={max_groups};"
            f"smaller_row_state={require_smaller_row_state}"
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
