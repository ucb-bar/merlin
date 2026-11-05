"""Independent full-module correspondence for bounded source recomputation.

This does not relax the layout pass's prohibition on extra arithmetic. The
original host must separately permit repeated evaluation and storage removal.
No target execution, allocation removal or performance qualification is issued.
"""

from __future__ import annotations

from collections import Counter
from math import prod

from xdsl.dialects import tensor
from xdsl.dialects.linalg.ops import GenericOp, YieldOp

from merlin.compile.scheduling.fusion_semantics import ScalarGraph, UnsupportedElementwise
from merlin.compile.scheduling.multioutput_fusion import read_source_multioutput
from merlin.compile.scheduling.pointwise_source import read_pointwise_source
from merlin.llvmlower.layout_conformance import _attribute, _metadata, layout_expression


def _blocks(module):
    """Stable structural paths, independent of SSA names and object identity."""
    result = {}

    def visit(op, path):
        for r, region in enumerate(op.regions):
            for b, block in enumerate(region.blocks):
                key = (*path, r, b)
                result[key] = block
                for i, child in enumerate(block.ops):
                    visit(child, (*key, i))

    visit(module, ())
    return result


def _site(module, operations):
    paths = _blocks(module)
    path = next((key for key, value in paths.items() if value is operations[0].parent), None)
    if path is None:
        raise ValueError("recomputation members are not in the current source")
    block = paths[path]
    index = tuple(block.ops).index(operations[0])
    empty = operations[0].outputs[0].owner
    if (
        type(empty) is not tensor.EmptyOp
        or empty.parent is not block
        or empty.attributes
        or empty.properties
        or empty.operands
        or len(empty.results) != 1
        or len(tuple(empty.results[0].uses)) != 1
        or next(iter(empty.results[0].uses)).operation is not operations[0]
    ):
        raise UnsupportedElementwise("recomputation requires a sole-use fresh empty producer destination")
    for operation in operations:
        body = operation.body.block
        local = set(body.args) | {value for scalar in body.ops for value in scalar.results}
        for scalar in body.ops:
            for value in scalar.operands:
                if value in local:
                    continue
                owner, anchor = value.owner, operations[0]
                while anchor is not None:
                    if owner.parent is anchor.parent and owner.is_before_in_block(anchor):
                        break
                    anchor = anchor.parent_op()
                if anchor is None:
                    raise UnsupportedElementwise("recomputation scalar captures must dominate the whole fan-out")
    if any(not operation.results[0].uses for operation in operations[1:]):
        raise UnsupportedElementwise("recomputation requires two live consumer results")
    return path, index, empty


def _inputs(producer, consumer):
    return tuple(dict.fromkeys((*producer.inputs, *(v for v in consumer.inputs if v is not producer.results[0]))))


def _composed_graph(source, member, inputs, producer, consumer):
    """Reconstruct exact ordered applications and shared constant captures.

    Scalar captures remain original dominating constants; they are never moved
    into an element loop or evaluated twice merely to simplify correspondence.
    """
    operations, captures = [], {}

    def application(op, values):
        for value in op.operands:
            if value not in values:
                if value not in captures:
                    constant = value.owner
                    captures[value] = application(constant, {})
                values[value] = captures[value]
        operations.append(
            (
                op.name,
                str(op.results[0].type),
                tuple(sorted((name, str(value)) for name, value in op.properties.items())),
                tuple(values[v] for v in op.operands),
            )
        )
        return len(inputs) + len(operations) - 1

    def append(op, arguments):
        body = op.body.block
        values = dict(zip(tuple(body.args)[:-1], arguments, strict=True))
        for scalar in tuple(body.ops)[:-1]:
            values[scalar.results[0]] = application(scalar, values)
        return values[body.last_op.operands[0]]

    returned = append(producer, [inputs.index(value) for value in producer.inputs])
    result = append(
        consumer, [returned if value is producer.results[0] else inputs.index(value) for value in consumer.inputs]
    )
    return ScalarGraph(
        tuple(value.type.element_type.width.data for value in inputs),
        source.epilogues[member].graph.output_bits,
        tuple(operations),
        result,
    )


def _skeleton(module, omitted, epilogues):
    """Check every other op, metadata, region and complete operand wiring."""
    values = {}
    labels = {operation: ("epilogue", i) for i, operation in enumerate(epilogues)}

    def operation(op, path):
        values.update({v: (labels.get(op, path), i, _attribute(v.type)) for i, v in enumerate(op.results)})
        if op in omitted or op in labels:
            return None
        operands = tuple(values.get(v, ("foreign", id(v))) for v in op.operands)
        regions = []
        for r, region in enumerate(op.regions):
            blocks = []
            for b, block in enumerate(region.blocks):
                key = (*path, "region", r, b)
                values.update({v: (key, i, _attribute(v.type)) for i, v in enumerate(block.args)})
                rows, ordinal = [], 0
                for child in block.ops:
                    if child in omitted:
                        continue
                    item = operation(child, (*key, ordinal))
                    if child not in labels:
                        ordinal += 1
                    if item is not None:
                        rows.append(item)
                blocks.append((tuple(_attribute(v.type) for v in block.args), tuple(rows)))
            regions.append(tuple(blocks))
        return (op.name, _metadata(op), tuple(_attribute(v.type) for v in op.results), operands, tuple(regions))

    return operation(module, ()), values


def _observable(value):
    # These frames describe scalar-body packaging, not an arithmetic operation.
    # Exact local bodies/order/work and unchanged remaining IR are checked below.
    if type(value) is tuple:
        if len(value) == 4 and value[0] == "pointwise":
            return _observable(value[3])
        return tuple(_observable(item) for item in value)
    return value


def _work(operations):
    result = Counter()
    for op in operations:
        for scalar in op.body.block.ops:
            if type(scalar) is YieldOp:
                continue
            metadata = tuple(
                tuple((name, value) for name, value in fields if not name.startswith("prov."))
                for fields in _metadata(scalar)
            )
            result[scalar.name, tuple(_attribute(v.type) for v in scalar.results), metadata] += prod(
                op.results[0].type.get_shape()
            )
    return result


def verify_source_recomputation(original, candidate, operations, permissions, budgets):
    """Reconstruct a sole fan-out rewrite without trusting emitter reports.

    Both modules must be independently parsed/verified current IR. Shape,
    ordered modular operations, captures, complete outputs and opaque effect
    epochs are reconstructed; hashes alone never establish repeated work.
    """
    from .recomputation import RecomputationBudgets, RecomputationPermissions

    if type(permissions) is not RecomputationPermissions or type(budgets) is not RecomputationBudgets:
        raise ValueError("recomputation checking requires original typed permissions and bounds")
    permissions.__post_init__()
    budgets.__post_init__()
    for module in (original, candidate):
        for ordinal, _ in enumerate(module.walk()):
            if ordinal >= budgets.max_constructed_operations:
                raise UnsupportedElementwise("recomputation checking exceeds the original construction ceiling")
    original.verify()
    candidate.verify()
    source = read_source_multioutput(
        original,
        operations,
        max_source_elements=budgets.max_source_elements,
        max_scalar_operations=budgets.max_producer_scalar_ops,
        max_scalar_ir_bytes=budgets.max_scalar_ir_bytes,
    )
    permissions.require()
    path, index, empty = _site(original, operations)
    blocks = _blocks(candidate)
    if path not in blocks:
        raise ValueError("recomputation changed the enclosing source region")
    position = index - (tuple(operations[0].parent.ops).index(empty) < index)
    members = tuple(blocks[path].ops)[position : position + 2]
    if len(members) != 2 or any(type(op) is not GenericOp for op in members):
        raise ValueError("recomputation did not retain both consumer operations")
    old, old_values = _skeleton(original, {operations[0], empty}, operations[1:])
    new, new_values = _skeleton(candidate, set(), members)
    if old != new:
        raise ValueError("recomputation changed unrelated IR, metadata, outputs or opaque ordering")
    for i, (before, after) in enumerate(zip(operations[1:], members, strict=True)):
        inputs = _inputs(operations[0], before)
        expected = _composed_graph(source, i, inputs, operations[0], before)
        actual = read_pointwise_source(
            after,
            max_elements=budgets.max_source_elements,
            max_operations=2 * budgets.max_producer_scalar_ops,
            max_ir_bytes=budgets.max_scalar_ir_bytes,
            allow_constant_captures=True,
        )
        if (
            actual.shape != source.shape
            or actual.graph != expected
            or tuple(new_values.get(v) for v in after.inputs) != tuple(old_values.get(v) for v in inputs)
            or tuple(new_values.get(v) for v in after.outputs) != tuple(old_values.get(v) for v in before.outputs)
            or after.attributes != before.attributes
        ):
            raise ValueError("recomputation changed exact consumer inputs, scalar order, widths or destination")
    constructed = len(tuple(original.walk())) + sum(len(tuple(op.walk())) for op in members)
    if constructed > budgets.max_constructed_operations:
        raise UnsupportedElementwise("recomputation candidate exceeds the original construction ceiling")
    before_expression, before_work = layout_expression(original, include_work=True)
    after_expression, after_work = layout_expression(candidate, include_work=True)
    if _observable(before_expression) != _observable(after_expression):
        raise ValueError("recomputation changed complete observable expressions or effect epochs")
    before_local, after_local = _work(operations), _work(members)

    def flatten(work):
        result = Counter()
        for key, count in work.items():
            result[key[1:]] += count
        return result

    if dict(flatten(after_work) - flatten(before_work)) != dict(after_local - before_local) or dict(
        flatten(before_work) - flatten(after_work)
    ) != dict(before_local - after_local):
        raise ValueError("recomputation work changed outside its independently checked region")
    added = sum(after_work.values()) - sum(before_work.values())
    if added < 0 or added > budgets.max_added_scalar_evaluations:
        raise UnsupportedElementwise("recomputation exceeds the original added-evaluation ceiling")
    elements = prod(source.shape)
    if budgets.max_repeats < 2:
        raise UnsupportedElementwise("recomputation exceeds the original producer-repeat ceiling")
    return {
        "schema": "merlin.source_recomputation_check.v1",
        "scope": "conditional source correspondence; no allocation, execution or cost qualification",
        "shape": source.shape,
        "before_scalar_evaluations": sum(before_work.values()),
        "after_scalar_evaluations": sum(after_work.values()),
        "added_scalar_evaluations": added,
        "before_producer_evaluations": elements,
        "after_producer_evaluations": 2 * elements,
        "logical_producer_bytes_removed": elements * source.producer.graph.output_bits // 8,
        "producer_scalar_operations": len(source.producer.graph.operations),
        "semantics": "exact ordered fixed-width operations, full outputs and opaque epochs",
    }
