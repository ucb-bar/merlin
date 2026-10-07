"""Retain an explicit typed observation boundary without numeric permission.

This is source analysis only. Integer contractions keep their scalar bodies,
initializers, maps and reduction order. A later provider must independently prove
every observation, overflow behavior, rounding point, effect and buffer lifetime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ordered_fma_groups import _context_snapshot, _snapshot
from .quantized_consumer_frontier import _allowed, _complete_assembly, _strict


@dataclass(frozen=True)
class ObservationBoundary:
    """A live source DAG; external uses remain mandatory observations."""

    inputs: tuple
    source_values: tuple
    requested_outputs: tuple
    operations: tuple
    external_escapes: tuple
    _block: object = field(repr=False)
    _order: tuple = field(repr=False)
    _inputs: tuple = field(repr=False)
    _witnesses: tuple = field(repr=False)
    _contexts: tuple = field(repr=False)
    _selection: tuple = field(repr=False)

    @property
    def observation_outputs(self) -> tuple:
        return (*self.requested_outputs, *self.external_escapes)

    @property
    def requested_boundary_closed(self) -> bool:
        return not self.external_escapes


def _input_snapshot(value):
    return value, value.owner, value.type, frozenset((u.operation, u.index) for u in value.uses)


def _static_tensor(value) -> bool:
    from xdsl.dialects.builtin import TensorType

    return isinstance(value.type, TensorType) and all(n > 0 for n in value.type.get_shape())


def _retained_operation(operation) -> bool:
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.linalg.ops import GenericOp, IteratorType
    from xdsl.traits import Pure

    if isinstance(operation, GenericOp):
        # This admits retained source reductions, not a rescheduling theorem.
        if (
            _strict(operation)
            or len(operation.body.blocks) != 1
            or not all(_static_tensor(v) for v in (*operation.operands, *operation.results))
            or any(i.data not in (IteratorType.PARALLEL, IteratorType.REDUCTION)
                   for i in operation.iterator_types)
        ):
            return False
        # These exact integer extensions lack Pure on supported xDSL versions.
        # Recognize their typed semantics explicitly, never external calls.
        for scalar in operation.body.block.ops:
            if scalar.name == "linalg.yield":
                continue
            if scalar.regions or _strict(scalar) or not (
                scalar.has_trait(Pure) or isinstance(scalar, (arith.ExtSIOp, arith.ExtUIOp))
            ):
                return False
            if any(name == "fastmath" and str(value) != "#arith.fastmath<none>"
                   for name, value in (*scalar.attributes.items(), *scalar.properties.items())):
                return False
        for index, value in enumerate(operation.outputs, start=len(operation.inputs)):
            if isinstance(value.owner, tensor.EmptyOp) and tuple(operation.body.block.args[index].uses):
                return False
        operation.verify()
        return True
    return _allowed(operation)


def analyze_observation_boundary(*, inputs, outputs, source_values=None) -> ObservationBoundary:
    """Close explicit tensor observations to explicit tensor inputs.

    Inputs may include immutable tensor function arguments or prior tensor
    results in the same block. Undeclared arguments, calls, strict FP and unknown
    effects refuse. Other uses of inputs or intermediate results remain explicit
    observations. ``source_values`` identifies the input producers whose values
    a future replacement would change; other inputs remain unchanged parameters.
    It defaults to every input. No physical immutability or no-alias permission
    is inferred. No IR is cloned, erased, reordered or numerically relaxed.
    """
    from xdsl.dialects import tensor
    from xdsl.ir import BlockArgument, OpResult

    inputs, outputs = tuple(inputs), tuple(outputs)
    sources = inputs if source_values is None else tuple(source_values)
    if not inputs or not outputs or len(set(inputs)) != len(inputs) or len(set(outputs)) != len(outputs):
        raise ValueError("distinct nonempty boundary inputs and outputs required")
    if not sources or len(set(sources)) != len(sources) or not set(sources) <= set(inputs):
        raise ValueError("distinct nonempty source values must be boundary inputs")
    if any(not isinstance(v, OpResult) or not _static_tensor(v) for v in outputs):
        raise ValueError("static tensor result observations required")
    block = outputs[0].owner.parent
    if block is None or block.parent_op() is None or block.parent_op().name != "func.func":
        raise ValueError("single function block required")
    if any(v.owner.parent is not block for v in outputs):
        raise ValueError("observations must share a function block")
    positions = {op: i for i, op in enumerate(block.ops)}
    for value in inputs:
        if not _static_tensor(value) or not (
            isinstance(value, OpResult) and value.owner.parent is block
            or isinstance(value, BlockArgument) and value.owner is block
        ):
            raise ValueError("static tensor inputs in the same block required")
    needed, reached = set(), set()

    def need(value):
        if value in inputs:
            reached.add(value)
            return
        if not isinstance(value, OpResult) or value.owner.parent is not block:
            raise ValueError("unbound or cross-block dependency")
        operation = value.owner
        if operation in needed:
            return
        if not _retained_operation(operation):
            raise ValueError("unsupported source operation or effect")
        needed.add(operation)
        for operand in operation.operands:
            if isinstance(operand, OpResult) and operand.owner.parent is block:
                if positions[operand.owner] >= positions[operation]:
                    raise ValueError("source dependency does not dominate its use")
            need(operand)

    for value in outputs:
        need(value)
    if reached != set(inputs):
        raise ValueError("every declared input must reach an observation")
    operations = tuple(sorted(needed, key=positions.__getitem__))
    for operation in operations:
        if isinstance(operation, tensor.InsertSliceOp) and any(
            not isinstance(u.operation, tensor.InsertSliceOp) or u.index != 1
            for u in operation.results[0].uses
        ):
            _complete_assembly(operation, inputs)
    descendants = {child for op in operations for child in op.walk()}
    dependent = set(sources)
    for operation in operations:
        if any(v in dependent for v in operation.operands):
            dependent.update(operation.results)
    escapes = tuple(
        v for v in (*inputs, *(v for op in operations for v in op.results))
        if v in dependent and v not in outputs and any(u.operation not in descendants for u in v.uses)
    )
    contexts = []
    current = block.parent_op()
    while current is not None:
        if _strict(current):
            raise ValueError("strict FP context requires separate effect proof")
        contexts.append(_context_snapshot(current))
        current = current.parent_op()
    owners = tuple(dict.fromkeys(v.owner for v in inputs if isinstance(v, OpResult)))
    return ObservationBoundary(
        inputs, sources, outputs, operations, escapes, block, tuple(block.ops),
        tuple(_input_snapshot(v) for v in inputs),
        tuple(_snapshot(child) for op in (*owners, *operations) for child in op.walk()),
        tuple(contexts),
        (inputs, sources, outputs),
    )


def validate_observation_boundary(boundary: ObservationBoundary) -> None:
    """Refuse changed bodies, uses, input types, block order or FP context."""
    if (
        (boundary.inputs, boundary.source_values, boundary.requested_outputs) != boundary._selection
        or tuple(boundary._block.ops) != boundary._order
        or tuple(_input_snapshot(v) for v in boundary.inputs) != boundary._inputs
        or any(_snapshot(w.operation) != w for w in boundary._witnesses)
        or any(_context_snapshot(c.operation) != c for c in boundary._contexts)
    ):
        raise ValueError("observation boundary source changed after analysis")
    current = analyze_observation_boundary(
        inputs=boundary.inputs, outputs=boundary.requested_outputs,
        source_values=boundary.source_values,
    )
    if current != boundary:
        raise ValueError("observation boundary contents changed after analysis")
