"""Materialize original internal tensor values directly in caller-owned buffers.

Only private return descriptors are added. No arithmetic, copies, public result
or input roles change, and the ordinary native memory admission remains required.
"""

from __future__ import annotations

from merlin.compiler.backend.cabi import TensorPort
from merlin.compiler.backend.workspace import TensorWorkspace


def plan(operations, outputs):
    """Derive materialization barriers from actual structured iterator semantics."""
    from xdsl.dialects import builtin
    from xdsl.dialects.linalg.ops import LinalgStructuredOperation

    operations, outputs = tuple(operations), set(outputs)
    values, sources = [], []
    for ordinal, operation in enumerate(operations):
        if not isinstance(operation, LinalgStructuredOperation):
            continue
        for index, value in enumerate(operation.results):
            if value in outputs or type(value.type) is not builtin.TensorType:
                continue
            uses = tuple(use for use in value.uses if use.operation in operations)
            consumers = tuple(
                use.operation
                for use in uses
                if not isinstance(use.operation, LinalgStructuredOperation) or value in use.operation.inputs
            )
            barrier = any(
                isinstance(user, LinalgStructuredOperation)
                and any(iterator.data.value == "reduction" for iterator in user.get_iterator_types())
                for user in consumers
            )
            if barrier or len(set(consumers)) > 1:
                values.append(value)
                sources.append((ordinal, index))
    if any(
        operation.name
        in {
            "tensor.collapse_shape",
            "tensor.expand_shape",
            "tensor.extract_slice",
            "tensor.insert_slice",
            "tensor.cast",
        }
        for operation in operations
    ):
        raise ValueError("source workspace view/alias admission is unavailable")
    if not values:
        return None, ()
    ports = tuple(TensorPort.from_type(value.type) for value in values)
    return TensorWorkspace.pack(sources, ports), tuple(values)


def promote(module, *, limit):
    """Append private native outputs and replay the unchanged public SSA/body."""
    if type(limit) is not int or limit < 0:
        raise ValueError("scalar.workspace-bytes must be a nonnegative caller ceiling")
    if not limit:
        return module, None
    function = module.body.block.first_op
    original = module.clone()
    returned = tuple(function.get_return_op().arguments)
    workspace, values = plan(tuple(function.body.block.ops)[:-1], returned)
    if workspace is None:
        return module, None
    if workspace.bytes > limit:
        raise ValueError("source tensor workspace exceeds the original caller ceiling")
    transformed = module.clone()
    target = transformed.body.block.first_op
    mapping = dict(zip(function.body.block.ops, target.body.block.ops, strict=True))
    private = tuple(mapping[value.owner].results[value.index] for value in values)
    target.get_return_op().operands = (*target.get_return_op().operands, *private)
    target.update_function_type()
    transformed.verify()
    replay = transformed.clone()
    root = replay.body.block.first_op
    root.get_return_op().operands = root.get_return_op().operands[: len(returned)]
    root.update_function_type()
    if not original.is_structurally_equivalent(replay):
        raise ValueError("private workspace promotion changed public outputs or source computations")
    return transformed, workspace
