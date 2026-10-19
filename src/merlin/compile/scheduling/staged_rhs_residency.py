"""Place RHS panels in an already assigned private full-reduction stripe.

This is a finite proposal, not an allocation or ownership proof. The original
target must independently assign the expanded operand rows for this complete
source epoch, render the changed addresses, and check actual emitted effects.
No hardware capacity, target lookup, synchronization or timing is supplied here.
"""

from __future__ import annotations

from dataclasses import replace

from merlin.compile.scheduling.staged_dense import (
    AccumulatorReadout,
    AccumulatorZero,
    CompletionFence,
    ComputeConsume,
    DescriptorIssue,
    OperandLoad,
    StagedDenseContract,
    StagedDenseProblem,
)


def place_rhs_stripe(operations, problem, contract):
    """Preserve all effects while assigning each RHS K panel distinct rows.

    Compose through ``rewrite_staged_dense`` so the unchanged independent
    source/product/state checker verifies the original and proposed programs.
    Complete output-tile ordering and live-load elimination are separate passes;
    their order determines whether the new placement saves any transfers.
    """
    if (
        type(operations) is not tuple
        or type(problem) is not StagedDenseProblem
        or type(contract) is not StagedDenseContract
    ):
        raise ValueError("RHS placement requires original immutable staged source and assigned contract")
    dim = contract.dim
    required_rows = dim + ((problem.k + dim - 1) // dim) * dim
    if (
        not contract.external_buffers_disjoint
        or not contract.rhs_immutable
        or problem.k <= dim
        or required_rows > contract.operand_rows
    ):
        return operations
    # Refuse other placements and stationary-register lifetimes. They need
    # their own explicit assignment and independently checked transformation.
    for operation in operations:
        if type(operation) is OperandLoad:
            if operation.role == "lhs":
                if operation.local_row != 0:
                    return operations
            elif operation.role == "rhs":
                if operation.local_row != dim or operation.source_row % dim or operation.source_col % dim:
                    return operations
            else:
                raise ValueError("RHS placement encountered unknown operand ownership")
        elif type(operation) is DescriptorIssue:
            if operation.rhs_row != dim:
                return operations
        elif type(operation) is ComputeConsume:
            if operation.lhs_row != 0 or operation.retain_weights:
                return operations
        elif type(operation) not in (AccumulatorZero, AccumulatorReadout, CompletionFence):
            raise ValueError("RHS placement encountered an unknown staged effect")
    result, cells = [], {}
    pending = None
    for operation in operations:
        index = len(result)
        result.append(operation)
        if type(operation) is OperandLoad and operation.role == "rhs":
            for row in range(operation.rows):
                for col in range(operation.cols):
                    cells[operation.local_row + row, col] = (operation.source_row + row, operation.source_col + col)
            result[index] = replace(operation, local_row=dim + operation.source_row)
        elif type(operation) is DescriptorIssue:
            if pending is not None:
                raise ValueError("RHS placement cannot overwrite a pending source descriptor")
            pending = index, operation
        elif type(operation) is ComputeConsume:
            if pending is None:
                raise ValueError("RHS placement requires the original matching descriptor")
            position, descriptor = pending
            origin = cells.get((descriptor.rhs_row, 0))
            if origin is None:
                raise ValueError("RHS placement cannot derive consumed source panel")
            start_k, start_n = origin
            if (
                start_k % dim
                or start_n % dim
                or any(
                    cells.get((descriptor.rhs_row + row, col)) != (start_k + row, start_n + col)
                    for row in range(descriptor.k)
                    for col in range(descriptor.n)
                )
            ):
                raise ValueError("RHS placement requires the complete original consumed panel")
            # Descriptor visibility is at consume, not at issue or a fence.
            result[position] = replace(descriptor, rhs_row=dim + start_k)
            pending = None
    if pending is not None:
        raise ValueError("RHS placement leaves an original descriptor unconsumed")
    return tuple(result)
