"""Independent finite staged data/descriptor/accumulator state induction.

The target supplies checked effect projection from current actual emitted
operands. No plan claim, pass implementation or configuration label proves
these effects. This checker provides no asynchronous overlap or timing model.
"""

from __future__ import annotations

from .staged_dense import (
    AccumulatorReadout,
    AccumulatorZero,
    CompletionFence,
    ComputeConsume,
    DescriptorIssue,
    OperandLoad,
    StagedDenseContract,
    StagedDenseProblem,
)


def verify_staged_dense(operations, *, problem, contract, max_commands=1_000_000, max_products=4 << 20):
    if type(problem) is not StagedDenseProblem or type(contract) is not StagedDenseContract:
        raise ValueError("staged verification requires exact source and selected effect contracts")
    if not contract.external_buffers_disjoint:
        raise ValueError("staged verification lacks original external source/output ownership proof")
    if type(operations) is not tuple or any(
        type(value) is not int or value <= 0 for value in (max_commands, max_products)
    ):
        raise ValueError("staged verification requires bounded immutable actual effects")
    if len(operations) > max_commands or problem.m * problem.k * problem.n > max_products:
        raise ValueError("staged source exceeds its independent command/product checking budget")
    operands, accumulators, outputs = {}, {}, set()
    pending, stationary = None, None
    products = computes = loads = retained = fences_pending = 0
    cells_visited = 0

    def rectangle(row, rows, cols, capacity):
        if (
            any(type(value) is not int for value in (row, rows, cols))
            or row < 0
            or not 0 < rows <= contract.dim
            or not 0 < cols <= contract.dim
            or row + rows > capacity
        ):
            raise ValueError("actual staged operand rectangle exceeds selected primitive or assigned storage")

    for operation in operations:
        if type(operation) in (OperandLoad, AccumulatorZero, AccumulatorReadout):
            if type(operation.rows) is int and type(operation.cols) is int:
                cells_visited += operation.rows * operation.cols
            if cells_visited > 4 * max_products:
                raise ValueError("staged transfer/readout effects exceed independent checking budget")
        if type(operation) is CompletionFence:
            fences_pending += pending is not None
        elif type(operation) is OperandLoad:
            if operation.role not in {"lhs", "rhs"}:
                raise ValueError("staged load has unknown source arithmetic ownership")
            rectangle(operation.local_row, operation.rows, operation.cols, contract.operand_rows)
            rows, cols = (problem.m, problem.k) if operation.role == "lhs" else (problem.k, problem.n)
            if (
                any(type(value) is not int or value < 0 for value in (operation.source_row, operation.source_col))
                or operation.source_row + operation.rows > rows
                or operation.source_col + operation.cols > cols
            ):
                raise ValueError("actual staged DMA exceeds its current source tensor")
            for r in range(operation.rows):
                for c in range(operation.cols):
                    operands[operation.local_row + r, c] = (
                        operation.role,
                        (operation.source_row + r) * cols + operation.source_col + c,
                    )
            loads += 1
        elif type(operation) is AccumulatorZero:
            rectangle(operation.local_row, operation.rows, operation.cols, contract.accumulator_rows)
            for r in range(operation.rows):
                for c in range(operation.cols):
                    accumulators[operation.local_row + r, c] = None
        elif type(operation) is DescriptorIssue:
            if pending is not None or type(operation.continuation) is not bool:
                raise ValueError("actual staged descriptor overwrites pending ownership or has unknown continuation")
            rectangle(operation.accumulator_row, operation.m, operation.n, contract.accumulator_rows)
            if type(operation.k) is not int or not 0 < operation.k <= contract.dim:
                raise ValueError("staged descriptor reduction exceeds selected primitive")
            if operation.rhs_row is not None:
                rectangle(operation.rhs_row, operation.k, operation.n, contract.operand_rows)
            pending = operation
        elif type(operation) is ComputeConsume:
            if (
                pending is None
                or (operation.m, operation.k) != (pending.m, pending.k)
                or type(operation.retain_weights) is not bool
            ):
                raise ValueError("actual staged compute has no matching pending descriptor")
            rectangle(operation.lhs_row, operation.m, operation.k, contract.operand_rows)
            if operation.retain_weights:
                if not contract.retained_weights or stationary is None or stationary[:2] != (pending.k, pending.n):
                    raise ValueError("actual retained weights lack matching independently established ownership")
                weights = stationary[2]
                retained += 1
            else:
                if pending.rhs_row is None:
                    raise ValueError("actual staged preload has no RHS source cells")
                weights = tuple(
                    tuple(operands.get((pending.rhs_row + k, n)) for n in range(pending.n)) for k in range(pending.k)
                )
                stationary = pending.k, pending.n, weights
            for r in range(operation.m):
                for n in range(pending.n):
                    a = [operands.get((operation.lhs_row + r, k)) for k in range(operation.k)]
                    b = [weights[k][n] for k in range(operation.k)]
                    if None in a or None in b or a[0][0] != "lhs" or b[0][0] != "rhs":
                        raise ValueError("actual staged compute reads unowned/clobbered source cells")
                    m, k0 = divmod(a[0][1], problem.k)
                    kb, out_n = divmod(b[0][1], problem.n)
                    if (
                        kb != k0
                        or k0 + operation.k > problem.k
                        or a != [("lhs", m * problem.k + k0 + k) for k in range(operation.k)]
                        or b != [("rhs", (k0 + k) * problem.n + out_n) for k in range(operation.k)]
                    ):
                        raise ValueError("actual staged product cells differ from current source geometry")
                    location = pending.accumulator_row + r, n
                    if (
                        location not in accumulators
                        or accumulators[location] != (None if k0 == 0 else (m, out_n, k0))
                        or (k0 > 0 and not pending.continuation)
                    ):
                        raise ValueError(
                            "actual staged compute loses zero initialization or exact accumulator continuation"
                        )
                    accumulators[location] = m, out_n, k0 + operation.k
                    products += operation.k
            pending = None
            computes += 1
        elif type(operation) is AccumulatorReadout:
            rectangle(operation.accumulator_row, operation.rows, operation.cols, contract.accumulator_rows)
            if (
                any(type(value) is not int or value < 0 for value in (operation.output_row, operation.output_col))
                or operation.output_row + operation.rows > problem.m
                or operation.output_col + operation.cols > problem.n
            ):
                raise ValueError("actual staged readout exceeds current complete source outputs")
            for r in range(operation.rows):
                for c in range(operation.cols):
                    target = operation.output_row + r, operation.output_col + c
                    if (
                        accumulators.get((operation.accumulator_row + r, c)) != (*target, problem.k)
                        or target in outputs
                    ):
                        raise ValueError("actual staged readout omits, duplicates or changes source products")
                    outputs.add(target)
        else:
            raise ValueError("unknown staged effects cannot be erased or treated as a fence")
    if pending is not None or len(outputs) != problem.m * problem.n or products != problem.m * problem.k * problem.n:
        raise ValueError("actual staged final completion leaves pending descriptors or incomplete required source work")
    return {
        "status": "verified_finite_staged_source_effects",
        "products": products,
        "outputs": len(outputs),
        "computes": computes,
        "operand_loads": loads,
        "retained_computes": retained,
        "fences_preserving_pending_descriptor": fences_pending,
        "timing_authority": "NONE",
    }
