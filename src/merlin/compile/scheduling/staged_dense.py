"""Finite staged contraction proposals under independently checked effects.

Unlike dense-v1, a descriptor issue is not synchronous completion. The target
must independently bind these logical effects to its current emitted program;
this library supplies no target lookup, adapter admission or timing authority.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StagedDenseContract:
    hardware: str
    software: str
    numerical_policy: str
    execution: str
    dim: int
    operand_rows: int
    accumulator_rows: int
    operand_bits: int
    primitive_bits: int
    accumulator_bits: int
    preload_visibility: str
    fence_completion: str
    readout: str
    external_buffers_disjoint: bool
    rhs_immutable: bool
    retained_weights: bool
    schema: str = "merlin.staged_dense.v1"

    def __post_init__(self):
        if self.schema != "merlin.staged_dense.v1":
            raise ValueError("unknown staged dense contract version")
        if any(
            type(value) is not str or not value or len(value) > 512
            for value in (self.hardware, self.software, self.numerical_policy, self.execution)
        ):
            raise ValueError("staged dense needs exact selected hardware/software/numerical/execution identities")
        if (
            any(
                type(value) is not int or value <= 0
                for value in (
                    self.dim,
                    self.operand_rows,
                    self.accumulator_rows,
                    self.operand_bits,
                    self.primitive_bits,
                    self.accumulator_bits,
                )
            )
            or self.dim > 256
        ):
            raise ValueError("staged dense dimensions and arithmetic widths must be explicit and bounded")
        if (
            self.primitive_bits < 2 * self.operand_bits + (self.dim - 1).bit_length()
            or self.accumulator_bits < self.primitive_bits
        ):
            raise ValueError("staged primitive arithmetic cannot contain the selected full signed panel")
        if (
            self.preload_visibility != "read_at_compute_consume"
            or self.fence_completion != "data_commands_preserve_pending_descriptor"
            or self.readout != "raw_accumulator_identity"
        ):
            raise ValueError("unsupported staged descriptor, completion or readout semantics")
        if any(
            type(value) is not bool
            for value in (self.external_buffers_disjoint, self.rhs_immutable, self.retained_weights)
        ):
            raise ValueError("staged ownership and retention require explicit Boolean premises")


@dataclass(frozen=True)
class StagedDenseProblem:
    m: int
    k: int
    n: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in (self.m, self.k, self.n)):
            raise ValueError("staged source dimensions must be positive static extents")


@dataclass(frozen=True)
class OperandLoad:
    role: str
    source_row: int
    source_col: int
    local_row: int
    rows: int
    cols: int


@dataclass(frozen=True)
class AccumulatorZero:
    local_row: int
    rows: int
    cols: int


@dataclass(frozen=True)
class DescriptorIssue:
    rhs_row: int | None
    accumulator_row: int
    m: int
    k: int
    n: int
    continuation: bool


@dataclass(frozen=True)
class ComputeConsume:
    lhs_row: int
    m: int
    k: int
    retain_weights: bool = False


@dataclass(frozen=True)
class AccumulatorReadout:
    accumulator_row: int
    output_row: int
    output_col: int
    rows: int
    cols: int


@dataclass(frozen=True)
class CompletionFence:
    pass


StagedDenseOperation = (
    OperandLoad | AccumulatorZero | DescriptorIssue | ComputeConsume | AccumulatorReadout | CompletionFence
)


@dataclass(frozen=True)
class StagedDenseRewrite:
    name: str
    implementation: Callable

    def __post_init__(self):
        if type(self.name) is not str or not self.name or len(self.name) > 128 or not callable(self.implementation):
            raise ValueError("staged rewrites need a bounded name and an executable implementation")


def eliminate_redundant_rhs_loads(operations, problem, contract):
    """Keep an immutable panel only while its exact private cells remain live.

    This does not allocate another panel, prefetch, remove a fence or pretend a
    pending descriptor completed. LHS loads invalidate aliased operand cells.
    """
    if not contract.rhs_immutable or not contract.external_buffers_disjoint:
        return operations
    cells, result = {}, []
    for operation in operations:
        if type(operation) is OperandLoad:
            stride = problem.k if operation.role == "lhs" else problem.n
            values = {
                (operation.local_row + r, c): (
                    operation.role,
                    (operation.source_row + r) * stride + operation.source_col + c,
                )
                for r in range(operation.rows)
                for c in range(operation.cols)
            }
            if operation.role == "rhs" and all(cells.get(cell) == value for cell, value in values.items()):
                continue
            cells.update(values)
        result.append(operation)
    return tuple(result)


def rewrite_staged_dense(operations, *, problem, contract, rewrites=(), max_commands=1_000_000, max_products=4 << 20):
    """Apply actual ordered candidate code; independently check every result.

    The original target owner must later bind and check the final emitted code.
    A successful logical proposal is not that final executable evidence.
    """
    from .staged_dense_verification import verify_staged_dense

    if (
        type(operations) is not tuple
        or type(problem) is not StagedDenseProblem
        or type(contract) is not StagedDenseContract
        or type(rewrites) is not tuple
        or any(type(rewrite) is not StagedDenseRewrite for rewrite in rewrites)
    ):
        raise ValueError("staged source effects and ordered rewrites require explicit immutable tuples")
    original = operations
    context = (asdict(problem), asdict(contract))
    original_effects = [(type(operation).__name__, asdict(operation)) for operation in original]
    reports = [
        verify_staged_dense(
            original, problem=problem, contract=contract, max_commands=max_commands, max_products=max_products
        )
    ]
    for rewrite in rewrites:
        proposal = rewrite.implementation(operations, problem, contract)
        if context != (asdict(problem), asdict(contract)) or original_effects != [
            (type(operation).__name__, asdict(operation)) for operation in original
        ]:
            raise ValueError("candidate staged rewrite changed original source, effects or selected contracts")
        if type(proposal) is not tuple:
            raise ValueError("candidate staged rewrite must return actual immutable operations")
        reports.append(
            verify_staged_dense(
                proposal, problem=problem, contract=contract, max_commands=max_commands, max_products=max_products
            )
        )
        operations = proposal
    return operations, {
        "schema": "merlin.checked_staged_dense_proposal.v1",
        "rewrite_order": [rewrite.name for rewrite in rewrites],
        "before_commands": len(original),
        "after_commands": len(operations),
        "checks": reports,
        "final_emitted_program_check": "required from original qualified target owner",
        "timing_authority": "NONE",
    }


def order_output_tiles_n_before_m(operations, problem, contract):
    """Move complete output tiles to column-major traversal without new storage.

    Each movable tile includes its zero, full reduction, readout and following
    completion fence. No operation inside a tile is reordered, and no fence is
    removed. A later RHS-residency rewrite can use the resulting panel locality.
    If earlier rewrites removed a needed load, the independent checker must
    reject the resulting proposal; pass order is therefore observable.
    """
    del problem
    if not contract.external_buffers_disjoint or not contract.rhs_immutable:
        return operations
    if type(operations) is not tuple:
        raise ValueError("output-tile ordering requires immutable staged effects")
    cursor = 0
    while cursor < len(operations) and type(operations[cursor]) is CompletionFence:
        cursor += 1
    prefix = operations[:cursor]
    blocks = []
    while cursor < len(operations):
        if type(operations[cursor]) is CompletionFence:
            if not all(type(op) is CompletionFence for op in operations[cursor:]):
                raise ValueError("output-tile ordering cannot detach an inter-tile effect")
            break
        start = cursor
        if type(operations[cursor]) is not AccumulatorZero:
            raise ValueError("output-tile ordering requires complete independently zeroed tiles")
        cursor += 1
        while cursor < len(operations) and type(operations[cursor]) is not AccumulatorReadout:
            if type(operations[cursor]) is AccumulatorZero:
                raise ValueError("output-tile ordering cannot split overlapping accumulator lifetimes")
            cursor += 1
        if cursor + 1 >= len(operations) or type(operations[cursor + 1]) is not CompletionFence:
            raise ValueError("output-tile ordering requires complete readout and completion fence")
        readout = operations[cursor]
        cursor += 2
        fence_end = cursor
        while fence_end < len(operations) and type(operations[fence_end]) is CompletionFence:
            fence_end += 1
        if fence_end < len(operations) and fence_end != cursor:
            if type(operations[fence_end]) is not AccumulatorZero:
                raise ValueError("output-tile ordering cannot absorb an unknown inter-tile effect")
            # Retain every inter-tile fence with the completed tile. A final
            # all-fence suffix stays outside the movable blocks.
            cursor = fence_end
        blocks.append(((readout.output_col, readout.output_row), operations[start:cursor]))
    keys = [key for key, _ in blocks]
    if len(keys) != len(set(keys)):
        raise ValueError("output-tile ordering cannot duplicate a logical output tile")
    ordered = sorted(blocks, key=lambda block: block[0])
    return (*prefix, *(op for _, block in ordered for op in block), *operations[cursor:])
