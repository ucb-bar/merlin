"""Immutable structural analysis facts, with stable identities from actual IR."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from xdsl.dialects import builtin
from xdsl.ir import Block, Operation, SSAValue

from .certificates import ir_digest


@dataclass(frozen=True)
class ValueFacts:
    id: str
    type: str
    defining_operation: int | None
    defining_block: int
    shape: tuple[int, ...] | None
    element_bits: int | None
    logical_bytes: int | None
    memory: bool | None
    alias_roots: tuple[str, ...]
    alias_kind: str
    escapes: bool


@dataclass(frozen=True)
class EffectFacts:
    operation: int
    kind: str
    resource: str
    value: str | None


@dataclass(frozen=True)
class LiveInterval:
    value: str
    start: int
    end: int
    conservative: bool


@dataclass(frozen=True)
class BlockFacts:
    block: int
    operations: tuple[int, ...]
    live_intervals: tuple[LiveInterval, ...]
    dependencies: tuple[tuple[int, int], ...]
    unknown_effects: tuple[int, ...]
    peak_live_values: int
    peak_known_logical_bytes: int
    working_set_known_logical_bytes: int
    unknown_sizes: tuple[str, ...]


@dataclass(frozen=True)
class IRAnalysis:
    source_sha256: str
    index_bits: int | None
    values: tuple[ValueFacts, ...]
    effects: tuple[EffectFacts, ...]
    blocks: tuple[BlockFacts, ...]
    unknown: tuple[str, ...]
    schema: str = "merlin.structural_analysis.v1"

    def to_dict(self) -> dict:
        return asdict(self)

    def value(self, identity: str) -> ValueFacts:
        return next(value for value in self.values if value.id == identity)


class IRIndex:
    """Linear-size indexing; values retain no identity from another revision."""

    def __init__(self, module: builtin.ModuleOp):
        module.verify()
        self.source_sha256 = ir_digest(module)
        self.operations = tuple(module.walk())
        self.operation_ids = {operation: i for i, operation in enumerate(self.operations)}
        self.blocks = tuple(block for op in self.operations for region in op.regions for block in region.blocks)
        self.block_ids = {block: i for i, block in enumerate(self.blocks)}
        self.value_ids: dict[SSAValue, str] = {}
        for block, identity in self.block_ids.items():
            for i, value in enumerate(block.args):
                self.value_ids[value] = f"b{identity}.a{i}"
        for op, identity in self.operation_ids.items():
            for i, value in enumerate(op.results):
                self.value_ids[value] = f"o{identity}.r{i}"
        self.values = {identity: value for value, identity in self.value_ids.items()}

    def id(self, value: SSAValue) -> str:
        return self.value_ids[value]


def ancestor_in(operation: Operation, block: Block) -> Operation | None:
    while operation.parent is not block:
        parent = operation.parent_op()
        if parent is None:
            return None
        operation = parent
    return operation


def operation_name(operation: Operation) -> str:
    if isinstance(operation, builtin.UnregisteredOp):
        return operation.op_name.data
    return operation.name
