"""Plan connected SSA regions, complete live boundaries and source-owned lifetimes."""
from __future__ import annotations

from dataclasses import dataclass
from math import prod

from merlin.transforms.certificates import ir_digest


@dataclass(frozen=True)
class RegionStage:
    operation: object
    ordinal: int
    kind: str
    inputs: tuple[int, ...]
    output: int


@dataclass(frozen=True)
class ConnectedRegion:
    source: object
    source_sha256: str
    stages: tuple[RegionStage, ...]
    values: tuple
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    lifetimes: tuple[tuple[int, int], ...]

    def revalidate(self):
        if ir_digest(self.source) != self.source_sha256:
            raise ValueError('connected region source changed; replan')
        current = tuple(self.source.walk())
        if any(stage.ordinal >= len(current) or current[stage.ordinal] is not stage.operation
               for stage in self.stages):
            raise ValueError('connected region operation ownership changed; replan')
        rebuilt = plan_connected_region(self.source, tuple(stage.operation for stage in self.stages))
        if rebuilt != self:
            raise ValueError('connected region dataflow or lifetimes changed; replan')

    def record(self):
        return {'source_ir_sha256': self.source_sha256,
            'stages': [{'ordinal': s.ordinal, 'kind': s.kind, 'inputs': s.inputs, 'output': s.output}
                       for s in self.stages],
            'values': [{'type': str(value.type), 'lifetime': life} for value, life in
                       zip(self.values, self.lifetimes, strict=True)],
            'inputs': self.inputs, 'outputs': self.outputs,
            'maturity': 'typed_source_plan_not_physical_residency'}


def plan_connected_region(module, operations):
    """Plan one ordered same-block region, retaining every external consumer.

    Source operand identity, never equal shape, connects stages. Initializers
    are excluded only after native pointwise admission proves they are unread.
    Unknown intervening operations refuse rather than disappearing. Inputs stay
    borrowed/read-only for the full region; externally observed outputs stay live.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import IntegerType, TensorType
    from merlin.compiler.planning.frontiers import _structural_family
    from merlin.compiler.schedule.pointwise_source import read_pointwise_source

    operations = tuple(operations)
    members = tuple(module.walk())
    if not 2 <= len(operations) <= 64 or len(members) > 10000 or len(set(operations)) != len(operations):
        raise ValueError('connected region requires two to sixty-four distinct bounded source operations')
    ordinals = {op: i for i, op in enumerate(members)}
    block = operations[0].parent
    if block is None or any(op not in ordinals or op.parent is not block for op in operations):
        raise ValueError('connected region requires current operations from one source block')
    body = tuple(block.ops)
    positions = tuple(body.index(op) for op in operations)
    if positions != tuple(sorted(positions)):
        raise ValueError('connected region must retain original program order')
    # Admit only source-proven pure zero fills used by selected contractions.
    # This permits dense -> pointwise -> dense regions without treating an
    # arbitrary intervening operation as an ignorable initializer.
    initializers = set()
    from merlin.compiler.schedule.dense_source import read_zero_seeded_integer_matmul
    for op in operations:
        if _structural_family(op) == 'contraction':
            dtypes = tuple(str(value.type.element_type) for value in (*op.operands[:2], op.results[0]))
            read_zero_seeded_integer_matmul(op, dtypes=dtypes)
            seed = op.operands[2].owner
            if getattr(seed, 'name', None) == 'linalg.fill' and all(
                    use.operation in operations for value in seed.results for use in value.uses):
                initializers.add(seed)
    for op in body[positions[0]:positions[-1] + 1]:
        if op not in operations and op not in initializers and not (isinstance(op, (arith.ConstantOp, tensor.EmptyOp))
                                         or op.name == 'tensor.splat'):
            raise ValueError('connected region crosses an unmodeled operation or observer')
    values, inputs, stages = [], [], []
    created, last = [], []
    for index, op in enumerate(operations):
        kind = _structural_family(op)
        if kind == 'elementwise_map':
            read_pointwise_source(op, max_elements=64 << 20, max_operations=128, max_ir_bytes=65536)
            operands = tuple(op.inputs)
        elif kind == 'contraction':
            operands = tuple(op.operands[:2])
        else:
            raise ValueError('connected region needs native pointwise or declared contraction source')
        if len(op.results) != 1:
            raise ValueError('connected stage needs one explicit tensor result')
        used = []
        for value in (*operands, op.results[0]):
            typ = value.type
            if (not isinstance(typ, TensorType) or not isinstance(typ.element_type, IntegerType)
                    or typ.element_type.width.data % 8 or any(n <= 0 for n in typ.get_shape())
                    or prod(typ.get_shape()) > 64 << 20):
                raise ValueError('connected tensor requires bounded static byte-addressable integer storage')
            if value not in values:
                if value.owner in operations and value is not op.results[0]:
                    raise ValueError('connected input is not available in source program order')
                values.append(value)
                external = value is not op.results[0]
                created.append(-1 if external else index)
                last.append(len(operations) if external else index)
                if external:
                    inputs.append(len(values) - 1)
            ident = values.index(value)
            last[ident] = max(last[ident], index)
            used.append(ident)
        stages.append(RegionStage(op, ordinals[op], kind, tuple(used[:-1]), used[-1]))
    outputs = tuple(stage.output for stage in stages
                    if any(use.operation not in operations for use in values[stage.output].uses))
    for ident in outputs:
        last[ident] = len(operations)
    connected = {0}
    for _ in stages:
        for index, stage in enumerate(stages):
            for value in stage.inputs:
                producer = created[value]
                if producer >= 0 and (producer in connected or index in connected):
                    connected.update((producer, index))
    if not outputs or len(connected) != len(stages):
        raise ValueError('connected region needs one producer-consumer component and an externally visible output')
    return ConnectedRegion(module, ir_digest(module), tuple(stages), tuple(values), tuple(inputs), outputs,
                           tuple(zip(created, last, strict=True)))
