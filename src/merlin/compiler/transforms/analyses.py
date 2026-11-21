"""Callable structural analysis of actual xDSL IR, independent of target policies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import prod

from xdsl.dialects import builtin, func, memref, scf
from xdsl.ir import Block, Operation, SSAValue
from xdsl.rewriter import Rewriter
from xdsl.traits import DefaultResource, IsolatedFromAbove, IsTerminator, get_effects

from merlin.compiler.transforms.certificates import (
    BufferInterval,
    BufferReuseWitness,
    DependencyWitness,
    ir_digest,
)
from merlin.compiler.transforms.structural_rewrites import constant_integer

# ---- analysis_types ----------------------------------------------------------------------------
# Immutable structural analysis facts, with stable identities from actual IR.


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


# ---- analysis_liveness -------------------------------------------------------------------------
# Scoped structural liveness, bounded pressure accounting and dependency edges.


def block_facts(
    index: IRIndex, values: tuple[ValueFacts, ...], effects: tuple[EffectFacts, ...]
) -> tuple[BlockFacts, ...]:
    facts = {value.id: value for value in values}
    by_effect = defaultdict(list)
    for effect in effects:
        by_effect[effect.operation].append(effect)
    result = []
    for block in index.blocks:
        ops = tuple(block.ops)
        positions = {op: i for i, op in enumerate(ops)}
        local = {index.id(value): -1 for value in block.args}
        last: dict[str, int] = {}
        edges = set()
        unknown = set()
        previous: dict[str, int] = {}
        global_barrier: int | None = None
        memory_barrier: int | None = None
        for i, op in enumerate(ops):
            for value in op.results:
                local[index.id(value)] = i
            touched = []
            nested_operations = (op,) if op.has_trait(IsolatedFromAbove, value_if_unregistered=False) else op.walk()
            for nested in nested_operations:
                for value in nested.operands:
                    if isinstance(value.owner, Operation):
                        owner = ancestor_in(value.owner, block)
                        if owner is op:
                            continue
                        if owner is not None:
                            edges.add((positions[owner], i))
                    identity = index.id(value)
                    local.setdefault(identity, -1)
                    last[identity] = i
                touched.extend(by_effect[index.operation_ids[nested]])
            if op.has_trait(IsTerminator, value_if_unregistered=False):
                edges.update((j, i) for j in range(i))
            keys = set()
            for effect in touched:
                if effect.kind == "unknown":
                    keys.add("*")
                    unknown.add(index.operation_ids[op])
                elif effect.resource == "memory" and effect.value is not None:
                    value = facts[effect.value]
                    if value.alias_kind == "fresh":
                        keys.update("memory:" + root for root in value.alias_roots)
                    elif value.alias_kind == "external":
                        keys.add("memory:external")
                    else:
                        keys.add("memory:*")
                elif effect.resource == "memory":
                    keys.add("memory:*")
                else:
                    # Explicit resource effects are ordered even for read/read.
                    # Untyped default memory effects cannot establish disjointness.
                    keys.add("resource:" + effect.resource)
            if keys:
                if global_barrier is not None:
                    edges.add((global_barrier, i))
                if "*" in keys:
                    edges.update((prior, i) for prior in previous.values() if prior != i)
                    global_barrier = i
                else:
                    for key in keys:
                        if key.startswith("memory:"):
                            if memory_barrier is not None:
                                edges.add((memory_barrier, i))
                            if key == "memory:*":
                                edges.update(
                                    (prior, i)
                                    for k, prior in previous.items()
                                    if k.startswith("memory:") and prior != i
                                )
                                memory_barrier = i
                        if key in previous and previous[key] != i:
                            edges.add((previous[key], i))
                previous.update((key, i) for key in keys)
        # View aliases keep their underlying storage live, and escaped buffers
        # remain live through the end of this scope rather than guessed last use.
        for identity, start in tuple(local.items()):
            value = facts[identity]
            end = len(ops) if value.escapes else last.get(identity, start)
            last[identity] = max(end, start)
            for alias in value.alias_roots:
                if alias in local:
                    last[alias] = max(last.get(alias, local[alias]), last[identity])
        intervals = tuple(
            LiveInterval(
                identity,
                start,
                last[identity],
                facts[identity].escapes or facts[identity].alias_kind in {"unknown", "external"},
            )
            for identity, start in sorted(local.items())
        )
        peak_values, peak_bytes, working, unknown_sizes = _pressure(intervals, facts)
        result.append(
            BlockFacts(
                index.block_ids[block],
                tuple(index.operation_ids[op] for op in ops),
                intervals,
                tuple(sorted(edges)),
                tuple(sorted(unknown)),
                peak_values,
                peak_bytes,
                working,
                unknown_sizes,
            )
        )
    return tuple(result)


def _pressure(intervals: tuple[LiveInterval, ...], facts: dict[str, ValueFacts]):
    events: dict[int, list[tuple[int, int]]] = defaultdict(list)
    representatives = {}
    unknown = set()
    working = 0
    for interval in intervals:
        value = facts[interval.value]
        # Exact views count one underlying buffer. Unknown aliases/sizes are
        # reported; this subtotal is not an upper bound on physical occupancy.
        identity = value.alias_roots[0] if len(value.alias_roots) == 1 and value.alias_kind != "unknown" else value.id
        previous = representatives.get(identity)
        start = min(previous[0], interval.start) if previous else interval.start
        end = max(previous[1], interval.end) if previous else interval.end
        sized = facts[identity].logical_bytes if identity in facts else value.logical_bytes
        representatives[identity] = (start, end, sized)
        if sized is None:
            unknown.add(identity)
    for identity, (start, end, sized) in representatives.items():
        size = 0 if sized is None else sized
        events[start].append((1, size))
        events[end + 1].append((-1, -size))
        if facts[identity].shape is not None:
            working += size
    live = known_bytes = peak_values = peak_bytes = 0
    for point in sorted(events):
        for count, size in events[point]:
            live += count
            known_bytes += size
        peak_values = max(peak_values, live)
        peak_bytes = max(peak_bytes, known_bytes)
    return peak_values, peak_bytes, working, tuple(sorted(unknown))


# ---- analysis_memory ---------------------------------------------------------------------------
# Conservative aliases/effects; no instruction-name interpretation or guessed widths.


def _representation(typ, index_bits: int | None):
    if isinstance(typ, builtin.MemRefType | builtin.TensorType | builtin.VectorType):
        shape = tuple(typ.get_shape())
        element = typ.get_element_type()
    else:
        shape, element = None, typ
    if isinstance(element, builtin.IntegerType | builtin.AnyFloat):
        bits = element.bitwidth
    elif isinstance(element, builtin.IndexType):
        bits = index_bits
    else:
        bits = None
    memory = (
        True
        if isinstance(typ, builtin.MemRefType)
        else False
        if isinstance(
            typ, builtin.TensorType | builtin.VectorType | builtin.IntegerType | builtin.AnyFloat | builtin.IndexType
        )
        else None
    )
    known_layout = not isinstance(typ, builtin.MemRefType) or isinstance(typ.layout, builtin.NoneAttr)
    static = shape is None or all(dimension >= 0 for dimension in shape)
    size = prod(shape or ()) * ((bits + 7) // 8) if static and bits is not None and known_layout else None
    return shape, bits, size, memory


def value_facts(index: IRIndex, *, index_bits: int | None) -> tuple[ValueFacts, ...]:
    memo: dict[SSAValue, tuple[tuple[str, ...], str]] = {}

    def roots(value: SSAValue):
        if value in memo:
            return memo[value]
        owner = value.owner
        _, _, _, memory = _representation(value.type, index_bits)
        if memory is False:
            answer = ((), "not_memory")
        elif not isinstance(owner, Operation):
            parent = owner.parent_op()
            # Only function-entry arguments predate this activation's fresh
            # allocations. Region and CFG arguments can forward an allocation
            # from the enclosing program; without a forwarding analysis they
            # must remain possibly aliased with that allocation.
            entry_argument = isinstance(parent, func.FuncOp) and owner is next(iter(parent.body.blocks), None)
            answer = ((index.id(value),), "external" if memory is True and entry_argument else "unknown")
        elif isinstance(owner, memref.AllocOp | memref.AllocaOp):
            answer = ((index.id(value),), "fresh")
        elif isinstance(
            owner,
            (
                memref.CastOp,
                memref.SubviewOp,
                memref.CollapseShapeOp,
                memref.ExpandShapeOp,
                memref.ReinterpretCastOp,
                memref.MemorySpaceCastOp,
            ),
        ):
            answer = roots(owner.operands[0])
        else:
            answer = ((index.id(value),), "unknown")
        memo[value] = answer
        return answer

    escaped: set[str] = set()
    for value, identity in index.value_ids.items():
        shape, bits, size, memory = _representation(value.type, index_bits)
        for use in value.uses:
            op = use.operation
            if memory is not False and (isinstance(op, func.ReturnOp) or get_effects(op) is None):
                escaped.update(roots(value)[0])
    result = []
    for value, identity in index.value_ids.items():
        shape, bits, size, memory = _representation(value.type, index_bits)
        owner = value.owner
        block = owner.parent if isinstance(owner, Operation) else owner
        aliases, kind = roots(value)
        crosses_scope = any(ancestor_in(use.operation, block) is None for use in value.uses)
        unknown_use = any(get_effects(use.operation) is None for use in value.uses)
        result.append(
            ValueFacts(
                identity,
                str(value.type),
                index.operation_ids.get(owner) if isinstance(owner, Operation) else None,
                index.block_ids[block],
                shape,
                bits,
                size,
                memory,
                aliases,
                kind,
                bool(set(aliases) & escaped) or crosses_scope or unknown_use,
            )
        )
    return tuple(result)


def may_alias(left: ValueFacts, right: ValueFacts) -> bool:
    if left.memory is False or right.memory is False:
        return False
    if left.id == right.id or set(left.alias_roots) & set(right.alias_roots):
        return True
    if left.alias_kind == "unknown" or right.alias_kind == "unknown":
        return True
    # Fresh allocations are disjoint from other valid live allocations and
    # existing external arguments. Distinct external arguments may alias.
    return left.alias_kind != "fresh" and right.alias_kind != "fresh"


def effect_facts(index: IRIndex) -> tuple[EffectFacts, ...]:
    result = []
    for op, identity in index.operation_ids.items():
        if isinstance(
            op, builtin.ModuleOp | func.FuncOp | scf.ForOp | scf.IfOp | scf.WhileOp | scf.IndexSwitchOp
        ) or op.has_trait(IsTerminator, value_if_unregistered=False):
            # Regions are aggregated from their actual nested operations by
            # the block planner, not declared pure because they contain loops.
            continue
        value = None
        kind = None
        if isinstance(op, memref.LoadOp):
            kind, value = "read", op.memref
        elif isinstance(op, memref.StoreOp):
            kind, value = "write", op.memref
        elif isinstance(op, memref.AllocOp | memref.AllocaOp):
            kind, value = "allocate", op.memref
        elif isinstance(op, memref.DeallocOp):
            kind, value = "free", op.memref
        if kind is not None:
            result.append(EffectFacts(identity, kind, "memory", index.id(value)))
            continue
        effects = get_effects(op)
        if effects is None:
            result.append(EffectFacts(identity, "unknown", "*", None))
        else:
            for effect in effects:
                affected = index.value_ids.get(effect.value) if isinstance(effect.value, SSAValue) else None
                resource = "memory" if isinstance(effect.resource, DefaultResource) else effect.resource.name()
                result.append(EffectFacts(identity, effect.kind.name.lower(), resource, affected))
    return tuple(
        sorted(result, key=lambda effect: (effect.operation, effect.kind, effect.resource, effect.value or ""))
    )


# ---- analyses ----------------------------------------------------------------------------------


def analyze_ir(module: builtin.ModuleOp, *, index_bits: int | None = None) -> IRAnalysis:
    if index_bits is not None and (type(index_bits) is not int or index_bits <= 1):
        raise ValueError("index width must be explicit and positive when known")
    index = IRIndex(module)
    values, effects = value_facts(index, index_bits=index_bits), effect_facts(index)
    unknown = [f"value {value.id}: unknown memory aliasing" for value in values if value.alias_kind == "unknown"]
    unknown.extend(f"operation {effect.operation}: unknown effects" for effect in effects if effect.kind == "unknown")
    return IRAnalysis(
        index.source_sha256, index_bits, values, effects, block_facts(index, values, effects), tuple(unknown)
    )


def validate_analysis(module: builtin.ModuleOp, analysis: IRAnalysis) -> None:
    if analyze_ir(module, index_bits=analysis.index_bits) != analysis:
        raise ValueError("analysis does not describe the actual current IR")


# ---- planning ----------------------------------------------------------------------------------
# IR-derived storage lifetimes and conservative sequential effect dependencies.


@dataclass(frozen=True)
class StoragePlan:
    witnesses: tuple[BufferReuseWitness, ...]
    unknown: tuple[str, ...]

    @property
    def peak_reserved_bytes(self) -> int:
        # Blocks represent separate allocation scopes, so this is a per-block
        # reservation bound, not a whole-program concurrent memory claim.
        return max(
            (
                sum(
                    max(i.size_bytes for i in witness.intervals if i.slot == slot)
                    for slot in {i.slot for i in witness.intervals}
                )
                for witness in self.witnesses
            ),
            default=0,
        )


@dataclass(frozen=True)
class DependencyPlan:
    witnesses: tuple[DependencyWitness, ...]


def blocks_in(module: builtin.ModuleOp) -> tuple[Block, ...]:
    return tuple(block for op in module.walk() for region in op.regions for block in region.blocks)


def _ancestor_in(op: Operation, block: Block) -> Operation | None:
    while op.parent is not block:
        parent = op.parent_op()
        if parent is None:
            return None
        op = parent
    return op


def _full_write_initializer(op: Operation, value: SSAValue, shape: tuple[int, ...]) -> bool:
    """Recognize one complete rectangular write, without enumerating coordinates."""
    ivs: list[SSAValue] = []
    current = op
    for extent in shape:
        if not isinstance(current, scf.ForOp) or current.iter_args:
            return False
        if tuple(constant_integer(v) for v in (current.lb, current.ub, current.step)) != (0, extent, 1):
            return False
        ivs.append(current.body.block.args[0])
        significant = [child for child in current.body.block.ops if not isinstance(child, scf.YieldOp)]
        if not significant:
            return False
        # Scalar value computations can precede the next loop/store; no other
        # uses of this buffer or effects are allowed in that prefix.
        for child in significant[:-1]:
            if value in child.operands or get_effects(child) != set():
                return False
        current = significant[-1]
    return isinstance(current, memref.StoreOp) and current.memref is value and tuple(current.indices) == tuple(ivs)


def plan_storage(module: builtin.ModuleOp) -> StoragePlan:
    """Plan identical-type reuse in direct allocation scopes.

    Only static contiguous byte-addressable scalar integer buffers with a
    recognized complete initialization and explicit deallocation are eligible.
    Escapes, aliases, dynamic layouts and asynchronous lifetimes remain unknown.
    Distinct memref allocations are intrinsically disjoint; caller input alias
    assertions are neither needed nor accepted.
    """
    witnesses = []
    unknown = []
    for block_index, block in enumerate(blocks_in(module)):
        ops = tuple(block.ops)
        positions = {op: i for i, op in enumerate(ops)}
        intervals: list[BufferInterval] = []
        slot_end: list[int] = []
        slot_keys: list[tuple[builtin.MemRefType, int]] = []
        for alloc in ops:
            if not isinstance(alloc, memref.AllocOp):
                continue
            alloc_index = positions[alloc]
            tag = f"block {block_index} allocation {alloc_index}"
            dtype = alloc.memref.type
            shape = dtype.get_shape()
            element = dtype.get_element_type()
            if (
                not isinstance(dtype.layout, builtin.NoneAttr)
                or any(v <= 0 for v in shape)
                or alloc.dynamic_sizes
                or alloc.symbol_operands
                or not isinstance(element, builtin.IntegerType)
                or element.width.data <= 0
                or element.width.data % 8
                or element.width.data & (element.width.data - 1)
            ):
                unknown.append(f"{tag}: unsupported layout, shape or physical element size")
                continue
            alignment = alloc.alignment.value.data if alloc.alignment is not None else 1
            if alignment <= 0 or alignment & (alignment - 1):
                unknown.append(f"{tag}: unsupported alignment")
                continue
            access_ancestors: list[Operation] = []
            deallocations: list[memref.DeallocOp] = []
            valid = True
            for use in alloc.memref.uses:
                user = use.operation
                if isinstance(user, memref.DeallocOp) and user.parent is block:
                    deallocations.append(user)
                elif isinstance(user, memref.LoadOp | memref.StoreOp | memref.DimOp):
                    ancestor = _ancestor_in(user, block)
                    if ancestor is None:
                        valid = False
                        break
                    access_ancestors.append(ancestor)
                else:
                    valid = False
                    break
            if not valid or len(deallocations) != 1 or not access_ancestors:
                unknown.append(f"{tag}: alias, escape or missing explicit lifetime")
                continue
            end = positions[deallocations[0]]
            first = min(positions[user] for user in access_ancestors)
            last = max(positions[user] for user in access_ancestors)
            if not alloc_index < first <= last < end:
                unknown.append(f"{tag}: use outside its allocation lifetime")
                continue
            if not _full_write_initializer(ops[first], alloc.memref, tuple(shape)):
                unknown.append(f"{tag}: complete initialization before reads is unproved")
                continue
            # Additional reads/writes within the first enclosing initializer
            # would invalidate full initialization as an initial event.
            first_uses = [
                use.operation for use in alloc.memref.uses if _ancestor_in(use.operation, block) is ops[first]
            ]
            if len(first_uses) != 1 or not isinstance(first_uses[0], memref.StoreOp):
                unknown.append(f"{tag}: initial operation also reads or aliases the buffer")
                continue
            key = (dtype, alignment)
            slot = next(
                (i for i, (k, e) in enumerate(zip(slot_keys, slot_end, strict=True)) if k == key and e < alloc_index),
                None,
            )
            if slot is None:
                slot = len(slot_keys)
                slot_keys.append(key)
                slot_end.append(end)
            else:
                slot_end[slot] = end
            intervals.append(
                BufferInterval(
                    alloc_index,
                    first,
                    end,
                    tuple(shape),
                    element.width.data,
                    alignment,
                    prod(shape) * (element.width.data // 8),
                    slot,
                )
            )
        if intervals:
            witnesses.append(BufferReuseWitness(block_index, tuple(intervals)))
    return StoragePlan(tuple(witnesses), tuple(unknown))


def reuse_storage(module: builtin.ModuleOp, plan: StoragePlan) -> tuple[BufferReuseWitness, ...]:
    """Apply a freshly recomputed plan, never trust a caller's interval claims."""
    if plan != plan_storage(module):
        raise ValueError("storage plan does not match the current IR")
    blocks = blocks_in(module)
    changed = []
    for witness in plan.witnesses:
        block = blocks[witness.block_index]
        ops = tuple(block.ops)
        groups: dict[int, list[BufferInterval]] = {}
        for interval in witness.intervals:
            groups.setdefault(interval.slot, []).append(interval)
        for group in groups.values():
            if len(group) < 2:
                continue
            first_alloc = ops[group[0].allocation_index]
            assert isinstance(first_alloc, memref.AllocOp)
            for prior, later in zip(group, group[1:]):
                if prior.deallocation_index >= later.allocation_index:
                    raise ValueError("live allocations overlap in the requested reuse slot")
                Rewriter.erase_op(ops[prior.deallocation_index])
                later_alloc = ops[later.allocation_index]
                assert isinstance(later_alloc, memref.AllocOp)
                Rewriter.replace_op(later_alloc, [], [first_alloc.memref])
            changed.append(BufferReuseWitness(witness.block_index, tuple(group)))
    return tuple(changed)


def plan_dependencies(module: builtin.ModuleOp) -> DependencyPlan:
    """SSA edges plus an ordered chain of all side-effecting operations.

    This initial sequential planner permits independent pure scalar work to
    move, while conservatively ordering even read/read effects and all unknown
    device/configuration operations. It does not infer missing OOT effects.
    """
    witnesses = []
    for block_index, block in enumerate(blocks_in(module)):
        ops = tuple(block.ops)
        positions = {op: i for i, op in enumerate(ops)}
        edges: set[tuple[int, int]] = set()
        previous_effect: int | None = None
        unknown = []
        for i, op in enumerate(ops):
            for nested in op.walk():
                for value in nested.operands:
                    if isinstance(value.owner, Operation):
                        owner = _ancestor_in(value.owner, block)
                        if owner is not None and owner is not op:
                            edges.add((positions[owner], i))
            effectful = False
            for nested in op.walk():
                effects = get_effects(nested)
                # Terminators do not reorder program effects; all preceding
                # block operations must remain before the block terminator.
                if nested is op and op.has_trait(IsTerminator, value_if_unregistered=False):
                    edges.update((j, i) for j in range(i))
                    continue
                if nested.has_trait(IsTerminator, value_if_unregistered=False):
                    continue
                if effects is None:
                    unknown.append(i)
                    effectful = True
                elif effects:
                    effectful = True
            if effectful:
                if previous_effect is not None:
                    edges.add((previous_effect, i))
                previous_effect = i
        witnesses.append(
            DependencyWitness(
                block_index,
                tuple(ir_digest(op) for op in ops),
                tuple(sorted(edges)),
                tuple(range(len(ops))),
                tuple(sorted(set(unknown))),
            )
        )
    return DependencyPlan(tuple(witnesses))


def validate_order(witness: DependencyWitness, order: Sequence[int]) -> None:
    if sorted(order) != list(range(len(witness.operation_sha256))):
        raise ValueError("schedule must contain each original operation exactly once")
    positions = {node: index for index, node in enumerate(order)}
    if any(positions[before] >= positions[after] for before, after in witness.edges):
        raise ValueError("schedule violates an SSA, memory, state or terminator dependency")


def apply_order(module: builtin.ModuleOp, block_index: int, order: Sequence[int]) -> DependencyWitness:
    """Apply a permutation only after deriving dependencies from the current IR."""
    plan = plan_dependencies(module)
    witness = plan.witnesses[block_index]
    validate_order(witness, order)
    block = blocks_in(module)[block_index]
    ops = tuple(block.ops)
    for op in ops:
        block.detach_op(op)
    block.add_ops(ops[i] for i in order)
    module.verify()
    return DependencyWitness(
        block_index, witness.operation_sha256, witness.edges, tuple(order), witness.unknown_effect_indices
    )


__all__ = [
    "IRAnalysis",
    "IRIndex",
    "analyze_ir",
    "may_alias",
    "validate_analysis",
]
