"""IR-derived storage lifetimes and conservative sequential effect dependencies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

from xdsl.dialects import builtin, memref, scf
from xdsl.ir import Block, Operation, SSAValue
from xdsl.rewriter import Rewriter
from xdsl.traits import IsTerminator, get_effects

from .certificates import BufferInterval, BufferReuseWitness, DependencyWitness, ir_digest
from .normalization import constant_integer


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
