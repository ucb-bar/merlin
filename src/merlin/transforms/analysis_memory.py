"""Conservative aliases/effects; no instruction-name interpretation or guessed widths."""

from __future__ import annotations

from math import prod

from xdsl.dialects import builtin, func, memref, scf
from xdsl.ir import Operation, SSAValue
from xdsl.traits import DefaultResource, IsTerminator, get_effects

from .analysis_types import EffectFacts, IRIndex, ValueFacts, ancestor_in


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
