"""Scoped structural liveness, bounded pressure accounting and dependency edges."""

from __future__ import annotations

from collections import defaultdict

from xdsl.ir import Operation
from xdsl.traits import IsolatedFromAbove, IsTerminator

from .analysis_types import BlockFacts, EffectFacts, IRIndex, LiveInterval, ValueFacts, ancestor_in


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
