"""Forward private exact copies through physically shared addressable views.

The source must stay unchanged until the destination's last read. Both views
must address the same physical backing; no cross-space handoff, conversion,
asynchronous completion or in-place arithmetic capability is inferred.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import lcm

from merlin.sched.ir import Kernel

from .temporary_storage import TemporaryObject, _problem, _project


@dataclass(frozen=True)
class CopyElimination:
    kernel: Kernel
    objects: tuple[TemporaryObject, ...]
    removed_calls: tuple[int, ...]


def eliminate_private_copies(
    kernel, objects, banks, adapter, *, external_windows=(), private_ownership=False, max_calls=4096, max_objects=256
) -> CopyElimination | None:
    """Produce a bounded candidate; allocation and independent checking follow.

    A complete private destination is initialized by one synchronous exact
    copy, then only read. Forward those reads to the original byte interval
    when no intervening write can change it. Other consumers remain ordered.
    Unknown effects are handled by the ordinary storage admission barrier.
    """
    original = kernel
    removed = []
    source_positions = list(range(len(kernel.body)))
    while objects:
        bank_map, effects, bindings, intervals, *_ = _problem(
            kernel, objects, banks, adapter, external_windows, private_ownership, max_calls, max_objects
        )
        # The allocator handles actual asynchronous completion separately.
        # This forwarding family promises no asynchronous ownership extension.
        if any(effect.completion != "synchronous" for effect in effects):
            break
        object_map = {obj.name: obj for obj in objects}
        alignments = {obj.name: lcm(obj.alignment, bank_map[obj.bank].alignment) for obj in objects}
        for operation, row in zip(effects, bindings, strict=True):
            for access, binding in zip(operation.accesses, row, strict=True):
                if binding is not None:
                    name = binding[0]
                    alignments[name] = lcm(alignments[name], bank_map[access.space].alignment)
        changed = False
        for index, effect in enumerate(effects):
            if not effect.exact_copy or any(binding is None for binding in bindings[index]):
                continue
            source, destination = effect.accesses
            source_name, source_offset = bindings[index][0]
            destination_name, destination_offset = bindings[index][1]
            destination_object = object_map[destination_name]
            destination_alignment = alignments[destination_name]
            if (
                source_name == destination_name
                or bank_map[source.space].backing != bank_map[destination.space].backing
                or intervals[destination_name][0] != index
                or destination_offset != 0
                or destination.size != destination_object.size
                or source_offset % destination_alignment
            ):
                continue
            last_read = intervals[destination_name][1]
            if any(
                access.writes
                and binding is not None
                and (
                    binding[0] == destination_name
                    or binding[0] == source_name
                    and binding[1] < source_offset + source.size
                    and source_offset < binding[1] + access.size
                )
                for later in range(index + 1, last_read + 1)
                for access, binding in zip(effects[later].accesses, bindings[later], strict=True)
            ):
                continue
            calls = []
            for later, (call, operation, row) in enumerate(zip(kernel.body, effects, bindings, strict=True)):
                if later == index:
                    continue
                accesses = tuple(
                    replace(access, start=source.start + binding[1])
                    if binding is not None and binding[0] == destination_name
                    else access
                    for access, binding in zip(operation.accesses, row, strict=True)
                )
                calls.append(call if accesses == operation.accesses else adapter.rewrite(call, accesses))
                if accesses != operation.accesses:
                    projected = _project(replace(kernel, body=(calls[-1],)), adapter, max_calls)[0]
                    if projected != replace(operation, accesses=accesses):
                        raise ValueError("copy forwarding adapter changed decoded instruction semantics")
            kernel = replace(kernel, body=tuple(calls))
            objects = tuple(
                replace(obj, alignment=lcm(obj.alignment, destination_alignment)) if obj.name == source_name else obj
                for obj in objects
                if obj.name != destination_name
            )
            removed.append(source_positions.pop(index))
            changed = True
            break
        if not changed:
            break
    return CopyElimination(kernel, objects, tuple(sorted(removed))) if kernel != original else None
