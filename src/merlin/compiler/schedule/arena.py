"""Completion-aware bounded arena placement and independent witness checking.

Default dispatch planning retains its existing owner and behavior. This explicit
path consumes resource/completion requirements; it does not infer concurrency.
"""

from __future__ import annotations

from dataclasses import dataclass

from merlin.compiler.dialects.lowering.arena_plan import (
    ARENA_ALIGN,
    ArenaPlanError,
    MemoryPlan,
    _assert_reads_are_complete,
    _buf_bytes,
    pack_disjoint,
)
from merlin.compiler.dialects.lowering.dispatch_program import DispatchProgram


@dataclass(frozen=True)
class ArenaRequirements:
    """Explicit graph-level storage and completion requirements.

    Events are ordered application events, not guessed accelerator cycles.
    Aliases are caller-provided physical views; this planner checks bounds and
    lifetime retention but does not prove the source view's numerical meaning.
    Persistent buffers remain reserved throughout one invocation. The caller
    separately owns initialization and state retention across invocations.
    """

    capacity_bytes: int
    launch_events: tuple[int, ...]
    completion_events: tuple[int, ...]
    alignment: int = ARENA_ALIGN
    aliases: tuple[tuple[str, str, int], ...] = ()
    persistent: tuple[str, ...] = ()
    reserved: tuple[tuple[int, int], ...] = ()
    dispatch_digest: str = ""
    physical_sizes: tuple[tuple[str, int], ...] = ()


def _bounded_arena_inputs(prog: DispatchProgram, requirements: ArenaRequirements):
    """Recover storage and lifetime obligations from the current graph."""
    from merlin.compiler.dialects.lowering.dispatch_program import verify_program
    from merlin.compiler.dialects.lowering.global_plan_emission import dispatch_digest
    from merlin.compiler.semantic.allocate import Reservation, StorageBank
    from merlin.compiler.semantic.buffer_allocation import Buffer

    if requirements.dispatch_digest != dispatch_digest(prog):
        raise ArenaPlanError("memory requirements do not match the emitted dispatch graph")
    problems = verify_program(prog)
    if problems:
        raise ArenaPlanError("invalid dispatch graph: " + "; ".join(problems))
    _assert_reads_are_complete(prog)
    if (
        type(requirements.capacity_bytes) is not int
        or requirements.capacity_bytes <= 0
        or type(requirements.alignment) is not int
        or requirements.alignment <= 0
    ):
        raise ArenaPlanError("arena needs positive capacity and alignment")
    launches, completions = requirements.launch_events, requirements.completion_events
    if (
        len(launches) != len(prog.nodes)
        or len(completions) != len(prog.nodes)
        or any(type(n) is not int or n < 0 for n in (*launches, *completions))
        or any(a > b for a, b in zip(launches, launches[1:]))
        or any(start > end for start, end in zip(launches, completions))
    ):
        raise ArenaPlanError("each dispatch needs valid ordered launch and completion events")
    sizes = {name: _buf_bytes(buffer.shape, buffer.dtype) for name, buffer in prog.buffers.items()}
    physical_sizes = dict(requirements.physical_sizes)
    if (
        len(physical_sizes) != len(requirements.physical_sizes)
        or not set(physical_sizes) <= set(sizes)
        or any(type(size) is not int or size < sizes[name] for name, size in physical_sizes.items())
    ):
        raise ArenaPlanError("physical sizes must name buffers and cover their logical storage")
    sizes.update(physical_sizes)
    external = {
        name for name, buffer in prog.buffers.items() if buffer.kind in ("arg", "const") or name in prog.results
    }
    owned = set(prog.buffers) - external
    aliases = {name: (parent, offset) for name, parent, offset in requirements.aliases}
    if (
        len(aliases) != len(requirements.aliases)
        or not set(aliases) <= owned
        or any(parent not in owned for parent, _ in aliases.values())
        or not set(requirements.persistent) <= owned
    ):
        raise ArenaPlanError("arena aliases and persistent buffers must name owned intermediates")
    first, last, ready, producers = {}, {}, {}, {}
    for index, node in enumerate(prog.nodes):
        for name in node.inputs:
            if name in ready and ready[name] > launches[index]:
                raise ArenaPlanError("consumer launches before its input producer completes")
            last[name] = max(last.get(name, 0), completions[index])
        for name in node.outputs:
            if name in first or prog.buffers[name].kind == "arg":
                raise ArenaPlanError("dispatch buffer has more than one definition")
            first[name], ready[name] = launches[index], completions[index]
            producers[name] = node
            last[name] = max(last.get(name, 0), completions[index])
    for name, (parent, _) in aliases.items():
        producer = producers.get(name)
        if producer is None or producer.kind != "view" or parent not in producer.inputs:
            raise ArenaPlanError("declared alias must be a view of its source input")
    requests = []
    for name, buffer in prog.buffers.items():
        if name not in owned:
            continue
        if name not in first:
            raise ArenaPlanError("intermediate has no producing dispatch")
        parent, offset = aliases.get(name, (None, 0))
        requests.append(
            Buffer(
                name,
                "arena",
                sizes[name],
                first[name],
                last[name],
                alignment=requirements.alignment if parent is None else 1,
                view_of=parent,
                view_offset=offset,
                persistent=name in requirements.persistent,
            )
        )
    return (
        sizes,
        tuple(requests),
        (StorageBank("arena", "arena", requirements.capacity_bytes, "byte"),),
        tuple(Reservation("arena", start, extent) for start, extent in requirements.reserved),
    )


def _bounded_arena_stats(prog, requirements, requests, arena_bytes):
    return {
        "n_intermediate_buffers": len(requests),
        "naive_total_bytes": sum(item.extent for item in requests),
        "arena_bytes": arena_bytes,
        "reuse_factor": round(sum(item.extent for item in requests) / arena_bytes, 2) if arena_bytes else 0.0,
        "n_result_buffers": len(prog.results),
        "align": requirements.alignment,
        "lifetime_basis": "explicit launch/completion events",
        "launch_events": list(requirements.launch_events),
        "completion_events": list(requirements.completion_events),
        "aliases": [list(row) for row in requirements.aliases],
        "persistent": list(requirements.persistent),
        "reserved": [list(row) for row in requirements.reserved],
        "capacity_bytes": requirements.capacity_bytes,
        "dispatch_digest": requirements.dispatch_digest,
        "physical_sizes": dict(requirements.physical_sizes),
        "execution_scope": "placement only; target lowering and runtime binding remain required",
    }


def check_bounded_arena(prog: DispatchProgram, requirements: ArenaRequirements, memory: MemoryPlan) -> None:
    """Check an emitted placement directly; do not invoke the allocator."""
    from merlin.compiler.semantic.buffer_allocation import check_buffer_allocation

    sizes, requests, banks, reservations = _bounded_arena_inputs(prog, requirements)
    if memory.sizes != sizes:
        raise ArenaPlanError("memory plan sizes differ from the current graph and physical footprints")
    if type(memory.arena_bytes) is not int or not 0 <= memory.arena_bytes <= requirements.capacity_bytes:
        raise ArenaPlanError("memory plan exceeds its arena capacity")
    check_buffer_allocation(requests, banks, memory.offsets, reservations=reservations)
    high_water = max(
        [memory.offsets[item.name] + item.extent for item in requests]
        + [item.start + item.extent for item in reservations],
        default=0,
    )
    if high_water > memory.arena_bytes:
        raise ArenaPlanError("memory plan offsets exceed the emitted arena extent")
    if memory.stats != _bounded_arena_stats(prog, requirements, requests, memory.arena_bytes):
        raise ArenaPlanError("memory plan statistics differ from the current requirements")


def plan_bounded_arena(prog: DispatchProgram, requirements: ArenaRequirements) -> MemoryPlan:
    """Place completion-aware temporaries; leave external buffers caller-owned."""
    from merlin.compiler.semantic.buffer_allocation import BufferAllocation, allocate_buffers, check_buffer_allocation

    sizes, requests, banks, reservations = _bounded_arena_inputs(prog, requirements)
    result = None
    # Reuse the existing deterministic planner for its supported case. Check
    # its witness against the completion-aware constraints before accepting it.
    # A failed heuristic is inconclusive; the finite-domain solver still runs.
    if not reservations and all(
        item.view_of is None and item.fixed_start is None and ARENA_ALIGN % item.alignment == 0 for item in requests
    ):
        conflicts = {
            item.name: {
                other.name
                for other in requests
                if item.name != other.name
                and (
                    item.persistent
                    or other.persistent
                    or item.first_event <= other.last_event
                    and other.first_event <= item.last_event
                )
            }
            for item in requests
        }
        offsets, _ = pack_disjoint([(item.name, item.extent) for item in requests], conflicts)
        try:
            check_buffer_allocation(requests, banks, offsets, reservations=reservations)
        except ValueError:
            pass
        else:
            result = BufferAllocation("sat", tuple(offsets.items()))
    if result is None:
        result = allocate_buffers(requests, banks, reservations=reservations)
    if result.status != "sat":
        raise ArenaPlanError(f"bounded arena allocation {result.status}: {result.reason}")
    offsets = dict(result.addresses)
    arena_bytes = max(
        [offset + sizes[name] for name, offset in offsets.items()]
        + [start + extent for start, extent in requirements.reserved],
        default=0,
    )
    memory = MemoryPlan(
        arena_bytes,
        offsets,
        sizes,
        _bounded_arena_stats(prog, requirements, requests, arena_bytes),
    )
    check_bounded_arena(prog, requirements, memory)
    return memory
