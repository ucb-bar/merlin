"""Plan direct producer-to-consumer handoffs in a whole-model dispatch DAG.

The program supplies def-use and liveness facts. A selected target supplies the
physical handoff, capacity, numerical-equivalence, and cost facts. This pass
only records a plan; a backend must report that it realized every selected
handoff before the optimized program can be used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .dispatch_program import Buffer, DispatchProgram, Node, verify_program


@dataclass(frozen=True)
class HandoffFacts:
    """Target evidence for keeping one logical buffer between two dispatches.

    ``equivalent_to_materialized`` means the direct path produces the same
    values as the existing materialized path under the selected numeric policy.
    A new rounding policy must be qualified separately, not enabled here.
    """

    interface: str
    resident_bytes: int
    capacity_bytes: int
    equivalent_to_materialized: bool
    saved_cycles: int


@dataclass(frozen=True)
class Handoff:
    producer: int
    consumer: int
    buffer: str
    interface: str
    saved_cycles: int


@dataclass(frozen=True)
class ConnectedPlan:
    handoffs: tuple[Handoff, ...]
    groups: tuple[tuple[int, ...], ...]


HandoffProvider = Callable[[Node, Node, Buffer], HandoffFacts | None]


def plan_connected_dispatches(program: DispatchProgram, facts_for: HandoffProvider) -> ConnectedPlan:
    """Select adjacent, single-consumer dispatch chains with proven direct handoffs.

    Adjacent nodes are intentional: the dispatch DAG does not describe effects
    inside kernels, so moving an intervening operation would need another
    proof. Region-carrying nodes are excluded for the same reason.
    """
    problems = verify_program(program)
    if problems:
        raise ValueError("invalid dispatch program: " + "; ".join(problems))

    producers: dict[str, int] = {}
    uses: dict[str, list[int]] = {}
    for index, node in enumerate(program.nodes):
        for buffer in node.outputs:
            if buffer not in program.buffers:
                raise ValueError(f"node {index} produces unknown buffer {buffer}")
            if buffer in producers:
                raise ValueError(f"buffer {buffer} has multiple producers")
            producers[buffer] = index
        for buffer in node.inputs:
            uses.setdefault(buffer, []).append(index)

    selected: list[Handoff] = []
    for index, (producer, consumer) in enumerate(zip(program.nodes, program.nodes[1:])):
        if producer.kind != "dispatch" or consumer.kind != "dispatch" or producer.regions or consumer.regions:
            continue
        # Multi-result kernels need a bundle-wide capacity and lifetime proof.
        # A separate proof for each output could overcommit the same store.
        if len(producer.outputs) != 1:
            continue
        for buffer_id in producer.outputs:
            if (
                buffer_id in program.results
                or uses.get(buffer_id) != [index + 1]
                or program.buffers[buffer_id].kind != "intermediate"
            ):
                continue
            buffer = program.buffers[buffer_id]
            facts = facts_for(producer, consumer, buffer)
            if facts is None:
                continue
            if not facts.interface:
                raise ValueError(f"handoff for {buffer_id} has no interface")
            if facts.resident_bytes <= 0 or facts.capacity_bytes <= 0:
                raise ValueError(f"handoff for {buffer_id} has invalid capacity evidence")
            if (
                facts.resident_bytes > facts.capacity_bytes
                or not facts.equivalent_to_materialized
                or facts.saved_cycles <= 0
            ):
                continue
            selected.append(Handoff(index, index + 1, buffer_id, facts.interface, facts.saved_cycles))

    # Maximal chains provide a dispatch-grouping decision; each edge still has
    # its own capability and can be audited independently by the backend.
    groups: list[tuple[int, ...]] = []
    if selected:
        chain = [selected[0].producer, selected[0].consumer]
        for handoff in selected[1:]:
            if handoff.producer == chain[-1]:
                chain.append(handoff.consumer)
            else:
                groups.append(tuple(chain))
                chain = [handoff.producer, handoff.consumer]
        groups.append(tuple(chain))
    return ConnectedPlan(tuple(selected), tuple(groups))


def require_realized(plan: ConnectedPlan, realized: set[tuple[int, int, str]]) -> None:
    """Refuse a backend that silently ignores a selected direct handoff."""
    expected = {(h.producer, h.consumer, h.buffer) for h in plan.handoffs}
    missing = expected - realized
    if missing:
        raise ValueError(f"selected handoffs not realized by backend: {sorted(missing)}")
    extra = realized - expected
    if extra:
        raise ValueError(f"backend realized handoffs absent from plan: {sorted(extra)}")
