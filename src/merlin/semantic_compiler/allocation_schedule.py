"""Native command ordering, storage alias legality and live ranges."""
from __future__ import annotations

from collections.abc import Iterator
from itertools import chain

from .allocation_types import CandidateGraph, StorageBank, Value
from .instruction import AddressConstraint

def _instruction_predecessors(graph: CandidateGraph) -> dict[int, frozenset[int]]:
    """Find producers through non-emitting nodes as well as direct operands."""
    predecessors: dict[int, frozenset[int]] = {}
    for value in graph.values:
        if value.id < 0 or value.id >= len(graph.values) or graph.value(value.id) is not value:
            raise ValueError("candidate graph value IDs differ from their positions")
        ancestors: set[int] = set()
        for child_id in value.children:
            if child_id < 0 or child_id >= value.id:
                raise ValueError("candidate graph is not in topological value order")
            ancestors.update(predecessors[child_id])
            if graph.value(child_id).kind in {"instruction", "event"}:
                ancestors.add(child_id)
        if value.kind == "projection":
            if value.projection_of is None or value.projection_of >= value.id:
                raise ValueError("multi-result projection has no earlier instruction owner")
            ancestors.add(value.projection_of)
        predecessors[value.id] = frozenset(ancestors)
    return predecessors


def _defer_boundary_copies(graph, order, predecessors):
    """Shorten loaded-value lifetimes without changing computational order.

    Only declared value-preserving moves from boundary values qualify. Their
    storage/address constraints are still solved for the resulting order. A
    failed proposal neither prunes the original order nor proves infeasibility.
    """
    copies = tuple(
        key
        for key in order
        if graph.value(key).value_preserving_copy
        and graph.value(graph.value(key).children[0]).kind in {"input", "constant"}
    )
    pending, result = set(copies), []
    for key in order:
        if key in copies:
            continue
        for load in copies:
            if load in pending and load in predecessors[key]:
                result.append(load)
                pending.remove(load)
        result.append(key)
    result.extend(load for load in copies if load in pending)
    return tuple(result)


def topological_orders(graph: CandidateGraph, *, limit: int) -> Iterator[tuple[int, ...]]:
    """Enumerate bounded legal instruction orders after a low-pressure first order."""
    if limit <= 0:
        raise ValueError("order limit must be positive")
    instructions = {value.id for value in graph.values if value.kind in {"instruction", "event"}}
    predecessors = _instruction_predecessors(graph)
    emitted = 0

    def walk(done: tuple[int, ...], pending: frozenset[int]) -> Iterator[tuple[int, ...]]:
        nonlocal emitted
        if emitted >= limit:
            return
        if not pending:
            emitted += 1
            yield done
            return
        ready = [item for item in pending if not predecessors[item].intersection(pending)]
        # Prefer consuming larger inputs, reducing the expected live set.
        ready.sort(key=lambda item: (-sum(graph.value(c).extent for c in graph.value(item).children), item))
        for item in ready:
            if emitted >= limit:
                return
            yield from walk((*done, item), pending - {item})

    orders = walk((), frozenset(instructions))
    first = next(orders, None)
    if first is None:
        return
    deferred = _defer_boundary_copies(graph, first, predecessors)
    seen = set()
    for order in chain((first, deferred), orders):
        if order in seen:
            continue
        seen.add(order)
        yield order
        if len(seen) == limit:
            return


def instruction_schedule(graph: CandidateGraph, order: tuple[int, ...]) -> dict[int, tuple[int, int]]:
    """Issue serially until target timing has a reviewed overlap contract."""
    instructions = {value.id for value in graph.values if value.kind in {"instruction", "event"}}
    if len(order) != len(instructions) or set(order) != instructions:
        raise ValueError("order omits or duplicates an instruction")
    predecessors = _instruction_predecessors(graph)
    position = {value_id: index for index, value_id in enumerate(order)}
    if any(
        position[producer] >= position[value.id]
        for value in graph.values
        if value.kind in {"instruction", "event"}
        for producer in predecessors[value.id]
    ):
        raise ValueError("order executes a consumer before its instruction producer")
    schedule: dict[int, tuple[int, int]] = {}
    last_completion = 0
    for value_id in order:
        value = graph.value(value_id)
        if value.completion_offset < 0 or (
            value.input_read_offsets and len(value.input_read_offsets) != len(value.children)
        ):
            raise ValueError("instruction has malformed execution timing")
        if any(
            value.read_offset(index) < 0 or value.read_offset(index) > value.completion_offset
            for index in range(len(value.children))
        ):
            raise ValueError("instruction reads an input outside its execution interval")
        issue = last_completion + 1
        last_completion = issue + value.completion_offset
        schedule[value_id] = (issue, last_completion)
    return schedule


def live_ranges(graph: CandidateGraph, order: tuple[int, ...]) -> dict[int, tuple[int, int]]:
    schedule = instruction_schedule(graph, order)
    available: dict[int, int] = {}
    ends: dict[int, int] = {}
    finish = max((completion for _, completion in schedule.values()), default=0) + 1
    for value in graph.values:
        if value.kind in {"instruction", "event"}:
            ready = schedule[value.id][1]
        elif value.kind == "projection" and value.projection_of is not None:
            ready = schedule[value.projection_of][1]
        else:
            ready = max((available[child] for child in value.children), default=0)
        available[value.id] = ready
        ends[value.id] = ready
        if value.id in graph.outputs:
            ends[value.id] = finish
    for parent in graph.values:
        if parent.kind in {"instruction", "event"}:
            for index, child in enumerate(parent.children):
                ends[child] = max(ends[child], schedule[parent.id][0] + parent.read_offset(index))
    # A view or other non-emitting dependency has no issue time. Its source
    # stays live until every consumer of that dependency has read it.
    for parent in reversed(graph.values):
        if parent.kind not in {"instruction", "event", "projection"}:
            for child in parent.children:
                ends[child] = max(ends[child], ends[parent.id])
    # A multi-result bundle may write any of its results anywhere in its
    # execution interval, so every projection reserves storage from the shared
    # issue time, while consumers still see it only at completion.
    starts = {
        value.id: (
            schedule[value.id][0]
            if value.kind in {"instruction", "event"}
            else schedule[value.projection_of][0]
            if value.kind == "projection" and value.projection_of is not None
            else available[value.id]
        )
        for value in graph.values
    }
    return {value.id: (starts[value.id], ends[value.id]) for value in graph.values}


def _qualified_in_place_pair(graph: CandidateGraph, left: Value, right: Value) -> bool:
    """Permit exact reuse only for a declared, last-use produced operand."""
    for child, parent in ((left, right), (right, left)):
        if child.kind != "instruction" or parent.kind != "instruction" or child.id in graph.outputs:
            continue
        if child.extent != parent.extent or parent.children.count(child.id) != 1:
            continue
        port = parent.children.index(child.id)
        if port not in parent.in_place_inputs or parent.read_offset(port) >= parent.completion_offset:
            continue
        if sum(value.children.count(child.id) for value in graph.values) != 1:
            continue
        return True
    return False


def _view_alias_roots(graph: CandidateGraph) -> dict[int, int]:
    roots: dict[int, int] = {}
    for value in graph.values:
        roots[value.id] = roots[value.children[0]] if value.kind == "view" else value.id
    return roots


def _direct_view_alias(left: Value, right: Value) -> bool:
    """A view intentionally overlaps each of its physical source spans."""
    return left.kind == "view" and right.id in left.children or right.kind == "view" and left.id in right.children


def interference_edges(
    graph: CandidateGraph,
    order: tuple[int, ...],
    banks: tuple[StorageBank, ...],
) -> frozenset[tuple[int, int]]:
    """Canonical storage-conflict edges; order affects only this part of our formula."""
    bank_map = _geometry(banks)
    view_problem = _view_problem(graph)
    if view_problem:
        raise ValueError(view_problem)
    alias_roots = _view_alias_roots(graph)
    ranges = live_ranges(graph, order)
    edges = set()
    for left_index, left in enumerate(graph.values):
        for right in graph.values[left_index + 1 :]:
            if left.extent == 0 or right.extent == 0:
                continue
            if bank_map[left.storage].backing != bank_map[right.storage].backing:
                continue
            if alias_roots[left.id] == alias_roots[right.id] or _direct_view_alias(left, right):
                continue
            lo1, hi1 = ranges[left.id]
            lo2, hi2 = ranges[right.id]
            if lo1 <= hi2 and lo2 <= hi1 and not _qualified_in_place_pair(graph, left, right):
                edges.add((left.id, right.id))
    return frozenset(edges)


def _geometry(banks: tuple[StorageBank, ...]) -> dict[str, StorageBank]:
    by_name = {bank.name: bank for bank in banks}
    if len(by_name) != len(banks):
        raise ValueError("duplicate storage bank")
    by_backing: dict[str, str] = {}
    for bank in banks:
        existing = by_backing.setdefault(bank.backing, bank.unit)
        if existing != bank.unit:
            raise ValueError("typed aliases need one explicit common address unit")
    return by_name


def _view_problem(graph: CandidateGraph) -> str:
    for index, value in enumerate(graph.values):
        if value.id != index:
            return "candidate graph value IDs differ from their positions"
        if value.kind not in {"input", "constant", "instruction", "event", "projection", "view"}:
            return "candidate contains an unselected semantic value"
        if value.kind == "projection":
            if (
                value.projection_of is None
                or value.projection_of >= value.id
                or graph.value(value.projection_of).kind != "instruction"
                or value.instruction_instance is None
                or value.instruction_instance != graph.value(value.projection_of).instruction_instance
                or value.children != graph.value(value.projection_of).children
                or value.input_read_offsets
                or value.completion_offset
                or value.in_place_inputs
            ):
                return "multi-result projection differs from its shared instruction"
        if value.kind in {"input", "constant"} and value.children:
            return "candidate boundary value has unexpected dependencies"
        if value.kind == "view":
            if not value.children or any(child < 0 or child >= value.id for child in value.children):
                return "non-emitting view has no earlier source value"
            sources = tuple(graph.value(child) for child in value.children)
            offset = value.validity[0].value if value.validity else None
            if len(sources) == 1:
                expected = (AddressConstraint("eq_offset", "out", "in0", offset),) if type(offset) is int else ()
                geometry_ok = type(offset) is int and offset >= 0 and offset + value.extent <= sources[0].extent
            else:
                prefix = 0
                expected_list = [AddressConstraint("eq_offset", "out", "in0", 0)]
                for index, source in enumerate(sources):
                    if index:
                        expected_list.append(AddressConstraint("eq_offset", f"in{index}", "out", prefix))
                    prefix += source.extent
                expected = tuple(expected_list)
                geometry_ok = value.extent == prefix
            if (
                any(value.storage != source.storage for source in sources)
                or not geometry_ok
                or value.validity != expected
                or value.input_read_offsets
                or value.completion_offset
                or value.in_place_inputs
            ):
                return "non-emitting view lacks a checked contiguous address relation"
    return ""

