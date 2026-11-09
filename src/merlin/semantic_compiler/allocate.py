"""Native finite-domain allocation with an independent witness checker.

Z3 solves Merlin's generated placement formula. The checker below evaluates
the selected graph and storage geometry directly without reusing the formula.
An UNSAT result applies only to one candidate, order and bounded model.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import chain
from typing import TYPE_CHECKING

from .allocation_schedule import (
    _defer_boundary_copies,
    _direct_view_alias,
    _geometry,
    _instruction_predecessors,
    _qualified_in_place_pair,
    _view_alias_roots,
    _view_problem,
    instruction_schedule,
    interference_edges,
    live_ranges,
    topological_orders,
)
from .allocation_types import AllocationResult, CandidateGraph, Reservation, StorageBank, Value

if TYPE_CHECKING:
    from .extract import Candidate, Choice
from .rules import AddressConstraint

if TYPE_CHECKING:
    from .rules import RuleProgram


def lower_candidate(candidate: Candidate, program: RuleProgram) -> CandidateGraph:
    if not candidate.bundles_complete():
        raise ValueError("multi-result instruction selection omitted or changed a projection")
    values: list[Value] = []
    ids: dict[tuple, int] = {}
    leaders: dict[tuple, Choice] = {}

    def collect(choice: Choice) -> None:
        instance = choice.physical_instance()
        if instance is not None and choice.bundle_result_index == 0:
            leaders[instance] = choice
        for child in choice.children:
            collect(child)

    for output in candidate.outputs:
        collect(output)

    def visit(choice: Choice) -> int:
        key = choice.key()
        if key in ids:
            return ids[key]
        instance = choice.physical_instance()
        owner = None
        if instance is not None and choice.bundle_result_index != 0:
            owner = visit(leaders[instance])
        children = tuple(visit(child) for child in choice.children)
        metadata = program.symbols[choice.symbol]
        kind = "projection" if owner is not None else metadata["kind"]
        extent = (
            metadata["descriptor"]["extent"]
            if kind in {"instruction", "projection", "view"}
            else (metadata.get("extent"))
        )
        if type(extent) is not int or extent <= 0:
            raise ValueError("selected value has no positive physical extent")
        if kind in {"instruction", "projection"}:
            validity = tuple(AddressConstraint(**row) for row in metadata["descriptor"]["validity"])
        elif kind == "view":
            descriptor = metadata["descriptor"]
            if descriptor.get("kind") == "concat":
                prefix = 0
                conditions = [AddressConstraint("eq_offset", "out", "in0", 0)]
                for index, source_extent in enumerate(descriptor["source_extents"]):
                    if index:
                        conditions.append(AddressConstraint("eq_offset", f"in{index}", "out", prefix))
                    prefix += source_extent
                validity = tuple(conditions)
            else:
                validity = (AddressConstraint("eq_offset", "out", "in0", descriptor.get("physical_offset", 0)),)
        else:
            validity = ()
        value_id = len(values)
        ids[key] = value_id
        values.append(
            Value(
                id=value_id,
                symbol=choice.symbol,
                storage=choice.storage,
                extent=extent,
                children=children,
                source_node=metadata.get("source_node"),
                kind=kind,
                validity=validity,
                input_read_offsets=tuple(metadata["descriptor"]["input_read_offsets"]) if kind == "instruction" else (),
                completion_offset=int(metadata["descriptor"]["completion_offset"]) if kind == "instruction" else 0,
                in_place_inputs=tuple(metadata["descriptor"]["in_place_inputs"]) if kind == "instruction" else (),
                preserve_input=metadata.get("preserve_input", True),
                instruction_instance=instance,
                projection_of=owner,
                value_preserving_copy=kind == "instruction"
                and metadata["descriptor"].get("value_preserving_copy", False),
            )
        )
        return value_id

    outputs = tuple(visit(choice) for choice in candidate.outputs)
    return CandidateGraph(tuple(values), outputs)


def _address_unit_problem(graph: CandidateGraph, bank_map: dict[str, StorageBank]) -> str:
    for value in graph.values:
        if value.storage not in bank_map:
            return "unknown storage bank"
        ports = {
            "out": value.storage,
            **{f"in{index}": graph.value(child).storage for index, child in enumerate(value.children)},
        }
        for condition in value.validity:
            if condition.kind == "eq_offset" and (
                bank_map[ports[condition.lhs]].unit != bank_map[ports[condition.rhs]].unit
            ):
                return "address relation mixes physical units"
    return ""


def _reservation_problem(reservations: tuple[Reservation, ...], bank_map: dict[str, StorageBank]) -> str:
    for index, reservation in enumerate(reservations):
        bank = bank_map.get(reservation.storage)
        if bank is None:
            return "reservation names an unknown storage bank"
        if reservation.start < bank.start or reservation.start + reservation.extent > bank.capacity:
            return "reservation exceeds its physical storage bank"
        for earlier in reservations[:index]:
            other = bank_map[earlier.storage]
            if bank.backing == other.backing and (
                reservation.start < earlier.start + earlier.extent
                and earlier.start < reservation.start + reservation.extent
            ):
                return "reservations overlap through physical aliases"
    return ""


def _boundary_problem(
    graph: CandidateGraph,
    banks: dict[str, StorageBank],
    fixed_inputs: dict[str, int],
    fixed_outputs: tuple[int | None, ...] | None,
) -> str:
    # A declared constant is an initialized boundary value too. Its external
    # address may be fixed by the materializer before selection, just like an
    # invocation input; the payload itself remains a compiler input.
    boundaries = {value.source_node: value for value in graph.values if value.kind in {"input", "constant"}}
    for name, address in fixed_inputs.items():
        if name not in boundaries:
            return "fixed boundary ABI names an unknown source"
        if type(address) is not int or address < 0:
            return "fixed boundary ABI has an invalid address"
        value = boundaries[name]
        bank = banks.get(value.storage)
        if bank is not None and (
            address < bank.start or address + value.extent > bank.capacity or address % bank.alignment
        ):
            return "fixed boundary ABI exceeds physical storage"
    if fixed_outputs is None:
        return ""
    if len(fixed_outputs) != len(graph.outputs):
        return "fixed output ABI differs from ordered roots"
    for value_id, address in zip(graph.outputs, fixed_outputs):
        if address is None:
            continue
        if type(address) is not int or address < 0:
            return "fixed output ABI has an invalid address"
        value = graph.value(value_id)
        bank = banks.get(value.storage)
        if bank is not None and (
            address < bank.start or address + value.extent > bank.capacity or address % bank.alignment
        ):
            return "fixed output ABI exceeds physical storage"
    return ""


def check_assignment(
    graph: CandidateGraph,
    order: tuple[int, ...],
    addresses: dict[int, int],
    banks: tuple[StorageBank, ...],
    *,
    fixed_inputs: dict[str, int] | None = None,
    reservations: tuple[Reservation, ...] = (),
    fixed_outputs: tuple[int | None, ...] | None = None,
) -> tuple[bool, str]:
    """Recompute original geometry/lifetimes without consulting Z3 expressions."""
    bank_map = _geometry(banks)
    view_problem = _view_problem(graph)
    if view_problem:
        return False, view_problem
    alias_roots = _view_alias_roots(graph)
    unit_problem = _address_unit_problem(graph, bank_map)
    if unit_problem:
        return False, unit_problem
    reservation_problem = _reservation_problem(reservations, bank_map)
    if reservation_problem:
        return False, reservation_problem
    fixed_inputs = fixed_inputs or {}
    boundary_problem = _boundary_problem(graph, bank_map, fixed_inputs, fixed_outputs)
    if boundary_problem:
        return False, boundary_problem
    try:
        ranges = live_ranges(graph, order)
    except ValueError as exc:
        return False, str(exc)
    if set(addresses) != {value.id for value in graph.values}:
        return False, "assignment omits a value"
    if fixed_outputs is not None and any(
        address is not None and addresses[value_id] != address
        for value_id, address in zip(graph.outputs, fixed_outputs)
    ):
        return False, "output moved from fixed external address"
    for value in graph.values:
        bank = bank_map.get(value.storage)
        if bank is None:
            return False, "unknown storage bank"
        address = addresses[value.id]
        if not isinstance(address, int) or isinstance(address, bool):
            return False, "non-integer address"
        if (
            address < (bank.start if value.extent else 0)
            or address + value.extent > bank.capacity
            or address % bank.alignment
        ):
            return False, "out-of-range or misaligned address"
        if value.kind == "view":
            offset = value.validity[0].value
            if address != addresses[value.children[0]] + offset:
                return False, (
                    "concat view moved from its first source address"
                    if len(value.children) > 1
                    else "identity view moved from its source address"
                    if offset == 0
                    else "slice view has wrong source offset"
                )
        for reservation in reservations:
            if bank.backing != bank_map[reservation.storage].backing:
                continue
            if (
                value.extent
                and address < reservation.start + reservation.extent
                and reservation.start < address + value.extent
            ):
                return False, "assignment overlaps reserved physical storage"
        if value.source_node in fixed_inputs and value.kind in {"input", "constant"}:
            if address != fixed_inputs[value.source_node]:
                return False, "boundary moved from fixed external address"
        for child_id in () if value.kind == "projection" else value.children:
            child = graph.value(child_id)
            if child.id not in ranges or ranges[child.id][1] < ranges[value.id][0]:
                return False, "consumer reads a dead value"
        ports = {"out": address, **{f"in{index}": addresses[child] for index, child in enumerate(value.children)}}
        for condition in value.validity:
            if condition.kind == "eq_offset" and ports[condition.lhs] != ports[condition.rhs] + condition.value:
                return False, "instruction address map or validity failed"
            if condition.kind == "aligned" and ports[condition.lhs] % condition.value:
                return False, "instruction alignment validity failed"
            if condition.kind == "at_least" and ports[condition.lhs] < condition.value:
                return False, "instruction lower address bound failed"
            if condition.kind == "at_most" and ports[condition.lhs] > condition.value:
                return False, "instruction upper address bound failed"
    for left_index, left in enumerate(graph.values):
        for right in graph.values[left_index + 1 :]:
            if left.extent == 0 or right.extent == 0:
                continue
            if bank_map[left.storage].backing != bank_map[right.storage].backing:
                continue
            a = addresses[left.id]
            b = addresses[right.id]
            overlap = a < b + right.extent and b < a + left.extent
            if overlap and (
                (left.kind == "input" and left.preserve_input and right.kind in {"instruction", "projection"})
                or (right.kind == "input" and right.preserve_input and left.kind in {"instruction", "projection"})
            ):
                return False, "boundary input overlaps an instruction write"
            lo1, hi1 = ranges[left.id]
            lo2, hi2 = ranges[right.id]
            if overlap and lo1 <= hi2 and lo2 <= hi1:
                if alias_roots[left.id] == alias_roots[right.id] or _direct_view_alias(left, right):
                    continue
                # Replay the exact-reuse exception without relying on solver constraints.
                allowed = False
                for child, parent in ((left, right), (right, left)):
                    if child.kind != "instruction" or parent.kind != "instruction":
                        continue
                    if child.id in graph.outputs or child.extent != parent.extent:
                        continue
                    if parent.children.count(child.id) != 1:
                        continue
                    port = parent.children.index(child.id)
                    if port not in parent.in_place_inputs or parent.read_offset(port) >= parent.completion_offset:
                        continue
                    if sum(value.children.count(child.id) for value in graph.values) != 1:
                        continue
                    allowed = a == b
                if not allowed:
                    return False, "simultaneously live physical views overlap"
    return True, ""


def _live_capacity_problem(
    graph: CandidateGraph,
    bank_map: dict[str, StorageBank],
    ranges: dict[int, tuple[int, int]],
) -> str:
    """Reject only a necessary capacity violation in one candidate and order.

    Without views or in-place permissions, simultaneously live values in a
    backing store must occupy disjoint intervals. Their total extent cannot
    exceed its largest address domain. Ignore reservations and alignment here;
    the full formula still checks those and all cases that permit aliasing.
    """
    for backing in sorted({bank.backing for bank in bank_map.values()}):
        values = [value for value in graph.values if bank_map[value.storage].backing == backing]
        if any(value.kind == "view" or value.in_place_inputs for value in values):
            continue
        capacity = max(bank.capacity for bank in bank_map.values() if bank.backing == backing)
        events: dict[int, int] = {}
        for value in values:
            start, end = ranges[value.id]
            events[start] = events.get(start, 0) + value.extent
            events[end + 1] = events.get(end + 1, 0) - value.extent
        live = 0
        for time in sorted(events):
            live += events[time]
            if live > capacity:
                return f"live disjoint storage in {backing} needs {live} units, capacity is {capacity}"
    return ""


def allocate(
    graph: CandidateGraph,
    order: tuple[int, ...],
    banks: tuple[StorageBank, ...],
    *,
    fixed_inputs: dict[str, int] | None = None,
    reservations: tuple[Reservation, ...] = (),
    fixed_outputs: tuple[int | None, ...] | None = None,
    timeout_ms: int = 5000,
) -> AllocationResult:
    if timeout_ms <= 0:
        raise ValueError("solver timeout must be positive")
    try:
        import z3
    except ImportError as exc:
        return AllocationResult("tool_unavailable", order, {}, f"z3-solver missing: {exc}")
    bank_map = _geometry(banks)
    view_problem = _view_problem(graph)
    if view_problem:
        return AllocationResult("modeling_failure", order, {}, view_problem)
    alias_roots = _view_alias_roots(graph)
    unit_problem = _address_unit_problem(graph, bank_map)
    if unit_problem:
        return AllocationResult("unqualified_target", order, {}, unit_problem)
    reservation_problem = _reservation_problem(reservations, bank_map)
    if reservation_problem:
        return AllocationResult("unqualified_target", order, {}, reservation_problem)
    fixed_inputs = fixed_inputs or {}
    boundary_problem = _boundary_problem(graph, bank_map, fixed_inputs, fixed_outputs)
    if boundary_problem:
        return AllocationResult("modeling_failure", order, {}, boundary_problem)
    ranges = live_ranges(graph, order)
    capacity_problem = _live_capacity_problem(graph, bank_map, ranges)
    if capacity_problem:
        return AllocationResult("infeasible_candidate", order, {}, capacity_problem)
    # A plain SAT model may select different valid physical registers on
    # repeated invocations. Lexicographic minimization fixes one canonical
    # assignment in value-ID order, so replayed emission has stable bytes.
    solver = z3.Optimize()
    solver.set(timeout=timeout_ms, priority="lex")
    variables = {value.id: z3.Int(f"address_{value.id}") for value in graph.values}
    if fixed_outputs is not None:
        for value_id, address in zip(graph.outputs, fixed_outputs):
            if address is not None:
                solver.add(variables[value_id] == address)
    for value in graph.values:
        bank = bank_map.get(value.storage)
        if bank is None:
            return AllocationResult("unqualified_target", order, {}, "unknown storage bank")
        addr = variables[value.id]
        solver.add(
            addr >= (bank.start if value.extent else 0),
            addr + value.extent <= bank.capacity,
            addr % bank.alignment == 0,
        )
        for reservation in reservations:
            if value.extent and bank.backing == bank_map[reservation.storage].backing:
                solver.add(
                    z3.Or(
                        addr + value.extent <= reservation.start,
                        addr >= reservation.start + reservation.extent,
                    )
                )
        if value.kind in {"input", "constant"} and value.source_node in fixed_inputs:
            solver.add(addr == fixed_inputs[value.source_node])
        if value.kind == "view":
            solver.add(addr == variables[value.children[0]] + value.validity[0].value)
        ports = {"out": addr, **{f"in{index}": variables[child] for index, child in enumerate(value.children)}}
        for condition in value.validity:
            if condition.kind == "eq_offset":
                solver.add(ports[condition.lhs] == ports[condition.rhs] + condition.value)
            elif condition.kind == "aligned":
                solver.add(ports[condition.lhs] % condition.value == 0)
            elif condition.kind == "at_least":
                solver.add(ports[condition.lhs] >= condition.value)
            elif condition.kind == "at_most":
                solver.add(ports[condition.lhs] <= condition.value)
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
            preserve_input = (
                left.kind == "input" and left.preserve_input and right.kind in {"instruction", "projection"}
            ) or (right.kind == "input" and right.preserve_input and left.kind in {"instruction", "projection"})
            if preserve_input or lo1 <= hi2 and lo2 <= hi1:
                alternatives = [
                    variables[left.id] + left.extent <= variables[right.id],
                    variables[right.id] + right.extent <= variables[left.id],
                ]
                if not preserve_input and _qualified_in_place_pair(graph, left, right):
                    alternatives.append(variables[left.id] == variables[right.id])
                solver.add(z3.Or(*alternatives))
    for value_id in sorted(variables):
        solver.minimize(variables[value_id])
    status = solver.check()
    if status == z3.unsat:
        return AllocationResult("infeasible_candidate", order, {}, "bounded placement formula is UNSAT")
    if status != z3.sat:
        return AllocationResult("search_timeout", order, {}, f"solver returned {status}: {solver.reason_unknown()}")
    model = solver.model()
    addresses = {value_id: model[variable].as_long() for value_id, variable in variables.items()}
    checked, reason = check_assignment(
        graph,
        order,
        addresses,
        banks,
        fixed_inputs=fixed_inputs,
        reservations=reservations,
        fixed_outputs=fixed_outputs,
    )
    if not checked:
        return AllocationResult("modeling_failure", order, addresses, reason)
    schedule = instruction_schedule(graph, order)
    issue_times = tuple((value_id, schedule[value_id][0]) for value_id in order)
    return AllocationResult("feasible", order, addresses, issue_times=issue_times)


def may_prune_interference(
    failed_base: str,
    current_base: str,
    failed_edges: frozenset[tuple[int, int]],
    current_edges: frozenset[tuple[int, int]],
    *,
    failed_status: str,
) -> bool:
    """Only a same-base UNSAT graph's supergraph inherits the contradiction."""
    return failed_status == "infeasible_candidate" and failed_base == current_base and failed_edges <= current_edges
