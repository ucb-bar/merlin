"""Native finite-domain allocation with an independent witness checker.

Z3 solves Merlin's generated placement formula. The checker below evaluates
the selected graph and storage geometry directly without reusing the formula.
An UNSAT result applies only to one candidate, order and bounded model.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from .extract import Candidate, Choice
from .rules import AddressConstraint, RuleProgram


@dataclass(frozen=True)
class StorageBank:
    name: str
    backing: str
    capacity: int
    unit: str
    alignment: int = 1

    def __post_init__(self) -> None:
        if not self.name or not self.backing or not self.unit or self.capacity <= 0 or self.alignment <= 0:
            raise ValueError("storage bank needs positive geometry and explicit units")

    def record(self) -> dict[str, str | int]:
        return vars(self).copy()

    @classmethod
    def from_record(cls, row: dict[str, str | int]) -> StorageBank:
        if set(row) != {"name", "backing", "capacity", "unit", "alignment"}:
            raise ValueError("storage bank has missing or unknown fields")
        return cls(**row)


@dataclass(frozen=True)
class Reservation:
    """Physical interval unavailable to every typed view of one backing store.

    The target contract names the storage view and supplies the interval in its
    explicit address unit. Reservations last for this entire candidate. Shorter
    lifetimes require a qualified temporal model, not an assumed issue order.
    """

    storage: str
    start: int
    extent: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.storage, str)
            or not self.storage
            or not isinstance(self.start, int)
            or isinstance(self.start, bool)
            or self.start < 0
            or not isinstance(self.extent, int)
            or isinstance(self.extent, bool)
            or self.extent <= 0
        ):
            raise ValueError("reservation needs a storage, nonnegative start and positive extent")


@dataclass(frozen=True)
class Value:
    id: int
    symbol: str
    storage: str
    extent: int
    children: tuple[int, ...]
    source_node: str | None
    kind: str
    validity: tuple[AddressConstraint, ...] = ()
    input_read_offsets: tuple[int, ...] = ()
    completion_offset: int = 0
    in_place_inputs: tuple[int, ...] = ()
    # A source input is immutable across the kernel unless its boundary says reusable.
    preserve_input: bool = True

    def __post_init__(self) -> None:
        if type(self.preserve_input) is not bool or (self.kind != "input" and not self.preserve_input):
            raise ValueError("input retention must be a boolean on input values")

    def read_offset(self, index: int) -> int:
        return self.input_read_offsets[index] if self.input_read_offsets else 0


@dataclass(frozen=True)
class CandidateGraph:
    values: tuple[Value, ...]
    outputs: tuple[int, ...]

    def value(self, value_id: int) -> Value:
        return self.values[value_id]


def lower_candidate(candidate: Candidate, program: RuleProgram) -> CandidateGraph:
    values: list[Value] = []
    ids: dict[tuple, int] = {}

    def visit(choice: Choice) -> int:
        key = choice.key()
        if key in ids:
            return ids[key]
        children = tuple(visit(child) for child in choice.children)
        metadata = program.symbols[choice.symbol]
        kind = metadata["kind"]
        extent = int(metadata["descriptor"]["extent"]) if kind == "instruction" else 1
        validity = (
            tuple(AddressConstraint(**row) for row in metadata["descriptor"]["validity"])
            if kind == "instruction"
            else ()
        )
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
            )
        )
        return value_id

    outputs = tuple(visit(choice) for choice in candidate.outputs)
    return CandidateGraph(tuple(values), outputs)


def topological_orders(graph: CandidateGraph, *, limit: int) -> Iterator[tuple[int, ...]]:
    """Enumerate bounded legal instruction orders after a low-pressure first order."""
    if limit <= 0:
        raise ValueError("order limit must be positive")
    instructions = {value.id for value in graph.values if value.kind == "instruction"}
    emitted = 0

    def walk(done: tuple[int, ...], pending: frozenset[int]) -> Iterator[tuple[int, ...]]:
        nonlocal emitted
        if emitted >= limit:
            return
        if not pending:
            emitted += 1
            yield done
            return
        ready = [item for item in pending if not any(child in pending for child in graph.value(item).children)]
        # Prefer consuming larger inputs, reducing the expected live set.
        ready.sort(key=lambda item: (-sum(graph.value(c).extent for c in graph.value(item).children), item))
        for item in ready:
            if emitted >= limit:
                return
            yield from walk((*done, item), pending - {item})

    yield from walk((), frozenset(instructions))


def instruction_schedule(graph: CandidateGraph, order: tuple[int, ...]) -> dict[int, tuple[int, int]]:
    """Issue serially until target timing has a reviewed overlap contract."""
    instructions = {value.id for value in graph.values if value.kind == "instruction"}
    if len(order) != len(instructions) or set(order) != instructions:
        raise ValueError("order omits or duplicates an instruction")
    position = {value_id: index for index, value_id in enumerate(order)}
    if any(
        child in position and position[child] >= position[value.id]
        for value in graph.values
        if value.kind == "instruction"
        for child in value.children
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
    ranges: dict[int, tuple[int, int]] = {}
    for value in graph.values:
        start = schedule[value.id][0] if value.kind == "instruction" else 0
        uses = [
            schedule[parent.id][0] + parent.read_offset(index)
            for parent in graph.values
            if parent.id in schedule
            for index, child in enumerate(parent.children)
            if child == value.id
        ]
        own_completion = schedule[value.id][1] if value.kind == "instruction" else start
        end = max((*uses, own_completion))
        if value.id in graph.outputs:
            end = max((completion for _, completion in schedule.values()), default=0) + 1
        ranges[value.id] = (start, end)
    return ranges


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


def interference_edges(
    graph: CandidateGraph,
    order: tuple[int, ...],
    banks: tuple[StorageBank, ...],
) -> frozenset[tuple[int, int]]:
    """Canonical storage-conflict edges; order affects only this part of our formula."""
    bank_map = _geometry(banks)
    ranges = live_ranges(graph, order)
    edges = set()
    for left_index, left in enumerate(graph.values):
        for right in graph.values[left_index + 1 :]:
            if bank_map[left.storage].backing != bank_map[right.storage].backing:
                continue
            lo1, hi1 = ranges[left.id]
            lo2, hi2 = ranges[right.id]
            if lo1 <= hi2 and lo2 <= hi1 and not _qualified_in_place_pair(graph, left, right):
                edges.add((left.id, right.id))
    return frozenset(edges)


@dataclass(frozen=True)
class AllocationResult:
    status: str
    order: tuple[int, ...]
    addresses: dict[int, int]
    reason: str = ""
    issue_times: tuple[tuple[int, int], ...] = ()


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
        if reservation.start + reservation.extent > bank.capacity:
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
    inputs = {value.source_node: value for value in graph.values if value.kind == "input"}
    for name, address in fixed_inputs.items():
        if name not in inputs:
            return "fixed input ABI names an unknown source"
        if type(address) is not int or address < 0:
            return "fixed input ABI has an invalid address"
        value = inputs[name]
        bank = banks.get(value.storage)
        if bank is not None and (address + value.extent > bank.capacity or address % bank.alignment):
            return "fixed input ABI exceeds physical storage"
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
        if bank is not None and (address + value.extent > bank.capacity or address % bank.alignment):
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
        if address < 0 or address + value.extent > bank.capacity or address % bank.alignment:
            return False, "out-of-range or misaligned address"
        for reservation in reservations:
            if bank.backing != bank_map[reservation.storage].backing:
                continue
            if address < reservation.start + reservation.extent and reservation.start < address + value.extent:
                return False, "assignment overlaps reserved physical storage"
        if value.source_node in fixed_inputs and value.kind == "input":
            if address != fixed_inputs[value.source_node]:
                return False, "input moved from fixed external address"
        for child_id in value.children:
            child = graph.value(child_id)
            if child.id not in ranges or ranges[child.id][1] < ranges[value.id][0]:
                return False, "consumer reads a dead value"
        ports = {"out": address, **{f"in{index}": addresses[child] for index, child in enumerate(value.children)}}
        for condition in value.validity:
            if condition.kind == "eq_offset" and ports[condition.lhs] != ports[condition.rhs] + condition.value:
                return False, "instruction address map or validity failed"
            if condition.kind == "aligned" and ports[condition.lhs] % condition.value:
                return False, "instruction alignment validity failed"
    for left_index, left in enumerate(graph.values):
        for right in graph.values[left_index + 1 :]:
            if bank_map[left.storage].backing != bank_map[right.storage].backing:
                continue
            a = addresses[left.id]
            b = addresses[right.id]
            overlap = a < b + right.extent and b < a + left.extent
            if overlap and (
                (left.kind == "input" and left.preserve_input and right.kind == "instruction")
                or (right.kind == "input" and right.preserve_input and left.kind == "instruction")
            ):
                return False, "boundary input overlaps an instruction write"
            lo1, hi1 = ranges[left.id]
            lo2, hi2 = ranges[right.id]
            if overlap and lo1 <= hi2 and lo2 <= hi1:
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
        solver.add(addr >= 0, addr + value.extent <= bank.capacity, addr % bank.alignment == 0)
        for reservation in reservations:
            if bank.backing == bank_map[reservation.storage].backing:
                solver.add(
                    z3.Or(
                        addr + value.extent <= reservation.start,
                        addr >= reservation.start + reservation.extent,
                    )
                )
        if value.kind == "input" and value.source_node in fixed_inputs:
            solver.add(addr == fixed_inputs[value.source_node])
        ports = {"out": addr, **{f"in{index}": variables[child] for index, child in enumerate(value.children)}}
        for condition in value.validity:
            if condition.kind == "eq_offset":
                solver.add(ports[condition.lhs] == ports[condition.rhs] + condition.value)
            elif condition.kind == "aligned":
                solver.add(ports[condition.lhs] % condition.value == 0)
    for left_index, left in enumerate(graph.values):
        for right in graph.values[left_index + 1 :]:
            if bank_map[left.storage].backing != bank_map[right.storage].backing:
                continue
            lo1, hi1 = ranges[left.id]
            lo2, hi2 = ranges[right.id]
            preserve_input = (left.kind == "input" and left.preserve_input and right.kind == "instruction") or (
                right.kind == "input" and right.preserve_input and left.kind == "instruction"
            )
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
