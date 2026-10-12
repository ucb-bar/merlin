"""Source-bound Lean obligations for the actual checked SSA resource annotation.

The projection and source-to-epoch interpretation are trusted. No saved status,
solver result or projection object issues accepted evidence. Hardware sizes,
units, effects and qualified overwrite meanings remain independent obligations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from xdsl.dialects import builtin

from merlin.common.paths import module_source_path
from merlin.transforms.certificates import ir_digest
from merlin.transforms.resource_allocation import ATTRIBUTE, _project, validate_allocated_ir

SCHEMA = "merlin.resource_geometry_obligations.v1"
SCOPE = (
    "actual_source_and_emitted_assignment_identity",
    "explicit_capacity_alignment_and_address_units",
    "shared_backing_reservations_and_closed_live_intervals",
    "declared_final_use_overwrite_exceptions",
    "complete_boundary_and_instruction_port_constraints",
    "sequential_instruction_and_result_free_event_reads",
)
_SOURCE_MODULES = (
    "merlin.verify.lean_resource_geometry",
    "merlin.transforms.resource_allocation",
    "merlin.transforms.analysis_memory",
    "merlin.transforms.analysis_types",
    "merlin.transforms.certificates",
    "merlin.semantic_compiler.allocate",
    "merlin.semantic_compiler.allocation_types",
    "merlin.semantic_compiler.rules",
    "merlin.semantic_compiler.model",
)


def _digest(record) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def resource_semantic_identity() -> str:
    """Bind the trusted projector/checker and mathematical model source bytes."""
    sources = [(name, hashlib.sha256(module_source_path(name).read_bytes()).hexdigest()) for name in _SOURCE_MODULES]
    mathematical = Path(__file__).parent / "lean/Merlin/ResourceGeometry.lean"
    return _digest((SCHEMA, SCOPE, sources, hashlib.sha256(mathematical.read_bytes()).hexdigest()))


@dataclass(frozen=True)
class GeometryObligation:
    kind: str
    subjects: tuple[str, ...]
    numbers: tuple[int, ...]


@dataclass(frozen=True)
class ResourceGeometryProjection:
    schema: str
    source_sha256: str
    emitted_sha256: str
    constraints_sha256: str
    semantics_sha256: str
    obligations: tuple[GeometryObligation, ...]
    scope: tuple[str, ...] = SCOPE

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())


def _overwrite(graph, left, right):
    """Recover the actual declared last-use exception, not a solver permission."""
    for child, parent in ((left, right), (right, left)):
        if child.kind != "instruction" or parent.kind != "instruction" or child.id in graph.outputs:
            continue
        references = parent.children.count(child.id)
        uses = sum(value.children.count(child.id) for value in graph.values)
        if child.extent != parent.extent or references != 1 or uses != 1:
            continue
        port = parent.children.index(child.id)
        if port in parent.in_place_inputs and parent.read_offset(port) < parent.completion_offset:
            return child, parent, references, uses
    return None


def project_resource_geometry(
    original: builtin.ModuleOp,
    emitted: builtin.ModuleOp,
    *,
    entry: str,
    bindings: tuple,
    banks: tuple,
    reservations: tuple = (),
    instruction_constraints: tuple = (),
    fixed_inputs: dict[str, int] | None = None,
    fixed_outputs: tuple[int | None, ...] | None = None,
    index_bits: int | None = None,
    max_obligations: int = 16384,
) -> ResourceGeometryProjection:
    """Validate and project actual assignments, including all result-free events.

    Bounds count IR values/relations, never tensor elements or address units.
    No placement solver is run. A host checker must call this function freshly
    and bind its identities; accepting a caller-supplied object is insufficient.
    """
    from merlin.semantic_compiler.allocate import instruction_schedule, live_ranges

    if type(max_obligations) is not int or max_obligations < 1:
        raise ValueError("resource verification requires a positive obligation budget")
    model = dict(
        entry=entry,
        bindings=bindings,
        banks=banks,
        reservations=reservations,
        instruction_constraints=instruction_constraints,
        fixed_inputs=fixed_inputs,
        fixed_outputs=fixed_outputs,
        index_bits=index_bits,
    )
    validate_allocated_ir(original, emitted, **model)
    graph, order, ids, outputs, _ = _project(original, entry, bindings, banks, instruction_constraints, index_bits)
    header = emitted.attributes[ATTRIBUTE].data
    addresses = {value.id: 0 for value in graph.values if value.kind == "event"}
    addresses.update(
        (ids[binding.value], row.data["address"].value.data) for binding, row in zip(bindings, header["values"])
    )
    bank_map, schedule, ranges = (
        {bank.name: bank for bank in banks},
        instruction_schedule(graph, order),
        live_ranges(graph, order),
    )
    previous_completion = {
        value: schedule[order[position - 1]][1] if position else 0 for position, value in enumerate(order)
    }
    rows = []

    def add(kind, subjects, *numbers):
        if len(rows) >= max_obligations:
            raise ValueError("resource verification exceeds its explicit obligation budget")
        if any(type(number) is not int for number in numbers):
            raise ValueError("resource projection requires exact integer geometry")
        rows.append(GeometryObligation(kind, tuple(subjects), tuple(numbers)))

    for value in graph.values:
        bank, address = bank_map[value.storage], addresses[value.id]
        subject = value.source_node or f"event:{value.id}"
        add("fits", (subject, bank.name), address, value.extent, bank.capacity, bank.alignment)
        add("live_interval", (subject,), *ranges[value.id])
        for reservation in reservations:
            if value.extent and bank.backing == bank_map[reservation.storage].backing:
                add(
                    "disjoint",
                    (subject, "reservation:" + reservation.storage),
                    address,
                    value.extent,
                    reservation.start,
                    reservation.extent,
                )
        if value.kind == "input" and value.source_node in (fixed_inputs or {}):
            add("fixed_address", (subject, "input"), address, fixed_inputs[value.source_node])
        if value.kind == "event":
            add("zero_extent_event", (subject,), value.extent)
        if value.id in schedule:
            issue, complete = schedule[value.id]
            add("live_interval", (subject, "execution"), issue, complete)
            add("serial_issue", (subject,), previous_completion[value.id], issue)
            add("equal_offset", (subject, "completion", "issue"), complete, issue, value.completion_offset)
            for port, child in enumerate(value.children):
                ready = schedule[child][1] if child in schedule else 0
                add(
                    "read_within",
                    (subject, f"in{port}"),
                    ready,
                    issue,
                    value.read_offset(port),
                    complete,
                    ranges[child][1],
                )
        ports = {"out": address, **{f"in{port}": addresses[child] for port, child in enumerate(value.children)}}
        for condition in value.validity:
            if condition.kind == "aligned":
                add("aligned", (subject, condition.lhs), ports[condition.lhs], condition.value)
            else:
                add(
                    "equal_offset",
                    (subject, condition.lhs, condition.rhs),
                    ports[condition.lhs],
                    ports[condition.rhs],
                    condition.value,
                )
    for ordinal, address in enumerate(fixed_outputs or ()):
        if address is not None:
            add("fixed_address", (outputs[ordinal], "output"), addresses[ids[outputs[ordinal]]], address)
    for index, reservation in enumerate(reservations):
        bank = bank_map[reservation.storage]
        add("fits", ("reservation:" + str(index), bank.name), reservation.start, reservation.extent, bank.capacity, 1)
        for earlier in reservations[:index]:
            if bank.backing == bank_map[earlier.storage].backing:
                add(
                    "disjoint",
                    ("reservation:" + str(index), earlier.storage),
                    reservation.start,
                    reservation.extent,
                    earlier.start,
                    earlier.extent,
                )
    for position, left in enumerate(graph.values):
        for right in graph.values[position + 1 :]:
            if not left.extent or not right.extent or bank_map[left.storage].backing != bank_map[right.storage].backing:
                continue
            subjects = (left.source_node, right.source_node)
            geometry = (addresses[left.id], left.extent, addresses[right.id], right.extent)
            if (left.kind == "input" and left.preserve_input and right.kind == "instruction") or (
                right.kind == "input" and right.preserve_input and left.kind == "instruction"
            ):
                add("disjoint", (*subjects, "preserved_input"), *geometry)
            qualified = _overwrite(graph, left, right)
            if qualified is not None and addresses[left.id] == addresses[right.id]:
                child, parent, references, uses = qualified
                add(
                    "final_use_overwrite",
                    subjects,
                    addresses[child.id],
                    addresses[parent.id],
                    child.extent,
                    parent.extent,
                    references,
                    uses,
                    int(child.id in graph.outputs),
                    ranges[child.id][1],
                    schedule[parent.id][1],
                )
            else:
                add(
                    "separated",
                    subjects,
                    addresses[left.id],
                    left.extent,
                    *ranges[left.id],
                    addresses[right.id],
                    right.extent,
                    *ranges[right.id],
                )
    return ResourceGeometryProjection(
        SCHEMA,
        ir_digest(original),
        ir_digest(emitted),
        header["constraints_sha256"].data,
        resource_semantic_identity(),
        tuple(rows),
    )


def _statement(row: GeometryObligation) -> str:
    sizes = {
        "fits": 4,
        "disjoint": 4,
        "separated": 8,
        "final_use_overwrite": 9,
        "read_within": 5,
        "equal_offset": 3,
        "aligned": 2,
        "fixed_address": 2,
        "live_interval": 2,
        "zero_extent_event": 1,
        "serial_issue": 2,
    }
    if (
        row.kind not in sizes
        or len(row.numbers) != sizes[row.kind]
        or any(type(value) is not int for value in row.numbers)
    ):
        raise ValueError("unmodeled or malformed resource geometry obligation")
    if any(value < 0 for value in row.numbers[:-1] if row.kind == "equal_offset") or (
        row.kind != "equal_offset" and any(value < 0 for value in row.numbers)
    ):
        raise ValueError("resource geometry cannot silently coerce negative addresses or extents")
    values = " ".join(str(value) for value in row.numbers)
    predicates = {
        "fits": "Fits",
        "disjoint": "Disjoint",
        "separated": "Separated",
        "final_use_overwrite": "FinalUseOverwrite",
        "read_within": "ReadWithin",
    }
    if row.kind in predicates:
        return "Merlin.ResourceGeometry." + predicates[row.kind] + " " + values
    a = row.numbers[0]
    if row.kind == "zero_extent_event":
        return f"({a} : Nat) = 0"
    b = row.numbers[1]
    if row.kind == "aligned":
        return f"(0 < ({b} : Nat)) ∧ ({a} : Nat) % {b} = 0"
    if row.kind == "equal_offset":
        return f"Merlin.ResourceGeometry.EqualOffset {a} {b} ({row.numbers[2]})"
    if row.kind == "serial_issue":
        return f"({a} : Nat) < {b}"
    return f"({a} : Nat) {'=' if row.kind == 'fixed_address' else '≤'} {b}"


def lean_resource_query(projection: ResourceGeometryProjection) -> str:
    """Render one kernel-checkable conjunction; this does not issue a receipt."""
    if projection.schema != SCHEMA or projection.scope != SCOPE or not projection.obligations:
        raise ValueError("resource geometry lacks the fixed modeled scope")
    declarations = [
        f"theorem resource_{index} : {_statement(row)} := by decide" for index, row in enumerate(projection.obligations)
    ]

    def join(first, limit):
        if limit - first == 1:
            return _statement(projection.obligations[first]), f"resource_{first}"
        middle = (first + limit) // 2
        left, a = join(first, middle)
        right, b = join(middle, limit)
        return f"({left}) ∧ ({right})", f"And.intro ({a}) ({b})"

    statement, proof = join(0, len(projection.obligations))
    return (
        "import Merlin.ResourceGeometry\nset_option warningAsError true\n"
        + "\n".join(declarations)
        + f"\ntheorem merlin_transform : {statement} := {proof}\n#print axioms merlin_transform\n"
    )
