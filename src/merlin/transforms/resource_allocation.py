"""Adapt actual sequential xDSL SSA to the existing checked generic allocator.

The result is a resource assignment in an explicitly supplied representation,
not an instruction encoding, hardware timing estimate or numerical proof.
Annotation validation recomputes graph/lifetimes and independently checks the
actual recorded addresses; it never accepts a solver status string alone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from xdsl.dialects import builtin, func
from xdsl.traits import get_effects

from .analysis_memory import value_facts
from .analysis_types import IRIndex, operation_name
from .certificates import ir_digest, ir_text

ATTRIBUTE = "merlin.resource_allocation"
SCHEMA = "merlin.resource_allocation.v1"


@dataclass(frozen=True)
class ResourceBinding:
    value: str
    storage: str
    extent: int
    alignment: int = 1

    def __post_init__(self):
        if (
            not isinstance(self.value, str)
            or not self.value
            or not isinstance(self.storage, str)
            or not self.storage
            or type(self.extent) is not int
            or self.extent <= 0
        ):
            raise ValueError("resource binding requires an actual SSA identity, bank and positive extent")
        if type(self.alignment) is not int or self.alignment <= 0:
            raise ValueError("resource alignment must be positive in the bank's explicit unit")


@dataclass(frozen=True)
class InstructionConstraints:
    operation: int
    validity: tuple[Any, ...] = ()
    input_read_offsets: tuple[int, ...] = ()
    completion_offset: int = 0
    in_place_inputs: tuple[int, ...] = ()

    def __post_init__(self):
        if type(self.operation) is not int or self.operation < 0:
            raise ValueError("instruction constraints must identify an actual nonnegative operation index")
        if type(self.completion_offset) is not int or self.completion_offset < 0:
            raise ValueError("completion epochs must be nonnegative integers")
        if any(type(offset) is not int or offset < 0 for offset in self.input_read_offsets):
            raise ValueError("input read epochs must be nonnegative integers")
        if any(type(port) is not int or port < 0 for port in self.in_place_inputs):
            raise ValueError("in-place permissions must identify actual operand indices")


@dataclass(frozen=True)
class ResourceAllocation:
    status: str
    source_sha256: str
    constraints_sha256: str
    entry: str
    order: tuple[int, ...]
    addresses: tuple[tuple[int, int], ...]
    value_ids: tuple[tuple[str, int], ...]
    logical_outputs: tuple[str, ...]
    retained_effect_operands: tuple[str, ...]
    live_ranges: tuple[tuple[str, int, int], ...]
    reason: str = ""


def _identity(entry, bindings, banks, reservations, constraints, fixed_inputs, fixed_outputs, index_bits):
    record = {
        "entry": entry,
        "bindings": [asdict(binding) for binding in bindings],
        "banks": [bank.record() for bank in banks],
        "reservations": [asdict(row) for row in reservations],
        "instructions": [asdict(row) for row in constraints],
        "fixed_inputs": fixed_inputs or {},
        "fixed_outputs": fixed_outputs,
        "index_bits": index_bits,
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _project(module, entry, bindings, banks, constraints, index_bits):
    from merlin.semantic_compiler.allocation_types import CandidateGraph, Value
    from merlin.semantic_compiler.rules import AddressConstraint

    if index_bits is not None and (type(index_bits) is not int or index_bits <= 1):
        raise ValueError("index representation requires an explicit positive width")
    index = IRIndex(module)
    selected = [op for op in module.walk() if isinstance(op, func.FuncOp) and op.sym_name.data == entry]
    if len(selected) != 1 or len(selected[0].body.blocks) != 1:
        raise ValueError("resource allocation requires one explicitly selected single-block function")
    block = selected[0].body.block
    if not banks or len({bank.name for bank in banks}) != len(banks):
        raise ValueError("resource banks must be explicit, nonempty and uniquely named")
    for bank in banks:
        if any(not isinstance(value, str) or not value for value in (bank.name, bank.backing, bank.unit)):
            raise ValueError("resource banks require explicit names, backing identities and units")
        if (
            type(bank.capacity) is not int
            or type(bank.alignment) is not int
            or bank.capacity <= 0
            or bank.alignment <= 0
        ):
            raise ValueError("resource bank capacity and alignment must be positive explicit integers")
    supplied = {binding.value: binding for binding in bindings}
    actual = {index.id(value): value for value in block.args}
    actual.update((index.id(value), value) for op in block.ops for value in op.results)
    if len(supplied) != len(bindings) or set(supplied) != set(actual):
        raise ValueError("resource bindings must exactly cover the selected function's SSA values")
    facts = {value.id: value for value in value_facts(index, index_bits=index_bits)}
    if any(facts[identity].memory is not False or facts[identity].logical_bytes is None for identity in actual):
        raise ValueError("unknown, pointer, dynamic or aliased representations cannot be allocated by this model")
    if any(binding.storage not in {bank.name for bank in banks} for binding in bindings):
        raise ValueError("resource binding names an unknown bank")
    conditions = {row.operation: row for row in constraints}
    if len(conditions) != len(constraints):
        raise ValueError("duplicate instruction constraint")
    values, ids, order, outputs, retained = [], {}, [], [], set()

    def add(value, op=None):
        identity = index.id(value)
        binding = supplied[identity]
        node = len(values)
        metadata = (
            conditions.get(index.operation_ids[op], InstructionConstraints(index.operation_ids[op]))
            if op is not None
            else None
        )
        validity = (AddressConstraint("aligned", "out", value=binding.alignment),)
        children = tuple(ids[index.id(operand)] for operand in op.operands) if op is not None else ()
        values.append(
            Value(
                node,
                operation_name(op) if op is not None else identity,
                binding.storage,
                binding.extent,
                children,
                identity,
                "instruction" if op is not None else "input",
                validity + (metadata.validity if metadata is not None else ()),
                metadata.input_read_offsets if metadata is not None else (),
                metadata.completion_offset if metadata is not None else 0,
                metadata.in_place_inputs if metadata is not None else (),
            )
        )
        ids[identity] = node
        return node

    for value in block.args:
        add(value)
    seen_constraints = set()
    returned = False
    for op in block.ops:
        identity = index.operation_ids[op]
        if returned:
            raise ValueError("operations after return are outside the resource model")
        if isinstance(op, func.ReturnOp):
            outputs = [index.id(value) for value in op.arguments]
            returned = True
            continue
        if op.regions or op.successors or len(op.results) > 1:
            raise ValueError("control-flow and multiple simultaneous results need a qualified resource model")
        effects = get_effects(op)
        if effects is None:
            raise ValueError("instruction has unknown effects; resource choices must remain constrained")
        if effects and identity not in conditions:
            raise ValueError("effectful instruction needs explicit sequential completion constraints")
        if effects:
            retained.update(index.id(value) for value in op.operands)
        if identity in conditions:
            seen_constraints.add(identity)
            if any(not isinstance(condition, AddressConstraint) for condition in conditions[identity].validity):
                raise ValueError("instruction validity requires typed address constraints")
            ports = {"out", *(f"in{i}" for i in range(len(op.operands)))}
            if any(
                condition.lhs not in ports
                or (condition.rhs and condition.rhs not in ports)
                or type(condition.value) is not int
                for condition in conditions[identity].validity
            ) or any(port >= len(op.operands) for port in conditions[identity].in_place_inputs):
                raise ValueError("instruction constraints reference absent ports or invalid numeric values")
        if op.results:
            order.append(add(op.results[0], op))
        else:
            # An event extends its operand live ranges at its true source-order
            # position. Its zero extent occupies no storage and is never encoded.
            metadata = conditions.get(identity, InstructionConstraints(identity))
            if metadata.in_place_inputs or any(
                condition.lhs == "out" or condition.rhs == "out" for condition in metadata.validity
            ):
                raise ValueError("result-free event has no physical output address or in-place result")
            children = tuple(ids[index.id(value)] for value in op.operands)
            order.append(len(values))
            values.append(
                Value(
                    len(values),
                    operation_name(op),
                    banks[0].name,
                    0,
                    children,
                    None,
                    "event",
                    metadata.validity,
                    metadata.input_read_offsets,
                    metadata.completion_offset,
                )
            )
    if not returned or seen_constraints != set(conditions):
        raise ValueError("resource projection has missing outputs or constraints outside actual instructions")
    if any(type(row.completion_offset) is not int or row.completion_offset < 0 for row in constraints):
        raise ValueError("completion epochs must be explicit nonnegative integers")
    if any(any(type(offset) is not int or offset < 0 for offset in row.input_read_offsets) for row in constraints):
        raise ValueError("input read epochs must be explicit nonnegative integers")
    roots = tuple(ids[value] for value in outputs) + tuple(ids[value] for value in sorted(retained - set(outputs)))
    return CandidateGraph(tuple(values), roots), tuple(order), ids, tuple(outputs), tuple(sorted(retained))


def allocate_instruction_resources(
    module: builtin.ModuleOp,
    *,
    entry: str,
    bindings: tuple[ResourceBinding, ...],
    banks: tuple,
    reservations: tuple = (),
    instruction_constraints: tuple[InstructionConstraints, ...] = (),
    fixed_inputs: dict[str, int] | None = None,
    fixed_outputs: tuple[int | None, ...] | None = None,
    timeout_ms: int = 5000,
    index_bits: int | None = None,
) -> ResourceAllocation:
    """Solve one source-order allocation, never infer banks/units from op names.

    Extents are supplied by the admitted physical representation; logical bytes
    alone cannot qualify them. Effect operands are conservatively retained to
    scope completion. This sequential epoch model makes no latency/overlap claim.
    """
    from merlin.semantic_compiler.allocate import allocate, live_ranges

    signature = _identity(
        entry, bindings, banks, reservations, instruction_constraints, fixed_inputs, fixed_outputs, index_bits
    )
    source = ir_digest(module)
    try:
        graph, order, ids, outputs, retained = _project(
            module, entry, bindings, banks, instruction_constraints, index_bits
        )
        if fixed_outputs is not None and len(fixed_outputs) != len(outputs):
            raise ValueError("fixed outputs must match complete source output order")
        pinned_outputs = (
            None if fixed_outputs is None else (*fixed_outputs, *(None for _ in graph.outputs[len(outputs) :]))
        )
        selected = allocate(
            graph,
            order,
            banks,
            fixed_inputs=fixed_inputs,
            reservations=reservations,
            fixed_outputs=pinned_outputs,
            timeout_ms=timeout_ms,
        )
        ranges = live_ranges(graph, order)
    except (ValueError, KeyError) as exc:
        return ResourceAllocation("unknown", source, signature, entry, (), (), (), (), (), (), str(exc))
    return ResourceAllocation(
        selected.status,
        source,
        signature,
        entry,
        order,
        tuple(sorted(selected.addresses.items())),
        tuple(sorted(ids.items())),
        outputs,
        retained,
        tuple((value, *ranges[node]) for value, node in sorted(ids.items())),
        selected.reason,
    )


def _integer(value: int):
    return builtin.IntegerAttr(value, builtin.IntegerType(max(1, value.bit_length() + 1)))


def _metadata(allocation, bindings):
    addresses, ids = dict(allocation.addresses), dict(allocation.value_ids)
    ranges = {row[0]: row[1:] for row in allocation.live_ranges}
    return builtin.DictionaryAttr(
        {
            "schema": builtin.StringAttr(SCHEMA),
            "source_sha256": builtin.StringAttr(allocation.source_sha256),
            "constraints_sha256": builtin.StringAttr(allocation.constraints_sha256),
            "entry": builtin.StringAttr(allocation.entry),
            "values": builtin.ArrayAttr(
                [
                    builtin.DictionaryAttr(
                        {
                            "value": builtin.StringAttr(binding.value),
                            "storage": builtin.StringAttr(binding.storage),
                            "extent": _integer(binding.extent),
                            "alignment": _integer(binding.alignment),
                            "address": _integer(addresses[ids[binding.value]]),
                            "live_start": _integer(ranges[binding.value][0]),
                            "live_end": _integer(ranges[binding.value][1]),
                        }
                    )
                    for binding in bindings
                ]
            ),
        }
    )


def emit_resource_allocation(original: builtin.ModuleOp, allocation: ResourceAllocation, **model) -> builtin.ModuleOp:
    if allocation.status != "feasible" or allocation.source_sha256 != ir_digest(original):
        raise ValueError("allocation is not feasible for the actual original IR")
    if ATTRIBUTE in original.attributes:
        raise ValueError("source already contains a resource assignment")
    emitted = original.clone()
    emitted.attributes[ATTRIBUTE] = _metadata(allocation, model["bindings"])
    validate_allocated_ir(original, emitted, **model)
    return emitted


def validate_allocated_ir(
    original: builtin.ModuleOp,
    emitted: builtin.ModuleOp,
    *,
    entry: str,
    bindings: tuple[ResourceBinding, ...],
    banks: tuple,
    reservations: tuple = (),
    instruction_constraints: tuple[InstructionConstraints, ...] = (),
    fixed_inputs: dict[str, int] | None = None,
    fixed_outputs: tuple[int | None, ...] | None = None,
    index_bits: int | None = None,
) -> None:
    """Check actual emitted assignment independently of the solver and plan object."""
    from merlin.semantic_compiler.allocate import check_assignment, live_ranges

    emitted.verify()
    actual = emitted.attributes.get(ATTRIBUTE)
    if not isinstance(actual, builtin.DictionaryAttr):
        raise ValueError("emitted IR has no typed resource assignment")
    stripped = emitted.clone()
    del stripped.attributes[ATTRIBUTE]
    if ir_text(stripped) != ir_text(original):
        raise ValueError("resource assignment changed the actual source program")
    graph, order, ids, outputs, retained = _project(
        original, entry, bindings, banks, instruction_constraints, index_bits
    )
    if fixed_outputs is not None and len(fixed_outputs) != len(outputs):
        raise ValueError("fixed outputs differ from complete source output order")
    signature = _identity(
        entry, bindings, banks, reservations, instruction_constraints, fixed_inputs, fixed_outputs, index_bits
    )
    header = actual.data
    if set(header) != {"schema", "source_sha256", "constraints_sha256", "entry", "values"}:
        raise ValueError("emitted assignment has unexpected fields")
    expected = {"schema": SCHEMA, "source_sha256": ir_digest(original), "constraints_sha256": signature, "entry": entry}
    if any(
        not isinstance(header[key], builtin.StringAttr) or header[key].data != value for key, value in expected.items()
    ):
        raise ValueError("emitted resource assignment has stale source or constraints")
    encoded = header["values"]
    if not isinstance(encoded, builtin.ArrayAttr) or len(encoded) != len(bindings):
        raise ValueError("emitted assignment omits actual SSA bindings")
    ranges = live_ranges(graph, order)
    addresses = {value.id: 0 for value in graph.values if value.kind == "event"}
    for binding, row in zip(bindings, encoded, strict=True):
        if not isinstance(row, builtin.DictionaryAttr) or set(row.data) != {
            "value",
            "storage",
            "extent",
            "alignment",
            "address",
            "live_start",
            "live_end",
        }:
            raise ValueError("emitted resource row has unexpected fields")
        fields = row.data
        if fields["value"] != builtin.StringAttr(binding.value) or fields["storage"] != builtin.StringAttr(
            binding.storage
        ):
            raise ValueError("emitted resource row names a different value or bank")
        numbers = {}
        for key in ("extent", "alignment", "address", "live_start", "live_end"):
            if not isinstance(fields[key], builtin.IntegerAttr):
                raise ValueError("emitted addresses and lifetimes must be typed integers")
            numbers[key] = fields[key].value.data
        node = ids[binding.value]
        if (numbers["extent"], numbers["alignment"], numbers["live_start"], numbers["live_end"]) != (
            binding.extent,
            binding.alignment,
            *ranges[node],
        ):
            raise ValueError("emitted resource sizes or lifetimes differ from actual IR")
        addresses[node] = numbers["address"]
    pinned_outputs = None if fixed_outputs is None else (*fixed_outputs, *(None for _ in graph.outputs[len(outputs) :]))
    valid, reason = check_assignment(
        graph,
        order,
        addresses,
        banks,
        fixed_inputs=fixed_inputs,
        reservations=reservations,
        fixed_outputs=pinned_outputs,
    )
    if not valid:
        raise ValueError("emitted physical assignment is invalid: " + reason)
