"""Exact target-neutral physical-layout planning.

This module chooses physical encodings over a complete producer/consumer graph.  It deliberately
does not predict cycles: targets contribute legal operator alternatives, exact conversion
capabilities, storage extents, and capacity facts.  The result records physical traffic and live
storage so a later target emitter has an auditable contract instead of an implicit layout choice.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Any, Iterable


PHYSICAL_LAYOUT_RESULT_SCHEMA = "target_neutral_physical_layout_result_v1"


class PhysicalLayoutError(ValueError):
    """The supplied graph is malformed, rather than merely infeasible."""


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise PhysicalLayoutError(f"{name} must be a positive integer")
    return value


def _dtype_bytes(dtype: str) -> int:
    if dtype == "bf16":
        return 2
    if len(dtype) > 1 and dtype[0] in {"i", "u", "f"} and dtype[1:].isdigit():
        bits = int(dtype[1:])
        if bits > 0 and bits % 8 == 0:
            return bits // 8
    raise PhysicalLayoutError(f"byte-addressable scalar width is unknown for {dtype!r}")


def _nonempty(items: Iterable[str], name: str) -> tuple[str, ...]:
    result = tuple(items)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise PhysicalLayoutError(f"{name} must contain nonempty strings")
    if len(set(result)) != len(result):
        raise PhysicalLayoutError(f"{name} must not contain duplicates")
    return result


@dataclass(frozen=True)
class PhysicalEncoding:
    """One exact allocation/addressing choice for a logical value."""

    id: str
    layout: str
    storage_space: str
    storage_bytes: int
    requires_capacity: bool = False
    required_capabilities: tuple[str, ...] = ()
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.layout.strip() or not self.storage_space.strip():
            raise PhysicalLayoutError("an encoding requires id, layout, and storage space")
        _positive_int(self.storage_bytes, "encoding storage_bytes")
        _nonempty(self.required_capabilities, "encoding capabilities")
        _nonempty(self.provenance, "encoding provenance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "layout": self.layout,
            "storage_space": self.storage_space,
            "storage_bytes": self.storage_bytes,
            "requires_capacity": self.requires_capacity,
            "required_capabilities": list(self.required_capabilities),
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class LayoutValue:
    """A logical tensor and every exact physical encoding available to it."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    encodings: tuple[PhysicalEncoding, ...]

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.dtype.strip():
            raise PhysicalLayoutError("a layout value requires a name and dtype")
        if not self.shape or any(type(dim) is not int or dim <= 0 for dim in self.shape):
            raise PhysicalLayoutError(f"value {self.name!r} requires positive static extents")
        if not self.encodings:
            raise PhysicalLayoutError(f"value {self.name!r} has no physical encodings")
        ids = [encoding.id for encoding in self.encodings]
        if len(ids) != len(set(ids)):
            raise PhysicalLayoutError(f"value {self.name!r} repeats an encoding id")
        if any(encoding.storage_bytes < self.logical_bytes for encoding in self.encodings):
            raise PhysicalLayoutError(
                f"value {self.name!r} has an encoding smaller than its logical payload")

    @property
    def logical_bytes(self) -> int:
        return prod(self.shape) * _dtype_bytes(self.dtype)

    def encoding(self, identifier: str) -> PhysicalEncoding | None:
        return next((item for item in self.encodings if item.id == identifier), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "shape": list(self.shape), "dtype": self.dtype,
            "logical_bytes": self.logical_bytes,
            "encodings": [item.to_dict() for item in self.encodings],
        }


@dataclass(frozen=True)
class CapacityDemand:
    storage_space: str
    bytes: int
    provenance: str

    def __post_init__(self) -> None:
        if not self.storage_space.strip() or not self.provenance.strip():
            raise PhysicalLayoutError("capacity demand requires storage space and provenance")
        if type(self.bytes) is not int or self.bytes < 0:
            raise PhysicalLayoutError("capacity demand bytes must be a nonnegative integer")

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class OperatorEncoding:
    """An exact implementation/encoding pair supplied by a target adapter."""

    id: str
    encoding_id: str
    preference_weight: int = 0
    required_capabilities: tuple[str, ...] = ()
    capacity_demands: tuple[CapacityDemand, ...] = ()
    port_encodings: tuple[tuple[str, str], ...] = ()
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.encoding_id.strip():
            raise PhysicalLayoutError("operator encoding requires id and encoding_id")
        if type(self.preference_weight) is not int or self.preference_weight < 0:
            raise PhysicalLayoutError("operator preference weight must be a nonnegative integer")
        _nonempty(self.required_capabilities, "operator capabilities")
        _nonempty(self.provenance, "operator provenance")
        names = [name for name, _ in self.port_encodings]
        if (len(names) != len(set(names))
                or any(not name.strip() or not encoding.strip()
                       for name, encoding in self.port_encodings)):
            raise PhysicalLayoutError("operator port encodings require unique named ports")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "encoding_id": self.encoding_id,
            "preference_weight": self.preference_weight,
            "required_capabilities": list(self.required_capabilities),
            "capacity_demands": [item.to_dict() for item in self.capacity_demands],
            "port_encodings": dict(self.port_encodings),
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class LayoutOp:
    """Physical-layout behavior of one ordered logical operation.

    A coupled operation requires every port to use the selected encoding and is how elementwise,
    residual, and layout-polymorphic accelerator regions propagate one graph-level decision.  A
    non-coupled operation is a fixed physical boundary and therefore has exactly one native option;
    conversions are explicit at every mismatched port.
    """

    name: str
    kind: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    couple: bool
    options: tuple[OperatorEncoding, ...]
    provenance: tuple[str, ...] = ()
    coupled_ports: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.kind.strip():
            raise PhysicalLayoutError("a layout operation requires name and kind")
        ports = (*self.inputs, *self.outputs)
        if not ports or any(not isinstance(item, str) or not item.strip() for item in ports):
            raise PhysicalLayoutError("operation ports must be nonempty value names")
        if not self.options:
            raise PhysicalLayoutError(f"operation {self.name!r} has no exact encoding option")
        option_ids = [item.id for item in self.options]
        encoding_ids = [item.encoding_id for item in self.options]
        if len(option_ids) != len(set(option_ids)):
            raise PhysicalLayoutError(f"operation {self.name!r} repeats an option id")
        if len(encoding_ids) != len(set(encoding_ids)):
            raise PhysicalLayoutError(f"operation {self.name!r} repeats an encoding option")
        if not self.couple and len(self.options) != 1:
            raise PhysicalLayoutError(
                f"fixed-boundary operation {self.name!r} must have exactly one native encoding")
        if self.coupled_ports:
            if not self.couple:
                raise PhysicalLayoutError("a fixed-boundary operation cannot declare coupled ports")
            if len(set(self.coupled_ports)) != len(self.coupled_ports):
                raise PhysicalLayoutError("operation coupled ports must not contain duplicates")
            if set(self.coupled_ports) - set(ports):
                raise PhysicalLayoutError("operation couples a value absent from its ports")
        for option in self.options:
            if set(dict(option.port_encodings)) - set(ports):
                raise PhysicalLayoutError("operator option encodes a value absent from its ports")
            if set(dict(option.port_encodings)) & set(self.propagated_ports):
                raise PhysicalLayoutError(
                    "operator option must not override a graph-propagated port encoding")
        _nonempty(self.provenance, "operation provenance")

    @property
    def ports(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.inputs, *self.outputs)))

    @property
    def propagated_ports(self) -> tuple[str, ...]:
        if not self.couple:
            return ()
        return self.coupled_ports or self.ports

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "inputs": list(self.inputs),
            "outputs": list(self.outputs), "couple": self.couple,
            "coupled_ports": list(self.propagated_ports),
            "options": [item.to_dict() for item in self.options],
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class ConversionCapability:
    """Exact bit-preserving conversion between two named physical encodings."""

    id: str
    source_encoding: str
    destination_encoding: str
    read_scope: str
    write_scope: str
    required_capabilities: tuple[str, ...] = ()
    workspace_space: str | None = None
    workspace_bytes: int = 0
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (not self.id.strip() or not self.source_encoding.strip()
                or not self.destination_encoding.strip()):
            raise PhysicalLayoutError("conversion requires id and endpoint encodings")
        if self.source_encoding == self.destination_encoding:
            raise PhysicalLayoutError("an identity conversion must not be materialized")
        if self.read_scope not in {"logical_payload", "full_physical"}:
            raise PhysicalLayoutError("conversion read scope is not exact")
        if self.write_scope not in {"logical_payload", "full_physical"}:
            raise PhysicalLayoutError("conversion write scope is not exact")
        _nonempty(self.required_capabilities, "conversion capabilities")
        _nonempty(self.provenance, "conversion provenance")
        if type(self.workspace_bytes) is not int or self.workspace_bytes < 0:
            raise PhysicalLayoutError("conversion workspace bytes must be nonnegative")
        if (self.workspace_space is None) != (self.workspace_bytes == 0):
            raise PhysicalLayoutError(
                "conversion workspace space must be present exactly when workspace is nonzero")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "source_encoding": self.source_encoding,
            "destination_encoding": self.destination_encoding,
            "read_scope": self.read_scope, "write_scope": self.write_scope,
            "required_capabilities": list(self.required_capabilities),
            "workspace_space": self.workspace_space, "workspace_bytes": self.workspace_bytes,
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class CapacityLimit:
    storage_space: str
    bytes: int
    provenance: str

    def __post_init__(self) -> None:
        if not self.storage_space.strip() or not self.provenance.strip():
            raise PhysicalLayoutError("capacity limit requires storage space and provenance")
        _positive_int(self.bytes, "capacity limit bytes")

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True, order=True)
class LayoutRefusal:
    subject: str
    code: str
    detail: str
    provenance: tuple[str, ...] = ()
    blocks_plan: bool = False

    def __post_init__(self) -> None:
        if not self.subject.strip() or not self.code.strip() or not self.detail.strip():
            raise PhysicalLayoutError("a refusal requires subject, code, and detail")
        _nonempty(self.provenance, "refusal provenance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "code": self.code,
            "detail": self.detail,
            "provenance": list(self.provenance),
            "blocks_plan": self.blocks_plan,
        }


@dataclass(frozen=True)
class LayoutGraph:
    values: tuple[LayoutValue, ...]
    ops: tuple[LayoutOp, ...]
    capabilities: tuple[str, ...]
    conversions: tuple[ConversionCapability, ...]
    capacity_limits: tuple[CapacityLimit, ...] = ()
    target_refusals: tuple[LayoutRefusal, ...] = ()
    max_assignments: int = 100_000

    def __post_init__(self) -> None:
        _nonempty(self.capabilities, "graph capabilities")
        if not self.values or not self.ops:
            raise PhysicalLayoutError("layout planning requires values and operations")
        if len({item.name for item in self.values}) != len(self.values):
            raise PhysicalLayoutError("layout graph repeats a value name")
        if len({item.name for item in self.ops}) != len(self.ops):
            raise PhysicalLayoutError("layout graph repeats an operation name")
        if len({item.id for item in self.conversions}) != len(self.conversions):
            raise PhysicalLayoutError("layout graph repeats a conversion id")
        if len({item.storage_space for item in self.capacity_limits}) != len(self.capacity_limits):
            raise PhysicalLayoutError("layout graph repeats a capacity limit")
        _positive_int(self.max_assignments, "max_assignments")
        names = {item.name for item in self.values}
        producers: dict[str, str] = {}
        producer_positions: dict[str, int] = {}
        consumer_positions: dict[str, list[int]] = {name: [] for name in names}
        for position, op in enumerate(self.ops):
            missing = set(op.ports) - names
            if missing:
                raise PhysicalLayoutError(
                    f"operation {op.name!r} references unknown values {sorted(missing)}")
            for output in op.outputs:
                if output in producers:
                    raise PhysicalLayoutError(
                        f"value {output!r} has multiple producers: "
                        f"{producers[output]!r}, {op.name!r}")
                producers[output] = op.name
                producer_positions[output] = position
            for input_name in op.inputs:
                consumer_positions[input_name].append(position)
        for name, producer_position in producer_positions.items():
            earlier = [position for position in consumer_positions[name]
                       if position < producer_position]
            if earlier:
                raise PhysicalLayoutError(
                    f"value {name!r} is consumed before its producer in the ordered graph")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "target_neutral_physical_layout_graph_v1",
            "values": [item.to_dict() for item in self.values],
            "ops": [item.to_dict() for item in self.ops],
            "capabilities": list(self.capabilities),
            "conversions": [item.to_dict() for item in self.conversions],
            "capacity_limits": [item.to_dict() for item in self.capacity_limits],
            "target_refusals": [item.to_dict() for item in self.target_refusals],
            "max_assignments": self.max_assignments,
        }


@dataclass(frozen=True)
class ValueLifetime:
    value: str
    first_op: int
    last_op: int
    storage_space: str
    storage_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class LayoutConversion:
    op: str
    port: str
    value: str
    capability: str
    source_encoding: str
    destination_encoding: str
    logical_payload_bytes: int
    source_physical_bytes: int
    destination_physical_bytes: int
    physical_read_bytes: int
    physical_write_bytes: int
    movement_bytes: int
    materialized_bytes: int
    temporary_bytes: int
    workspace_space: str | None
    workspace_bytes: int
    provenance: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = self.__dict__.copy()
        result["provenance"] = list(self.provenance)
        return result


@dataclass(frozen=True)
class PhysicalLayoutCost:
    preferred_weight: int
    physical_movement_bytes: int
    conversion_count: int
    peak_capacity_bytes: int
    selected_storage_bytes: int

    @property
    def objective(self) -> tuple[int, int, int, int, int]:
        return (-self.preferred_weight, self.physical_movement_bytes, self.conversion_count,
                self.peak_capacity_bytes, self.selected_storage_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "selection_order": [
            "maximize_preferred_weight", "minimize_physical_movement_bytes",
            "minimize_conversion_count", "minimize_peak_capacity_bytes",
            "minimize_selected_storage_bytes",
        ]}


@dataclass(frozen=True)
class PhysicalLayoutPlan:
    assignments: tuple[tuple[str, str], ...]
    selected_options: tuple[tuple[str, str], ...]
    conversions: tuple[LayoutConversion, ...]
    lifetimes: tuple[ValueLifetime, ...]
    peak_capacity_bytes: tuple[tuple[str, int], ...]
    capacity_limits: tuple[tuple[str, int], ...]
    required_capabilities: tuple[str, ...]
    cost: PhysicalLayoutCost

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignments": dict(self.assignments),
            "selected_options": dict(self.selected_options),
            "conversions": [item.to_dict() for item in self.conversions],
            "lifetimes": [item.to_dict() for item in self.lifetimes],
            "peak_capacity_bytes": dict(self.peak_capacity_bytes),
            "capacity_limits": dict(self.capacity_limits),
            "required_capabilities": list(self.required_capabilities),
            "cost": self.cost.to_dict(),
            "cycle_estimate": None,
            "performance_claim": "UNPROVEN",
        }


@dataclass(frozen=True)
class PhysicalLayoutResult:
    plan: PhysicalLayoutPlan | None
    refusals: tuple[LayoutRefusal, ...]
    searched_assignments: int
    candidate_assignments: int
    exhausted: bool
    schema: str = PHYSICAL_LAYOUT_RESULT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        histogram: dict[str, int] = {}
        for refusal in self.refusals:
            histogram[refusal.code] = histogram.get(refusal.code, 0) + 1
        return {
            "schema": self.schema,
            "plan": self.plan.to_dict() if self.plan is not None else None,
            "refusals": [item.to_dict() for item in self.refusals],
            "refusal_histogram": dict(sorted(histogram.items())),
            "searched_assignments": self.searched_assignments,
            "candidate_assignments": self.candidate_assignments,
            "exhausted": self.exhausted,
            "optimal_for_supplied_exact_alternatives": (
                self.exhausted and self.plan is not None
                and not any(item.blocks_plan for item in self.refusals)),
            "cycle_estimate": None,
            "performance_claim": "UNPROVEN",
        }


class _DisjointSet:
    def __init__(self, names: Iterable[str]):
        self.parents = {name: name for name in names}

    def find(self, name: str) -> str:
        root = name
        while self.parents[root] != root:
            root = self.parents[root]
        while self.parents[name] != name:
            parent = self.parents[name]
            self.parents[name] = root
            name = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self.parents[right_root] = left_root


def _dedupe_refusals(items: Iterable[LayoutRefusal]) -> tuple[LayoutRefusal, ...]:
    return tuple(sorted(set(items)))


def _capability_refusal(subject: str, required: Iterable[str], available: set[str],
                        provenance: tuple[str, ...]) -> LayoutRefusal | None:
    missing = sorted(set(required) - available)
    if not missing:
        return None
    return LayoutRefusal(subject, "missing_required_capability",
                         f"missing capabilities: {', '.join(missing)}", provenance)


def _component_domains(graph: LayoutGraph) -> tuple[list[list[str]], dict[str, int]]:
    values = {item.name: item for item in graph.values}
    joined = _DisjointSet(values)
    for op in graph.ops:
        if len(op.propagated_ports) > 1:
            for port in op.propagated_ports[1:]:
                joined.union(op.propagated_ports[0], port)
    members: dict[str, list[str]] = {}
    for name in values:
        members.setdefault(joined.find(name), []).append(name)
    ordered_members = [sorted(members[root]) for root in sorted(members)]
    component_index = {name: index for index, names in enumerate(ordered_members) for name in names}
    domains: list[list[str]] = []
    for names in ordered_members:
        common = set(encoding.id for encoding in values[names[0]].encodings)
        for name in names[1:]:
            common &= {encoding.id for encoding in values[name].encodings}
        for op in graph.ops:
            if op.propagated_ports and op.propagated_ports[0] in names:
                common &= {option.encoding_id for option in op.options}
        order = [encoding.id for encoding in values[names[0]].encodings if encoding.id in common]
        domains.append(order)
    return domains, component_index


def _lifetimes(graph: LayoutGraph, assignments: dict[str, str]) -> tuple[ValueLifetime, ...]:
    values = {item.name: item for item in graph.values}
    producer: dict[str, int] = {}
    consumers: dict[str, list[int]] = {name: [] for name in values}
    for index, op in enumerate(graph.ops):
        producer.update((name, index) for name in op.outputs)
        for name in op.inputs:
            consumers[name].append(index)
    last_index = len(graph.ops) - 1
    result = []
    for name in sorted(values):
        start = producer.get(name, 0)
        end = max(consumers[name], default=(last_index if name not in producer else start))
        encoding = values[name].encoding(assignments[name])
        assert encoding is not None
        result.append(ValueLifetime(name, start, end, encoding.storage_space,
                                    encoding.storage_bytes))
    return tuple(result)


def _conversion(graph: LayoutGraph, op: LayoutOp, port: str, value: LayoutValue,
                source_id: str, destination_id: str, available: set[str]) -> tuple[
                    LayoutConversion | None, LayoutRefusal | None]:
    matches = [item for item in graph.conversions
               if item.source_encoding == source_id and item.destination_encoding == destination_id]
    if not matches:
        return None, LayoutRefusal(
            f"{op.name}:{port}:{value.name}", "missing_exact_conversion",
            f"no exact conversion from {source_id!r} to {destination_id!r}", op.provenance)
    for capability in matches:
        missing = sorted(set(capability.required_capabilities) - available)
        if missing:
            continue
        source = value.encoding(source_id)
        destination = value.encoding(destination_id)
        if source is None or destination is None:
            continue
        read_bytes = (value.logical_bytes if capability.read_scope == "logical_payload"
                      else source.storage_bytes)
        write_bytes = (value.logical_bytes if capability.write_scope == "logical_payload"
                       else destination.storage_bytes)
        return LayoutConversion(
            op.name, port, value.name, capability.id, source_id, destination_id,
            value.logical_bytes, source.storage_bytes, destination.storage_bytes,
            read_bytes, write_bytes, read_bytes + write_bytes, destination.storage_bytes,
            destination.storage_bytes if port == "input" else source.storage_bytes,
            capability.workspace_space, capability.workspace_bytes, capability.provenance,
        ), None
    return None, LayoutRefusal(
        f"{op.name}:{port}:{value.name}", "conversion_capability_unavailable",
        f"conversion {source_id!r} to {destination_id!r} lacks a supplied capability",
        tuple(item.id for item in matches))


def _evaluate(graph: LayoutGraph, assignments: dict[str, str]) -> tuple[
        PhysicalLayoutPlan | None, tuple[LayoutRefusal, ...]]:
    values = {item.name: item for item in graph.values}
    available = set(graph.capabilities)
    refusals: list[LayoutRefusal] = []
    selected: dict[str, OperatorEncoding] = {}
    required: set[str] = set()

    for name, identifier in assignments.items():
        encoding = values[name].encoding(identifier)
        assert encoding is not None
        required.update(encoding.required_capabilities)
        refusal = _capability_refusal(name, encoding.required_capabilities, available,
                                      encoding.provenance)
        if refusal is not None:
            refusals.append(refusal)

    for op in graph.ops:
        identifier = (assignments[op.propagated_ports[0]] if op.propagated_ports
                      else op.options[0].encoding_id)
        option = next((item for item in op.options if item.encoding_id == identifier), None)
        if option is None:
            refusals.append(LayoutRefusal(
                op.name, "operator_encoding_unavailable",
                f"operation has no exact option for encoding {identifier!r}", op.provenance))
            continue
        port_encodings = dict(option.port_encodings)
        if any(values[port].encoding(port_encodings.get(port, identifier)) is None
               for port in op.ports):
            refusals.append(LayoutRefusal(
                op.name, "operator_port_encoding_unavailable",
                f"not every port represents native encoding {identifier!r}", op.provenance))
            continue
        selected[op.name] = option
        required.update(option.required_capabilities)
        refusal = _capability_refusal(op.name, option.required_capabilities, available,
                                      option.provenance)
        if refusal is not None:
            refusals.append(refusal)

    conversions: list[LayoutConversion] = []
    for op in graph.ops:
        option = selected.get(op.name)
        if option is None:
            continue
        port_encodings = dict(option.port_encodings)
        for port, names in (("input", op.inputs), ("output", op.outputs)):
            for name in names:
                selected_id = assignments[name]
                native_id = port_encodings.get(name, option.encoding_id)
                if selected_id == native_id:
                    continue
                source, destination = ((selected_id, native_id) if port == "input"
                                       else (native_id, selected_id))
                conversion, refusal = _conversion(
                    graph, op, port, values[name], source, destination, available)
                if refusal is not None:
                    refusals.append(refusal)
                elif conversion is not None:
                    conversions.append(conversion)
                    capability = next(item for item in graph.conversions
                                      if item.id == conversion.capability)
                    required.update(capability.required_capabilities)

    if refusals:
        return None, _dedupe_refusals(refusals)

    lifetimes = _lifetimes(graph, assignments)
    limits = {item.storage_space: item.bytes for item in graph.capacity_limits}
    provenance = {item.storage_space: item.provenance for item in graph.capacity_limits}
    peaks: dict[str, int] = {}
    for index, op in enumerate(graph.ops):
        usage: dict[str, int] = {}
        for lifetime in lifetimes:
            if lifetime.first_op <= index <= lifetime.last_op:
                usage[lifetime.storage_space] = (
                    usage.get(lifetime.storage_space, 0) + lifetime.storage_bytes)
        option = selected[op.name]
        for demand in option.capacity_demands:
            usage[demand.storage_space] = usage.get(demand.storage_space, 0) + demand.bytes
            if demand.storage_space not in limits:
                refusals.append(LayoutRefusal(
                    op.name, "capacity_unproven",
                    f"no limit supplied for transient storage space {demand.storage_space!r}",
                    (demand.provenance,)))
        for conversion in (item for item in conversions if item.op == op.name):
            temporary_id = (conversion.destination_encoding if conversion.port == "input"
                            else conversion.source_encoding)
            temporary = values[conversion.value].encoding(temporary_id)
            assert temporary is not None
            usage[temporary.storage_space] = (
                usage.get(temporary.storage_space, 0) + conversion.temporary_bytes)
            if temporary.requires_capacity and temporary.storage_space not in limits:
                refusals.append(LayoutRefusal(
                    f"{conversion.op}:{conversion.value}", "capacity_unproven",
                    f"no limit supplied for conversion temporary {temporary.storage_space!r}",
                    conversion.provenance))
            if conversion.workspace_space is not None:
                usage[conversion.workspace_space] = (
                    usage.get(conversion.workspace_space, 0) + conversion.workspace_bytes)
                if conversion.workspace_space not in limits:
                    refusals.append(LayoutRefusal(
                        f"{conversion.op}:{conversion.value}", "capacity_unproven",
                        "no limit supplied for conversion workspace "
                        f"{conversion.workspace_space!r}",
                        conversion.provenance))
        for space, amount in usage.items():
            peaks[space] = max(peaks.get(space, 0), amount)

    for lifetime in lifetimes:
        encoding = values[lifetime.value].encoding(assignments[lifetime.value])
        assert encoding is not None
        if encoding.requires_capacity and lifetime.storage_space not in limits:
            refusals.append(LayoutRefusal(
                lifetime.value, "capacity_unproven",
                f"no limit supplied for storage space {lifetime.storage_space!r}",
                encoding.provenance))
    for space, peak in peaks.items():
        if space in limits and peak > limits[space]:
            refusals.append(LayoutRefusal(
                space, "capacity_exceeded",
                f"peak {peak} bytes exceeds limit {limits[space]} bytes",
                (provenance[space],)))
    if refusals:
        return None, _dedupe_refusals(refusals)

    preferred = sum(option.preference_weight for option in selected.values())
    movement = sum(item.movement_bytes for item in conversions)
    selected_storage = sum(item.storage_bytes for item in lifetimes)
    constrained_peak = sum(peaks.get(space, 0) for space in limits)
    cost = PhysicalLayoutCost(preferred, movement, len(conversions), constrained_peak,
                              selected_storage)
    plan = PhysicalLayoutPlan(
        tuple(sorted(assignments.items())),
        tuple(sorted((name, option.id) for name, option in selected.items())),
        tuple(conversions), lifetimes, tuple(sorted(peaks.items())), tuple(sorted(limits.items())),
        tuple(sorted(required)), cost,
    )
    return plan, ()


def plan_physical_layout(graph: LayoutGraph) -> PhysicalLayoutResult:
    """Exhaustively optimize the supplied exact encoding alternatives, or fail closed."""

    domains, component_index = _component_domains(graph)
    empty = [index for index, domain in enumerate(domains) if not domain]
    if empty:
        refusal = LayoutRefusal(
            "layout_graph", "incompatible_coupled_encoding_domains",
            f"coupled components have no common exact encoding: {empty}")
        return PhysicalLayoutResult(None, _dedupe_refusals((*graph.target_refusals, refusal)),
                                    0, 0, True)
    candidates = prod(len(domain) for domain in domains)
    if candidates > graph.max_assignments:
        refusal = LayoutRefusal(
            "layout_graph", "assignment_limit_exceeded",
            f"{candidates} assignments exceed explicit limit {graph.max_assignments}")
        return PhysicalLayoutResult(None, _dedupe_refusals((*graph.target_refusals, refusal)),
                                    0, candidates, False)

    best: PhysicalLayoutPlan | None = None
    best_key: tuple[Any, ...] | None = None
    refusals: list[LayoutRefusal] = list(graph.target_refusals)
    searched = 0
    value_names = [item.name for item in graph.values]
    for choices in product(*domains):
        searched += 1
        assignments = {name: choices[component_index[name]] for name in value_names}
        plan, candidate_refusals = _evaluate(graph, assignments)
        refusals.extend(candidate_refusals)
        if plan is None:
            continue
        deterministic = tuple(assignments[name] for name in sorted(assignments))
        key = (*plan.cost.objective, deterministic)
        if best_key is None or key < best_key:
            best, best_key = plan, key
    return PhysicalLayoutResult(best, _dedupe_refusals(refusals), searched, candidates, True)
