"""Pinned expert evidence and fusion-aware, conservative region comparison.

Records are machine-readable inputs to reconstruction. Source syntax, decoded
issue streams, compiler decisions, executed instructions, and timing evidence
retain their different authority. Unknown semantics never establish parity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from merlin.compile.analysis.schedule_structure import (
    CountedLoop,
    ScheduleAccess,
    ScheduleDependency,
    ScheduleModel,
    ScheduleStage,
    StorageLifetime,
    analyze_counted_kernel,
    analyze_ir_schedule,
    analyze_normalized_commands,
)

__all__ = [
    "ArtifactPin", "CountedLoop", "EvidenceFact", "KernelEvidence", "MeasurementBoundary", "RegionComparison",
    "RegionDifference", "RegionOperation", "RegionTensor", "ScheduleAccess", "ScheduleDependency", "ScheduleModel",
    "ScheduleStage", "SemanticRegion", "StorageLifetime", "analyze_committed_trace", "analyze_counted_kernel",
    "analyze_disassembly", "analyze_ir_schedule", "analyze_native_commands", "analyze_normalized_commands",
    "cca_projection", "compare_regions", "semantic_region_from_ir",
]


def _identity(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _version(record):
    if type(record.get("version", 1)) is not int or record.get("version", 1) != 1:
        raise ValueError("unsupported kernel evidence version")


def _sha256(value):
    return type(value) is str and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


@dataclass(frozen=True)
class ArtifactPin:
    """An exact source, ELF, fact, contract, capture, tool or runtime input."""

    kind: str
    uri: str
    sha256: str
    size_bytes: int | None = None

    def __post_init__(self):
        if (type(self.kind) is not str or not self.kind or type(self.uri) is not str or not self.uri
                or not _sha256(self.sha256)
                or (self.size_bytes is not None and (type(self.size_bytes) is not int or self.size_bytes < 0))):
            raise ValueError("artifact pins require a kind, location and exact lowercase SHA256")

    @classmethod
    def from_record(cls, record):
        return cls(**dict(record))

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class EvidenceFact:
    key: str
    value: Any = None
    status: str = "unknown"
    evidence: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self):
        if (type(self.key) is not str or not self.key
                or self.status not in ("observed", "inferred", "contradicted", "unknown")
                or type(self.evidence) is not tuple or any(type(item) is not str or not item for item in self.evidence)
                or (self.status in ("observed", "inferred", "contradicted") and not self.evidence)):
            raise ValueError("known evidence facts require exact evidence pointers and a supported status")
        _identity(self.to_dict())

    @classmethod
    def from_record(cls, record):
        row = dict(record)
        row["evidence"] = tuple(row.get("evidence", ()))
        return cls(**row)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class KernelEvidence:
    evidence_id: str
    artifacts: tuple[ArtifactPin, ...] = ()
    facts: tuple[EvidenceFact, ...] = ()
    capture_kind: str = "static_disassembly"
    complete: bool = False
    visibility: str = "operator_only"
    missing: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self):
        if (type(self.evidence_id) is not str or not self.evidence_id or self.version != 1
                or type(self.artifacts) is not tuple or any(type(pin) is not ArtifactPin for pin in self.artifacts)
                or type(self.facts) is not tuple or any(type(fact) is not EvidenceFact for fact in self.facts)
                or len({fact.key for fact in self.facts}) != len(self.facts)
                or self.capture_kind not in ("source", "static_disassembly", "native_issue", "committed_trace",
                                             "normalized_commands", "compiler_payload", "timing")
                or self.visibility not in ("public", "operator_only", "reference_guided")
                or type(self.complete) is not bool):
            raise ValueError("kernel evidence requires typed, uniquely indexed selected inputs")
        _identity(self.to_dict())

    @classmethod
    def from_record(cls, record):
        _version(record)
        row = dict(record)
        row["artifacts"] = tuple(ArtifactPin.from_record(pin) for pin in row.get("artifacts", ()))
        row["facts"] = tuple(EvidenceFact.from_record(fact) for fact in row.get("facts", ()))
        row["missing"] = tuple(row.get("missing", ()))
        return cls(**row)

    def to_dict(self):
        return asdict(self)

    @property
    def identity(self):
        return _identity(self.to_dict())

    def kinds(self):
        return frozenset(pin.kind for pin in self.artifacts)


@dataclass(frozen=True)
class RegionTensor:
    key: str
    shape: tuple[int | str, ...]
    dtype: str | None
    layout: str | Mapping | None
    alias_group: str | None = None

    def __post_init__(self):
        if (type(self.key) is not str or not self.key or type(self.shape) is not tuple
                or any((type(dim) is int and dim < 0) or type(dim) not in (int, str) for dim in self.shape)
                or (self.dtype is not None and (type(self.dtype) is not str or not self.dtype))
                or (self.layout is not None and not isinstance(self.layout, (str, Mapping)))):
            raise ValueError("region tensors require an identifier and explicit shape/type/layout observations")

    @classmethod
    def from_record(cls, record):
        row = dict(record)
        row["shape"] = tuple(row.get("shape", ()))
        row.setdefault("dtype", None)
        row.setdefault("layout", None)
        return cls(**row)


_NUMERICAL_FIELDS = ("accumulator_dtype", "bias", "scale", "activation", "rounding", "saturation")


@dataclass(frozen=True)
class RegionOperation:
    key: str
    op: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)
    numeric: Mapping[str, Any] = field(default_factory=dict)
    numeric_complete: bool = False

    def __post_init__(self):
        if (any(type(value) is not str or not value for value in (self.key, self.op))
                or type(self.inputs) is not tuple or type(self.outputs) is not tuple or not self.outputs
                or any(type(key) is not str or not key for key in (*self.inputs, *self.outputs))
                or not isinstance(self.attributes, Mapping) or not isinstance(self.numeric, Mapping)
                or type(self.numeric_complete) is not bool):
            raise ValueError("semantic operations require explicit ordered value edges and numerical contracts")

    @classmethod
    def from_record(cls, record):
        row = dict(record)
        for key in ("inputs", "outputs"):
            row[key] = tuple(row.get(key, ()))
        row.setdefault("numeric_complete", all(name in row.get("numeric", {}) for name in _NUMERICAL_FIELDS))
        return cls(**row)


@dataclass(frozen=True)
class MeasurementBoundary:
    """Explicit cost boundary; unknown values cannot qualify a cycle comparison."""

    scope: str = "unknown"
    packing: str = "unknown"
    preparation: str = "unknown"
    conversions: str = "unknown"
    synchronization: str = "unknown"
    readout: str = "unknown"
    input_identity: str | None = None
    runtime_identity: str | None = None

    def __post_init__(self):
        if (self.scope not in ("unknown", "complete_region", "device_only")
                or any(getattr(self, field) not in ("included", "excluded", "not_applicable", "unknown")
                       for field in ("packing", "preparation", "conversions", "synchronization", "readout"))
                or any(value is not None and (type(value) is not str or not value)
                       for value in (self.input_identity, self.runtime_identity))):
            raise ValueError("measurement boundaries require explicit inclusion and input/runtime identities")

    @classmethod
    def from_record(cls, record):
        return cls(**dict(record))


@dataclass(frozen=True)
class SemanticRegion:
    """Complete semantic DAG including shared operands and observable intermediates.

    Operations appear in source topological order. Exact ordering is a
    conservative graph correspondence; schedule ordering and dispatch grouping
    live in ScheduleModel rather than changing semantic equivalence.
    """

    region_id: str
    tensors: tuple[RegionTensor, ...]
    operations: tuple[RegionOperation, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    external_uses: Mapping[str, int] = field(default_factory=dict)
    boundary: MeasurementBoundary = field(default_factory=MeasurementBoundary)
    complete: bool = False
    version: int = 1

    def __post_init__(self):
        if (type(self.region_id) is not str or not self.region_id or self.version != 1
                or type(self.complete) is not bool or type(self.boundary) is not MeasurementBoundary
                or type(self.tensors) is not tuple or any(type(tensor) is not RegionTensor for tensor in self.tensors)
                or type(self.operations) is not tuple or any(type(op) is not RegionOperation for op in self.operations)
                or type(self.inputs) is not tuple or type(self.outputs) is not tuple
                or not isinstance(self.external_uses, Mapping)):
            raise ValueError("semantic regions need typed source graphs and a declared boundary")
        tensors = {tensor.key for tensor in self.tensors}
        if (len(tensors) != len(self.tensors) or len({op.key for op in self.operations}) != len(self.operations)
                or len(set(self.inputs)) != len(self.inputs) or len(set(self.outputs)) != len(self.outputs)
                or not set((*self.inputs, *self.outputs, *self.external_uses)) <= tensors
                or any(type(count) is not int or count < 1 for count in self.external_uses.values())):
            raise ValueError("semantic region values, live outputs or observers are malformed")
        available = set(self.inputs)
        for operation in self.operations:
            if (not set(operation.inputs) <= available or not set(operation.outputs) <= tensors
                    or len(set(operation.outputs)) != len(operation.outputs) or set(operation.outputs) & available):
                raise ValueError("semantic operations must be a topological SSA graph")
            available.update(operation.outputs)
        if not set((*self.outputs, *self.external_uses)) <= available or tensors != available:
            raise ValueError("every declared semantic tensor must have an input or producer")
        _identity(self.to_dict())

    @classmethod
    def from_record(cls, record):
        _version(record)
        row = dict(record)
        row["tensors"] = tuple(RegionTensor.from_record(tensor) for tensor in row.get("tensors", ()))
        row["operations"] = tuple(RegionOperation.from_record(op) for op in row.get("operations", ()))
        for key in ("inputs", "outputs"):
            row[key] = tuple(row.get(key, ()))
        row["boundary"] = MeasurementBoundary.from_record(row.get("boundary", {}))
        return cls(**row)

    def to_dict(self):
        return asdict(self)

    @property
    def identity(self):
        return _identity(self.to_dict())

    def canonical_semantics(self):
        """Rename source SSA identities without losing shared inputs or aliases."""
        tensors = {tensor.key: tensor for tensor in self.tensors}
        names = {key: f"input:{index}" for index, key in enumerate(self.inputs)}
        for index, operation in enumerate(self.operations):
            names.update({key: f"value:{index}:{result}" for result, key in enumerate(operation.outputs)})
        alias_names = {}
        values = []
        for key, name in names.items():
            tensor = tensors[key]
            if tensor.alias_group is not None:
                alias_names.setdefault(tensor.alias_group, f"alias:{len(alias_names)}")
            values.append(dict(key=name, shape=tensor.shape, dtype=tensor.dtype, layout=tensor.layout,
                               alias_group=alias_names.get(tensor.alias_group)))
        operations = [dict(op=operation.op, inputs=tuple(names[key] for key in operation.inputs),
                           outputs=tuple(names[key] for key in operation.outputs),
                           attributes=dict(operation.attributes), numeric=dict(operation.numeric),
                           numeric_complete=operation.numeric_complete) for operation in self.operations]
        return dict(tensors=values, operations=operations, inputs=tuple(names[key] for key in self.inputs),
                    outputs=tuple(names[key] for key in self.outputs),
                    external_uses={names[key]: count for key, count in self.external_uses.items()})


@dataclass(frozen=True)
class RegionDifference:
    path: str
    reference: Any
    candidate: Any
    status: str
    reason: str


@dataclass(frozen=True)
class RegionComparison:
    status: str
    differences: tuple[RegionDifference, ...]
    measurement_status: str
    measurement_differences: tuple[RegionDifference, ...]
    reference_identity: str
    candidate_identity: str
    version: int = 1

    @property
    def comparable(self):
        return self.status == self.measurement_status == "equivalent"

    def to_dict(self):
        return {**asdict(self), "comparable": self.comparable}


def _differences(a, b, path=""):
    if a is None or b is None or a == "unknown" or b == "unknown":
        return [RegionDifference(path, a, b, "unknown", "unresolved semantic or measurement field")]
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        out = []
        for key in sorted(set(a) | set(b)):
            out.extend(_differences(a.get(key), b.get(key), f"{path}.{key}" if path else str(key)))
        return out
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return [RegionDifference(path, a, b, "mismatch", "different semantic graph or interface cardinality")]
        return [difference for index, (left, right) in enumerate(zip(a, b))
                for difference in _differences(left, right, f"{path}[{index}]",)]
    if type(a) is not type(b) or a != b:
        return [RegionDifference(path, a, b, "mismatch", "semantic graph, numerical or boundary mismatch")]
    return []


def _status(differences):
    statuses = {difference.status for difference in differences}
    return "mismatch" if "mismatch" in statuses else "unknown" if "unknown" in statuses else "equivalent"


def compare_regions(reference: SemanticRegion, candidate: SemanticRegion) -> RegionComparison:
    """Conservative source matching plus a separate apples-to-apples timing gate.

    Semantic matching can proceed without a runtime observation. Ranking cycles
    additionally requires ``result.comparable``, which includes preparation,
    packing, conversion, synchronization, readout and input/runtime identity.
    """
    if type(reference) is not SemanticRegion or type(candidate) is not SemanticRegion:
        raise ValueError("region comparison requires typed source regions")
    left, right = reference.canonical_semantics(), candidate.canonical_semantics()
    # None alias groups explicitly mean no declared alias; they are known values.
    for graph in (left, right):
        for tensor in graph["tensors"]:
            if tensor["alias_group"] is None:
                tensor["alias_group"] = "none"
        for operation in graph["operations"]:
            del operation["numeric_complete"]
    differences = _differences(left, right)
    if not reference.complete or not candidate.complete:
        differences.append(RegionDifference("complete", reference.complete, candidate.complete, "unknown",
                                            "complete fusion/semantic boundary has not been established"))
    for label, region in (("reference", reference), ("candidate", candidate)):
        for index, operation in enumerate(region.operations):
            if not operation.numeric_complete:
                differences.append(RegionDifference(f"{label}.operations[{index}].numeric_complete", True, False,
                                                    "unknown", "numerical boundary is incomplete"))
    boundary_differences = _differences(asdict(reference.boundary), asdict(candidate.boundary), "boundary")
    return RegionComparison(_status(differences), tuple(differences), _status(boundary_differences),
                            tuple(boundary_differences), reference.identity, candidate.identity)


def _attribute_record(attribute):
    """Stable structural xDSL attributes, retaining unsupported kinds as syntax."""
    from io import StringIO

    from xdsl.printer import Printer

    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_attribute(attribute)
    return stream.getvalue()


def semantic_region_from_ir(
    module, *, operations=None, numeric_projector=None, layout_projector=None, boundary=None,
    region_id="frontend", complete=True,
) -> SemanticRegion:
    """Extract a source region automatically from actual xDSL SSA.

    A selected adapter supplies a numerical-contract projector once for its
    source dialect. Without one, explicit complete ``numeric.*`` attributes are
    consumed; missing fields remain unknown. Nested bodies are retained as
    structural operation trees, including implicit captures and block arguments.
    No quantizer or fusion permission is guessed from an operation's name.
    """
    from xdsl.dialects.builtin import MemRefType, NoneAttr, TensorType
    from xdsl.traits import IsTerminator

    members = tuple(module.walk())
    if operations is None:
        functions = [op for op in members if op.name == "func.func"]
        subject = functions[0] if len(functions) == 1 else module
        if len(subject.regions) != 1 or len(subject.regions[0].blocks) != 1:
            raise ValueError("automatic region extraction needs one source block or explicit operations")
        operations = tuple(op for op in subject.regions[0].block.ops
                           if not op.has_trait(IsTerminator, value_if_unregistered=False))
    operations = tuple(operations)
    if (not operations or len(set(operations)) != len(operations)
            or any(op not in members or not op.results for op in operations)
            or any(op.parent is not operations[0].parent for op in operations)):
        raise ValueError("semantic extraction requires distinct result-producing roots in one current source block")
    positions = {op: index for index, op in enumerate(members)}
    if tuple(positions[op] for op in operations) != tuple(sorted(positions[op] for op in operations)):
        raise ValueError("semantic extraction preserves source topological order")
    selected = {child for root in operations for child in root.walk()}
    names, input_values, tensors = {}, [], []

    def value_name(value):
        if value not in names:
            names[value] = f"v{len(names)}"
            typ = value.type
            shape = tuple(dim if dim >= 0 else "?" for dim in typ.get_shape()) if hasattr(typ, "get_shape") else ()
            dtype = str(typ.get_element_type()) if hasattr(typ, "get_element_type") else str(typ)
            if layout_projector is not None:
                layout = layout_projector(value)
            elif isinstance(typ, TensorType):
                layout = "logical" if isinstance(typ.encoding, NoneAttr) else _attribute_record(typ.encoding)
            elif isinstance(typ, MemRefType):
                layout = "identity" if isinstance(typ.layout, NoneAttr) else _attribute_record(typ.layout)
            else:
                layout = "scalar"
            tensors.append(RegionTensor(names[value], shape, dtype, layout))
        return names[value]

    root_inputs = {}
    for root in operations:
        captured = []
        root_members = set(root.walk())
        for child in root.walk():
            for value in child.operands:
                owner = value.owner
                parent = owner.parent_op() if hasattr(owner, "parent_op") else None
                local = owner in selected or parent in selected
                root_local = owner in root_members or parent in root_members
                if value not in captured and (child is root or not root_local):
                    captured.append(value)
                if not local and value not in input_values:
                    input_values.append(value)
                    value_name(value)
        root_inputs[root] = captured

    def body_tree(root):
        def region_record(region, outer):
            blocks = []
            for block_index, block in enumerate(region.blocks):
                local = dict(outer)
                for index, arg in enumerate(block.args):
                    local[arg] = f"block:{block_index}:arg:{index}"
                children = []
                for index, child in enumerate(block.ops):
                    operands = [local.get(value, names.get(value, "unresolved_capture")) for value in child.operands]
                    for result_index, result in enumerate(child.results):
                        local[result] = f"block:{block_index}:op:{index}:result:{result_index}"
                    children.append(dict(op=child.name, operands=operands,
                                         types=[str(result.type) for result in child.results],
                                         attributes={key: _attribute_record(value)
                                                     for key, value in child.attributes.items()
                                                     if not key.startswith("prov.")},
                                         properties={key: _attribute_record(value)
                                                     for key, value in child.properties.items()},
                                         regions=[region_record(nested, local) for nested in child.regions]))
                blocks.append(dict(arguments=[str(arg.type) for arg in block.args], operations=children))
            return blocks
        return [region_record(region, names) for region in root.regions]

    region_operations = []
    for index, root in enumerate(operations):
        inputs = tuple(value_name(value) for value in root_inputs[root])
        outputs = tuple(value_name(value) for value in root.results)
        attributes = {key: _attribute_record(value) for key, value in root.attributes.items()
                      if not key.startswith("prov.") and not key.startswith("numeric.")}
        attributes.update({f"property:{key}": _attribute_record(value) for key, value in root.properties.items()})
        if root.regions:
            attributes["regions"] = body_tree(root)
        if numeric_projector is not None:
            numerical = numeric_projector(root)
            if numerical is None:
                numerical = {}
            if not isinstance(numerical, Mapping):
                raise ValueError("numerical projectors must return explicit contract mappings")
            numerical = dict(numerical)
            numeric_complete = numerical.pop("complete", all(key in numerical for key in _NUMERICAL_FIELDS))
        else:
            numerical = {key.removeprefix("numeric."): _attribute_record(value)
                         for key, value in root.attributes.items() if key.startswith("numeric.")}
            numeric_complete = all(key in numerical for key in _NUMERICAL_FIELDS)
        region_operations.append(RegionOperation(f"op{index}", root.name, inputs, outputs,
                                                 attributes, numerical, numeric_complete))
    outputs, external = [], {}
    for root in operations:
        for value in root.results:
            uses = sum(use.operation not in selected for use in value.uses)
            if uses:
                key = value_name(value)
                outputs.append(key)
                external[key] = uses
    return SemanticRegion(region_id, tuple(tensors), tuple(region_operations),
                          tuple(value_name(value) for value in input_values), tuple(outputs), external,
                          boundary if boundary is not None else MeasurementBoundary(), complete)


def analyze_disassembly(text: str, *, table: Mapping, evidence: KernelEvidence, roles_of=None) -> ScheduleModel:
    """Use the existing tokenizer and the explicitly selected source-derived decoder."""
    from merlin.compile.analysis.disassembly import tokenize_text
    from merlin.compile.analysis.rocc_commands import audit_decoded, decode_stream

    if evidence.capture_kind != "static_disassembly" or "disassembly" not in evidence.kinds():
        raise ValueError("static analysis requires a pinned disassembly evidence input")
    if not {"facts", "isa"} & evidence.kinds():
        raise ValueError("static decoding requires a pinned selected fact or ISA input")
    decoded = decode_stream(tokenize_text(text), table, roles_of=roles_of)
    audit = audit_decoded(decoded, table=table)
    unaccounted = {item["index"] for item in audit.unaccounted}
    records = [dict(index=item.index, identity=item.identity, roles=item.roles,
                    parameters=dict(fields=item.fields, operands=item.operands, pc=item.addr,
                                    unaccounted=item.index in unaccounted))
               for item in decoded if item.from_endpoint or item.index in unaccounted]
    missing_decoder = table.get("custom_opcode") is None or not table.get("legal_funct")
    model = analyze_normalized_commands(records, evidence_id=evidence.evidence_id, source_kind="static_disassembly",
                                        complete=evidence.complete and not unaccounted and not missing_decoder)
    unknowns = set(model.unknowns)
    if unaccounted:
        unknowns.add("unaccounted_instruction_decode")
    if missing_decoder:
        unknowns.add("selected_instruction_decoder_unavailable")
    return replace(model, unknowns=tuple(sorted(unknowns)))


def analyze_native_commands(
    commands: Iterable[Mapping], *, isa: Mapping, evidence: KernelEvidence,
    project_command: Callable[[Mapping], Mapping] | None = None, max_commands=100_000,
) -> ScheduleModel:
    """Decode concrete native issue operands; this capture supplies no target cycles."""
    from merlin.targetgen.rocc.semantics import decode_instruction

    if (evidence.capture_kind != "native_issue" or "capture" not in evidence.kinds()
            or not {"facts", "isa"} & evidence.kinds()):
        raise ValueError("native decoding requires pinned capture and selected ISA inputs")

    def decoded():
        for index, command in enumerate(commands):
            if not isinstance(command, Mapping):
                raise ValueError("native issue captures must contain typed operand records")
            if command.get("kind") == "fence":
                row = dict(index=index, **{"class": "FENCE"}, decoded={})
            else:
                values = [command.get(name) for name in ("funct", "rs1", "rs2")]
                if any(type(value) is not int or value < 0 for value in values):
                    raise ValueError("native commands require actual nonnegative register values")
                funct, rs1, rs2 = values
                cls, operands = decode_instruction(funct, dict(raw=rs1, kind="const"),
                                                    dict(raw=rs2, kind="const"), dict(isa))
                row = dict(command, index=index, **{"class": cls}, decoded=operands)
            normalized = dict(project_command(row) if project_command else {})
            normalized.setdefault("index", index)
            normalized.setdefault("identity", row["class"])
            normalized.setdefault("parameters", dict(decoded=row["decoded"]))
            normalized.setdefault("kind", "fence" if command.get("kind") == "fence" else "command")
            yield normalized

    return analyze_normalized_commands(decoded(), evidence_id=evidence.evidence_id, complete=evidence.complete,
                                       source_kind="native_issue", max_commands=max_commands)


def analyze_committed_trace(
    view, *, isa: Mapping, evidence: KernelEvidence, project_command: Callable[[Mapping], Mapping] | None = None,
    max_retired=200_000_000, max_commands=100_000,
) -> ScheduleModel:
    """Replay an explicitly complete single-hart trace through the existing reader.

    A partial snippet cannot initialize registers and is refused. Completeness
    is a pinned capture assertion, independent of modeled/physical timing.
    """
    from merlin.perf.streamed_executed_trace import iter_source_commands

    if (evidence.capture_kind != "committed_trace" or not evidence.complete
            or "capture" not in evidence.kinds() or not {"facts", "isa"} & evidence.kinds()):
        raise ValueError("register replay requires pinned complete original capture and selected ISA")

    def normalized():
        for index, row in enumerate(iter_source_commands(view, isa=dict(isa), max_retired=max_retired,
                                                         max_commands=max_commands)):
            record = dict(project_command(row) if project_command else {})
            record.setdefault("index", index)
            record.setdefault("identity", row["class"])
            record.setdefault("parameters", dict(decoded=row.get("decoded", {})))
            record.setdefault("kind", "fence" if row["class"] == "FENCE" else "command")
            yield record

    return analyze_normalized_commands(normalized(), evidence_id=evidence.evidence_id, complete=True,
                                       source_kind="committed_trace", max_commands=max_commands)


def cca_projection(model: ScheduleModel, *, op: str, source: str = "reference", backend=(), facts=(), scope="kernel"):
    """Populate only proven summary fields, retaining the existing CCA API."""
    from dataclasses import fields, is_dataclass
    from typing import Any, get_args, get_origin, get_type_hints

    from merlin.kernels.cca import CCA, DispatchFacet

    result = CCA(op=op, backend=list(backend), provenance=dict(level=model.source_kind, source=source,
                                                           schedule_sha256=model.identity,
                                                           unknowns=model.unknowns, axis_evidence={}), scope=scope)
    axes = result.provenance["axis_evidence"]
    if model.source_kind not in ("schedule_dialect", "static_disassembly"):
        count = model.event_counts.get("command") if model.complete else None
        result.dispatch = DispatchFacet(n_dispatches=count)
        axes["dispatch.n_dispatches"] = dict(status="observed" if count is not None else "unknown",
                                             evidence=(f"schedule:{model.identity}#event_counts.command",),
                                             reason="complete bounded issue/payload count, not elapsed timing")
    types = get_type_hints(CCA)

    def compatible(value, annotation):
        if annotation is Any:
            return True
        if annotation is tuple:
            return isinstance(value, (tuple, list))
        origin = get_origin(annotation)
        if origin is None:
            return type(value) is annotation
        if origin in (tuple, list):
            if not isinstance(value, (tuple, list)):
                return False
            arguments = get_args(annotation)
            if not arguments:
                return True
            if len(arguments) == 2 and arguments[1] is Ellipsis:
                return all(compatible(item, arguments[0]) for item in value)
            return len(value) == len(arguments) and all(compatible(item, typ) for item, typ in zip(value, arguments))
        return any(compatible(value, argument) for argument in get_args(annotation))

    seen = set()
    for fact in facts:
        if type(fact) is not EvidenceFact or fact.key in seen:
            raise ValueError("CCA projection requires uniquely indexed typed evidence facts")
        seen.add(fact.key)
        facet_name, separator, field_name = fact.key.partition(".")
        candidates = get_args(types.get(facet_name, None)) or (types.get(facet_name),)
        classes = [candidate for candidate in candidates if is_dataclass(candidate)]
        if (not separator or len(classes) != 1
                or field_name not in {item.name for item in fields(classes[0])}):
            raise ValueError(f"evidence axis {fact.key!r} is not an existing CCA facet field")
        axes[fact.key] = dict(status=fact.status, evidence=fact.evidence, reason=fact.reason)
        if fact.status not in ("observed", "inferred") or fact.value is None:
            continue
        annotation = get_type_hints(classes[0])[field_name]
        if not compatible(fact.value, annotation):
            raise ValueError(f"evidence axis {fact.key!r} has a value incompatible with its existing CCA type")
        facet = getattr(result, facet_name)
        if facet is None:
            facet = classes[0]()
            setattr(result, facet_name, facet)
        current = getattr(facet, field_name)
        if current is not None and current != fact.value:
            raise ValueError(f"contradictory evidence for CCA axis {fact.key!r}")
        candidates = get_args(annotation) or (annotation,)
        is_tuple = any(candidate is tuple or get_origin(candidate) is tuple for candidate in candidates)
        setattr(facet, field_name, tuple(fact.value) if is_tuple and isinstance(fact.value, list) else fact.value)
    result.provenance["unknown_axes"] = tuple(sorted(key for key, value in axes.items()
                                                     if value["status"] in ("unknown", "contradicted")))
    if result.scope_problems():
        raise ValueError("CCA projection facts contradict the selected semantic scope")
    return result
