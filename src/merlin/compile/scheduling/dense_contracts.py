"""Fact-bound admission for the existing compact dense scheduling family.

RTL references constrain geometry and physical windows. Software declares
ordered modular arithmetic, effects and execution. Assigned windows come from
allocation; this module never treats a whole memory as free scratch space.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType

from merlin.sched.ir import Ptr, as_expr
from merlin.sched.ir.kernel import DTYPE_BYTES
from merlin.targetgen.isa_spec import resolve_fact_integer
from merlin.transforms.dense_planning import DenseProblem, DenseResources

from .block_schedule import Compute, Load, Preload, Store
from .program_limits import AddressWindow

ROLES = ("lhs", "rhs", "accumulator")
OPERATIONS = ("lhs_load", "rhs_load", "preload", "compute", "store")

DENSE_FIELDS = ("schema", "primitive", "formats", "storage", "external", "instructions", "semantics", "support")
DENSE_SEMANTICS = MappingProxyType({
    "layout": "row_major", "arithmetic": "signed_integer_mac_modular", "reduction": "source_order",
    "initialization": "zero", "continuation": "exact", "completion": "synchronous",
    "accumulator": "addressable", "ownership": "invocation_private",
})
DENSE_SUPPORT = ("max_m_blocks", "max_k_blocks", "max_n_blocks", "partial_output_tiles",
                 "partial_output_blocks", "partial_reduction_tiles")
DENSE_POINTER_FIELDS = ("schema", "layout", "alignment_bytes", "access", "bounds", "completion")
DENSE_STORAGE_FIELDS = ("resource", "backing", "row_bytes", "origin_bytes", "alignment_rows")


def dense_contract_draft(*, schema):
    """Editable shape of the existing resolver, with every target assertion unknown.

    Supported family assumptions are exposed separately as DENSE_SEMANTICS; they
    are not inferred target properties. No instruction, geometry, allocation,
    policy, completion behavior or qualification is selected by this template.
    """
    if schema not in ("merlin.dense_schedule.v1", "merlin.dense_schedule.v2"):
        raise ValueError("unknown dense scheduling contract version")
    row = {
        "schema": schema,
        "primitive": {axis: {"fact": None} for axis in ("m", "k", "n")},
        "formats": {role: {"dtype": None, "bits": {"fact": None}} for role in ROLES},
        "storage": {role: {key: ({"fact": None} if key in
                      ("row_bytes", "origin_bytes", "alignment_rows") else None)
                      for key in DENSE_STORAGE_FIELDS} for role in ROLES},
        "external": {role: None for role in ("lhs", "rhs", "output")},
        "instructions": {operation: None for operation in OPERATIONS},
        "semantics": {key: None for key in DENSE_SEMANTICS},
        "support": {key: None for key in DENSE_SUPPORT},
    }
    if schema == "merlin.dense_schedule.v2":
        row["logical_pointer_abi"] = {key: None for key in DENSE_POINTER_FIELDS}
        row["logical_pointer_abi"]["schema"] = "merlin.logical_pointer_abi.v1"
        row["external_pointer_arguments"] = {
            operation: {"argument": None, "space": None} for operation in ("lhs_load", "rhs_load", "store")}
    return row


class DenseSchedulingUnavailable(ValueError):
    """Declared capabilities cannot implement this family; retain a legal source route."""


def _fields(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"dense {label}: expected exactly {sorted(keys)}")
    return value


@dataclass(frozen=True)
class DenseStoreContract:
    resource: str
    backing: str
    row_bytes: int
    origin_bytes: int
    capacity_bytes: int | None
    alignment_rows: int


@dataclass(frozen=True)
class DenseLogicalPointerABI:
    """Declared logical pointer access, separate from physical tile storage.

    These software declarations do not qualify an implementation. Original
    independent format/effect qualification remains a separate caller input.
    """

    layout: str
    alignment_bytes: tuple[int, int, int]
    access: tuple[str, str, str]
    bounds: str
    completion: str


@dataclass(frozen=True)
class DenseIntegerReadout:
    """One declared unary integer output path; qualification is separate."""

    scalar_ir: str
    output_dtype: str
    instruction_name: str
    instruction_arguments: tuple[tuple[str, int], ...]

    @property
    def identity(self):
        import hashlib

        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def graph(self):
        from .fusion_semantics import scalar_graph

        # A signed-i8 monotone readout has at most255 transitions. Its
        # complete scalar decision representation requires at most1021 ops.
        # Keep both operation and byte bounds; no loops/effects become legal.
        return scalar_graph(self.scalar_ir.encode(), 1024, 65536)


def _f32(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


@dataclass(frozen=True)
class TiledIntegerStore(Store):
    """An actual store names its original coefficient-table tile explicitly."""

    readout_tile: object = None


@dataclass(frozen=True)
class DenseTiledIntegerReadout:
    """Exact i32 -> binary32 scale -> nearest-even saturated i8 semantics.

    The complete original column vector is retained as bits, including repeated
    coefficients. Tile width is supplied by the selected primitive, never guessed
    from a model or target name. No reassociation or coefficient fitting occurs.
    """

    column_scale_bits: tuple[int, ...]
    tile_columns: int
    relu: bool
    instruction_name: str
    instruction_arguments: tuple[tuple[str, int], ...] = ()

    def __post_init__(self):
        if (type(self.column_scale_bits) is not tuple or not self.column_scale_bits
                or len(self.column_scale_bits) > 262144
                or type(self.tile_columns) is not int or not 0 < self.tile_columns < 2**31
                or type(self.relu) is not bool
                or type(self.instruction_name) is not str or not self.instruction_name
                or type(self.instruction_arguments) is not tuple
                or any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not str
                       or not row[0] or type(row[1]) is not int or row[1] <= 0
                       for row in self.instruction_arguments)):
            raise ValueError("tiled integer readout requires bounded explicit source and instruction data")
        for bits in self.column_scale_bits:
            if type(bits) is not int or not 0 <= bits < 2**32:
                raise ValueError("source scale must retain its exact binary32 bits")
            value = struct.unpack("<f", struct.pack("<I", bits))[0]
            if not math.isfinite(value) or value <= 0:
                raise ValueError("tiled readout requires positive finite binary32 source scales")
        for start in range(0, self.columns, self.tile_columns):
            if len(set(self.column_scale_bits[start:start + self.tile_columns])) != 1:
                raise ValueError("source scales vary within a selected output tile; retain the original route")

    @classmethod
    def from_column_scales(cls, scales, *, tile_columns, relu=False,
                           instruction_name="store_tiled_i8", instruction_arguments=()):
        if type(scales) not in (tuple, list) or any(type(x) not in (int, float) for x in scales):
            raise ValueError("one explicit numeric binary32 scale per source column required")
        bits = tuple(struct.unpack("<I", struct.pack("<f", _f32(x)))[0] for x in scales)
        return cls(bits, tile_columns, relu, instruction_name, instruction_arguments)

    @property
    def columns(self):
        return len(self.column_scale_bits)

    @property
    def output_dtype(self):
        return "i8"

    @property
    def semantics(self):
        return "signed-i32:relu-before-convert:binary32-rne:single-scale-binary32-rne:signed-i8-rne-saturate"

    @property
    def identity(self):
        record = {"schema": "merlin.dense_tiled_integer_readout.v1", "semantics": self.semantics, **asdict(self)}
        return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()

    @property
    def table_words(self):
        """Immutable source coefficients, one u32 word per original output tile."""
        return self.column_scale_bits[::self.tile_columns]

    def validate_problem(self, problem):
        if (self.columns != problem.n or self.tile_columns != problem.tile_n
                or problem.accumulator_bytes != 4):
            raise ValueError("tiled readout source columns, tile or accumulator width differ from the contraction")

    def evaluate(self, accumulator, column):
        """Neutral exact source oracle; never emitted as host tensor arithmetic."""
        if (type(accumulator) is not int or not -(1 << 31) <= accumulator < 1 << 31
                or type(column) is not int or not 0 <= column < self.columns):
            raise ValueError("readout oracle requires original signed-i32 and column domains")
        scale = struct.unpack("<f", struct.pack("<I", self.column_scale_bits[column]))[0]
        value = _f32(_f32(max(0, accumulator) if self.relu else accumulator) * scale)
        low = 0 if self.relu else -128
        return low if value <= low else 127 if value >= 127 else max(low, min(127, round(value)))


def _logical_pointer_abi(row):
    if row is None:
        return None
    row = _fields(row, DENSE_POINTER_FIELDS, "logical pointer ABI")
    alignments, access = row["alignment_bytes"], row["access"]
    if (
        row["schema"] != "merlin.logical_pointer_abi.v1"
        or type(alignments) is not list
        or len(alignments) != 3
        or any(type(a) is not int or not 0 < a < 1 << 63 or a & (a - 1) for a in alignments)
        or type(access) is not list
        or len(access) != 3
        or any(type(v) is not str or not v for v in (*access, row["layout"], row["bounds"], row["completion"]))
    ):
        raise ValueError("dense logical pointer ABI requires versioned explicit layout, alignment and effects")
    return DenseLogicalPointerABI(row["layout"], tuple(alignments), tuple(access), row["bounds"], row["completion"])


@dataclass(frozen=True)
class DenseSchedulingContract:
    primitive: tuple[int, int, int]
    dtypes: tuple[str, str, str]
    stores: tuple[DenseStoreContract, ...]
    instruction_names: tuple[str, ...]
    instruction_arguments: tuple[tuple[tuple[str, int], ...], ...]
    semantics: tuple[tuple[str, str], ...]
    block_limits: tuple[int, int, int]
    partial_output_tiles: bool
    partial_output_blocks: bool
    partial_reduction_tiles: bool
    logical_pointer_abi: DenseLogicalPointerABI | None = None
    output_readout: DenseIntegerReadout | DenseTiledIntegerReadout | None = None
    external_pointer_arguments: tuple[tuple[str, str], ...] = ()

    def problem(self, m, k, n):
        """Build a problem from computation extents and selected RTL geometry."""
        return DenseProblem(m, k, n, *self.primitive, *(DTYPE_BYTES[dtype] for dtype in self.dtypes))

    def _resource_limits(self, storage, resources=None):
        """Return policy ceilings, without admitting storage or execution.

        Assigned byte windows and declared block limits bound optional search
        restrictions. This calculation establishes no legality or qualification;
        bind still checks physical backing, reservations, numerical semantics
        and the independently qualified adapter before using these ceilings.
        """
        views = (storage.lhs, storage.rhs, storage.accumulator)
        if storage.accumulator is None:
            raise ValueError("dense resource limits require assigned accumulator storage")
        derived = DenseResources(
            sum(v.capacity * v.unit_bytes for v in views[:2]),
            views[2].capacity * views[2].unit_bytes,
            *self.block_limits,
            partial_output_tiles=self.partial_output_tiles,
            partial_output_blocks=self.partial_output_blocks,
        )
        if resources is not None:
            derived = DenseResources(
                *(
                    min(getattr(resources, name), getattr(derived, name))
                    for name in ("operand_bytes", "accumulator_bytes", "max_m_blocks", "max_n_blocks", "max_k_blocks")
                ),
                candidate_budget=resources.candidate_budget,
                partial_output_tiles=resources.partial_output_tiles and self.partial_output_tiles,
                partial_output_blocks=resources.partial_output_blocks and self.partial_output_blocks,
            )

        return derived

    def bind(
        self,
        problem,
        storage,
        adapter,
        *,
        dtypes,
        hardware,
        policy,
        resources=None,
        reservations=(),
        output_readout=None,
    ):
        """Check allocation outputs and clamp optional search budgets to capabilities."""
        expected = DENSE_SEMANTICS
        for name, value in self.semantics:
            if value != expected[name]:
                raise DenseSchedulingUnavailable(f"dense family requires {name}={expected[name]}; selected {value}")
        if output_readout is not None and output_readout != self.output_readout:
            raise ValueError("dense output readout differs from the original selected declaration")
        if type(output_readout) is DenseTiledIntegerReadout:
            output_readout.validate_problem(problem)
        if problem != self.problem(problem.m, problem.k, problem.n) or tuple(dtypes) != self.dtypes:
            raise ValueError("dense problem or dtypes disagree with selected RTL geometry and software formats")
        if (
            adapter.configuration_identity != hardware
            or adapter.numerical_policy_identity != policy
            or not adapter.qualification_identity
            or not adapter.synchronous_completion
            or not adapter.exact_accumulator_continuation
        ):
            raise ValueError("dense adapter requires matching selected contracts and independent qualification")
        if not self.partial_reduction_tiles and problem.k % problem.tile_k:
            raise DenseSchedulingUnavailable("selected dense instructions do not support a partial reduction tile")
        if any(bits < adapter.index_bits for arguments in self.instruction_arguments for _, bits in arguments):
            raise ValueError("dense instruction integer operands are narrower than the checked index representation")
        if output_readout is not None and any(
            bits < adapter.index_bits for _, bits in output_readout.instruction_arguments
        ):
            raise ValueError("dense output readout operands are narrower than the checked index representation")
        views = (storage.lhs, storage.rhs, storage.accumulator)
        if storage.accumulator is None:
            raise ValueError("dense family requires assigned addressable accumulator storage")
        if type(reservations) is not tuple or any(type(r) is not AddressWindow for r in reservations):
            raise ValueError("dense reservations require explicit canonical backing byte windows")
        for view, selected in zip(views, self.stores):
            if selected.capacity_bytes is None:
                raise DenseSchedulingUnavailable(f"dense storage {selected.resource}: RTL capacity is unknown")
            start = view.base * view.unit_bytes
            end = start + view.capacity * view.unit_bytes
            if (
                view.backing != selected.backing
                or view.unit_bytes != selected.row_bytes
                or view.base % selected.alignment_rows
                or view.alignment % selected.alignment_rows
                or start < selected.origin_bytes
                or end > selected.origin_bytes + selected.capacity_bytes
            ):
                raise ValueError("dense assigned storage disagrees with selected RTL window, backing or alignment")
            if any(r.space == view.backing and start < r.start + r.size and r.start < end for r in reservations):
                raise ValueError("dense assigned storage overlaps packing, staging or live storage reservations")
        derived = self._resource_limits(storage, resources)

        pointer_arguments = dict(self.external_pointer_arguments)

        def pointer(operation):
            if type(operation) is Load:
                if operation.role == "lhs":
                    return "lhs_load", Ptr(
                        "lhs", (as_expr(operation.dram_row) * problem.k + operation.dram_col) * problem.lhs_bytes
                    )
                if operation.role == "weight":
                    return "rhs_load", Ptr(
                        "rhs", (as_expr(operation.dram_row) * problem.n + operation.dram_col) * problem.rhs_bytes
                    )
            if type(operation) in (Store, TiledIntegerStore):
                output_bytes = problem.accumulator_bytes if output_readout is None else 1
                return "store", Ptr(
                    "output", (as_expr(operation.dram_row) * problem.n + operation.dram_col) * output_bytes
                )
            return None, None

        def encode(operation):
            call = adapter.encode(operation)
            kind, value = pointer(operation)
            if kind in pointer_arguments:
                key = pointer_arguments[kind]
                keys = [name for name, _ in call.args]
                if key not in keys:
                    arguments = (
                        output_readout.instruction_arguments
                        if kind == "store" and output_readout is not None
                        else self.instruction_arguments[OPERATIONS.index(kind)]
                    )
                    position = tuple(name for name, _ in arguments).index(key)
                    call = replace(call, args=(*call.args[:position], (key, value), *call.args[position:]))
                elif keys.count(key) != 1 or dict(call.args)[key] != value:
                    raise ValueError("dense encoded external pointer disagrees with its bound logical buffer")
            return call

        def project(call):
            operation = adapter.project(call)
            if isinstance(operation, Load) and operation.role in ("lhs", "weight"):
                kind = "lhs_load" if operation.role == "lhs" else "rhs_load"
            elif type(operation) in (Preload, Compute, Store, TiledIntegerStore):
                kinds = {Preload: "preload", Compute: "compute", Store: "store", TiledIntegerStore: "store"}
                kind = kinds[type(operation)]
            else:
                raise ValueError("dense projection has an unsupported operation or memory effect")
            readout = output_readout if kind == "store" else None
            expected_instruction = (
                readout.instruction_name if readout else self.instruction_names[OPERATIONS.index(kind)]
            )
            if call.instr != expected_instruction:
                raise ValueError("dense actual instruction disagrees with selected software operation binding")
            if kind == "store" and operation.readout != (readout.identity if readout else None):
                raise ValueError("dense actual store readout differs from the checked output path")
            keys = tuple(key for key, _ in call.args)
            arguments = readout.instruction_arguments if readout else self.instruction_arguments[OPERATIONS.index(kind)]
            expected_keys = tuple(key for key, _ in arguments)
            if len(set(keys)) != len(keys) or set(keys) != set(expected_keys):
                raise ValueError("dense actual operands disagree with selected ISA argument bindings")
            if pointer_arguments:
                _, expected_pointer = pointer(operation)
                pointer_key = pointer_arguments.get(kind)
                for key, value in call.args:
                    if key == pointer_key:
                        if type(value) is not Ptr or value != expected_pointer:
                            raise ValueError("dense actual external pointer disagrees with its bound logical buffer")
                    elif type(value) is Ptr:
                        raise ValueError("dense actual instruction has an undeclared external pointer")
            return operation

        return derived, replace(adapter, encode=encode if pointer_arguments else adapter.encode, project=project)


def resolve_dense_contract(declaration, facts, isa):
    """Read only the explicit versioned dense extension, never infer semantics."""
    if type(declaration) is not dict:
        raise ValueError("dense contract requires a mapping")
    version = declaration.get("schema")
    if version not in ("merlin.dense_schedule.v1", "merlin.dense_schedule.v2"):
        raise ValueError("unknown dense scheduling contract version")
    optional = ("logical_pointer_abi", "output_readout")
    if version == "merlin.dense_schedule.v2":
        optional += ("external_pointer_arguments",)
        if "external_pointer_arguments" not in declaration:
            raise ValueError("dense v2 requires explicit external pointer arguments")
    logical_pointer = _logical_pointer_abi(declaration.get("logical_pointer_abi"))
    if version == "merlin.dense_schedule.v2" and (
        logical_pointer is None
        or (logical_pointer.layout, logical_pointer.access, logical_pointer.bounds, logical_pointer.completion)
        != ("dense_row_major", ("read_only", "read_only", "fully_written"), "logical_elements", "synchronous")
    ):
        raise DenseSchedulingUnavailable("dense v2 requires a synchronous dense row-major logical pointer ABI")
    row = _fields(
        {key: value for key, value in declaration.items() if key not in optional},
        DENSE_FIELDS,
        "contract",
    )
    primitive = _fields(row["primitive"], ("m", "k", "n"), "primitive")
    dimensions = tuple(
        resolve_fact_integer(primitive[key], facts, label="dense primitive " + key, minimum=1)
        for key in ("m", "k", "n")
    )
    formats = _fields(row["formats"], ROLES, "formats")
    dtypes = []
    for role in ROLES:
        item = _fields(formats[role], ("dtype", "bits"), "format")
        if type(item["dtype"]) is not str or item["dtype"] not in ("i8", "i16", "i32"):
            raise ValueError("selected dense family supports signed integer formats only")
        bits = resolve_fact_integer(item["bits"], facts, label="dense element width", minimum=1)
        if bits != 8 * DTYPE_BYTES[item["dtype"]]:
            raise ValueError("dense software dtype disagrees with selected RTL element width")
        dtypes.append(item["dtype"])
    semantics = _fields(
        row["semantics"],
        DENSE_SEMANTICS,
        "semantics",
    )
    if any(type(value) is not str or not value for value in semantics.values()):
        raise ValueError("dense semantics must be explicit strings; missing facts remain unknown")
    support = _fields(
        row["support"],
        DENSE_SUPPORT,
        "support",
    )
    limits = tuple(support[key] for key in ("max_m_blocks", "max_k_blocks", "max_n_blocks"))
    if any(type(v) is not int or v <= 0 for v in limits):
        raise ValueError("dense block limits must be positive software instruction limits")
    for key in ("partial_output_tiles", "partial_output_blocks", "partial_reduction_tiles"):
        if type(support[key]) is not bool:
            raise ValueError("dense tail support must be explicit Boolean")
    storage = _fields(row["storage"], ROLES, "storage")
    resources = {r.name: r for r in isa.resources}
    stores = []
    for role in ROLES:
        item = _fields(
            storage[role], DENSE_STORAGE_FIELDS, "storage role"
        )
        resource = resources.get(item["resource"]) if type(item["resource"]) is str else None
        if resource is None or resource.kind != "memory" or resource.unit != "bytes":
            raise ValueError("dense storage requires declared ISA memory resources in bytes")
        if type(item["backing"]) is not str or not item["backing"]:
            raise ValueError("dense storage requires a canonical physical backing identity")
        stores.append(
            DenseStoreContract(
                resource.name,
                item["backing"],
                resolve_fact_integer(item["row_bytes"], facts, label="dense row bytes", minimum=1),
                resolve_fact_integer(item["origin_bytes"], facts, label="dense origin bytes"),
                resource.capacity,
                resolve_fact_integer(item["alignment_rows"], facts, label="dense row alignment", minimum=1),
            )
        )
    for a in stores:
        for b in stores:
            if a.resource == b.resource and (a.backing, a.origin_bytes) != (b.backing, b.origin_bytes):
                raise ValueError("dense roles cannot disguise aliases of the same ISA memory resource")
    external = _fields(row["external"], ("lhs", "rhs", "output"), "external resources")
    for name in external.values():
        resource = resources.get(name) if type(name) is str else None
        if resource is None or resource.kind != "memory" or resource.unit != "bytes":
            raise ValueError("dense external operands require declared byte memory effects")
    if set(external.values()) & {s.resource for s in stores}:
        raise ValueError("dense external buffers cannot share local storage resource identities")
    pointer_bindings = {}
    if version == "merlin.dense_schedule.v2":
        pointers = _fields(
            declaration["external_pointer_arguments"], ("lhs_load", "rhs_load", "store"), "external pointer arguments"
        )
        for kind, role in (("lhs_load", "lhs"), ("rhs_load", "rhs"), ("store", "output")):
            item = _fields(pointers[kind], ("argument", "space"), "external pointer argument")
            if type(item["argument"]) is not str or not item["argument"] or item["space"] != external[role]:
                raise ValueError("dense external pointer must name its exact declared external resource")
            pointer_bindings[kind] = item["argument"]
    instructions = _fields(row["instructions"], OPERATIONS, "instruction bindings")
    if any(type(name) is not str for name in instructions.values()) or len(set(instructions.values())) != len(
        OPERATIONS
    ):
        raise ValueError("dense operation bindings require distinct explicit instruction names")
    lhs, rhs, acc = (s.resource for s in stores)
    memory_effects = (
        ({external["lhs"]}, {lhs}),
        ({external["rhs"]}, {rhs}),
        ({rhs, acc}, set()),
        ({lhs, rhs, acc}, {acc}),
        ({acc}, {external["output"]}),
    )
    arguments = []
    for kind, (reads, writes) in zip(OPERATIONS, memory_effects):
        instruction = isa.instruction(instructions[kind])
        pointer_name = pointer_bindings.get(kind)
        if pointer_name is not None and pointer_name not in {a.name for a in instruction.arguments()}:
            raise ValueError("dense declared external pointer is missing from the actual ISA instruction")
        for arg in instruction.arguments():
            if arg.name == pointer_name:
                role = {"lhs_load": "lhs", "rhs_load": "rhs", "store": "output"}[kind]
                if arg.type.kind != "address" or arg.type.space != external[role] or arg.alignment != 1:
                    raise ValueError(
                        "dense pointer argument requires the matching declared address space and byte alignment"
                    )
            elif arg.type.kind != "integer" or arg.alignment != 1:
                raise ValueError("dense logical instruction bindings require unaligned integer coordinate operands")
            fields = [field for field in instruction.fields if field.argument == arg.name]
            if len(fields) != 1 or fields[0].width != arg.type.bits:
                raise ValueError("dense coordinate arguments require one full-width encoded field")
        arguments.append(tuple((arg.name, arg.type.bits) for arg in instruction.arguments()))
        effects = json.loads(instruction.effects_json)
        if effects == {"status": "unknown"}:
            raise ValueError("dense instructions require complete ISA effects")
        for effect, expected in (("reads", reads), ("writes", writes)):
            actual = effects[effect]
            if (
                any(resources[name].kind == "state" for name in actual)
                or {name for name in actual if resources[name].kind == "memory"} != expected
            ):
                raise ValueError("dense instruction has memory or state effects outside its operation binding")
    output_readout = None
    if "output_readout" in declaration:
        if type(declaration["output_readout"]) is not dict:
            raise ValueError("dense output readout requires an explicit mapping")
        tiled = declaration["output_readout"].get("schema") == "merlin.dense_tiled_integer_readout.v1"
        item = _fields(
            declaration["output_readout"],
            (("schema", "column_scales", "tile_columns", "relu", "output", "instruction") if tiled
             else ("schema", "scalar_ir", "output", "instruction")),
            "integer output readout"
        )
        output = _fields(item["output"], ("dtype", "bits"), "output readout format")
        if (
            item["schema"] not in ("merlin.dense_integer_readout.v1", "merlin.dense_tiled_integer_readout.v1")
            or dtypes[2] != "i32"
            or output["dtype"] != "i8"
            or (not tiled and (type(item["scalar_ir"]) is not str or len(item["scalar_ir"].encode()) > 65536))
            or resolve_fact_integer(output["bits"], facts, label="dense output readout width", minimum=1) != 8
            or type(item["instruction"]) is not str
            or item["instruction"] in instructions.values()
        ):
            raise ValueError(
                "dense integer output readout requires distinct i32-to-i8 instruction and declared RTL width"
            )
        instruction = isa.instruction(item["instruction"])
        args = instruction.arguments()
        pointer_name = pointer_bindings.get("store")
        if pointer_name is not None and pointer_name not in {arg.name for arg in args}:
            raise ValueError("dense output readout pointer is missing from its actual ISA instruction")
        for argument in args:
            fields = [field for field in instruction.fields if field.argument == argument.name]
            if argument.name == pointer_name:
                if argument.type.kind != "address" or argument.type.space != external["output"]:
                    raise ValueError("dense output readout pointer requires its declared output address space")
            elif argument.type.kind != "integer":
                raise ValueError("dense output readout requires integer coordinates and its declared pointer only")
            if argument.alignment != 1 or len(fields) != 1 or fields[0].width != argument.type.bits:
                raise ValueError("dense output readout requires full-width unaligned coordinate and pointer operands")
        effects = json.loads(instruction.effects_json)
        if effects != {"reads": [acc], "writes": [external["output"]]}:
            raise ValueError("dense output readout must read only its accumulator and write only its output")
        if tiled:
            if "readout_tile" not in {arg.name for arg in args} or item["tile_columns"] != dimensions[2]:
                raise ValueError("tiled readout requires an actual coefficient index and selected column primitive")
            output_readout = DenseTiledIntegerReadout.from_column_scales(
                item["column_scales"], tile_columns=item["tile_columns"], relu=item["relu"],
                instruction_name=item["instruction"],
                instruction_arguments=tuple((arg.name, arg.type.bits) for arg in args),
            )
        else:
            output_readout = DenseIntegerReadout(
                item["scalar_ir"], "i8", item["instruction"], tuple((arg.name, arg.type.bits) for arg in args)
            )
            graph = output_readout.graph()
            if graph.input_bits != (32,) or graph.output_bits != 8:
                raise ValueError("dense integer readout scalar semantics must be unary i32 to i8")
    return DenseSchedulingContract(
        dimensions,
        tuple(dtypes),
        tuple(stores),
        tuple(instructions[k] for k in OPERATIONS),
        tuple(arguments),
        tuple(sorted(semantics.items())),
        (limits[0], limits[2], limits[1]),
        support["partial_output_tiles"],
        support["partial_output_blocks"],
        support["partial_reduction_tiles"],
        logical_pointer,
        output_readout,
        tuple(pointer_bindings.items()),
    )
