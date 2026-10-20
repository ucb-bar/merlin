"""Bind schedule passes to selected RTL facts and software/ISA bytes.

RTL references supply structural bounds; software supplies computation,
layouts, effects and ordering for the existing planners. Neither a declaration
nor a review label qualifies execution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from merlin.sched.ir import Call, evaluate
from merlin.targetgen.isa_dialect_gen import decode_instruction, encode_instruction
from merlin.targetgen.isa_spec import ResolvedISA, resolve_isa

from .configuration import ConfigurationAdapter, ConfigurationEffects
from .dense_contracts import DenseSchedulingContract, resolve_dense_contract
from .fusion_contracts import SelectedFusionContract, resolve_fusion_contract
from .program_contracts import SelectedProgramContract, resolve_program_contract
from .row_contracts import SelectedRowContract, resolve_row_contract
from .spill_contracts import SelectedSpillContract, resolve_spill_contract
from .storage_contracts import TemporaryStorageContract, resolve_storage_contract

SCHEMA = "merlin.schedule_effects.v1"


def _hash(payload):
    return hashlib.sha256(payload).hexdigest()


def _fields(value, required, label):
    if type(value) is not dict or set(value) != set(required):
        raise ValueError(f"{label}: expected exactly {sorted(required)}")
    return value


def _read(path, budget):
    with path.open("rb") as stream:
        payload = stream.read(budget + 1)
    if len(payload) > budget:
        raise ValueError("selected scheduling input exceeds its byte budget")
    return payload


def _resolver_sources():
    """Resolved declarations expire with their shared consumers and checkers."""
    from .schedule_pipeline import shared_library_sources

    return tuple(sorted(shared_library_sources().items()))


@dataclass(frozen=True)
class SelectedSchedulingContracts:
    facts_path: Path
    software_path: Path
    facts_bytes: bytes
    software_bytes: bytes
    target: str
    isa: ResolvedISA = field(init=False)
    effects_json: str = field(init=False)
    numerical_policy: str = field(init=False)
    instruction_byteorder: str = field(init=False)
    dense: DenseSchedulingContract | None = field(init=False)
    storage: TemporaryStorageContract | None = field(init=False)
    fusion: SelectedFusionContract | None = field(init=False)
    spills: SelectedSpillContract | None = field(init=False)
    row_reduction: SelectedRowContract | None = field(init=False)
    program: SelectedProgramContract | None = field(init=False)
    resolution_sources: tuple[tuple[str, str], ...] = field(init=False)

    def __post_init__(self):
        if (
            type(self.facts_bytes) is not bytes or type(self.software_bytes) is not bytes
            or not isinstance(self.facts_path, Path) or not isinstance(self.software_path, Path)
        ):
            raise ValueError("selected scheduling inputs require immutable bytes and explicit paths")
        if len(self.facts_bytes) > 64 << 20 or len(self.software_bytes) > 1 << 20:
            raise ValueError("selected scheduling input exceeds its byte budget")
        sources = _resolver_sources()
        isa, effects, policy, byteorder, dense, storage, fusion, spills, row_reduction, program = _parse(
            self.facts_bytes, self.software_bytes, self.target
        )
        if _resolver_sources() != sources:
            raise ValueError("contract resolver changed during selection; reselect and replan")
        object.__setattr__(self, "resolution_sources", sources)
        object.__setattr__(self, "isa", isa)
        object.__setattr__(self, "effects_json", effects)
        object.__setattr__(self, "numerical_policy", policy)
        object.__setattr__(self, "instruction_byteorder", byteorder)
        object.__setattr__(self, "dense", dense)
        object.__setattr__(self, "storage", storage)
        object.__setattr__(self, "fusion", fusion)
        object.__setattr__(self, "spills", spills)
        object.__setattr__(self, "row_reduction", row_reduction)
        object.__setattr__(self, "program", program)

    @property
    def hardware_identity(self):
        return "rtl:" + _hash(self.facts_bytes)

    def record(self, *, scopes=("instruction_encoding", "configuration_fields")):
        return {
            "schema": SCHEMA,
            "consumed_scopes": list(scopes),
            "rtl_facts": {"path": str(self.facts_path), "sha256": _hash(self.facts_bytes)},
            "software_spec": {"path": str(self.software_path), "sha256": _hash(self.software_bytes)},
            "resolved_isa_identity": self.isa.identity(),
            "numerical_policy": self.numerical_policy,
            "instruction_byteorder": self.instruction_byteorder,
            "execution_qualification": "independent_evidence_required",
            "resolution_library_sources": dict(self.resolution_sources),
        }

    def revalidate(self):
        if _resolver_sources() != self.resolution_sources:
            raise ValueError("selected contract resolver changed; reselect and replan")
        for path, payload, budget in (
            (self.facts_path, self.facts_bytes, 64 << 20),
            (self.software_path, self.software_bytes, 1 << 20),
        ):
            if _read(path, budget) != payload:
                raise ValueError("selected RTL facts or software specification changed; reselect and replan")

    def configuration_adapter(self, qualification_identity):
        """Use actual operands with independently qualified declaration semantics."""
        declarations = json.loads(self.effects_json)
        resources = {resource.name: resource for resource in self.isa.resources}

        def project(call):
            if type(call) is not Call or call.instr not in declarations:
                return None
            if call.unit or call.produces or call.awaits or call.stages or call.sets or call.assumes:
                return None
            try:
                values = {name: evaluate(value, {}) for name, value in call.args}
            except ValueError:
                return None
            if len(values) != len(call.args):
                raise ValueError("configuration operands contain duplicate bindings")
            encode_instruction(self.isa, call.instr, values)
            row = declarations[call.instr]

            def value(binding):
                return values[binding["argument"]] if "argument" in binding else binding["constant"]

            def field(item):
                capacity = resources[item["resource"]].capacity
                if capacity is None:
                    return None
                index = value(item["index"])
                if not 0 <= index < capacity:
                    raise ValueError("configuration index is outside the selected RTL state capacity")
                return item["resource"] + ":" + str(index)

            decoded = {}
            for kind in ("reads", "writes", "clobbers"):
                entries = []
                for item in row[kind]:
                    key = field(item)
                    if key is None:
                        return None
                    entries.append(key if kind == "clobbers" else (key, value(item["value"])))
                decoded[kind] = tuple(entries)
            return ConfigurationEffects(**decoded, ordered=row["ordered"], pure_write=row["pure_write"])

        return ConfigurationAdapter(project, self.hardware_identity, qualification_identity)

    def verify_image(self, data, calls):
        """Decode actual words independently of the provider's Call projection.

        This family is fixed-width and byte-aligned. It does not interpret an
        arbitrary opaque program, infer framing or qualify instruction effects.
        """
        width = self.isa.word_bits // 8
        if self.isa.word_bits % 8 or len(data) % width or len(data) // width != len(calls):
            raise ValueError("selected ISA requires one complete byte-aligned word per decoded call")
        for i, call in enumerate(calls):
            name, actual = decode_instruction(
                self.isa, int.from_bytes(data[i * width:(i + 1) * width], self.instruction_byteorder)
            )
            projected = {key: evaluate(value, {}) for key, value in call.args}
            if name != call.instr or actual != projected or len(call.args) != len(projected):
                raise ValueError("provider decoding disagrees with actual selected ISA words")


def read_scheduling_contracts(facts_path, software_path, *, target):
    """Read explicit inputs once and reuse the existing fact-bound ISA resolver.

    Only this scheduling extension is interpreted here. Operation admission,
    numerical references and execution qualification remain separate owners.
    Unknown instructions are barriers, not inferred from their names.
    """
    facts_path, software_path = Path(facts_path).resolve(strict=True), Path(software_path).resolve(strict=True)
    facts_bytes, software_bytes = _read(facts_path, 64 << 20), _read(software_path, 1 << 20)
    return SelectedSchedulingContracts(facts_path, software_path, facts_bytes, software_bytes, target)


def _parse(facts_bytes, software_bytes, target):
    facts, software = json.loads(facts_bytes), yaml.safe_load(software_bytes)
    if (
        type(software) is not dict or software.get("schema") != "merlin.software_spec.v1"
        or software.get("target") != target or software.get("status") != "reviewed"
    ):
        raise ValueError("scheduling requires an explicitly selected reviewed software/ISA specification")
    isa = resolve_isa(software.get("isa"), facts, target=target, facts_sha256=_hash(facts_bytes))
    extension = software.get("scheduling")
    required = {"schema", "numerical_policy", "instruction_byteorder", "configuration"}
    if (type(extension) is not dict or not required <= set(extension)
            or set(extension) - required - {"dense", "storage", "fusion", "spills", "row_reduction", "program"}):
        raise ValueError("scheduling: expected versioned effects with optional declared scheduling families")
    if (
        extension["schema"] != SCHEMA
        or type(extension["numerical_policy"]) is not str or not extension["numerical_policy"]
    ):
        raise ValueError("scheduling requires its versioned effects and selected numerical policy")
    if extension["instruction_byteorder"] not in ("little", "big"):
        raise ValueError("selected instruction byte order must be explicit little or big")
    rows = extension["configuration"]
    if type(rows) is not dict or len(rows) > 4096:
        raise ValueError("configuration declaration must be a bounded instruction mapping")
    resources = {resource.name: resource for resource in isa.resources}
    for name, row in rows.items():
        instruction = isa.instruction(name)
        _fields(row, ("reads", "writes", "clobbers", "ordered", "pure_write"), "configuration effects")
        if any(type(row[key]) is not bool for key in ("ordered", "pure_write")):
            raise ValueError("configuration ordering and purity must be explicit Booleans")
        args = {arg.name: arg for arg in instruction.arguments()}
        encoded = {item.argument for item in instruction.fields if item.argument is not None}

        def binding(item):
            if type(item) is not dict or len(item) != 1:
                raise ValueError("configuration values require one argument or integer constant")
            if "constant" in item and type(item["constant"]) is int:
                return
            if (
                "argument" in item and type(item["argument"]) is str
                and item["argument"] in encoded and args[item["argument"]].type.kind == "integer"
            ):
                return
            raise ValueError("configuration binding must name an encoded typed integer argument")

        for kind in ("reads", "writes", "clobbers"):
            if type(row[kind]) is not list or len(row[kind]) > 256:
                raise ValueError("configuration field declarations must be bounded lists")
            for item in row[kind]:
                _fields(item, ("resource", "index") if kind == "clobbers" else ("resource", "index", "value"), kind)
                resource = resources.get(item["resource"]) if type(item["resource"]) is str else None
                if resource is None or resource.kind != "state" or resource.unit != "fields":
                    raise ValueError("configuration fields require declared ISA state resources in field units")
                binding(item["index"])
                if kind != "clobbers":
                    binding(item["value"])
        effects = json.loads(instruction.effects_json)
        if effects == {"status": "unknown"}:
            raise ValueError("configuration declarations require complete ISA resource effects")
        state = {name for name, resource in resources.items() if resource.kind == "state"}
        for kind, actual in (("reads", row["reads"]), ("writes", row["writes"] + row["clobbers"])):
            if set(effects[kind]) & state != {item["resource"] for item in actual}:
                raise ValueError("configuration fields disagree with declared ISA state effects")
        if row["pure_write"] and (
            not row["ordered"] or not row["writes"] or row["reads"] or row["clobbers"]
            or (set(effects["reads"]) | set(effects["writes"])) - state
        ):
            raise ValueError("pure configuration write contradicts declared state or memory effects")
    storage = (resolve_storage_contract(extension["storage"], facts, isa, configuration=rows)
               if "storage" in extension else None)
    return (
        isa, json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False),
        extension["numerical_policy"], extension["instruction_byteorder"],
        resolve_dense_contract(extension["dense"], facts, isa) if "dense" in extension else None,
        storage,
        resolve_fusion_contract(extension["fusion"], facts, isa, storage) if "fusion" in extension else None,
        resolve_spill_contract(extension["spills"], facts, isa, storage) if "spills" in extension else None,
        resolve_row_contract(extension["row_reduction"], facts, isa, storage, extension["numerical_policy"])
        if "row_reduction" in extension else None,
        resolve_program_contract(extension["program"], facts, isa, storage) if "program" in extension else None,
    )
