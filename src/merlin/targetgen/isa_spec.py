"""Minimal instruction declarations resolved against one explicitly selected facts document.

This module does not discover hardware, infer instruction meaning from names, or
establish correspondence between mathematical declarations and RTL behavior.
Resolved identities are suitable for independent checking and proof attribution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Any

SCHEMA = "merlin.isa_spec.v1"
RESOLVED_SCHEMA = "merlin.resolved_isa.v1"


class IsaSpecError(ValueError):
    """An instruction declaration cannot be resolved exactly."""


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _mapping(value: Any, label: str, allowed: set[str], required: set[str]) -> dict:
    if not isinstance(value, dict) or set(value) - allowed or required - set(value):
        raise IsaSpecError(f"{label}: expected fields {sorted(required)}; allowed {sorted(allowed)}")
    return value


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.isascii() or not value.isidentifier():
        raise IsaSpecError(f"{label}: expected an ASCII identifier")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise IsaSpecError(f"{label}: expected an integer >= {minimum}")
    return value


def _reference(value: Any, label: str) -> tuple[str | int, ...]:
    row = _mapping(value, label, {"fact"}, {"fact"})
    path = row["fact"]
    if (
        not isinstance(path, list)
        or not path
        or any(not ((isinstance(part, str) and part) or (type(part) is int and part >= 0)) for part in path)
    ):
        raise IsaSpecError(f"{label}: fact must be a nonempty structural path")
    return tuple(path)


def _number(value: Any, label: str, facts: dict | None, *, fact_only: bool = False, minimum: int = 0) -> int:
    if isinstance(value, dict) or fact_only:
        path = _reference(value, label)
        if facts is None:
            return minimum
        current: Any = facts
        for part in path:
            try:
                if isinstance(current, dict) and isinstance(part, str):
                    current = current[part]
                elif isinstance(current, list) and type(part) is int:
                    current = current[part]
                else:
                    raise KeyError(part)
            except (KeyError, IndexError) as exc:
                raise IsaSpecError(f"{label}: unresolved fact path {list(path)!r}") from exc
        return _integer(current, label, minimum=minimum)
    return _integer(value, label, minimum=minimum)


def resolve_fact_integer(reference: dict, facts: dict, *, label: str, minimum: int = 0) -> int:
    """Resolve a structural RTL reference without accepting a software literal.

    Shared consumers use the same traversal and integer validation as the ISA
    resolver. An absent, noninteger or undersized fact is a diagnostic.
    """
    if type(facts) is not dict:
        raise IsaSpecError(f"{label}: an explicit RTL facts mapping is required")
    return _number(reference, label, facts, fact_only=True, minimum=minimum)


@dataclass(frozen=True)
class IsaType:
    kind: str
    bits: int
    signed: bool
    space: str | None


@dataclass(frozen=True)
class Argument:
    name: str
    type: IsaType
    unit: str
    alignment: int


@dataclass(frozen=True)
class EncodingField:
    offset: int
    width: int
    fixed: int | None
    argument: str | None


@dataclass(frozen=True)
class ResourceDeclaration:
    name: str
    kind: str
    unit: str
    capacity: int | None


@dataclass(frozen=True)
class Instruction:
    name: str
    operands: tuple[Argument, ...]
    results: tuple[Argument, ...]
    immediates: tuple[Argument, ...]
    fields: tuple[EncodingField, ...]
    semantics_json: str
    effects_json: str

    def arguments(self) -> tuple[Argument, ...]:
        return self.operands + self.results + self.immediates


@dataclass(frozen=True)
class ResolvedISA:
    target: str
    namespace: str
    word_bits: int
    facts_sha256: str
    declaration_sha256: str
    instructions: tuple[Instruction, ...]
    resources: tuple[ResourceDeclaration, ...] = ()

    def instruction(self, name: str) -> Instruction:
        for instruction in self.instructions:
            if instruction.name == name:
                return instruction
        raise IsaSpecError(f"instruction {name!r} is not declared")

    def record(self) -> dict:
        return json.loads(_json({"schema": RESOLVED_SCHEMA, **asdict(self)}))

    def identity(self) -> str:
        return _digest(self.record())


def _arguments(value: Any, label: str, facts: dict | None, *, immediate: bool = False) -> tuple[Argument, ...]:
    if not isinstance(value, list):
        raise IsaSpecError(f"{label}: expected a list")
    rows = []
    for row in value:
        _mapping(row, label, {"name", "type", "unit", "alignment"}, {"name", "type", "unit"})
        name = _name(row["name"], label)
        spec = _mapping(row["type"], label, {"kind", "bits", "signed", "space"}, {"kind", "bits", "signed"})
        kind = spec["kind"]
        if kind not in {"integer", "address"} or (immediate and kind != "integer"):
            raise IsaSpecError(f"{label}: unsupported type kind")
        if type(spec["signed"]) is not bool or (kind == "address" and spec["signed"]):
            raise IsaSpecError(f"{label}: signed must be explicit; addresses are unsigned")
        space = spec.get("space")
        if kind == "address":
            space = _name(space, label)
        elif space is not None:
            raise IsaSpecError(f"{label}: integer types cannot name an address space")
        bits = _number(spec["bits"], label, facts, minimum=1)
        if bits > 65536:
            raise IsaSpecError("argument width exceeds the generator's supported input-size limit")
        unit = _name(row["unit"], label)
        alignment = _number(row.get("alignment", 1), label, facts, minimum=1)
        rows.append(Argument(name, IsaType(kind, bits, spec["signed"], space), unit, alignment))
    return tuple(rows)


def _semantics(value: Any, names: set[str]) -> str:
    if value is None:
        return _json({"status": "unknown"})
    row = _mapping(value, "semantics", {"primitive", "bindings", "parameters"}, {"primitive", "bindings"})
    primitive = row["primitive"]
    if not isinstance(primitive, str) or not primitive or any(not part.isidentifier() for part in primitive.split(".")):
        raise IsaSpecError("semantics: expected an explicit primitive reference")
    bindings = row["bindings"]
    if not isinstance(bindings, dict) or any(
        not isinstance(key, str) or not key.isidentifier() or not isinstance(item, str) or item not in names
        for key, item in bindings.items()
    ):
        raise IsaSpecError("semantics: bindings must name declared arguments/results")
    parameters = row.get("parameters", {})
    if not isinstance(parameters, dict):
        raise IsaSpecError("semantics: parameters must be a mapping")
    try:
        return _json({"primitive": primitive, "bindings": bindings, "parameters": parameters})
    except (ValueError, TypeError) as exc:
        raise IsaSpecError("semantics: parameters must be finite JSON values") from exc


def _effects(value: Any, resources: set[str]) -> str:
    if value is None:
        return _json({"status": "unknown"})
    row = _mapping(value, "effects", {"reads", "writes"}, {"reads", "writes"})
    for key in ("reads", "writes"):
        if not isinstance(row[key], list) or any(not isinstance(item, str) or not item for item in row[key]):
            raise IsaSpecError(f"effects.{key}: expected explicit resource identifiers")
        if len(set(row[key])) != len(row[key]):
            raise IsaSpecError(f"effects.{key}: duplicate resource")
        if set(row[key]) - resources:
            raise IsaSpecError(f"effects.{key}: resource must be explicitly declared")
    return _json(row)


def _resources(value: Any, facts: dict | None) -> tuple[ResourceDeclaration, ...]:
    if not isinstance(value, list):
        raise IsaSpecError("resources: expected a list")
    resources = []
    seen = set()
    for row in value:
        _mapping(row, "resource", {"name", "kind", "capacity", "unit"}, {"name", "kind", "unit"})
        name = _name(row["name"], "resource.name")
        if name in seen or row["kind"] not in {"memory", "state", "execution"}:
            raise IsaSpecError("resource: duplicate name or unsupported kind")
        seen.add(name)
        capacity = None
        if "capacity" in row:
            capacity = _number(row["capacity"], "resource.capacity", facts, fact_only=True, minimum=1)
        resources.append(ResourceDeclaration(name, row["kind"], _name(row["unit"], "resource.unit"), capacity))
    return tuple(resources)


def _resolve(declaration: Any, facts: dict | None, target: str) -> ResolvedISA:
    spec = _mapping(
        declaration,
        "isa",
        {"schema", "namespace", "facts_sha256", "word_bits", "instructions", "resources"},
        {"schema", "namespace", "facts_sha256", "word_bits", "instructions"},
    )
    if spec["schema"] != SCHEMA:
        raise IsaSpecError(f"isa: expected schema {SCHEMA}")
    namespace = _name(spec["namespace"], "isa.namespace")
    digest = spec["facts_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise IsaSpecError("isa.facts_sha256: expected a SHA-256 digest")
    word_bits = _number(spec["word_bits"], "isa.word_bits", facts, fact_only=True, minimum=1)
    # This is a generator input-size limit, not an assumed hardware width.
    if word_bits > 65536:
        raise IsaSpecError("instruction width exceeds the generator's supported input-size limit")
    resources = _resources(spec.get("resources", []), facts)
    resource_names = {resource.name for resource in resources}
    if not isinstance(spec["instructions"], list) or not spec["instructions"]:
        raise IsaSpecError("isa.instructions: expected a nonempty list")
    instructions = []
    seen = set()
    for row in spec["instructions"]:
        _mapping(
            row,
            "instruction",
            {"name", "operands", "results", "immediates", "encoding", "semantics", "effects"},
            {"name", "operands", "results", "immediates", "encoding"},
        )
        name = _name(row["name"], "instruction.name")
        if name in seen:
            raise IsaSpecError(f"instruction: duplicate name {name!r}")
        seen.add(name)
        operands = _arguments(row["operands"], name, facts)
        results = _arguments(row["results"], name, facts)
        immediates = _arguments(row["immediates"], name, facts, immediate=True)
        arguments = operands + results + immediates
        memory_spaces = {resource.name for resource in resources if resource.kind == "memory"}
        if any(argument.type.kind == "address" and argument.type.space not in memory_spaces for argument in arguments):
            raise IsaSpecError(f"{name}: address type requires an explicitly declared memory resource")
        names = {arg.name for arg in arguments}
        if len(names) != len(arguments):
            raise IsaSpecError(f"{name}: duplicate argument/result name")
        fields = []
        occupied = 0
        if not isinstance(row["encoding"], list) or not row["encoding"]:
            raise IsaSpecError(f"{name}: encoding must contain explicit fields")
        for field in row["encoding"]:
            _mapping(field, name, {"offset", "width", "fixed", "argument"}, {"offset", "width"})
            if ("fixed" in field) == ("argument" in field):
                raise IsaSpecError(f"{name}: each encoding field needs fixed OR argument")
            offset = _number(field["offset"], name, facts, fact_only=True)
            width = _number(field["width"], name, facts, fact_only=True, minimum=1)
            fixed = _number(field["fixed"], name, facts, fact_only=True) if "fixed" in field else None
            argument = field.get("argument")
            if "argument" in field and (not isinstance(argument, str) or argument not in names):
                raise IsaSpecError(f"{name}: encoding references an undeclared argument")
            if facts is not None:
                if offset + width > word_bits:
                    raise IsaSpecError(f"{name}: encoding exceeds instruction width")
                mask = ((1 << width) - 1) << offset
                if mask & occupied or (fixed is not None and fixed >= 1 << width):
                    raise IsaSpecError(f"{name}: encoding overflow or overlapping fields")
                occupied |= mask
            fields.append(EncodingField(offset, width, fixed, argument))
        if facts is not None and occupied != (1 << word_bits) - 1:
            raise IsaSpecError(f"{name}: encoding must explicitly cover the complete instruction word")
        variable_names = [field.argument for field in fields if field.argument is not None]
        if len(set(variable_names)) != len(variable_names):
            raise IsaSpecError(f"{name}: split argument encoding is unsupported")
        instructions.append(
            Instruction(
                name,
                operands,
                results,
                immediates,
                tuple(fields),
                _semantics(row.get("semantics"), names),
                _effects(row.get("effects"), resource_names),
            )
        )
    model = ResolvedISA(target, namespace, word_bits, digest, _digest(spec), tuple(instructions), resources)
    if facts is not None:
        for i, left in enumerate(model.instructions):
            left_mask, left_value = fixed_encoding(left)
            for right in model.instructions[i + 1 :]:
                right_mask, right_value = fixed_encoding(right)
                if (left_value ^ right_value) & (left_mask & right_mask) == 0:
                    raise IsaSpecError("instruction encodings are ambiguous")
    return model


def validate_isa_spec(declaration: Any) -> None:
    """Validate authored structure; unresolved references confer no hardware authority."""
    _resolve(declaration, None, "")


def resolve_isa(declaration: dict, facts: dict, *, target: str, facts_sha256: str) -> ResolvedISA:
    """Resolve only supplied facts after exact byte identity and target selection checks.

    ``facts_sha256`` is the hash of the selected raw facts bytes, not a hardware
    qualification receipt. The caller owns observation and freezing of those bytes.
    """
    validate_isa_spec(declaration)
    if facts_sha256 != declaration["facts_sha256"]:
        raise IsaSpecError("selected facts bytes differ from the ISA declaration pin")
    if (
        not isinstance(target, str)
        or not target
        or not isinstance(facts, dict)
        or not isinstance(facts.get("facts"), dict)
        or facts["facts"].get("target") != target
    ):
        raise IsaSpecError("selected facts must explicitly identify the selected target")
    return _resolve(declaration, facts, target)


def fixed_encoding(instruction: Instruction) -> tuple[int, int]:
    mask = value = 0
    for field in instruction.fields:
        if field.fixed is not None:
            mask |= ((1 << field.width) - 1) << field.offset
            value |= field.fixed << field.offset
    return mask, value


def resolved_isa_from_record(record: dict) -> ResolvedISA:
    """Recheck rehydrated structure; records themselves are not grading authority."""
    _mapping(
        record,
        "resolved ISA",
        {
            "schema",
            "target",
            "namespace",
            "word_bits",
            "facts_sha256",
            "declaration_sha256",
            "instructions",
            "resources",
            "identity",
        },
        {"schema", "target", "namespace", "word_bits", "facts_sha256", "declaration_sha256", "instructions"},
    )
    if record["schema"] != RESOLVED_SCHEMA or not isinstance(record["target"], str) or not record["target"]:
        raise IsaSpecError("unsupported resolved ISA record")
    digest = record["declaration_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise IsaSpecError("invalid declaration digest")
    evidence: dict = {"values": []}

    def ref(value):
        index = len(evidence["values"])
        evidence["values"].append(value)
        return {"fact": ["values", index]}

    spec = {
        "schema": SCHEMA,
        "namespace": record["namespace"],
        "facts_sha256": record["facts_sha256"],
        "word_bits": ref(record["word_bits"]),
        "instructions": [],
        "resources": [],
    }
    for resource in record.get("resources", []):
        item = dict(resource)
        if item.get("capacity") is not None:
            item["capacity"] = ref(item["capacity"])
        else:
            item.pop("capacity", None)
        spec["resources"].append(item)
    for row in record["instructions"]:
        _mapping(
            row,
            "resolved instruction",
            {"name", "operands", "results", "immediates", "fields", "semantics_json", "effects_json"},
            {"name", "operands", "results", "immediates", "fields", "semantics_json", "effects_json"},
        )
        item = {"name": row["name"], "encoding": []}
        for kind in ("operands", "results", "immediates"):
            item[kind] = row[kind]
        for field in row["fields"]:
            _mapping(
                field,
                "resolved field",
                {"offset", "width", "fixed", "argument"},
                {"offset", "width", "fixed", "argument"},
            )
            encoding = {"offset": ref(field["offset"]), "width": ref(field["width"])}
            if field["fixed"] is not None:
                if field["argument"] is not None:
                    raise IsaSpecError("resolved field cannot have fixed and variable values")
                encoding["fixed"] = ref(field["fixed"])
            else:
                encoding["argument"] = field["argument"]
            item["encoding"].append(encoding)
        for kind in ("semantics", "effects"):
            content = json.loads(row[f"{kind}_json"])
            if content != {"status": "unknown"}:
                item[kind] = content
        spec["instructions"].append(item)
    model = replace(_resolve(spec, evidence, record["target"]), declaration_sha256=digest)
    if "identity" in record and record["identity"] != model.identity():
        raise IsaSpecError("resolved ISA identity differs from actual model")
    return model
