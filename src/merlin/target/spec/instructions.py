"""Selected OOT instruction semantics for exploratory, target-independent selection.

This is an authored description, not a proof that RTL implements the arithmetic.
The selected software specification owns numerical legality; the selected CIRCT
facts own only the hardware facts they actually extracted.  Both input identities
travel with the normalized description so Phase 0 can qualify claims separately.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from merlin.common.schemas import validate_or_raise
from merlin.target.spec.computation import (  # noqa: F401 -- compatibility reexports
    UNKNOWN,
    _computation,
    _names,
    _nonempty,
    _scalar_body,
)
from merlin.target.spec.signature import (  # noqa: F401 -- compatibility reexports
    _CHECKED_SW_SIGNATURE_FIELDS,
    _DTYPE_ALIASES,
    _mlir_dtype_rank,
    _software_signature_unknowns,
)

SCHEMA = "merlin.instruction_semantics.v1"
_SPACE_FIELDS = ("capacity_bytes", "alignment_bytes", "banks", "bank_width_bytes")
_PARAM_FIELDS = frozenset({"type", "min", "max", "multiple_of", "choices"})
_CONSTRAINT_FIELDS = frozenset({"operand_spaces", "result_spaces", "distinct_banks"})


def _canonical(value: Any) -> bytes:
    """Canonical JSON identity; rejects YAML-only scalars and non-finite numbers."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _snapshot(value: bytes | Mapping[str, Any] | None, *, label: str) -> tuple[dict[str, Any] | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, bytes):
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise ValueError(f"{label}: cannot parse selected bytes: {exc}") from exc
        digest = hashlib.sha256(value).hexdigest()
    elif isinstance(value, Mapping):
        parsed = copy.deepcopy(dict(value))
        digest = hashlib.sha256(_canonical(parsed)).hexdigest()
    else:
        raise ValueError(f"{label}: expected selected bytes or a mapping")
    if not isinstance(parsed, dict):
        raise ValueError(f"{label}: expected a mapping")
    _canonical(parsed)
    return parsed, digest


def _parameters(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    result = {}
    for name, domain in sorted(value.items()):
        _nonempty(name, f"{label} name")
        if not isinstance(domain, dict) or set(domain) - _PARAM_FIELDS or domain.get("type") != "integer":
            raise ValueError(f"{label}.{name} requires an integer domain")
        for field in ("min", "max", "multiple_of"):
            if field in domain and (type(domain[field]) is not int or (field == "multiple_of" and domain[field] < 1)):
                raise ValueError(f"{label}.{name}.{field} must be an integer of the required range")
        if "min" in domain and "max" in domain and domain["min"] > domain["max"]:
            raise ValueError(f"{label}.{name}: min exceeds max")
        if "choices" in domain and (
            not isinstance(domain["choices"], list)
            or not domain["choices"]
            or any(type(choice) is not int for choice in domain["choices"])
        ):
            raise ValueError(f"{label}.{name}.choices must be nonempty integer list")
        result[name] = copy.deepcopy(domain)
    return result


def _memory_spaces(value: Any, unknowns: list[str]) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("memory_spaces must be a mapping")
    result = {}
    for name, declaration in sorted(value.items()):
        _nonempty(name, "memory space name")
        if not isinstance(declaration, dict) or set(declaration) - set(_SPACE_FIELDS):
            raise ValueError(f"memory_spaces.{name} has unsupported fields")
        row = {}
        for field in _SPACE_FIELDS:
            item = declaration.get(field, UNKNOWN)
            if item != UNKNOWN and (type(item) is not int or item < 1):
                raise ValueError(f"memory_spaces.{name}.{field} must be positive or UNKNOWN")
            if item == UNKNOWN:
                unknowns.append(f"memory_spaces.{name}.{field}")
            row[field] = item
        result[name] = row
    return result


def _constraints(value: Any, *, label: str, names: set[str], spaces: dict[str, Any]) -> dict[str, Any]:
    if value is None:
        return {"operand_spaces": {}, "result_spaces": {}, "distinct_banks": []}
    if not isinstance(value, dict) or set(value) - _CONSTRAINT_FIELDS:
        raise ValueError(f"{label} has unsupported fields")
    result = {}
    for field in ("operand_spaces", "result_spaces"):
        locations = value.get(field, {})
        if not isinstance(locations, dict):
            raise ValueError(f"{label}.{field} must be a mapping")
        for name, space in locations.items():
            if name not in names or space not in spaces:
                raise ValueError(f"{label}.{field} references undeclared value or space")
        result[field] = dict(sorted(locations.items()))
    distinct = value.get("distinct_banks", [])
    if (
        not isinstance(distinct, list)
        or any(name not in names for name in distinct)
        or len(set(distinct)) != len(distinct)
    ):
        raise ValueError(f"{label}.distinct_banks must name distinct declared values")
    result["distinct_banks"] = list(distinct)
    return result


def _effects(
    value: Any, *, label: str, names: set[str], spaces: dict[str, Any], unknowns: list[str]
) -> list[dict[str, Any]] | None:
    if value is None:
        unknowns.append(f"{label}.missing")
        return None
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list or null")
    result = []
    for index, effect in enumerate(value):
        site = f"{label}[{index}]"
        if not isinstance(effect, dict) or set(effect) - {"kind", "space", "operand", "result", "range"}:
            raise ValueError(f"{site} has unsupported fields")
        kind = effect.get("kind")
        space = effect.get("space")
        if kind not in {"read", "write"} or space not in spaces:
            raise ValueError(f"{site} requires read/write and a declared memory space")
        reference = effect.get("operand" if kind == "read" else "result")
        if reference not in names or ("operand" in effect and "result" in effect):
            raise ValueError(f"{site} requires one declared read operand or write result")
        span = effect.get("range", UNKNOWN)
        if span == UNKNOWN:
            unknowns.append(f"{site}.range")
        elif (
            not isinstance(span, dict)
            or set(span) != {"offset_bytes", "length_bytes"}
            or any(
                type(span[field]) is not int or span[field] < (1 if field == "length_bytes" else 0)
                for field in ("offset_bytes", "length_bytes")
            )
        ):
            raise ValueError(f"{site}.range requires nonnegative offset and positive length")
        result.append(
            {"kind": kind, "space": space, "operand" if kind == "read" else "result": reference, "range": span}
        )
    return result


def _effect_coverage_unknowns(constraints: dict[str, Any], effects: list[dict[str, Any]] | None) -> list[str]:
    if effects is None:
        return []
    observed_reads = {(row["operand"], row["space"]) for row in effects if row["kind"] == "read"}
    observed_writes = {(row["result"], row["space"]) for row in effects if row["kind"] == "write"}
    unknowns = [
        f"effects.uncovered_read.{name}"
        for name, space in constraints["operand_spaces"].items()
        if (name, space) not in observed_reads
    ]
    unknowns.extend(
        f"effects.uncovered_write.{name}"
        for name, space in constraints["result_spaces"].items()
        if (name, space) not in observed_writes
    )
    return unknowns


def normalize_instruction_semantics(
    document: Mapping[str, Any],
    *,
    software_spec: bytes | Mapping[str, Any] | None,
    rtl_facts: bytes | Mapping[str, Any] | None,
    target: str,
    source_bytes: bytes | None = None,
    software_source_bytes: bytes | None = None,
    rtl_source_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Validate an authored v1 description against exactly supplied selected inputs.

    No target lookup or RTL extraction occurs here. Mapping inputs use canonical JSON
    identities; callers with frozen input bytes should pass those bytes for exact hashes.
    Unknowns are retained so exploratory search cannot silently become admission.
    """
    if not isinstance(document, Mapping):
        raise ValueError("instruction semantics must be a mapping")
    source = copy.deepcopy(dict(document))
    validate_or_raise(source, "instruction_semantics")
    if source.get("schema") != SCHEMA:
        raise ValueError(f"instruction semantics must declare schema {SCHEMA}")
    if source.get("target") != _nonempty(target, "target"):
        raise ValueError("instruction semantics target differs from selected target")
    if set(source) - {"schema", "target", "instructions", "memory_spaces"}:
        raise ValueError("instruction semantics has unsupported top-level fields")
    sw, sw_digest = _snapshot(software_spec, label="selected software spec")
    facts, facts_digest = _snapshot(rtl_facts, label="selected CIRCT facts")
    # Phase 0's absent optional observations are represented as empty mappings.
    if sw == {} and software_source_bytes is None:
        sw, sw_digest = None, None
    if facts == {} and rtl_source_bytes is None:
        facts, facts_digest = None, None
    if sw is not None and (sw.get("target") != target or sw.get("schema") != "merlin.software_spec.v1"):
        raise ValueError("selected software spec does not match the target or schema")
    if facts is not None and not isinstance(facts.get("facts"), dict):
        raise ValueError("selected CIRCT facts require a facts mapping")
    if software_source_bytes is not None:
        if sw is None:
            raise ValueError("software source bytes require a selected software spec")
        from merlin.targetgen.software_spec import validate_software_spec

        raw_software = yaml.safe_load(software_source_bytes)
        if validate_software_spec(raw_software, target=target) != sw:
            raise ValueError("selected software spec differs from supplied source bytes")
        sw_digest = hashlib.sha256(software_source_bytes).hexdigest()
    if rtl_source_bytes is not None:
        if facts is None or yaml.safe_load(rtl_source_bytes) != facts:
            raise ValueError("selected CIRCT facts differ from supplied source bytes")
        facts_digest = hashlib.sha256(rtl_source_bytes).hexdigest()
    unknowns: list[str] = []
    if sw is None:
        unknowns.append("selected_software_spec_missing")
    elif sw.get("status") != "reviewed":
        unknowns.append("selected_software_spec_unreviewed")
    if facts is None:
        unknowns.append("selected_circt_facts_missing")
    spaces = _memory_spaces(source.get("memory_spaces"), unknowns)
    instructions = source.get("instructions")
    if not isinstance(instructions, list) or not instructions:
        raise ValueError("instructions must be a nonempty list")
    declared_operations = sw.get("operations") if sw is not None else None
    if isinstance(declared_operations, dict):
        sw_operations = {
            identity: declaration
            for identity, declaration in declared_operations.items()
            if isinstance(identity, str) and isinstance(declaration, dict)
        }
    elif isinstance(declared_operations, list):
        sw_operations = {
            row["id"]: row for row in declared_operations if isinstance(row, dict) and isinstance(row.get("id"), str)
        }
    else:
        sw_operations = {}
    normalized = []
    identities = set()
    for index, instruction in enumerate(instructions):
        site = f"instructions[{index}]"
        if not isinstance(instruction, dict) or set(instruction) - {
            "id",
            "operands",
            "results",
            "computation",
            "parameters",
            "constraints",
            "effects",
            "software_operation",
        }:
            raise ValueError(f"{site} has unsupported fields")
        identity = _nonempty(instruction.get("id"), f"{site}.id")
        if identity in identities:
            raise ValueError(f"duplicate instruction id {identity!r}")
        identities.add(identity)
        operands = _names(instruction.get("operands"), label=f"{site}.operands")
        results = _names(instruction.get("results"), label=f"{site}.results")
        names = {row["name"] for row in operands + results}
        if len(names) != len(operands) + len(results):
            raise ValueError(f"{site}: operands and results share a name")
        row_unknowns: list[str] = []
        computation = _computation(instruction.get("computation"), label=f"{site}.computation", unknowns=row_unknowns)
        software_operation = instruction.get("software_operation")
        software_declaration = None
        if software_operation is None:
            row_unknowns.append("software_operation_unlinked")
        elif sw is None and isinstance(software_operation, str) and software_operation:
            row_unknowns.append("software_operation_unverifiable")
        elif not isinstance(software_operation, str) or software_operation not in sw_operations:
            raise ValueError(f"{site}.software_operation must reference a selected SW operation ID")
        else:
            software_declaration = copy.deepcopy(sw_operations[software_operation])
            software_declaration["id"] = software_operation
            if software_declaration.get("placement") not in {"accelerator", "fused_accelerator"}:
                row_unknowns.append("software_operation_not_accelerator")
            row_unknowns.extend(_software_signature_unknowns(software_declaration, operands, results))
        constraints = _constraints(
            instruction.get("constraints"), label=f"{site}.constraints", names=names, spaces=spaces
        )
        effects = _effects(
            instruction.get("effects"), label=f"{site}.effects", names=names, spaces=spaces, unknowns=row_unknowns
        )
        row_unknowns.extend(_effect_coverage_unknowns(constraints, effects))
        row = {
            "id": identity,
            "operands": operands,
            "results": results,
            "computation": computation,
            "parameters": _parameters(instruction.get("parameters"), label=f"{site}.parameters"),
            "constraints": constraints,
            "effects": effects,
            "software_operation": software_operation,
            "software_declaration": software_declaration,
            "status": UNKNOWN if row_unknowns else "described",
            "unknowns": sorted(set(row_unknowns)),
        }
        normalized.append(row)
        unknowns.extend(f"{identity}.{item}" for item in row_unknowns)
    normalized.sort(key=lambda row: row["id"])
    digest = hashlib.sha256(source_bytes if source_bytes is not None else _canonical(source)).hexdigest()
    model = {
        "schema": SCHEMA,
        "target": target,
        "source_sha256": digest,
        "software_spec_sha256": sw_digest,
        "circt_facts_sha256": facts_digest,
        "memory_spaces": spaces,
        "instructions": normalized,
        "status": UNKNOWN if unknowns else "described",
        "unknowns": sorted(set(unknowns)),
    }
    model["canonical_sha256"] = hashlib.sha256(_canonical(model)).hexdigest()
    return model


def load_instruction_semantics(
    path: str | Path,
    *,
    software_spec: bytes | Mapping[str, Any] | None,
    rtl_facts: bytes | Mapping[str, Any] | None,
    target: str,
    software_source_bytes: bytes | None = None,
    rtl_source_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Read only one explicitly selected OOT resource path."""
    selected = Path(path)
    source_bytes = selected.read_bytes()
    try:
        document = yaml.safe_load(source_bytes)
    except yaml.YAMLError as exc:
        raise ValueError(f"{selected}: invalid instruction semantics: {exc}") from exc
    return normalize_instruction_semantics(
        document,
        software_spec=software_spec,
        rtl_facts=rtl_facts,
        target=target,
        source_bytes=source_bytes,
        software_source_bytes=software_source_bytes,
        rtl_source_bytes=rtl_source_bytes,
    )


def _digest_field(value: Any, *, label: str, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")


def _unknown_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{label} must be a string list")
    if value != sorted(set(value)):
        raise ValueError(f"{label} must be sorted and unique")
    return value


def validate_normalized_instruction_model(
    model: Mapping[str, Any],
    *,
    expected_target: str | None = None,
    expected_source_sha256: str | None = None,
    expected_software_spec_sha256: str | None = None,
    expected_circt_facts_sha256: str | None = None,
) -> dict[str, Any]:
    """Check a frozen model's structure and self-consistency before search.

    Optional expected identities bind the model to independently selected bytes.
    Without them, a self-asserted digest establishes no source authenticity.
    """
    if not isinstance(model, Mapping):
        raise ValueError("normalized instruction model must be a mapping")
    result = copy.deepcopy(dict(model))
    required = {
        "schema",
        "target",
        "source_sha256",
        "software_spec_sha256",
        "circt_facts_sha256",
        "memory_spaces",
        "instructions",
        "status",
        "unknowns",
        "canonical_sha256",
    }
    if set(result) != required or result.get("schema") != SCHEMA:
        raise ValueError("normalized instruction model has unsupported schema or fields")
    target = _nonempty(result["target"], "normalized instruction target")
    if expected_target is not None and target != expected_target:
        raise ValueError("normalized instruction model target differs from selected target")
    for field in ("source_sha256", "software_spec_sha256", "circt_facts_sha256", "canonical_sha256"):
        _digest_field(result[field], label=field, optional=field in {"software_spec_sha256", "circt_facts_sha256"})
    for field, expected in (
        ("source_sha256", expected_source_sha256),
        ("software_spec_sha256", expected_software_spec_sha256),
        ("circt_facts_sha256", expected_circt_facts_sha256),
    ):
        if expected is not None and result[field] != expected:
            raise ValueError(f"normalized instruction model {field} differs from selected bytes")
    checksum = result.pop("canonical_sha256")
    if hashlib.sha256(_canonical(result)).hexdigest() != checksum:
        raise ValueError("normalized instruction model canonical_sha256 mismatch")
    result["canonical_sha256"] = checksum
    generated_unknowns: list[str] = []
    spaces = _memory_spaces(result["memory_spaces"], generated_unknowns)
    if spaces != result["memory_spaces"]:
        raise ValueError("normalized instruction model has noncanonical memory spaces")
    instructions = result["instructions"]
    if not isinstance(instructions, list) or not instructions:
        raise ValueError("normalized instruction model requires instructions")
    ids = []
    for index, row in enumerate(instructions):
        site = f"instructions[{index}]"
        if not isinstance(row, dict) or set(row) != {
            "id",
            "operands",
            "results",
            "computation",
            "parameters",
            "constraints",
            "effects",
            "software_operation",
            "software_declaration",
            "status",
            "unknowns",
        }:
            raise ValueError(f"{site} has noncanonical fields")
        identity = _nonempty(row["id"], f"{site}.id")
        ids.append(identity)
        operands = _names(row["operands"], label=f"{site}.operands")
        outputs = _names(row["results"], label=f"{site}.results")
        names = {item["name"] for item in operands + outputs}
        if len(names) != len(operands) + len(outputs):
            raise ValueError(f"{site} shares operand/result names")
        locally_generated: list[str] = []
        if (
            _computation(row["computation"], label=f"{site}.computation", unknowns=locally_generated)
            != row["computation"]
        ):
            raise ValueError(f"{site} has noncanonical computation")
        if _parameters(row["parameters"], label=f"{site}.parameters") != row["parameters"]:
            raise ValueError(f"{site} has noncanonical parameters")
        if (
            _constraints(row["constraints"], label=f"{site}.constraints", names=names, spaces=spaces)
            != row["constraints"]
        ):
            raise ValueError(f"{site} has noncanonical constraints")
        if (
            _effects(row["effects"], label=f"{site}.effects", names=names, spaces=spaces, unknowns=locally_generated)
            != row["effects"]
        ):
            raise ValueError(f"{site} has noncanonical effects")
        locally_generated.extend(_effect_coverage_unknowns(row["constraints"], row["effects"]))
        software_operation = row["software_operation"]
        if software_operation is None:
            locally_generated.append("software_operation_unlinked")
        else:
            _nonempty(software_operation, f"{site}.software_operation")
        software_declaration = row["software_declaration"]
        if software_declaration is not None:
            if not isinstance(software_declaration, dict) or software_declaration.get("id") != software_operation:
                raise ValueError(f"{site}.software_declaration must match its SW operation ID")
            if software_declaration.get("placement") not in {"accelerator", "fused_accelerator"}:
                locally_generated.append("software_operation_not_accelerator")
            locally_generated.extend(_software_signature_unknowns(software_declaration, operands, outputs))
        elif software_operation is not None:
            locally_generated.append("software_operation_unverifiable")
        row_unknowns = _unknown_list(row["unknowns"], label=f"{site}.unknowns")
        if not set(locally_generated) <= set(row_unknowns):
            raise ValueError(f"{site} suppresses semantic unknowns")
        if row["status"] != (UNKNOWN if row_unknowns else "described"):
            raise ValueError(f"{site} status differs from unknowns")
        generated_unknowns.extend(f"{identity}.{item}" for item in row_unknowns)
    if ids != sorted(set(ids)):
        raise ValueError("normalized instruction IDs must be sorted and unique")
    if result["software_spec_sha256"] is None:
        generated_unknowns.append("selected_software_spec_missing")
    if result["circt_facts_sha256"] is None:
        generated_unknowns.append("selected_circt_facts_missing")
    unknowns = _unknown_list(result["unknowns"], label="normalized instruction unknowns")
    if not set(generated_unknowns) <= set(unknowns):
        raise ValueError("normalized instruction model suppresses selected-input or instruction unknowns")
    if result["status"] != (UNKNOWN if unknowns else "described"):
        raise ValueError("normalized instruction model status differs from unknowns")
    return result


# Keep public callable pickle spellings while their code has one canonical owner.
normalize_instruction_semantics.__module__ = "merlin.target.spec.instructions"
load_instruction_semantics.__module__ = "merlin.target.spec.instructions"
validate_normalized_instruction_model.__module__ = "merlin.target.spec.instructions"
