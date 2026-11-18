"""Exact-source composition of instruction semantics and their scalar bodies."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# ---- source ------------------------------------------------------------------------------------
# Exact-source, non-executable semantic-library composition.

SCHEMA = "merlin.software_spec.v2"
LIBRARY_SCHEMA = "merlin.software_spec_library.v1"
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_SOURCE_BYTES = 16 * 1024 * 1024
MAX_NODES = 100_000
MAX_EXPANSION_BYTES = 32 * 1024 * 1024
KINDS = frozenset(
    {
        "numerical_semantics",
        "signature",
        "operation",
        "quantization_format",
        "isa",
        "transfer_contracts",
        "computation",
        "scalar_body",
        "quantization_element_formats",
        "quantization_output_requantization",
    }
)


class SpecError(ValueError):
    """The selected source cannot be expanded or projected without loss."""


def read_source(path: Path) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(MAX_SOURCE_BYTES + 1)
    if len(raw) > MAX_SOURCE_BYTES:
        raise SpecError(f"{path}: source exceeds the supported byte limit")
    return raw


def measure(value: object) -> tuple[int, int]:
    """Bound actual serialization work, including repeated YAML alias occurrences."""
    stack = [(iter((value,)), None)]
    active = set()
    count = 0
    size = 0
    while stack:
        iterator, identity = stack[-1]
        try:
            member = next(iterator)
        except StopIteration:
            stack.pop()
            if identity is not None:
                active.remove(identity)
            continue
        count += 1
        # A conservative JSON upper bound avoids expanding a long scalar alias
        # many times while still staying under the node limit.
        size += 6 * len(member) + 2 if isinstance(member, str) else 32
        if isinstance(member, dict):
            size += sum(6 * len(key) + 4 for key in member)
        if count > MAX_NODES or len(stack) > 128:
            raise SpecError("specification exceeds the supported node/depth limit")
        if size > MAX_EXPANSION_BYTES:
            raise SpecError("specification exceeds the supported expanded byte limit")
        if isinstance(member, (dict, list, tuple)):
            if id(member) in active:
                raise SpecError("specification contains a cyclic value")
            active.add(id(member))
            stack.append((iter(member.values() if isinstance(member, dict) else member), id(member)))
    return count, size


def canonical(value: object) -> bytes:
    measure(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
    except (ValueError, TypeError, RecursionError) as exc:
        raise SpecError("specification must contain finite, acyclic JSON values") from exc


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def fields(
    value: object, allowed: set[str] | frozenset[str], label: str, required: set[str] | frozenset[str] = frozenset()
) -> dict:
    if not isinstance(value, dict) or set(value) - allowed or required - set(value):
        raise SpecError(f"{label}: expected {sorted(required)}; allowed {sorted(allowed)}")
    return value


def identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.isascii() or not value.isidentifier():
        raise SpecError(f"{label}: expected an ASCII identifier")
    return value


def sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise SpecError(f"{label}: expected a lowercase SHA-256 digest")
    return value


class _Loader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise SpecError("specification mapping keys must be unique strings; YAML merge keys are unsupported")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


class _LegacyLoader(yaml.SafeLoader):
    """Preserve v1 merge/override semantics while refusing authored duplicates."""

    def construct_document(self, node):
        # Bound repeated aliases before SafeLoader flattens merges. Check keys
        # before flattening so intentional merged overrides stay distinguishable
        # from duplicate explicitly authored keys, including chained merges.
        stack = [(iter((node,)), None)]
        active = set()
        count, size = 0, 0
        while stack:
            iterator, identity = stack[-1]
            try:
                member = next(iterator)
            except StopIteration:
                stack.pop()
                if identity is not None:
                    active.remove(identity)
                continue
            count += 1
            size += 6 * len(member.value) + 32
            if count > MAX_NODES or len(stack) > 128 or size > MAX_EXPANSION_BYTES:
                raise SpecError("legacy specification exceeds the supported alias/node/depth limit")
            children = []
            if isinstance(member, yaml.MappingNode):
                explicit = set()
                for key, value in member.value:
                    if key.tag != "tag:yaml.org,2002:merge":
                        if key.tag != "tag:yaml.org,2002:str" or key.value in explicit:
                            raise SpecError("legacy specification mapping keys must be unique strings")
                        explicit.add(key.value)
                    children.extend((key, value))
            elif isinstance(member, yaml.SequenceNode):
                children = member.value
            if children:
                if id(member) in active:
                    raise SpecError("legacy specification contains a cyclic alias")
                active.add(id(member))
                stack.append((iter(children), id(member)))
        return super().construct_document(node)


def parse(raw: bytes, label: str, *, legacy: bool = False) -> dict:
    if len(raw) > MAX_SOURCE_BYTES:
        raise SpecError(f"{label}: source exceeds the supported byte limit")
    try:
        raw.decode("utf-8")
        document = yaml.load(raw, Loader=_LegacyLoader if legacy else _Loader)
        canonical(document)
    except (yaml.YAMLError, UnicodeError, RecursionError) as exc:
        raise SpecError(f"{label}: invalid UTF-8 YAML source") from exc
    if not isinstance(document, dict):
        raise SpecError(f"{label}: expected a mapping")
    return document


@dataclass(frozen=True)
class Source:
    path: Path
    raw: bytes
    namespace: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()

    def record(self) -> dict:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "namespace": self.namespace,
            "authored_text": self.raw.decode("utf-8"),
        }

    def verify(self) -> None:
        if read_source(self.path) != self.raw:
            raise SpecError(f"selected source changed after expansion: {self.path}")


class Composition:
    """Typed references over existing record kinds, with no inheritance or overrides."""

    def __init__(self, path: str | Path):
        self.sources: list[Source] = []
        self.definitions: dict[str, tuple[str, object, str]] = {}
        self.used: set[str] = set()
        self.source_bytes = 0
        self.expansion_work = 0
        self.expansion_bytes = 0
        self.document = self._read(Path(path).expanduser().resolve(), "", (), library=False)

    def _read(self, path: Path, scope: str, stack: tuple[Path, ...], *, library: bool, pin: str | None = None) -> dict:
        if path in stack or len(stack) >= 32 or len(self.sources) >= 128:
            raise SpecError("semantic imports contain a cycle or exceed the supported import bound")
        raw = read_source(path)
        self.source_bytes += len(raw)
        if self.source_bytes > MAX_TOTAL_SOURCE_BYTES:
            raise SpecError("semantic imports exceed the supported total source byte limit")
        source = Source(path, raw, scope)
        if pin is not None and source.sha256 != pin:
            raise SpecError(f"{path}: semantic-library source differs from its SHA-256 pin")
        document = parse(raw, str(path))
        expected = LIBRARY_SCHEMA if library else SCHEMA
        if document.get("schema") != expected:
            raise SpecError(f"{path}: expected schema {expected}; importing a complete legacy spec is unsupported")
        if library:
            fields(document, {"schema", "imports", "definitions"}, str(path), {"schema", "definitions"})
        self.sources.append(source)
        imports = document.get("imports", {})
        if not isinstance(imports, dict):
            raise SpecError("imports must map local namespace identifiers to pinned library sources")
        for alias, declaration in imports.items():
            identifier(alias, "import namespace")
            row = fields(declaration, {"path", "sha256"}, f"import {alias}", {"path", "sha256"})
            if not isinstance(row["path"], str) or not row["path"].strip():
                raise SpecError(f"import {alias}: path must be nonempty")
            member = Path(row["path"]).expanduser()
            self._read(
                (path.parent / member).resolve(),
                f"{scope}{alias}.",
                stack + (path,),
                library=True,
                pin=sha256(row["sha256"], f"import {alias}"),
            )
        definitions = document.get("definitions", {})
        if not isinstance(definitions, dict):
            raise SpecError("definitions must map names to typed records")
        for name, definition in definitions.items():
            identifier(name, "definition name")
            row = fields(definition, {"kind", "value"}, f"definition {name}", {"kind", "value"})
            if not isinstance(row["kind"], str) or row["kind"] not in KINDS:
                raise SpecError(f"definition {name}: unsupported kind {row['kind']!r}")
            qualified = scope + name
            if qualified in self.definitions:
                raise SpecError(f"duplicate definition {qualified}")
            self.definitions[qualified] = row["kind"], row["value"], scope
        return document

    def expand(self, value: object, kind: str, *, scope: str = "", stack: tuple[str, ...] = ()) -> dict:
        nodes, size = measure(value)
        self.expansion_work += nodes
        self.expansion_bytes += size
        if self.expansion_work > MAX_NODES:
            raise SpecError("semantic definitions exceed the supported total expansion work limit")
        if self.expansion_bytes > MAX_EXPANSION_BYTES:
            raise SpecError("semantic definitions exceed the supported total expanded byte limit")
        if isinstance(value, dict) and "$ref" in value:
            row = fields(value, {"$ref"}, "typed reference", {"$ref"})
            reference = row["$ref"]
            if not isinstance(reference, str) or any(
                not part.isidentifier() or not part.isascii() for part in reference.split(".")
            ):
                raise SpecError("typed reference must name a local or imported definition")
            name = scope + reference
            if name not in self.definitions:
                raise SpecError(f"unknown semantic definition {name!r}")
            actual, authored, definition_scope = self.definitions[name]
            if actual != kind:
                raise SpecError(f"definition {name}: expected kind {kind}, observed {actual}")
            if name in stack or len(stack) >= 128:
                raise SpecError(f"cyclic or excessively deep semantic reference {name}")
            self.used.add(name)
            return self.expand(authored, kind, scope=definition_scope, stack=stack + (name,))
        if not isinstance(value, dict):
            raise SpecError(f"{kind}: expected a typed record or reference")
        # Reuse only slots whose v1 representation has this same meaning.
        # Canonical serialization pins values but must not reorder named
        # transfer declarations before they become an ordered consumer list.
        canonical(value)
        result = copy.deepcopy(value)
        if kind == "numerical_semantics" and scope:
            model = result.get("model")
            model_path = model.get("source_root_path") if isinstance(model, dict) else None
            if isinstance(model_path, str) and not Path(model_path).expanduser().is_absolute():
                raise SpecError(
                    "relative model source_root_path in an imported definition requires an explicit absolute binding"
                )
        slots = {
            "operation": {"signature": "signature"},
            "quantization_format": {"numerical_semantics": "numerical_semantics"},
            "computation": {"scalar_body": "scalar_body"},
        }.get(kind, {})
        for key, child_kind in slots.items():
            if key in result:
                result[key] = self.expand(result[key], child_kind, scope=scope, stack=stack)

        # References cannot masquerade as ignored data in an unsupported slot.
        def reject_unexpanded(member):
            if isinstance(member, dict):
                if "$ref" in member:
                    raise SpecError(f"{kind}: reference appears in an unsupported slot")
                for child in member.values():
                    reject_unexpanded(child)
            elif isinstance(member, list):
                for child in member:
                    reject_unexpanded(child)

        reject_unexpanded(result)
        return result


# ---- computation -------------------------------------------------------------------------------
# Typed scalar bodies and linalg computation records, without execution claims.

UNKNOWN = "UNKNOWN"


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _names(rows: Any, *, label: str) -> list[dict[str, str]]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be a list")
    result = []
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"name", "type"}:
            raise ValueError(f"{label}[{index}] requires only name and type")
        name = _nonempty(row["name"], f"{label}[{index}].name")
        typ = _nonempty(row["type"], f"{label}[{index}].type")
        if name in seen:
            raise ValueError(f"{label}: duplicate name {name!r}")
        seen.add(name)
        result.append({"name": name, "type": typ})
    return result


def _scalar_body(body: Any, *, label: str, unknowns: list[str]) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError(f"{label} must be a mapping")
    arguments = body.get("arguments")
    captures = body.get("captures", [])
    operations = body.get("operations")
    yields = body.get("yields")
    if not isinstance(arguments, list) or any(not isinstance(t, str) or not t for t in arguments):
        raise ValueError(f"{label}.arguments must be MLIR type strings")
    if not isinstance(captures, list):
        raise ValueError(f"{label}.captures must be a list")
    normalized_captures = []
    for index, capture in enumerate(captures):
        site = f"{label}.captures[{index}]"
        if not isinstance(capture, dict) or set(capture) - {"source", "shape", "dtype", "result_index", "const_value"}:
            raise ValueError(f"{site} has unsupported fields")
        source = capture.get("source")
        shape = capture.get("shape")
        if (
            not isinstance(source, (list, tuple))
            or len(source) != 2
            or source[0] not in {"arg", "op", "init", "const", "other"}
            or not isinstance(source[1], (int, str))
        ):
            raise ValueError(f"{site}.source must identify a captured value")
        if not isinstance(shape, list) or any(type(dim) is not int for dim in shape):
            raise ValueError(f"{site}.shape must be integer dimensions")
        dtype = _nonempty(capture.get("dtype"), f"{site}.dtype")
        result_index = capture.get("result_index")
        if result_index is not None and (type(result_index) is not int or result_index < 0):
            raise ValueError(f"{site}.result_index must be nonnegative")
        const_value = capture.get("const_value")
        if "const_value" in capture and (type(const_value) not in (int, float) or isinstance(const_value, bool)):
            raise ValueError(f"{site}.const_value must be numeric")
        normalized_capture = {"source": list(source), "shape": list(shape), "dtype": dtype}
        if result_index is not None:
            normalized_capture["result_index"] = result_index
        if "const_value" in capture:
            normalized_capture["const_value"] = const_value
        normalized_captures.append(normalized_capture)
    if not isinstance(operations, list) or not isinstance(yields, list):
        raise ValueError(f"{label} requires operations and yields lists")
    available = {f"arg:{index}" for index in range(len(arguments))}
    available.update(f"capture:{index}" for index in range(len(captures)))
    normalized_ops = []
    for index, operation in enumerate(operations):
        site = f"{label}.operations[{index}]"
        if not isinstance(operation, dict):
            raise ValueError(f"{site} must be a mapping")
        op = _nonempty(operation.get("op"), f"{site}.op")
        operands = operation.get("operands")
        results = operation.get("results")
        attrs = operation.get("attributes", {})
        regions = operation.get("regions", 0)
        if not isinstance(operands, list) or any(ref not in available for ref in operands):
            raise ValueError(f"{site}.operands must reference preceding scalar values")
        if not isinstance(results, list) or any(not isinstance(t, str) or not t for t in results):
            raise ValueError(f"{site}.results must be MLIR type strings")
        if not isinstance(attrs, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in attrs.items()
        ):
            raise ValueError(f"{site}.attributes must map names to MLIR attribute strings")
        if type(regions) is not int or regions < 0:
            raise ValueError(f"{site}.regions must be a nonnegative integer")
        effects = operation.get("effects")
        if effects is not None and (
            not isinstance(effects, list) or any(not isinstance(effect, str) or not effect for effect in effects)
        ):
            raise ValueError(f"{site}.effects must be a string list or null")
        if effects is None:
            unknowns.append(f"{site}.effects")
        if regions:
            unknowns.append(f"{site}.nested_regions")
        normalized_ops.append(
            {
                "op": op,
                "operands": list(operands),
                "results": list(results),
                "attributes": dict(sorted(attrs.items())),
                "effects": effects,
                "regions": regions,
            }
        )
        available.update(f"op:{index}:{result}" for result in range(len(results)))
    if any(ref not in available for ref in yields):
        raise ValueError(f"{label}.yields must reference scalar values")
    effects = body.get("effects")
    if effects is not None and (
        not isinstance(effects, list) or any(not isinstance(effect, str) or not effect for effect in effects)
    ):
        raise ValueError(f"{label}.effects must be a string list or null")
    if effects is None:
        unknowns.append(f"{label}.effects")
    return {
        "arguments": list(arguments),
        "captures": normalized_captures,
        "operations": normalized_ops,
        "yields": list(yields),
        "effects": effects,
    }


def _computation(value: Any, *, label: str, unknowns: list[str]) -> dict[str, Any]:
    if value is None:
        unknowns.append(f"{label}.missing")
        return {"status": UNKNOWN, "reason": "instruction computation was not described"}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    if value.get("status") == UNKNOWN:
        reason = _nonempty(value.get("reason"), f"{label}.reason")
        unknowns.append(f"{label}.{reason}")
        return {"status": UNKNOWN, "reason": reason}
    kind = _nonempty(value.get("kind"), f"{label}.kind")
    if kind != "linalg.generic":
        raise ValueError(f"{label}.kind {kind!r} is not described by v1")
    maps = value.get("indexing_maps")
    maps_explicit = value.get("indexing_maps_explicit")
    iterators = value.get("iterator_types")
    if maps is not None and (not isinstance(maps, list) or any(not isinstance(item, str) or not item for item in maps)):
        raise ValueError(f"{label}.indexing_maps must be MLIR attribute strings or null")
    if iterators is not None and (
        not isinstance(iterators, list) or any(item not in {"parallel", "reduction"} for item in iterators)
    ):
        raise ValueError(f"{label}.iterator_types must be parallel/reduction strings or null")
    if maps is None:
        unknowns.append(f"{label}.indexing_maps")
    if maps_explicit is not None and type(maps_explicit) is not bool:
        raise ValueError(f"{label}.indexing_maps_explicit must be Boolean or null")
    if iterators is None:
        unknowns.append(f"{label}.iterator_types")
    return {
        "kind": kind,
        "indexing_maps": maps,
        "indexing_maps_explicit": maps_explicit,
        "iterator_types": iterators,
        "scalar_body": _scalar_body(value.get("scalar_body"), label=f"{label}.scalar_body", unknowns=unknowns),
    }
