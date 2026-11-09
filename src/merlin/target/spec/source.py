"""Exact-source, non-executable semantic-library composition."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

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
