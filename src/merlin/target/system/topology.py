"""Explicit selected execution resources and synchronous shared memory access.

The original caller selects this contract; provider proposals cannot amend it.
No relationship is inferred from a backend name or from a device's presence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.common.yaml import safe_load_text

SCHEMA = "merlin.execution_topology.v1"


def _name(value, field):
    if type(value) is not str or not value.strip():
        raise ValueError(f"topology requires an explicit {field}")


def _names(value, field):
    if type(value) is not tuple or len(set(value)) != len(value):
        raise ValueError(f"topology {field} must be an immutable unique tuple")
    for item in value:
        _name(item, field)


@dataclass(frozen=True)
class Resource:
    id: str
    kind: str
    features: tuple[str, ...]
    shared_group: str

    def __post_init__(self):
        for field in ("id", "kind", "shared_group"):
            _name(getattr(self, field), field)
        _names(self.features, "features")


@dataclass(frozen=True)
class MemoryDomain:
    """Available selected byte budget after external/runtime reservations."""

    id: str
    capacity_bytes: int
    alignment_bytes: int

    def __post_init__(self):
        _name(self.id, "memory domain")
        if type(self.capacity_bytes) is not int or self.capacity_bytes <= 0:
            raise ValueError("memory domain requires an explicit positive available byte capacity")
        alignment = self.alignment_bytes
        if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("memory domain alignment must be an explicit positive power of two")


@dataclass(frozen=True)
class MemoryAccess:
    resource: str
    domain: str
    read: bool
    write: bool
    addressability: str
    visibility: str
    completion: str

    def __post_init__(self):
        for field in ("resource", "domain", "addressability", "visibility", "completion"):
            _name(getattr(self, field), field)
        if type(self.read) is not bool or type(self.write) is not bool:
            raise ValueError("memory access read/write permissions must be explicit booleans")
        if self.addressability not in ("shared", "separate", "unknown"):
            raise ValueError("unknown memory addressability vocabulary")
        if self.visibility not in ("coherent", "explicit", "unknown"):
            raise ValueError("unknown memory visibility vocabulary")
        if self.completion not in ("synchronous", "explicit", "unknown"):
            raise ValueError("unknown memory completion vocabulary")


def _document(resources, memories, access):
    return json.dumps(
        {
            "resources": [asdict(r) for r in resources],
            "memory_domains": [asdict(m) for m in memories],
            "access": [asdict(a) for a in access],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse(payload):
    document = safe_load_text(payload.decode("utf-8"))
    if type(document) is not dict or set(document) != {"schema", "resources", "memory_domains", "access"}:
        raise ValueError("execution topology requires its exact schema and resource/memory/access fields")
    if document["schema"] != SCHEMA:
        raise ValueError("unsupported execution topology schema")
    for field in ("resources", "memory_domains", "access"):
        if type(document[field]) is not list or len(document[field]) > 4096:
            raise ValueError(f"topology {field} must be a bounded list")
    resources = []
    for row in document["resources"]:
        if type(row) is not dict or set(row) != {"id", "kind", "features", "shared_group"}:
            raise ValueError("resource requires explicit identity, kind, features and shared group")
        if type(row["features"]) is not list:
            raise ValueError("resource features must be an explicit list")
        resources.append(Resource(**{**row, "features": tuple(row["features"])}))
    try:
        memories = tuple(MemoryDomain(**row) for row in document["memory_domains"])
        access = tuple(MemoryAccess(**row) for row in document["access"])
    except TypeError as error:
        raise ValueError("memory domains/access require exact explicitly typed fields") from error
    resources = tuple(resources)
    if not resources or not memories:
        raise ValueError("execution topology requires resources and memory domains")
    if len({r.id for r in resources}) != len(resources) or len({m.id for m in memories}) != len(memories):
        raise ValueError("execution topology repeats a resource or memory identity")
    if len({(a.resource, a.domain) for a in access}) != len(access):
        raise ValueError("execution topology repeats a resource/domain access edge")
    for edge in access:
        if edge.resource not in {r.id for r in resources} or edge.domain not in {m.id for m in memories}:
            raise ValueError("memory access names an undeclared resource or domain")
    return resources, memories, access


def _read(path):
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("execution topology requires an original unlinked regular file")
    if path.stat().st_size > 1 << 20:
        raise ValueError("execution topology exceeds its byte budget")
    payload = path.read_bytes()
    if len(payload) > 1 << 20:
        raise ValueError("execution topology exceeds its byte budget")
    return payload


@dataclass(frozen=True)
class SelectedTopology:
    path: Path
    sha256: str
    resources: tuple[Resource, ...]
    memory_domains: tuple[MemoryDomain, ...]
    access: tuple[MemoryAccess, ...]
    _record: str

    def revalidate(self):
        payload = _read(self.path)
        if hashlib.sha256(payload).hexdigest() != self.sha256:
            raise ValueError("selected execution topology changed; replan")
        parsed = _parse(payload)
        if (
            _document(*parsed) != self._record
            or _document(self.resources, self.memory_domains, self.access) != self._record
        ):
            raise ValueError("selected execution topology bindings changed; replan")

    def resource(self, identity: str) -> Resource:
        self.revalidate()
        found = next((row for row in self.resources if row.id == identity), None)
        if found is None:
            raise ValueError(f"undeclared execution resource {identity!r}")
        return found

    def memory(self, identity: str) -> MemoryDomain:
        self.revalidate()
        found = next((row for row in self.memory_domains if row.id == identity), None)
        if found is None:
            raise ValueError(f"undeclared memory domain {identity!r}")
        return found

    def require_access(self, resource: str, domain: str, *, read: bool, write: bool):
        """Admit this slice's synchronous coherent shared-address operation."""
        self.resource(resource)
        self.memory(domain)
        edge = next((a for a in self.access if (a.resource, a.domain) == (resource, domain)), None)
        if edge is None or read and not edge.read or write and not edge.write:
            raise ValueError("resource has no declared access to its boundary memory")
        if (edge.addressability, edge.visibility, edge.completion) != ("shared", "coherent", "synchronous"):
            raise ValueError("selected execution requires an unimplemented transfer, visibility or completion action")
        return edge


def select_topology(path: str | Path) -> SelectedTopology:
    """Bind an original caller-selected hardware contract, never a provider claim."""
    path = Path(path).absolute()
    payload = _read(path)
    resources, memories, access = _parse(payload)
    result = SelectedTopology(
        path, hashlib.sha256(payload).hexdigest(), resources, memories, access, _document(resources, memories, access)
    )
    result.revalidate()
    return result
