"""Exact package payload inventories and external publication bookkeeping.

An inventory identifies bytes, not an executed certification input. Historical
receipts without producer binding never acquire authority through this module.
The filesystem must be trusted and stable during inventory/publication.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from merlin.common.jsonio import canonical_json, canonical_sha256


def component(value: str) -> str:
    if not value or value in {".", "..", ".publication"} or Path(value).name != value or "\\" in value:
        raise ValueError(f"expected a single path component: {value!r}")
    return value


def safe_path(path: Path) -> Path:
    path = path.absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"symlink path component: {path}")
    return path


def payload_inventory(root: Path) -> dict[str, Any]:
    """Bind all regular members, empty directories and executable permission bits."""
    root = safe_path(root)
    if not root.is_dir():
        raise ValueError(f"payload is not a directory: {root}")
    members = []
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        entry: dict[str, Any] = {"path": path.relative_to(root).as_posix()}
        if stat.S_ISDIR(mode):
            entry.update(kind="directory", executable=mode & 0o111)
        elif stat.S_ISREG(mode):
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            entry.update(kind="file", executable=mode & 0o111, sha256=digest)
        else:
            raise ValueError(f"non-regular payload member: {path}")
        members.append(entry)
    body = {"version": 1, "root_executable": root.stat().st_mode & 0o111, "members": members}
    return {**body, "sha256": canonical_sha256(body)}


def record_path(package: Path) -> Path:
    component(package.name)
    component(package.parent.name)
    return safe_path(package.parent / ".publication" / f"{package.name}.json")


def valid_inventory(value: Any) -> bool:
    """Validate the exact inventory vocabulary emitted by :func:`payload_inventory`."""
    if not isinstance(value, dict) or set(value) != {"version", "root_executable", "members", "sha256"}:
        return False
    if type(value["version"]) is not int or value["version"] != 1 or not isinstance(value["members"], list):
        return False

    def mode(bits):
        return type(bits) is int and bits >= 0 and bits & ~0o111 == 0

    def digest(text):
        return isinstance(text, str) and len(text) == 64 and all(c in "0123456789abcdef" for c in text)

    if not mode(value["root_executable"]) or not digest(value["sha256"]):
        return False
    seen = {}
    for member in value["members"]:
        if not isinstance(member, dict):
            return False
        kind = member.get("kind")
        if not isinstance(kind, str):
            return False
        expected = {"path", "kind", "executable"} | ({"sha256"} if kind == "file" else set())
        if kind not in {"file", "directory"} or set(member) != expected or not mode(member["executable"]):
            return False
        name = member["path"]
        if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
            return False
        path = Path(name)
        if path.is_absolute() or path.as_posix() != name or ".." in path.parts or name == "." or name in seen:
            return False
        if any(parent.as_posix() != "." and seen.get(parent.as_posix()) != "directory" for parent in path.parents):
            return False
        if kind == "file" and not digest(member["sha256"]):
            return False
        seen[name] = kind
    if list(seen) != sorted(seen, key=Path):
        return False
    return value["sha256"] == canonical_sha256({k: v for k, v in value.items() if k != "sha256"})


def bound_inputs(identity: Any, payload: dict[str, Any], compiler_package_id: Any) -> bool:
    """Validate a producer's scoped input commitment; never invent absent evidence."""
    if not isinstance(identity, dict):
        return False
    if (
        type(identity.get("version")) is not int
        or identity.get("version") != 1
        or identity.get("scope") != "package-payload"
        or identity.get("external_dependency_closure") != "not-attested"
        or identity.get("compiler_package_id") != compiler_package_id
        or identity.get("source") != payload
        or not isinstance(identity.get("environment"), dict)
        or not isinstance(identity.get("execution_path"), str)
    ):
        return False
    execution = identity.get("execution")
    if not valid_inventory(execution):
        return False
    tool = identity.get("tool")
    if not isinstance(tool, str) or not tool or Path(tool).is_absolute() or ".." in Path(tool).parts:
        return False
    return any(
        isinstance(member, dict) and member.get("path") == tool and member.get("kind") == "file"
        for member in execution["members"]
    )


def read_record(package: Path) -> dict[str, Any] | None:
    safe_path(package)
    path = record_path(package)
    if not path.exists():
        return None
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"publication record is not a regular file: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or type(data.get("version")) is not int or data.get("version") != 1:
        raise ValueError(f"invalid publication record: {path}")
    if data.get("target") != package.parent.name or data.get("package_slot") != package.name:
        raise ValueError(f"publication record identity mismatch: {path}")
    if not isinstance(data.get("publication"), dict):
        raise ValueError(f"invalid publication metadata: {path}")
    if data.get("payload") != payload_inventory(package):
        raise ValueError(f"publication payload drift: {package}")
    return data


def write_record(package: Path, manifest: dict[str, Any], publication: dict[str, Any], **metadata: Any) -> None:
    path = record_path(package)
    # Ordinary bookkeeping preserves installation provenance. A replacement
    # installation supplies its own metadata and must not inherit old claims.
    if not metadata and path.exists():
        previous = read_record(package)
        metadata = {k: v for k, v in previous.items() if k not in {"publication", "payload"}}
    document = {
        **metadata,
        "version": 1,
        "target": package.parent.name,
        "package_slot": package.name,
        "compiler_package_id": manifest.get("package_id"),
        "payload": payload_inventory(package),
        "publication": publication,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".record-", dir=path.parent)
    with os.fdopen(fd, "wb") as stream:
        stream.write(canonical_json(document) + b"\n")
    os.replace(temporary, path)
