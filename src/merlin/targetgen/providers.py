"""Read-only identity of OOT support, candidate compiler, and host schedule packages.

Roles describe ownership, not trust or executability. Compiler ABI validation, plugin loading,
and grading remain the responsibility of their existing consumers. No provider code is imported.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

SCHEMA = "merlin.provider.v1"
METADATA_FILE = "provider.yaml"


class ProviderError(ValueError):
    """An explicitly declared provider has an invalid identity or resource boundary."""


class ProviderRole(StrEnum):
    SUPPORT = "support"
    CANDIDATE_COMPILER = "candidate_compiler"
    HOST_SCHEDULE = "host_schedule"


@dataclass(frozen=True)
class Provider:
    id: str
    target: str
    role: ProviderRole
    root: Path
    contract_path: Path | None = None
    declared: bool = False


def _mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProviderError(f"{path}: cannot read provider data: {exc}") from exc
    if not isinstance(value, dict):
        raise ProviderError(f"{path}: expected a mapping")
    return value


def _identity(value: Any, *, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProviderError(f"{path}: {field} must be a non-empty string")
    return value


def contained_resource(root: Path, value: str) -> Path:
    """Resolve a declared relative resource, allowing symlinks only within the provider root.

    Explicit user-selected provider roots may themselves be symlinks. Relative traversal which
    ultimately stays inside that canonical root is valid; absolute paths and escapes are not.
    """
    relative = Path(value)
    if relative.is_absolute() or not relative.parts:
        raise ProviderError(f"{root}: resource must be a non-empty relative path: {value!r}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ProviderError(f"{root}: resource escapes provider root: {value!r}")
    if not resolved.is_file():
        raise ProviderError(f"{root}: resource is not a file: {value!r}")
    return resolved


def read_provider(root: str | Path) -> Provider | None:
    """Describe a package without importing it or qualifying any claimed capability.

    Explicit metadata is authoritative and validated fail-closed. Legacy contract packages are
    support providers. Legacy schedule payloads take precedence over ABI-looking wrappers, since
    such wrappers do not make schedules standalone compilers. Unrecognized directories return None.
    """
    root = Path(root).resolve()
    metadata = root / METADATA_FILE
    if metadata.is_file():
        doc = _mapping(metadata)
        if doc.get("schema") != SCHEMA:
            raise ProviderError(f"{metadata}: expected schema {SCHEMA!r}")
        unknown = set(doc) - {"schema", "id", "target", "role", "contract"}
        if unknown:
            raise ProviderError(f"{metadata}: unknown fields {sorted(unknown)}")
        identity = _identity(doc.get("id"), field="id", path=metadata)
        target = _identity(doc.get("target"), field="target", path=metadata)
        try:
            role = ProviderRole(doc.get("role"))
        except (TypeError, ValueError) as exc:
            raise ProviderError(f"{metadata}: invalid provider role {doc.get('role')!r}") from exc
        contract = None
        if role == ProviderRole.SUPPORT:
            resource = _identity(doc.get("contract", "contracts/target_contract.yaml"), field="contract", path=metadata)
            contract = contained_resource(root, resource)
            name = _mapping(contract).get("name")
            if name != target:
                raise ProviderError(f"{metadata}: target {target!r} differs from contract name {name!r}")
        elif "contract" in doc:
            raise ProviderError(f"{metadata}: only support providers may declare a contract")
        return Provider(identity, target, role, root, contract, declared=True)

    contract = root / "contracts" / "target_contract.yaml"
    if contract.is_file():
        contract = contained_resource(root, "contracts/target_contract.yaml")
        doc = _mapping(contract)
        name = str(doc.get("name") or root.name)
        return Provider(name, name, ProviderRole.SUPPORT, root, contract)
    manifest = root / "manifest.yaml"
    if not manifest.is_file():
        return None
    doc = _mapping(manifest)
    target = str(doc.get("target") or root.name)
    identity = str(doc.get("package_id") or doc.get("run_id") or root.name)
    if (root / "knobs.yaml").is_file() or (root / "payload" / "knobs.yaml").is_file():
        return Provider(identity, target, ProviderRole.HOST_SCHEDULE, root)
    if doc.get("artifact_type") == "mlir_oot_target_backend":
        return Provider(identity, target, ProviderRole.CANDIDATE_COMPILER, root)
    return None
