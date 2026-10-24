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

from merlin.common.yaml import safe_load_text

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
    source_dependencies: tuple[tuple[str, Path], ...] = ()


def _mapping(path: Path) -> dict[str, Any]:
    try:
        value = safe_load_text(path.read_text(encoding="utf-8"))
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


def _source_dependencies(root: Path, value: Any, metadata: Path) -> tuple[tuple[str, Path], ...]:
    """Resolve explicitly declared sibling source owners, never infer a parent tree."""
    if not isinstance(value, dict):
        raise ProviderError(f"{metadata}: source_dependencies must be a mapping")
    dependencies: list[tuple[str, Path]] = []
    for name, relative in sorted(value.items(), key=lambda item: str(item[0])):
        if type(name) is not str or not name.isascii() or not name.isidentifier():
            raise ProviderError(f"{metadata}: source dependency name must be an ASCII identifier")
        if type(relative) is not str or "\\" in relative:
            raise ProviderError(f"{metadata}: source dependency requires canonical ../sibling spelling")
        parts = relative.split("/")
        if (
            len(parts) != 2
            or parts[0] != ".."
            or parts[1] in {"", ".", ".."}
            or not parts[1].isascii()
            or any(not (char.isalnum() or char in "_.-") for char in parts[1])
        ):
            raise ProviderError(f"{metadata}: source dependency requires canonical ../sibling spelling")
        path = root.parent / parts[1]
        if any(member.is_symlink() for member in (path, *path.parents)):
            raise ProviderError(f"{metadata}: source dependency path contains a symlink")
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ProviderError(f"{metadata}: source dependency is absent") from exc
        if resolved != path or not path.is_dir():
            raise ProviderError(f"{metadata}: source dependency must be an ordinary canonical directory")
        if (
            path.is_relative_to(root)
            or root.is_relative_to(path)
            or any(path.is_relative_to(other) or other.is_relative_to(path) for _, other in dependencies)
        ):
            raise ProviderError(f"{metadata}: source dependency ownership overlaps")
        dependencies.append((name, path))
    return tuple(dependencies)


def support_source_dependencies(root: str | Path) -> dict[str, Path]:
    """Read explicit support source ownership; this grants no executable or access authority."""
    provider = read_provider(root)
    if provider is None or provider.role != ProviderRole.SUPPORT:
        raise ProviderError(f"{root}: source dependencies require a support provider")
    return dict(provider.source_dependencies)


def read_provider(root: str | Path) -> Provider | None:
    """Describe a package without importing it or qualifying any claimed capability.

    Explicit metadata is authoritative and validated fail-closed. Legacy contract packages are
    support providers. Legacy schedule payloads take precedence over ABI-looking wrappers, since
    such wrappers do not make schedules standalone compilers. Unrecognized directories return None.
    """
    selected_root = Path(root).absolute()
    root = selected_root.resolve()
    metadata = root / METADATA_FILE
    if metadata.exists() or metadata.is_symlink():
        if metadata.is_symlink() or not metadata.is_file():
            raise ProviderError(f"{metadata}: provider metadata must be an ordinary file")
        doc = _mapping(metadata)
        if doc.get("schema") != SCHEMA:
            raise ProviderError(f"{metadata}: expected schema {SCHEMA!r}")
        unknown = set(doc) - {"schema", "id", "target", "role", "contract", "source_dependencies"}
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
        if "source_dependencies" in doc and role != ProviderRole.SUPPORT:
            raise ProviderError(f"{metadata}: only support providers may declare source dependencies")
        if doc.get("source_dependencies") and (
            selected_root != root or any(member.is_symlink() for member in (selected_root, *selected_root.parents))
        ):
            raise ProviderError(f"{metadata}: dependency provider root must be canonical and unlinked")
        dependencies = _source_dependencies(root, doc.get("source_dependencies", {}), metadata)
        return Provider(identity, target, role, root, contract, declared=True, source_dependencies=dependencies)

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
