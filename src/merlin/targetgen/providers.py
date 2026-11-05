"""Read-only identity of OOT support, candidate compiler, and host schedule packages.

Roles describe ownership, not trust or executability. Compiler ABI validation, plugin loading,
and grading remain the responsibility of their existing consumers. No provider code is imported.

A ``merlin.provider.v2`` support provider may declare ``extends``: it is a layer on top of another
support provider (its base), authoring only a contract overlay. The effective contract every consumer
reads is materialized from base + overlay by :mod:`merlin.targetgen.provider_compose`, so a layered
provider looks exactly like a flat one to everything downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from merlin.common.yaml import safe_load_text

SCHEMA = "merlin.provider.v1"
SCHEMA_V2 = "merlin.provider.v2"
SCHEMAS = (SCHEMA, SCHEMA_V2)
METADATA_FILE = "provider.yaml"
DEFAULT_CONTRACT = "contracts/target_contract.yaml"
DEFAULT_OVERLAY = "contracts/overlay.yaml"
_V1_FIELDS = frozenset({"schema", "id", "target", "role", "contract"})
_V2_FIELDS = _V1_FIELDS | {"extends"}
_EXTENDS_FIELDS = frozenset({"target", "commit", "repo", "overlay"})


class ProviderError(ValueError):
    """An explicitly declared provider has an invalid identity or resource boundary."""


class ProviderRole(StrEnum):
    SUPPORT = "support"
    CANDIDATE_COMPILER = "candidate_compiler"
    HOST_SCHEDULE = "host_schedule"


@dataclass(frozen=True)
class ProviderBase:
    """What a layered provider declares it builds on. ``commit`` pins the base's revision (full sha);
    ``repo`` is where to fetch it from when it is not already selected or vendored."""

    target: str
    overlay: str
    commit: str | None = None
    repo: str | None = None


@dataclass(frozen=True)
class Provider:
    id: str
    target: str
    role: ProviderRole
    root: Path
    contract_path: Path | None = None
    declared: bool = False
    extends: ProviderBase | None = None

    @property
    def overlay_path(self) -> Path | None:
        return None if self.extends is None else self.root / self.extends.overlay


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


def _extends(value: Any, *, target: str, path: Path) -> ProviderBase:
    if not isinstance(value, dict):
        raise ProviderError(f"{path}: extends must be a mapping")
    unknown = set(value) - _EXTENDS_FIELDS
    if unknown:
        raise ProviderError(f"{path}: extends has unknown fields {sorted(unknown)}")
    base = _identity(value.get("target"), field="extends.target", path=path)
    if base == target:
        raise ProviderError(f"{path}: a provider cannot extend its own target {target!r}")
    commit = value.get("commit")
    if commit is not None:
        commit = _identity(commit, field="extends.commit", path=path)
        if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
            raise ProviderError(f"{path}: extends.commit must be a full lowercase 40-char sha, got {commit!r}")
    repo = value.get("repo")
    if repo is not None:
        repo = _identity(repo, field="extends.repo", path=path)
    overlay = _identity(value.get("overlay", DEFAULT_OVERLAY), field="extends.overlay", path=path)
    return ProviderBase(target=base, overlay=overlay, commit=commit, repo=repo)


def read_provider(root: str | Path, *, materialized: bool = True) -> Provider | None:
    """Describe a package without importing it or qualifying any claimed capability.

    ``materialized=False`` describes a layered provider whose effective contract has not been composed
    yet (only :mod:`merlin.targetgen.provider_compose` needs that).

    Explicit metadata is authoritative and validated fail-closed. Legacy contract packages are
    support providers. Legacy schedule payloads take precedence over ABI-looking wrappers, since
    such wrappers do not make schedules standalone compilers. Unrecognized directories return None.
    """
    root = Path(root).resolve()
    metadata = root / METADATA_FILE
    if metadata.is_file():
        doc = _mapping(metadata)
        schema = doc.get("schema")
        if schema not in SCHEMAS:
            raise ProviderError(f"{metadata}: expected schema one of {list(SCHEMAS)}, got {schema!r}")
        unknown = set(doc) - (_V2_FIELDS if schema == SCHEMA_V2 else _V1_FIELDS)
        if unknown:
            raise ProviderError(f"{metadata}: unknown fields {sorted(unknown)} for {schema}")
        identity = _identity(doc.get("id"), field="id", path=metadata)
        target = _identity(doc.get("target"), field="target", path=metadata)
        try:
            role = ProviderRole(doc.get("role"))
        except (TypeError, ValueError) as exc:
            raise ProviderError(f"{metadata}: invalid provider role {doc.get('role')!r}") from exc
        extends = _extends(doc.get("extends"), target=target, path=metadata) if "extends" in doc else None
        if extends is not None and role != ProviderRole.SUPPORT:
            raise ProviderError(f"{metadata}: only support providers may declare extends")
        contract = None
        if role == ProviderRole.SUPPORT:
            resource = _identity(doc.get("contract", DEFAULT_CONTRACT), field="contract", path=metadata)
            if extends is not None:
                contained_resource(root, extends.overlay)
                if not materialized:
                    return Provider(identity, target, role, root, root / resource, declared=True, extends=extends)
                if not (root / resource).is_file():
                    raise ProviderError(
                        f"{metadata}: extends {extends.target!r} but its effective contract {resource} has not "
                        f"been materialized; run `merlin-target-compose {root}`"
                    )
            contract = contained_resource(root, resource)
            name = _mapping(contract).get("name")
            if name != target:
                raise ProviderError(f"{metadata}: target {target!r} differs from contract name {name!r}")
        elif "contract" in doc:
            raise ProviderError(f"{metadata}: only support providers may declare a contract")
        return Provider(identity, target, role, root, contract, declared=True, extends=extends)

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
