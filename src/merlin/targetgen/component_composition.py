"""Validate a chip backend's explicit composition declaration.

``provider.yaml`` and the composed contract remain the source of truth for the
base revision and executable plugin paths. ``backend.toml`` names ownership and
the chip's deliberate additions, alternatives, replacements, and restrictions.
This module never imports an implementation module, so import order cannot
select an owner or hide a conflicting declaration.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.common.yaml import safe_load_text

from .provider_compose import materialize
from .providers import read_provider

SCHEMA = "merlin.backend_composition.v1"
_ACTIONS = frozenset({"add", "alternative", "replace", "disable"})
_OPERATION_FIELDS = frozenset(
    {"name", "action", "owner", "unit", "pipeline", "alongside", "replaces", "preserves", "reason"}
)
_TOP_LEVEL_FIELDS = frozenset({"schema", "target", "base_target", "plugin_owners", "operations"})


class BackendCompositionError(ValueError):
    """The selected backend has an ambiguous or stale composition declaration."""


@dataclass(frozen=True)
class BackendComposition:
    target: str
    base_target: str
    base_commit: str
    plugin_owners: dict[str, str]
    operations: tuple[dict[str, str], ...]


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BackendCompositionError(f"backend.toml: {field} must be a non-empty string")
    return value


def _operations(raw: Any, *, target: str) -> tuple[dict[str, str], ...]:
    if not isinstance(raw, list):
        raise BackendCompositionError("backend.toml: operations must be an array of tables")
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for index, entry in enumerate(raw):
        where = f"operations[{index}]"
        if not isinstance(entry, dict):
            raise BackendCompositionError(f"backend.toml: {where} must be a table")
        unknown = set(entry) - _OPERATION_FIELDS
        if unknown:
            raise BackendCompositionError(f"backend.toml: {where} has unknown fields {sorted(unknown)}")
        name = _required_string(entry.get("name"), f"{where}.name")
        action = _required_string(entry.get("action"), f"{where}.action")
        owner = _required_string(entry.get("owner"), f"{where}.owner")
        if action not in _ACTIONS:
            raise BackendCompositionError(f"backend.toml: {where}.action {action!r} is not one of {sorted(_ACTIONS)}")
        if name in seen:
            raise BackendCompositionError(f"backend.toml: conflicting ownership for operation {name!r}")
        seen.add(name)
        if owner != target:
            raise BackendCompositionError(f"backend.toml: {where}.owner must be local target {target!r}, got {owner!r}")
        if action == "alternative":
            _required_string(entry.get("alongside"), f"{where}.alongside")
        elif "alongside" in entry:
            raise BackendCompositionError(f"backend.toml: {where}.alongside requires action='alternative'")
        if action == "replace":
            _required_string(entry.get("replaces"), f"{where}.replaces")
            _required_string(entry.get("preserves"), f"{where}.preserves")
        elif "replaces" in entry or "preserves" in entry:
            raise BackendCompositionError(f"backend.toml: {where}.replaces/preserves require action='replace'")
        if action == "disable":
            _required_string(entry.get("reason"), f"{where}.reason")
        elif "reason" in entry:
            raise BackendCompositionError(f"backend.toml: {where}.reason requires action='disable'")
        if action in {"add", "alternative", "replace"} and not (entry.get("unit") or entry.get("pipeline")):
            raise BackendCompositionError(f"backend.toml: {where} must name a unit or pipeline")
        for key, value in entry.items():
            if not isinstance(value, str):
                raise BackendCompositionError(f"backend.toml: {where}.{key} must be a string")
        result.append(dict(entry))
    return tuple(result)


def validate_backend_composition(root: str | Path) -> BackendComposition:
    """Check declared owners and operations against the exact composed base.

    The existing provider composer checks the full ``extends`` chain for cycles,
    verifies its pinned revision, and compares the generated contract and lock.
    We reuse that check instead of resolving providers in import order.
    """
    root = Path(root).resolve()
    path = root / "backend.toml"
    try:
        declaration = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise BackendCompositionError(f"{path}: cannot read composition: {exc}") from exc
    if declaration.get("schema") != SCHEMA:
        raise BackendCompositionError(f"{path}: schema must be {SCHEMA!r}")
    unknown = set(declaration) - _TOP_LEVEL_FIELDS
    if unknown:
        raise BackendCompositionError(f"{path}: unknown top-level fields {sorted(unknown)}")
    target = _required_string(declaration.get("target"), "target")
    base_target = _required_string(declaration.get("base_target"), "base_target")
    provider = read_provider(root)
    if provider is None or provider.extends is None:
        raise BackendCompositionError(f"{root}: expected a layered support provider")
    if provider.target != target or provider.extends.target != base_target:
        raise BackendCompositionError(
            f"{path}: target/base {target!r}/{base_target!r} disagree with provider.yaml "
            f"{provider.target!r}/{provider.extends.target!r}"
        )
    if target == base_target:
        raise BackendCompositionError(f"{path}: recursive composition of {target!r}")
    fresh, comp = materialize(root, check=True)
    if not fresh:
        raise BackendCompositionError(f"{root}: composed contract or lock is stale; run merlin-target-compose")
    if comp.base_revision is None:
        raise BackendCompositionError(f"{root}: base revision is not recorded")
    owners = declaration.get("plugin_owners")
    if not isinstance(owners, dict) or not owners:
        raise BackendCompositionError(f"{path}: plugin_owners must be a non-empty table")
    effective = safe_load_text(provider.contract_path.read_text(encoding="utf-8"))
    local_plugins = effective.get("plugin") or {}
    if not isinstance(local_plugins, dict):
        raise BackendCompositionError(f"{root}: effective plugin block is not a mapping")
    for key, owner in owners.items():
        _required_string(key, "plugin_owners key")
        owner = _required_string(owner, f"plugin_owners.{key}")
        if owner not in {target, base_target}:
            raise BackendCompositionError(f"{path}: plugin {key!r} has unknown owner {owner!r}")
        if owner == target and key not in local_plugins:
            raise BackendCompositionError(f"{path}: local plugin owner {key!r} has no effective plugin entry")
        if owner == base_target and key in local_plugins:
            raise BackendCompositionError(f"{path}: plugin {key!r} claims base ownership but layer declares it locally")
    for key in local_plugins:
        if owners.get(key) != target:
            raise BackendCompositionError(f"{path}: local plugin {key!r} has no explicit local owner")
    operations = _operations(declaration.get("operations", []), target=target)
    return BackendComposition(target, base_target, comp.base_revision, dict(owners), operations)
