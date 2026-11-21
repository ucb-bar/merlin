"""Read-only identity of OOT support, candidate compiler, and host schedule packages.

Roles describe ownership, not trust or executability. Compiler ABI validation, plugin loading,
and grading remain the responsibility of their existing consumers. No provider code is imported.

A ``merlin.provider.v2`` support provider may declare ``extends``: it is a layer on top of another
support provider (its base), authoring only a contract overlay. The effective contract every consumer
reads is materialized from base + overlay by :mod:`merlin.compiler.backend.providers`, so a layered
provider looks exactly like a flat one to everything downstream.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tomllib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

from merlin.common.record_formats import safe_load_text
from merlin.compiler.selection.features import ImprFeature

# ---- load --------------------------------------------------------------------------------------

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


def contained_resource(root: Path, value: str, *, must_exist: bool = True) -> Path:
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
    if must_exist and not resolved.is_file():
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
    yet (only :mod:`merlin.compiler.backend.providers` needs that).

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
                    contract = contained_resource(root, resource, must_exist=False)
                    return Provider(identity, target, role, root, contract, declared=True, extends=extends)
                if not (root / resource).is_file():
                    raise ProviderError(
                        f"{metadata}: extends {extends.target!r} but its effective contract {resource} has not "
                        "been materialized; call compiler.backend.composition.materialize with explicit bases"
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


def provider_revision(root: Path) -> str | None:
    """Read the explicitly selected provider's source revision without fetching."""
    import subprocess

    root = Path(root).resolve()
    source = root.parent / "SOURCE.yaml"
    if source.is_file():
        doc = _mapping(source)
        if str(doc.get("path") or "") == root.name:
            selected = doc.get("source") or {}
            commit = selected.get("commit") if isinstance(selected, dict) else None
            return str(commit) if commit else None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None
    if len(result) == 2 and Path(result[0]).resolve() == root:
        return result[1]
    return None


# ---- composition -------------------------------------------------------------------------------
# Explicit provider composition derived from the OPU integration implementation.
#
# Only caller-selected provider roots participate. Composition reads source
# identities and proposes an effective contract; writing requires an explicit
# materialize call. Backend declarations do not authorize hardware facts or
# numerical acceptance policy.

LOCK_FILE = "contracts/composed.lock.yaml"
LOCK_SCHEMA = "merlin.provider_compose.v1"
_DIRECTIVES = ("replace", "remove")
_NOT_INHERITED = ("plugin",)
_HEADER = (
    "# GENERATED by merlin.compiler.backend.providers.materialize.\n"
    "# Edit {overlay}, then call materialize(provider_root, bases=selected_provider_roots).\n"
    "# Base: {base_target} @ {base_revision} -- see {lock}.\n"
)


class ComposeError(ProviderError):
    """A layered provider cannot be composed (absent base, revision mismatch, cycle, bad overlay)."""


# ── merge ────────────────────────────────────────────────────────────────────────────────────────


def _is_named_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(i, dict) and "name" in i for i in value)


def _is_scalar_list(value: Any) -> bool:
    return isinstance(value, list) and all(not isinstance(i, (dict, list)) for i in value)


def merge(base: Any, overlay: Any) -> Any:
    """Deep-merge ``overlay`` onto ``base`` (neither is mutated)."""
    if isinstance(base, dict) and isinstance(overlay, dict):
        out = copy.deepcopy(base)
        for key, value in overlay.items():
            out[key] = merge(out[key], value) if key in out else copy.deepcopy(value)
        return out
    if _is_named_list(base) and _is_named_list(overlay) and base and overlay:
        out = [copy.deepcopy(i) for i in base]
        index = {item["name"]: n for n, item in enumerate(out)}
        for item in overlay:
            if item["name"] in index:
                out[index[item["name"]]] = merge(out[index[item["name"]]], item)
            else:
                index[item["name"]] = len(out)
                out.append(copy.deepcopy(item))
        return out
    if _is_scalar_list(base) and _is_scalar_list(overlay):
        out = list(base)
        for item in overlay:
            if item not in out:
                out.append(item)
        return out
    return copy.deepcopy(overlay)


def _split(path: str, *, where: str) -> list[str]:
    parts = path.split(".")
    if not path or any(not p for p in parts):
        raise ComposeError(f"{where}: invalid dotted path {path!r}")
    return parts


def _pop_path(doc: dict[str, Any], parts: list[str], *, where: str) -> Any:
    node: Any = doc
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise ComposeError(f"{where}: path {'.'.join(parts)!r} does not exist in the base")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise ComposeError(f"{where}: path {'.'.join(parts)!r} does not exist in the base")
    return node.pop(parts[-1])


def _get_path(doc: dict[str, Any], parts: list[str]) -> tuple[bool, Any]:
    node: Any = doc
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _set_path(doc: dict[str, Any], parts: list[str], value: Any) -> None:
    node = doc
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = copy.deepcopy(value)


def compose(base: dict[str, Any], overlay: dict[str, Any], *, target: str, where: str = "overlay") -> dict[str, Any]:
    """The effective contract of a layer named ``target`` over ``base``."""
    if not isinstance(base, dict) or not isinstance(overlay, dict):
        raise ComposeError(f"{where}: base contract and overlay must both be mappings")
    if "name" in overlay and overlay["name"] != target:
        raise ComposeError(f"{where}: overlay names {overlay['name']!r} but the provider declares {target!r}")
    directives = {key: overlay.get(key) or [] for key in _DIRECTIVES}
    for key, paths in directives.items():
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise ComposeError(f"{where}: {key} must be a list of dotted paths")
    body = {k: v for k, v in overlay.items() if k not in _DIRECTIVES}
    effective = {k: copy.deepcopy(v) for k, v in base.items() if k not in _NOT_INHERITED}
    for path in directives["remove"]:
        _pop_path(effective, _split(path, where=where), where=where)
    replaced: dict[str, Any] = {}
    for path in directives["replace"]:
        parts = _split(path, where=where)
        present, value = _get_path(body, parts)
        if not present:
            raise ComposeError(f"{where}: replace names {path!r} but the overlay does not set it")
        replaced[path] = value
    effective = merge(effective, body)
    for path, value in replaced.items():
        _set_path(effective, _split(path, where=where), value)
    effective["name"] = target
    return effective


# ── base resolution ──────────────────────────────────────────────────────────────────────────────


def resolve_base(provider: Provider, *, bases: Sequence[Path]) -> Provider:
    """Resolve one base from explicit provider roots; duplicate owners refuse."""
    if provider.extends is None:
        raise ComposeError(f"{provider.root}: provider does not extend another")
    matches: dict[Path, Provider] = {}
    for entry in bases:
        root = Path(entry).resolve()
        found = read_provider(root, materialized=False)
        if found is not None and found.role == ProviderRole.SUPPORT and found.target == provider.extends.target:
            matches[root] = found
    if not matches:
        raise ComposeError(f"{provider.root}: base {provider.extends.target!r} is not explicitly selected")
    if len(matches) != 1:
        raise ComposeError(f"{provider.root}: ambiguous base {provider.extends.target!r}: {list(matches)}")
    return next(iter(matches.values()))


def chain(provider: Provider, *, bases: Sequence[Path]) -> list[Provider]:
    """``[root-most base, ..., provider]``; refuses cycles."""
    seen = [provider]
    while seen[0].extends is not None:
        base = resolve_base(seen[0], bases=bases)
        if any(p.root.resolve() == base.root.resolve() or p.target == base.target for p in seen):
            names = " -> ".join(p.target for p in reversed([base, *seen]))
            raise ComposeError(f"{provider.root}: extends cycle: {names}")
        seen.insert(0, base)
    return seen


# ── materialization ──────────────────────────────────────────────────────────────────────────────


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        doc = safe_load_text(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ComposeError(f"{path}: cannot read: {exc}") from exc
    if not isinstance(doc, dict):
        raise ComposeError(f"{path}: expected a mapping")
    return doc


def _strip_header(text: str) -> str:
    lines = text.splitlines(keepends=True)
    while lines and lines[0].startswith("#"):
        lines.pop(0)
    return "".join(lines)


@dataclass(frozen=True)
class Composition:
    provider: Provider
    base: Provider
    base_revision: str | None
    text: str
    lock: dict[str, Any]


def composition(root: str | Path, *, bases: Sequence[Path]) -> Composition:
    """Compute (without writing) the effective contract of the layered provider at ``root``."""
    provider = _layered(root)
    base = resolve_base(provider, bases=bases)
    if base.contract_path is None:
        raise ComposeError(f"{base.root}: base provider has no contract")
    chain(provider, bases=bases)  # refuse cycles before trusting the base
    if base.extends is not None:
        fresh, _ = materialize(base.root, check=True, bases=bases)
        if not fresh:
            raise ComposeError(f"{base.root}: selected base composition is stale")
    revision = provider_revision(base.root)
    pinned = provider.extends.commit
    if pinned is not None and revision != pinned:
        raise ComposeError(
            f"{provider.root}: extends {base.target!r} pinned at {pinned} but the resolved base "
            f"{base.root} is at {revision or 'an unrecorded revision'}"
        )
    base_bytes = base.contract_path.read_bytes()
    overlay_path = provider.overlay_path
    overlay_bytes = overlay_path.read_bytes()
    effective = compose(_load(base.contract_path), _load(overlay_path), target=provider.target, where=str(overlay_path))
    body = yaml.safe_dump(effective, sort_keys=False, default_flow_style=False, width=100, allow_unicode=True)
    header = _HEADER.format(
        overlay=provider.extends.overlay,
        base_target=base.target,
        base_revision=revision or "unrecorded",
        lock=LOCK_FILE,
    )
    lock = {
        "schema": LOCK_SCHEMA,
        "target": provider.target,
        "provider_id": provider.id,
        "base": {
            "target": base.target,
            "provider_id": base.id,
            "revision": revision,
            "pinned": pinned,
            "repo": provider.extends.repo,
            "contract_sha256": _sha256(base_bytes),
            "not_inherited": list(_NOT_INHERITED),
        },
        "overlay": {"path": provider.extends.overlay, "sha256": _sha256(overlay_bytes)},
        "effective": {"path": _relative(provider.contract_path, provider.root), "sha256": _sha256(body.encode())},
    }
    return Composition(provider, base, revision, header + body, lock)


def _relative(path: Path | None, root: Path) -> str:
    assert path is not None
    return path.resolve().relative_to(root.resolve()).as_posix()


def _layered(root: str | Path) -> Provider:
    root = Path(root).resolve()
    provider = read_provider(root, materialized=False)
    if provider is None or provider.role != ProviderRole.SUPPORT:
        raise ComposeError(f"{root}: not a support provider")
    if provider.extends is None:
        raise ComposeError(f"{root}: provider does not declare extends")
    return provider


def materialize(root: str | Path, *, check: bool = False, bases: Sequence[Path]) -> tuple[bool, Composition]:
    """Write (or, with ``check``, only compare) the effective contract and lock. Returns ``(fresh, c)``:
    ``fresh`` is whether the files on disk already matched before this call."""
    comp = composition(root, bases=bases)
    contract = comp.provider.contract_path
    assert contract is not None
    lock_path = contained_resource(comp.provider.root, LOCK_FILE, must_exist=False)
    lock_text = yaml.safe_dump(comp.lock, sort_keys=False, default_flow_style=False, width=100)
    current = contract.read_text(encoding="utf-8") if contract.is_file() else ""
    current_lock = lock_path.read_text(encoding="utf-8") if lock_path.is_file() else ""
    fresh = _strip_header(current) == _strip_header(comp.text) and current_lock == lock_text
    if not check and not fresh:
        contract.parent.mkdir(parents=True, exist_ok=True)
        contract.write_text(comp.text, encoding="utf-8")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(lock_text, encoding="utf-8")
    return fresh, comp


BACKEND_COMPOSITION_SCHEMA = "merlin.backend_composition.v1"
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


def validate_backend_composition(root: str | Path, *, bases: Sequence[Path]) -> BackendComposition:
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
    if declaration.get("schema") != BACKEND_COMPOSITION_SCHEMA:
        raise BackendCompositionError(f"{path}: schema must be {BACKEND_COMPOSITION_SCHEMA!r}")
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
    fresh, comp = materialize(root, check=True, bases=bases)
    if not fresh:
        raise BackendCompositionError(f"{root}: composed contract or lock is stale; explicitly materialize again")
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


# ---- provided ----------------------------------------------------------------------------------
# Lowering components and named features implemented by the selected vector support provider.
#
# Register-block tables, panel-packed weight layouts, fused requantization epilogues, register-group
# widths and the accumulator micro-kernel runner are owned by a support provider, not by Merlin.
# Merlin reaches each component through the selected provider's ``<component>_api()`` accessor and
# resolves the features they implement from the provider's ``feature_definitions()``; it imports no
# provider module by name. Selection is the rule :func:`merlin.analysis.mining.from_strategy._codegen` applies.
#
# With no provider selected these components and features are unavailable. A caller that needs one
# gets :class:`ProviderUnavailable` naming what it asked for; a caller that only needs to know whether
# a provider feature could be present uses the ``optional_*`` forms, which return ``None`` only when
# nothing is selected and still raise when a selected provider cannot be loaded.


class ProviderUnavailable(LookupError):
    """A provider-implemented component or feature was requested and no vector provider is selected."""


def _selected() -> bool:
    from merlin.analysis.mining.from_strategy import _codegen_selected

    return _codegen_selected()


def provider(*, purpose: str) -> Any:
    """The selected vector support provider; ``purpose`` names the request in the refusal."""
    if not _selected():
        raise ProviderUnavailable(
            f"{purpose} is implemented by a vector support provider and none is selected: name one on "
            "MERLIN_TARGET_PATH (a provider declaring plugin.rvv_codegen) or install its backend package"
        )
    from merlin.analysis.mining.from_strategy import _codegen

    return _codegen()


def optional_provider() -> Any | None:
    """The selected provider, or ``None`` exactly when no provider is selected."""
    if not _selected():
        return None
    from merlin.analysis.mining.from_strategy import _codegen

    return _codegen()


def _component_of(selected: Any, name: str) -> Any:
    accessor = getattr(selected, f"{name}_api", None)
    if not callable(accessor):
        raise ProviderUnavailable(f"the selected vector support provider implements no {name!r} component")
    return accessor()


def component(name: str) -> Any:
    """The selected provider's implementation module for ``name`` (its ``<name>_api()``)."""
    return _component_of(provider(purpose=f"the {name!r} component"), name)


def optional_component(name: str) -> Any | None:
    """:func:`component`, or ``None`` exactly when no provider is selected."""
    selected = optional_provider()
    return None if selected is None else _component_of(selected, name)


def abi_bundle(model_dir: str | Path, work: str | Path, prepared: str | Path) -> tuple[Path, dict | None]:
    """The bundle an ABI table must be built from once ``work`` has been prepared.

    The selected provider's ``weight_panel`` component may have repacked weights during preparation;
    its ``abi_bundle`` returns the matching bundle and a report. With no provider selected nothing
    was packed, so the bundle is ``model_dir`` and there is no report.
    """
    panel = optional_component("weight_panel")
    return (Path(model_dir), None) if panel is None else panel.abi_bundle(model_dir, work, prepared)


def guard_abi_bundle(model_dir: str | Path, out_dir: str | Path) -> None:
    """Refuse a stock bundle for a build whose provider packed its weights (see :func:`abi_bundle`)."""
    panel = optional_component("weight_panel")
    if panel is not None:
        panel.guard_planned_pack(model_dir, out_dir)


_DEFINITIONS: dict[int, tuple[Any, dict[str, ImprFeature]]] = {}


def _definitions(selected: Any) -> dict[str, ImprFeature]:
    cached = _DEFINITIONS.get(id(selected))
    if cached is not None and cached[0] is selected:
        return cached[1]
    declare = getattr(selected, "feature_definitions", None)
    features = tuple(declare()) if callable(declare) else ()
    by_name: dict[str, ImprFeature] = {}
    for feature in features:
        if not isinstance(feature, ImprFeature):
            raise TypeError(f"provider feature definition must be ImprFeature, got {type(feature).__name__}")
        if feature.name in by_name:
            raise ValueError(f"provider declares feature {feature.name!r} twice")
        by_name[feature.name] = feature
    _DEFINITIONS[id(selected)] = (selected, by_name)
    return by_name


def feature_names() -> frozenset[str]:
    """Names the selected provider defines; empty when no provider is selected."""
    selected = optional_provider()
    return frozenset() if selected is None else frozenset(_definitions(selected))


def feature_definition(name: str) -> ImprFeature:
    """The selected provider's declaration of ``name``.

    Raises :class:`KeyError` when the selected provider defines no such feature, and
    :class:`ProviderUnavailable` (also a ``LookupError``) when no provider is selected to ask.
    """
    definitions = _definitions(provider(purpose=f"feature {name!r}"))
    if name not in definitions:
        raise KeyError(name)
    return definitions[name]


# ---- service -----------------------------------------------------------------------------------
# Resolve explicitly selected installed, trusted compiler/runtime services.
#
# This entry point is for ordinary compilation. Candidate grading retains its
# independent source and execution admission; these plugins cannot qualify it.


def _origin(distribution) -> dict[str, Any]:
    """Read declared installation origin, withholding URL credentials and queries."""
    if distribution is None:
        return {"status": "unknown"}
    raw = distribution.read_text("direct_url.json")
    if raw is None:
        return {"status": "unknown"}
    try:
        document = json.loads(raw)
        if not isinstance(document, dict) or not isinstance(document.get("url"), str):
            raise ValueError("invalid origin")
        url = urlsplit(document["url"])
        public_url = urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", ""))
        result: dict[str, Any] = {
            "status": "declared",
            "url": public_url,
            "url_redacted": public_url != document["url"],
        }
        directory = document.get("dir_info")
        vcs = document.get("vcs_info")
        archive = document.get("archive_info")
        if isinstance(directory, dict) and type(directory.get("editable", False)) is bool:
            result.update(kind="directory", editable=directory.get("editable", False))
        elif isinstance(vcs, dict):
            result.update(
                kind="vcs",
                vcs={
                    key: value
                    for key, value in vcs.items()
                    if key in {"vcs", "requested_revision", "commit_id"} and isinstance(value, str)
                },
            )
        elif isinstance(archive, dict):
            hashes = archive.get("hashes", {})
            result.update(
                kind="archive",
                hashes={key: value for key, value in hashes.items() if isinstance(key, str) and isinstance(value, str)}
                if isinstance(hashes, dict)
                else {},
            )
        else:
            result["kind"] = "unknown"
        return result
    except (ValueError, TypeError):
        return {"status": "invalid_metadata"}


def _description(entry) -> dict[str, Any]:
    distribution = getattr(entry, "dist", None)
    metadata = distribution.metadata if distribution is not None else {}
    return {
        "name": getattr(entry, "name", "unknown"),
        "entry_point": getattr(entry, "value", "unknown"),
        "distribution": {"name": metadata.get("Name", "unknown"), "version": getattr(distribution, "version", None)},
        "origin": _origin(distribution),
    }


def inventory(*, group: str = "merlin.backends") -> dict[str, Any]:
    """Describe installed services without importing, validating or executing them.

    Metadata and declared installation origins establish neither compatibility
    nor qualification. Checkout contracts and run-scoped candidates retain their
    separate explicit resolution and admission paths.
    """
    if group not in {"merlin.backends", "merlin.runners"}:
        raise ValueError("service inventory requires a compiler or runtime entry-point group")
    rows = [_description(entry) for entry in entry_points(group=group)]
    rows.sort(
        key=lambda row: (
            row["name"],
            row["distribution"]["name"],
            row["distribution"]["version"] or "",
            row["entry_point"],
        )
    )
    counts = Counter(row["name"] for row in rows)
    for row in rows:
        row.update(selection="ambiguous" if counts[row["name"]] > 1 else "unique", compatibility="not_checked")
    return {
        "schema": "merlin.service_inventory.v1",
        "group": group,
        "python": {"executable": sys.executable, "prefix": sys.prefix, "version": sys.version.split()[0]},
        "code_loaded_by_inspection": False,
        "scope": "installed_entry_point_metadata_only",
        "providers": rows,
    }


def resolve(selected: str | Any, *, group: str, methods: tuple[str, ...]) -> tuple[str, Any]:
    if isinstance(selected, str):
        matches = list(entry_points(group=group, name=selected))
        if len(matches) != 1:
            choices = sorted(
                f"{row['distribution']['name']}=={row['distribution']['version']} ({row['entry_point']})"
                for row in map(_description, matches)
            )
            detail = f"; candidates: {', '.join(choices)}" if choices else ""
            raise ValueError(
                f"expected one installed {group} provider named {selected!r}, found {len(matches)}{detail}"
            )
        provider = matches[0].load()
        name = selected
    else:
        provider = selected
        name = getattr(provider, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("an explicitly supplied service requires a nonempty name")
    for method in methods:
        if not callable(getattr(provider, method, None)):
            raise ValueError(f"service {name!r} lacks callable {method}")
    return name, provider
