"""Mandatory historical target-access declarations for the experiments sandbox.

The selected target's support tree is masked wholesale. These old import names are separate:
copies can survive in broadly bound Python installations even after the original source is gone.
Keep their identities as packaged data, and refuse to start the sandbox if that data is absent.
"""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path

from merlin.common import access as shared

RESOURCE = "resources/legacy_target_access.json"


class AccessPolicyUnavailable(RuntimeError):
    """The required experiments-owned deny registry cannot be used."""


def resource_path() -> Path:
    """Return the installed policy bytes for frozen-source inventory and audit receipts."""
    path = Path(str(files("merlin_experiments").joinpath(RESOURCE)))
    if path.is_symlink() or not path.is_file():
        raise AccessPolicyUnavailable(f"required access policy is absent or linked: {path}")
    return path


def load_legacy_access(path: Path | None = None) -> tuple[shared.ModuleAccess, ...]:
    """Load and validate required denies; missing/malformed data never means an empty policy."""
    source = path if path is not None else resource_path()
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise AccessPolicyUnavailable(f"cannot read required access policy: {source}") from exc
    if not isinstance(data, dict) or set(data) != {"schema", "modules"}:
        raise AccessPolicyUnavailable("invalid access policy structure")
    if data["schema"] != "merlin.access.legacy.v1" or not isinstance(data["modules"], list) or not data["modules"]:
        raise AccessPolicyUnavailable("invalid access policy schema or empty module set")
    result: list[shared.ModuleAccess] = []
    seen = {module for item in shared.MODULE_ACCESS for module in item.modules}
    for row in data["modules"]:
        if not isinstance(row, dict) or set(row) != {"identity", "origin", "directory", "aliases"}:
            raise AccessPolicyUnavailable("invalid access policy module row")
        identity, origin, directory, aliases = (row[key] for key in ("identity", "origin", "directory", "aliases"))
        if (
            not isinstance(identity, str)
            or not identity
            or not all(part.isidentifier() for part in identity.split("."))
            or origin not in {"grader", "oracle"}
            or type(directory) is not bool
            or not isinstance(aliases, list)
            or not all(
                isinstance(alias, str) and alias and all(part.isidentifier() for part in alias.split("."))
                for alias in aliases
            )
            or len(set(aliases)) != len(aliases)
        ):
            raise AccessPolicyUnavailable(f"invalid or duplicate access policy identity: {identity!r}")
        item = shared.module_access(identity, origin, directory=directory, aliases=tuple(aliases))
        if seen.intersection(item.modules):
            raise AccessPolicyUnavailable(f"invalid or duplicate access policy identity: {identity!r}")
        result.append(item)
        seen.update(item.modules)
    return tuple(result)


# Importing the experiments sandbox must refuse before any agent launch when this policy is missing.
MODULE_ACCESS = (*shared.MODULE_ACCESS, *load_legacy_access())


def _resource_digest() -> str:
    try:
        return hashlib.sha256(resource_path().read_bytes()).hexdigest()
    except OSError as exc:
        raise AccessPolicyUnavailable("required access policy cannot be read") from exc


_RESOURCE_SHA256 = _resource_digest()


def require_current_policy() -> None:
    """Refuse if packaged bytes changed after the deny set was loaded in this process."""
    if _resource_digest() != _RESOURCE_SHA256 or load_legacy_access() != MODULE_ACCESS[len(shared.MODULE_ACCESS) :]:
        raise AccessPolicyUnavailable("required access policy changed after sandbox initialization")


def declared_modules(origin: str) -> tuple[str, ...]:
    require_current_policy()
    return shared.declared_modules(origin, items=MODULE_ACCESS)


def legacy_module_paths(origin: str) -> tuple[str, ...]:
    require_current_policy()
    return shared.legacy_module_paths(origin, items=MODULE_ACCESS)


def unresolved_modules(root: Path) -> tuple[shared.ModuleAccess, ...]:
    require_current_policy()
    return shared.unresolved_modules(root, items=MODULE_ACCESS)
