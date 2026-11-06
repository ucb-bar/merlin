"""Per-object construction receipts for persistent host dispatch kernels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .host_cache_identity import _file

SCHEMA = "merlin.host_cache_object.v1"


def manifest_path(so: Path) -> Path:
    return so.with_suffix(".manifest.json")


def write_manifest(so: Path, *, key: str, kernel_text: str, identity: dict[str, Any], workdir: Path) -> dict:
    """Seal an object only after its lowerer and native linker have succeeded."""
    recipe = workdir / "lowering_recipe.json"
    record = json.loads(recipe.read_text())
    if (
        record.get("schema") != "merlin.lowering_recipe.v1"
        or record.get("status") != "returned"
        or not record.get("commands")
    ):
        raise ValueError(f"host cache needs a completed lowering receipt: {recipe}")
    obj = _file(so)
    payload = {
        "schema": SCHEMA,
        "key": key,
        "kernel_text_sha256": hashlib.sha256(kernel_text.encode()).hexdigest(),
        "compiler_identity_sha256": identity["sha256"],
        "object_sha256": obj["sha256"],
        "object_bytes": obj["bytes"],
        "lowering_recipe_sha256": _file(recipe)["sha256"],
        "lowering_commands": record["commands"],
        "builder": "merlin.llvmlower.kernel_backend.compile_host",
        "native_codegen": "build_host_shared: selected clang or llc -O2 -fPIC; cc -O2 -fPIC runtime; cc -shared -lm",
    }
    path = manifest_path(so)
    if path.is_symlink():
        raise ValueError(f"host cache refuses symlinked manifest: {path}")
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return payload


def verify_manifest(so: Path, *, key: str, kernel_text: str, identity: dict[str, Any]) -> dict | None:
    """Return verified receipt, or None for an absent/malformed/mismatched object."""
    path = manifest_path(so)
    if so.is_symlink() or path.is_symlink() or not so.is_file() or not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
        obj = _file(so)
        expected = {
            "schema": SCHEMA,
            "key": key,
            "kernel_text_sha256": hashlib.sha256(kernel_text.encode()).hexdigest(),
            "compiler_identity_sha256": identity["sha256"],
            "object_sha256": obj["sha256"],
            "object_bytes": obj["bytes"],
        }
        if any(record.get(field) != value for field, value in expected.items()):
            return None
        if record.get("builder") != "merlin.llvmlower.kernel_backend.compile_host":
            return None
        if not isinstance(record.get("lowering_recipe_sha256"), str):
            return None
        if not isinstance(record.get("lowering_commands"), list) or not record["lowering_commands"]:
            return None
        if not isinstance(record.get("native_codegen"), str):
            return None
        return {
            "manifest": record,
            "manifest_path": str(path.resolve(strict=True)),
            "manifest_sha256": _file(path)["sha256"],
        }
    except (OSError, ValueError, TypeError, KeyError):
        return None
