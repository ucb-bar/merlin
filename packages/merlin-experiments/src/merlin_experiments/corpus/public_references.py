"""Explicit pinned public examples staged before fresh release admission.

References are a separate readonly information treatment, never a compiler seed.
Only manifest-listed regular source leaves are copied; surrounding trees are not.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

from ..spec import SpecError

SCHEMA = "operator.allowed_reference_sources.v1"


def _regular(path: Path, root: Path):
    if path != root and not path.is_relative_to(root):
        raise SpecError("reference member escapes its selected source root")
    for parent in (path, *path.parents):
        mode = parent.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise SpecError("reference sources cannot contain symlinks")
        if parent == root:
            break
    if not stat.S_ISREG(path.lstat().st_mode):
        raise SpecError("reference member must be a regular file")


def stage(manifest: Path, destination: Path) -> dict:
    manifest = manifest.expanduser().absolute()
    root = manifest.parent
    if root.resolve() != root:
        raise SpecError("reference source root must be canonical")
    _regular(manifest, root)
    original = manifest.read_bytes()
    document = json.loads(original)
    if document.get("schema") != SCHEMA or not isinstance(document.get("members"), list) or not document["members"]:
        raise SpecError("reference manifest requires a nonempty pinned source roster")
    if destination.exists():
        raise SpecError("reference staging requires a fresh destination")
    selected = []
    names = set()
    for item in document["members"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise SpecError("reference member must name a relative source path")
        relative = Path(item["path"])
        if relative.is_absolute() or not relative.parts or ".." in relative.parts or "\\" in item["path"]:
            raise SpecError("reference member path must be contained and relative")
        if relative.as_posix() != item["path"] or item["path"] in names or item["path"] == "reference-manifest.json":
            raise SpecError("reference member paths must be unique canonical leaves")
        names.add(item["path"])
        source = root / relative
        _regular(source, root)
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != item.get("sha256"):
            raise SpecError("reference source differs from its explicit SHA-256 pin")
        # Public examples are text; reject executable binaries and embedded NUL payloads.
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SpecError("reference member must be UTF-8 source text") from exc
        if "\x00" in text:
            raise SpecError("reference member contains a binary payload")
        selected.append((source, relative, payload))
    destination.mkdir(parents=True)
    for source, relative, payload in selected:
        leaf = destination / relative
        leaf.parent.mkdir(parents=True, exist_ok=True)
        with leaf.open("xb") as handle:
            handle.write(payload)
        leaf.chmod(0o444)
        _regular(source, root)
        if source.read_bytes() != payload:
            raise SpecError("reference source changed during staging")
    _regular(manifest, root)
    if manifest.read_bytes() != original:
        raise SpecError("reference manifest changed during staging")
    record = destination / "reference-manifest.json"
    record.write_bytes(original)
    record.chmod(0o444)
    return {"path": str(manifest), "sha256": hashlib.sha256(original).hexdigest(),
            "members": len(selected), "role": "readonly public examples, not a compiler seed"}


def declare(document: dict, *, destination: Path, variants, source_sha256: str, repo: Path):
    try:
        relative = destination.relative_to(repo).as_posix()
    except ValueError as exc:
        raise SpecError("reference payload must have a repository-contained grant path") from exc
    information = document.setdefault("information_sets", {})
    for variant in variants:
        item = information.setdefault(variant, {"condition": "explicit_public_references"})
        item.setdefault("allowed", []).append({"path": relative + "/", "as": "reference-kernels", "mode": "ro",
                                               "note": "explicit pinned examples; compiler starts empty"})
        item.setdefault("source_pins", []).append("public_reference_manifest:" + source_sha256)
