"""Byte-level evidence primitives shared by Phase 2 authoring and telemetry."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from merlin.common import digest

ROUND_DEADLINE_EXIT = 124


class StageGateError(RuntimeError):
    """The candidate cannot be launched or admitted without weakening an experiment gate."""


class PerformanceExperimentError(RuntimeError):
    """Performance campaign admission or execution evidence fails its declared contract."""


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise StageGateError(f"content-addressed evidence is absent or linked: {path}")
    return digest.sha256_file(path)


def canonical_json(document: object) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(document))


def require_executable(name_or_path: str, *, label: str) -> Path:
    found = shutil.which(name_or_path)
    path = Path(found or name_or_path)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise StageGateError(f"required {label} executable is absent: {name_or_path}")
    return path.resolve()


def document_sha256(document: object) -> str:
    return digest.sha256_bytes(canonical_json(document).rstrip(b"\n"))


def exact_tree_record(root: Path) -> dict[str, Any]:
    """Hash all path and file bytes; reject links, special files, and emptiness."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise StageGateError(f"exact input is absent or linked: {root}")
    digest = hashlib.sha256()
    n_files = n_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise StageGateError(f"exact input contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise StageGateError(f"exact input contains a special file: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(relative + b"\0" + payload + b"\0")
        n_files += 1
        n_bytes += len(payload)
    if n_files <= 0:
        raise StageGateError(f"exact input contains zero files: {root}")
    return {"sha256": digest.hexdigest(), "n_files": n_files, "n_bytes": n_bytes}


def mapping_file(path: Path, *, yaml_file: bool = False) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StageGateError(f"required stage input is absent or linked: {path}")
    try:
        document = (
            yaml.safe_load(path.read_text(encoding="utf-8"))
            if yaml_file
            else json.loads(path.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise StageGateError(f"stage input is unreadable at {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise StageGateError(f"stage input must be a mapping: {path}")
    return document


def safe_component(value: str, *, label: str) -> str:
    if not value or Path(value).name != value or value in (".", ".."):
        raise StageGateError(f"{label} must be a simple non-empty path component")
    return value


def require_real_directory(path: Path, *, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink() or not raw.is_dir():
        raise StageGateError(f"{label} is absent or linked: {raw}")
    return raw.resolve()
