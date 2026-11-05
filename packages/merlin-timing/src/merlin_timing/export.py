"""Export the canonical C++ source payload without a compiler checkout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA_V1 = "merlin_timing.cpp_export.v1"
SCHEMA = "merlin_timing.cpp_export.v2"
MANIFEST = "source-manifest.json"
ASSETS_V1 = ("include/merlin_timing/engine.hpp", "CMakeLists.txt", "tests/test_engine.cpp", "README.md", "LICENSE")
ASSETS = (*ASSETS_V1, "include/merlin_timing/reservations.hpp", "tests/test_reservations.cpp")
MAX_SOURCE_BYTES = 1024 * 1024


def _asset_root() -> Path:
    package = Path(__file__).resolve().parent
    installed = package / "_portable"
    source = package.parent.parent
    for root in (installed, source):
        if (root / ASSETS[0]).is_file():
            return root
    raise ValueError("canonical portable timing sources are not installed")


def _payload(root: Path, name: str) -> bytes:
    path = root / name
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("portable source members must not be indirect")
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_SOURCE_BYTES:
        raise ValueError("portable source member is absent or exceeds its byte bound")
    with path.open("rb") as stream:
        payload = stream.read(MAX_SOURCE_BYTES + 1)
    if not 0 < len(payload) <= MAX_SOURCE_BYTES:
        raise ValueError("portable source member changed or exceeds its byte bound")
    return payload


def _row(name: str, payload: bytes) -> dict:
    return {"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def export_cpp_core(destination: str | Path) -> dict:
    """Copy the closed small source roster; hashes establish byte identity only."""
    root = _asset_root()
    payloads = [(name, _payload(root, name)) for name in ASSETS]
    if sum(len(payload) for _, payload in payloads) > MAX_SOURCE_BYTES:
        raise ValueError("complete portable source package exceeds its byte bound")
    target = Path(destination).absolute()
    if target.exists() or target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
        raise ValueError("portable source export requires a fresh ordinary directory")
    target.mkdir(parents=True)
    for name, payload in payloads:
        output = target / name
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(payload)
    document = {
        "schema": SCHEMA,
        "scope": "source byte identity only; no model, hardware or timing qualification",
        "members": [_row(name, payload) for name, payload in payloads],
    }
    (target / MANIFEST).write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    verify_cpp_core(target)
    return document


def verify_cpp_core(directory: str | Path) -> dict:
    """Reopen the exact exported source roster and every member's bytes."""
    root = Path(directory).absolute()
    if root.is_symlink() or any(parent.is_symlink() for parent in root.parents):
        raise ValueError("portable source export must be an ordinary directory")
    document = json.loads(_payload(root, MANIFEST))
    if (
        not isinstance(document, dict)
        or set(document) != {"schema", "scope", "members"}
        or document["schema"] not in (SCHEMA, SCHEMA_V1)
        or document["scope"] != "source byte identity only; no model, hardware or timing qualification"
        or not isinstance(document["members"], list)
    ):
        raise ValueError("portable source manifest is malformed")
    assets = ASSETS if document["schema"] == SCHEMA else ASSETS_V1
    if len(document["members"]) != len(assets):
        raise ValueError("portable source manifest has a different versioned roster")
    observed = []
    total = 0
    for name in assets:
        payload = _payload(root, name)
        total += len(payload)
        observed.append(_row(name, payload))
    expected = {*assets, MANIFEST}
    directories = {parent.as_posix() for name in assets for parent in Path(name).parents if parent != Path(".")}
    files = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or (path.is_dir() and relative not in directories):
            raise ValueError("portable source members are changed, missing or additional")
        if path.is_file():
            if relative not in expected:
                raise ValueError("portable source members are changed, missing or additional")
            files.add(relative)
    if total > MAX_SOURCE_BYTES or document["members"] != observed or files != expected:
        raise ValueError("portable source members are changed, missing or additional")
    return document
