"""Export, verify and build small standalone accelerator timing providers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path

SCHEMA = "merlin.timing_provider_sources.v1"
MANIFEST = "source-manifest.json"
BUILDER = "build_provider.py"
MAX_BYTES = 1024 * 1024
MAX_MEMBERS = 64
RUNTIME_HEADERS = ("timing/provider.hpp", "timing/core/include/merlin_timing/engine.hpp")


def ordinary_path(path: Path) -> Path:
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("paths and parents must not be symlinks")
    return Path(os.path.abspath(path))


def member_bytes(path: Path) -> bytes:
    ordinary_path(path)
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES:
        raise ValueError("bundle members must be ordinary files within the byte budget")
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) != info.st_size:
        raise ValueError("bundle member changed while reading or exceeded its byte budget")
    return data


def _name(value: str) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    if (
        not isinstance(value, str)
        or not value
        or value in {".", "..", MANIFEST}
        or len(value) > 128
        or any(char not in alphabet for char in value)
    ):
        raise ValueError("bundle member names must be simple relative filenames")
    return value


def _unique(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate manifest key")
        result[key] = value
    return result


def _row(name: str, data: bytes) -> dict:
    return {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _entrypoint(value: str, names: set[str]) -> str:
    value = _name(value)
    if value not in names or Path(value).suffix not in {".cc", ".cpp", ".cxx"}:
        raise ValueError("entrypoint must name a bundled C++ translation unit")
    return value


def verify_bundle(directory: Path) -> dict:
    """Verify declared members and exact byte identity; this does not qualify timing."""
    root = ordinary_path(directory)
    if not root.is_dir():
        raise ValueError("bundle must be an ordinary directory")
    actual = set()
    for path in root.iterdir():
        actual.add(path.name)
        if len(actual) > MAX_MEMBERS + 1:
            raise ValueError("bundle has too many members")
    manifest_raw = member_bytes(root / MANIFEST)
    manifest = json.loads(manifest_raw, object_pairs_hook=_unique)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema", "entrypoint", "members", "total_bytes"}
        or manifest["schema"] != SCHEMA
        or type(manifest["total_bytes"]) is not int
        or not isinstance(manifest["members"], list)
        or not 2 <= len(manifest["members"]) <= MAX_MEMBERS
    ):
        raise ValueError("unsupported bundle manifest")
    names = []
    for row in manifest["members"]:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"} or type(row["bytes"]) is not int:
            raise ValueError("invalid bundle member roster")
        names.append(_name(row["path"]))
    if names != sorted(set(names)) or BUILDER not in names or actual != {*names, MANIFEST}:
        raise ValueError("bundle has a missing, repeated or additional member")
    _entrypoint(manifest["entrypoint"], set(names))
    observed = []
    total = len(manifest_raw)
    for name in names:
        data = member_bytes(root / name)
        total += len(data)
        if total > MAX_BYTES:
            raise ValueError("bundle exceeds one MiB")
        observed.append(_row(name, data))
    if manifest["members"] != observed or manifest["total_bytes"] != total:
        raise ValueError("bundle member identity differs")
    return {
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "members": observed,
        "entrypoint": manifest["entrypoint"],
        "total_bytes": total,
    }


def export_bundle(payloads: dict[str, bytes], destination: Path, *, entrypoint: str) -> dict:
    """Copy caller-selected target sources and the canonical independent builder."""
    if not isinstance(payloads, dict) or not 1 <= len(payloads) < MAX_MEMBERS:
        raise ValueError("provider payload must be a bounded file mapping")
    files = {}
    for name, data in payloads.items():
        _name(name)
        if name == BUILDER or not isinstance(data, bytes) or len(data) > MAX_BYTES:
            raise ValueError("provider payload conflicts with the builder or exceeds its byte budget")
        files[name] = data
    files[BUILDER] = member_bytes(Path(__file__))
    _entrypoint(entrypoint, set(files))
    manifest = {
        "schema": SCHEMA,
        "entrypoint": entrypoint,
        "members": [_row(name, files[name]) for name in sorted(files)],
        "total_bytes": 0,
    }
    for _ in range(8):
        encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        total = sum(map(len, files.values())) + len(encoded)
        if total == manifest["total_bytes"]:
            break
        manifest["total_bytes"] = total
    else:
        raise ValueError("bundle manifest size did not converge")
    if total > MAX_BYTES:
        raise ValueError("bundle exceeds one MiB")
    destination = ordinary_path(destination)
    if destination.exists():
        raise ValueError("bundle destination already exists")
    destination.mkdir()
    for name, data in files.items():
        (destination / name).write_bytes(data)
    (destination / MANIFEST).write_bytes(encoded)
    return verify_bundle(destination)


def build(root: Path, spike_root: Path, output: Path, cxx: str) -> dict:
    """Compile one selected provider without importing its target implementation."""
    root = ordinary_path(root)
    before = verify_bundle(root)
    spike_root = ordinary_path(spike_root)
    output = ordinary_path(output)
    if root == output or root in output.parents:
        raise ValueError("output library must be outside the closed source bundle")
    if not output.parent.is_dir() or (output.exists() and not output.is_file()):
        raise ValueError("output parent must exist and output must be an ordinary file")
    dependencies = {name: hashlib.sha256(member_bytes(spike_root / name)).hexdigest() for name in RUNTIME_HEADERS}
    # A new inode prevents truncating an already mapped library.
    with tempfile.TemporaryDirectory(prefix=".provider-build-", dir=output.parent) as temporary:
        staged = Path(temporary) / "provider.so"
        env = {**os.environ, "TMPDIR": temporary}
        subprocess.run(
            [
                cxx,
                "-std=c++17",
                "-O2",
                "-g0",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-fPIC",
                "-shared",
                "-I",
                str(spike_root / "timing/core/include"),
                "-I",
                str(spike_root),
                str(root / before["entrypoint"]),
                "-o",
                str(staged),
            ],
            check=True,
            timeout=60,
            env=env,
        )
        if verify_bundle(root) != before:
            raise ValueError("bundle changed during compilation")
        if dependencies != {
            name: hashlib.sha256(member_bytes(spike_root / name)).hexdigest() for name in RUNTIME_HEADERS
        }:
            raise ValueError("generic timing headers changed during compilation")
        ordinary_path(output)
        os.replace(staged, output)
    return {
        **before,
        "generic_header_sha256": dependencies,
        "library": str(output),
        "library_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "jobs": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=Path(__file__).absolute().parent)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--spike-root", type=Path)
    parser.add_argument("--output-library", type=Path)
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--jobs", type=int, choices=(1,), default=1)
    args = parser.parse_args()
    if args.verify_only:
        if args.spike_root or args.output_library:
            parser.error("verification accepts no build paths")
        result = verify_bundle(args.bundle_dir)
    else:
        if not args.spike_root or not args.output_library:
            parser.error("building requires --spike-root and --output-library")
        result = build(args.bundle_dir, args.spike_root, args.output_library, args.cxx)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
