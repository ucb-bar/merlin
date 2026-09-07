"""Private, byte-verified source snapshots for resumable performance suites.

Source and corpus files are copies, not hard links. Runtime dependencies remain external and are
listed separately: their executable/hardware identities are verified by campaign preflight.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

SOURCE_ROOTS = (
    "merlin/python", "merlin/contract", "merlin/targets", "merlin/schemas", "build_tools",
    "merlin/experiments/gemmini_perf_bench", "merlin/experiments/capsule_bench",
)
OMIT = {"__pycache__", "_qa_ws"}
SCHEMA = "merlin.performance-source-snapshot.v1"


class SnapshotError(RuntimeError):
    pass


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def seal(root: Path, stem: str, document: object) -> Path:
    payload = (json.dumps(document, sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    path = root / f"{stem}.{digest}.json"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    return path


def load_seal(root: Path, stem: str) -> tuple[Path, dict]:
    files = list(root.glob(f"{stem}.*.json"))
    if len(files) != 1:
        raise SnapshotError(f"expected one {stem} seal in {root}")
    path = files[0]
    if (path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222
            or path.name != f"{stem}.{sha_file(path)}.json"):
        raise SnapshotError(f"invalid {stem} seal: {path}")
    return path, json.loads(path.read_text())


def create(source: Path, destination: Path, *, output_root: Path,
           source_roots: tuple[str, ...] = SOURCE_ROOTS,
           target_name: str | None = None) -> Path:
    """Publish a private snapshot. An interrupted destination cannot masquerade as a seal."""
    source, destination = source.resolve(), destination.absolute()
    if target_name is not None and (not target_name or Path(target_name).name != target_name
                                    or target_name in (".", "..")):
        raise SnapshotError("target_name must be one safe path component")
    if destination.exists() or destination.is_symlink():
        raise SnapshotError(f"snapshot destination already exists: {destination}")
    destination.mkdir(parents=True, mode=0o700)
    files = {}
    target_directories = {
        path.name for path in
        (source / "merlin/experiments/capsule_bench/targets").iterdir()
        if path.is_dir()
    } if target_name is not None else set()

    def ignored(directory: str, names: list[str]) -> set[str]:
        omitted = set(names) & OMIT
        # ``merlin._data`` contains runtime schemas and ABI sources, not merely a generated cache.
        # Only its multi-gigabyte capsule mirror is redundant with the separately snapped contract.
        # Globally ignoring ``_data`` removed quant_formats.registry.yaml and made a sealed process
        # unable to validate even one capsule although the live-tree preflight had passed.
        if Path(directory).resolve() == source / "merlin/python/merlin/_data/contract":
            omitted.add("capsules")
        current = Path(directory).resolve()
        if target_name is not None:
            if current in (source / "merlin/experiments/capsule_bench/targets",
                           source / "merlin/targets"):
                omitted.update(name for name in names if name != target_name)
            elif current == source / "merlin/contract/capsules":
                omitted.update(name for name in names
                               if name in target_directories and name != target_name)
        return omitted

    for relative in source_roots:
        src = source / relative
        if not src.is_dir():
            raise SnapshotError(f"missing snapshot source: {src}")
        shutil.copytree(src, destination / relative, ignore=ignored, symlinks=False)
    for relative in (".env", ".env.example", "pyproject.toml", "setup.cfg", "CMakeLists.txt"):
        src = source / relative
        if src.is_file():
            shutil.copy2(src, destination / relative)
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            relative = path.relative_to(destination).as_posix()
            digest = sha_file(path)
            if digest != sha_file(source / relative):
                raise SnapshotError(f"source changed during snapshot: {relative}")
            files[relative] = digest
            path.chmod(path.stat().st_mode & ~0o222)
    links = {}
    for relative, target in (("out", output_root), (".venv", source / ".venv"),
                             ("third_party", source / "third_party")):
        if target.exists():
            (destination / relative).symlink_to(target.resolve(), target_is_directory=True)
            links[relative] = str(target.resolve())
    receipt = seal(destination, "snapshot", {
        "schema": SCHEMA, "source_root": str(source), "source_roots": list(source_roots),
        "files": files, "external_links": links,
        "limits": "External tools and outputs are not immutable source; campaign pins verify engines.",
    })
    for path in destination.rglob("*"):
        if not path.is_symlink() and path.is_dir():
            path.chmod(path.stat().st_mode & ~0o222)
    destination.chmod(0o500)
    return receipt


def verify(root: Path) -> dict:
    path, receipt = load_seal(root, "snapshot")
    if receipt.get("schema") != SCHEMA:
        raise SnapshotError("unknown source snapshot schema")
    actual = set()
    for directory, dirs, names in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in list(dirs):
            candidate = base / name
            if candidate.is_symlink():
                dirs.remove(name)
                relative = candidate.relative_to(root).as_posix()
                if receipt["external_links"].get(relative) != str(candidate.resolve(strict=True)):
                    raise SnapshotError(f"snapshot dependency link changed: {relative}")
        for name in names:
            candidate = base / name
            if candidate == path:
                continue
            relative = candidate.relative_to(root).as_posix()
            actual.add(relative)
            if (candidate.is_symlink() or candidate.stat().st_mode & 0o222
                    or receipt["files"].get(relative) != sha_file(candidate)):
                raise SnapshotError(f"snapshot source changed: {relative}")
    if actual != set(receipt["files"]):
        raise SnapshotError("snapshot source set changed")
    for relative, target in receipt["external_links"].items():
        link = root / relative
        if not link.is_symlink() or str(link.resolve(strict=True)) != target:
            raise SnapshotError(f"snapshot dependency link missing: {relative}")
    return receipt
