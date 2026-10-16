"""One content-addressed copy of a sealed capture's runtime, hard-linked into each private guest root.

Every sealed capture copied the selected venv, base interpreter and system libraries (about 9 GB)
into its own guest root, although those bytes are identical for every capture that selects them --
the same pattern that once cost a campaign 235 GiB of byte-identical run inputs. The runtime is now
materialized once per selection identity under the regenerable cache and hard-linked into each run.

Nothing about the seal is relaxed: the per-run guest root is still a private directory whose every
byte the issuer re-hashes against the plan before executing anything, and the replay re-hashes it
again. A store entry is published only after its own trees match the plan's selected digests, and
is built under a private name and renamed into place, so a half-built store is never linked. A
per-entry advisory lock makes build/publish and verification exclusive and linking shared, so
parallel captures never remove or replace an entry another one is reading.
File modes are kept exactly as copied, because the selected tree digests bind them.
"""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import shutil
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.common import strict_json


def _identity(plan: dict[str, Any]) -> str:
    from .sealed_static import _digest, _json

    trees = plan["selected_trees"]
    return _digest(
        _json(
            {
                "venv": trees["venv"],
                "base": trees["base"],
                "base_path": plan["base"],
                "system_libs": plan["system_libs"],
            }
        )
    )


#: An explicit store location, for a process whose repository root is a frozen snapshot (whose own
#: cache would place the runtime inside the run). Must be on the run's filesystem.
STORE_ENV = "MERLIN_SEALED_RUNTIME_STORE"


def _store_root() -> Path:
    selected = os.environ.get(STORE_ENV)
    if selected:
        return Path(selected)
    from merlin.common.artifacts import cache_dir

    return Path(cache_dir("sealed-m2m-runtime"))


@contextmanager
def _entry_lock(root: Path, key: str, *, exclusive: bool) -> Iterator[None]:
    """Serialize publication of one store entry against everyone reading it.

    Builders hold the lock exclusively while they check, build and publish an entry, and so does
    a full verification, whose inventory must not see the ctime changes concurrent hard-links
    make. Linkers hold it shared with each other. An entry can therefore never be replaced or
    removed while a capture inventories or links it. The lock is advisory and per entry, so
    different selections never wait on each other.
    """
    descriptor = os.open(root / f"{key}.lock", os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        os.close(descriptor)  # closing the only descriptor releases the lock


def _published(entry: Path, marker: Path) -> bool:
    return entry.is_dir() and not entry.is_symlink() and marker.is_file() and not marker.is_symlink()


@dataclass(frozen=True)
class VerifiedRuntimeCache:
    """One exact, local store entry; never an authority to rebuild or change the plan."""

    entry: Path
    device: int
    inode: int
    marker_sha256: str
    runtime_bytes: int
    inventory_sha256: str


def verified_cached_entry(
    plan: dict[str, Any], destination: Path, selected_system_libraries: list[dict[str, Any]] | None = None
) -> VerifiedRuntimeCache | None:
    """Discount only a complete, owned, byte-verified runtime that can be hard-linked here.

    This is intentionally a full inventory, not a marker-presence check. The issuer
    rechecks the token immediately before and after linking; a discounted issue
    must never silently fall back to a fresh runtime copy.
    """
    from .sealed_static import _canonical_path

    try:
        root = _canonical_path(_store_root(), exists=False)
        if not root.is_dir() or root.is_symlink():
            return None
        # Exclusive: a concurrent hard-link changes member ctimes, which the inventory treats as
        # an unstable read. Verification therefore excludes linkers as well as builders.
        with _entry_lock(root, _identity(plan), exclusive=True):
            return _verified_cached_entry(plan, destination, selected_system_libraries, root)
    except (KeyError, TypeError, ValueError, OSError):
        return None


def _verified_cached_entry(
    plan: dict[str, Any],
    destination: Path,
    selected_system_libraries: list[dict[str, Any]] | None,
    root: Path,
) -> VerifiedRuntimeCache | None:
    from .sealed_static import _canonical_path, _digest, _file_digest, _json, _tree

    try:
        entry = root / _identity(plan)
        marker = root / f"{entry.name}.complete.json"
        destination = _canonical_path(destination, exists=True)
        device = destination.stat().st_dev
        uid = os.getuid()
        for path in (root, entry, marker):
            info = path.lstat()
            if info.st_dev != device or info.st_uid != uid or path.is_symlink():
                return None
        if not root.is_dir() or not entry.is_dir() or not marker.is_file():
            return None
        if strict_json.loads(marker.read_bytes()) != {
            "identity": entry.name,
            "base": plan["base"],
            "system_libs": plan["system_libs"],
        }:
            return None
        marker_sha256 = _file_digest(marker)
        entry_stat = entry.stat()
        entry_identity = (entry_stat.st_dev, entry_stat.st_ino, entry_stat.st_mtime_ns, entry_stat.st_ctime_ns)
        rows = _tree(entry)
        for name in rows:
            member = entry if name == "." else entry / name
            info = member.lstat()
            if info.st_dev != device or info.st_uid != uid:
                return None
        prefixes = {"venv": "opt/capture-venv", "base": Path(plan["base"]).relative_to("/").as_posix()}
        allowed: set[str] = {"."}
        for prefix in prefixes.values():
            parts = Path(prefix).parts
            allowed.update("/".join(parts[:index]) for index in range(1, len(parts) + 1))
        libraries = plan["system_libs"]
        if selected_system_libraries is not None and (
            not isinstance(selected_system_libraries, list)
            or [row.get("path") for row in selected_system_libraries] != libraries
        ):
            return None
        for index, name in enumerate(libraries):
            relative = Path(name).relative_to("/").as_posix()
            if any(relative == prefix or relative.startswith(f"{prefix}/") for prefix in prefixes.values()):
                return None
            parts = Path(relative).parts
            allowed.update("/".join(parts[:part]) for part in range(1, len(parts) + 1))
            row = rows.get(relative)
            if row is None or row["kind"] != "file":
                return None
            if selected_system_libraries is None:
                source = Path(name)
                if row["bytes"] != source.stat().st_size or row["sha256"] != _file_digest(source):
                    return None
            elif row["bytes"] != selected_system_libraries[index].get("bytes") or row["sha256"] != (
                selected_system_libraries[index].get("sha256")
            ):
                return None
        runtime_bytes = sum(rows[Path(name).relative_to("/").as_posix()]["bytes"] for name in libraries)
        for key, prefix in prefixes.items():
            subset = {
                "." if name == prefix else name[len(prefix) + 1 :]: row
                for name, row in rows.items()
                if name == prefix or name.startswith(f"{prefix}/")
            }
            if "." not in subset:
                return None
            digest = _digest(_json(sorted((name, *sorted(row.items())) for name, row in subset.items())))
            selected = plan["selected_trees"][key]
            size = sum(row.get("bytes", 0) for row in subset.values())
            if {"members": len(subset), "bytes": size, "sha256": digest} != selected:
                return None
            runtime_bytes += size
            allowed.update(name for name in rows if name == prefix or name.startswith(f"{prefix}/"))
        if set(rows) != allowed:
            return None
        after = entry.stat()
        if marker_sha256 != _file_digest(marker) or entry_identity != (
            after.st_dev,
            after.st_ino,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            return None
        return VerifiedRuntimeCache(
            entry=entry,
            device=device,
            inode=entry_stat.st_ino,
            marker_sha256=marker_sha256,
            runtime_bytes=runtime_bytes,
            inventory_sha256=_digest(_json(sorted((name, *sorted(row.items())) for name, row in rows.items()))),
        )
    except (KeyError, TypeError, ValueError, OSError):
        return None


def snapshot_space_requirement(
    plan: dict[str, Any], destination: Path, selected_system_libraries: list[dict[str, Any]] | None = None
) -> tuple[int, VerifiedRuntimeCache | None]:
    """Price unshared selected bytes plus conservative directory/inode block overhead."""
    cache = verified_cached_entry(plan, destination, selected_system_libraries)
    estimate = plan["estimate_bytes"]
    if isinstance(estimate, bool) or not isinstance(estimate, int) or estimate < 0:
        raise ValueError("invalid selected snapshot estimate")
    if cache is not None:
        if cache.runtime_bytes > estimate:
            cache = None
        else:
            estimate -= cache.runtime_bytes
    members = sum(row["members"] for row in plan.get("selected_trees", {}).values())
    members += len(plan.get("system_libs", ())) + len(plan.get("selected_inputs", ())) + 1024
    if isinstance(members, bool) or not isinstance(members, int) or members < 1024:
        raise ValueError("invalid selected snapshot member count")
    block_bytes = os.statvfs(destination).f_frsize
    return estimate + members * block_bytes, cache


def _build(plan: dict[str, Any], destination: Path) -> None:
    from .sealed_m2m import SealedM2MError, _snapshot_tree
    from .sealed_static import _file_digest

    venv = Path(plan["venv"])
    venv_copy = destination / "opt/capture-venv"
    venv_copy.parent.mkdir(parents=True)
    shutil.copytree(
        venv, venv_copy, symlinks=False, ignore=lambda directory, names: {"lib64"} if Path(directory) == venv else set()
    )
    base = Path(plan["base"])
    base_copy = destination / base.relative_to("/")
    base_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base.resolve(), base_copy, symlinks=False)
    libraries = []
    for name in plan["system_libs"]:
        path = Path(name)
        target = destination / path.relative_to("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        libraries.append({"path": name, "sha256": _file_digest(target)})
    if (
        _snapshot_tree(venv_copy) != plan["selected_trees"]["venv"]
        or _snapshot_tree(base_copy) != plan["selected_trees"]["base"]
    ):
        raise SealedM2MError("runtime store copy differs from the selected runtime trees")
    if any(_file_digest(Path(row["path"])) != row["sha256"] for row in libraries):
        raise SealedM2MError("runtime store library copy differs from the selected host bytes")


def store_entry(plan: dict[str, Any]) -> Path:
    """The published store directory for this plan's runtime, building it once if absent."""
    root = _store_root()
    root.mkdir(parents=True, exist_ok=True)
    key = _identity(plan)
    entry = root / key
    marker = root / f"{key}.complete.json"
    if _published(entry, marker):
        return entry
    with _entry_lock(root, key, exclusive=True):
        # Another capture may have published it while this one waited.
        if _published(entry, marker):
            return entry
        staging = root / f"{key}.building-{secrets.token_hex(8)}"
        staging.mkdir()
        try:
            _build(plan, staging)
            if entry.exists() or entry.is_symlink():
                # An unpublished leftover of an interrupted build. No reader links an entry without
                # its marker, and the exclusive lock excludes every reader, so removing it is safe.
                if entry.is_dir() and not entry.is_symlink():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            staging.rename(entry)
            pending = root / f"{key}.complete.json.{secrets.token_hex(8)}"
            pending.write_text(json.dumps({"identity": key, "base": plan["base"], "system_libs": plan["system_libs"]}))
            os.replace(pending, marker)  # the marker appears whole, after the entry is in place
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
    return entry


def link_runtime(
    plan: dict[str, Any],
    runtime: Path,
    *,
    verified_cache: VerifiedRuntimeCache | None = None,
    selected_system_libraries: list[dict[str, Any]] | None = None,
) -> None:
    """Hard-link the plan's stored runtime into ``runtime`` with the stored directory modes."""
    from .sealed_m2m import SealedM2MError

    if verified_cache is None:
        entry = store_entry(plan)
    else:
        if verified_cached_entry(plan, runtime, selected_system_libraries) != verified_cache:
            raise SealedM2MError("cached runtime changed after discounted admission")
        entry = verified_cache.entry
    # Link under the shared entry lock: no builder can replace the entry mid-walk.
    with _entry_lock(entry.parent, entry.name, exclusive=False):
        if not _published(entry, entry.parent / f"{entry.name}.complete.json"):
            raise SealedM2MError("runtime store entry is not published")
        _link_entry(entry, runtime)
    if verified_cache is not None and (
        verified_cached_entry(plan, runtime, selected_system_libraries) != verified_cache
    ):
        raise SealedM2MError("cached runtime changed during hard-link snapshot")


def _link_entry(entry: Path, runtime: Path) -> None:
    directories: list[tuple[Path, int]] = []
    for current, names, files in os.walk(entry, followlinks=False):
        here = Path(current)
        relative = here.relative_to(entry)
        target_dir = runtime / relative
        target_dir.mkdir(parents=True, exist_ok=True)
        directories.append((target_dir, stat.S_IMODE(here.stat().st_mode)))
        for name in names:
            if (here / name).is_symlink():
                raise ValueError(f"runtime store contains a directory link: {here / name}")
        for name in files:
            source = here / name
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"runtime store contains a nonregular member: {source}")
            os.link(source, target_dir / name)
    # Modes last, deepest first, so a read-only directory never blocks linking its own members.
    for path, mode in sorted(directories, key=lambda row: len(row[0].parts), reverse=True):
        if path != runtime:
            path.chmod(mode)
