"""Answer surfaces that host inputs create: private host files and every alias that reaches them.

A sandbox binds selected host inputs read-only; a private file among them, or a hardlink to one under a
bound tree, would expose an answer the arm must not read. This module derives those surfaces from the
mount plan and the bundle's own records (:func:`host_input_surfaces`, :func:`private_file_surfaces`);
:func:`merlin.targetgen.sandbox.bwrap.apply_final_answer_masks` masks them last.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from merlin.common.paths import repo_root
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface
from merlin.targetgen.sandbox.bwrap import (
    _SNAPSHOT_COMPLETE,
    _bind_alias,
    _host_input_paths,
    _mounts,
    _private_validation_paths,
    _snapshot_grants,
    _snapshot_host_inputs,
    _validate_host_locations,
    bundle_snapshot_root,
    path_kind,
    require_snapshot_ownership,
)


def _host_mount_devices() -> list[tuple[int, Path]]:
    """Linux host mount topology, including escaped mountpoint names.

    The namespace assembling bwrap is trusted and must remain stable through
    launch. Never infer disjointness solely from a directory's own device: a
    nested mount may lead back onto the private input's filesystem.
    """
    try:
        rows = Path("/proc/self/mountinfo").read_text().splitlines()
        if not rows:
            raise ValueError("empty mount topology")
        mounts = []
        for row in rows:
            fields = row.split()
            separator = fields.index("-")
            if separator < 6 or len(fields) != separator + 4:
                raise ValueError("malformed mount topology")
            major, minor = (int(part) for part in fields[2].split(":"))
            if major < 0 or minor < 0:
                raise ValueError("invalid mount device")
            encoded = fields[4]
            # mountinfo escapes these four bytes only. Decode backslash LAST,
            # so a literal backslash followed by digits cannot become a space.
            escaped = {"\\040": " ", "\\011": "\t", "\\012": "\n", "\\134": "\\"}
            rest = encoded
            for code in escaped:
                rest = rest.replace(code, "")
            if "\\" in rest:
                raise ValueError("unsupported mountpoint escape")
            for code, value in escaped.items():
                encoded = encoded.replace(code, value)
            path = Path(encoded)
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError("invalid mountpoint")
            mounts.append((os.makedev(major, minor), path))
        return mounts
    except (OSError, ValueError, OverflowError) as exc:
        raise RuntimeError("host mount topology cannot be verified for private inputs") from exc


def _host_hardlink_surfaces(
    argv: list[str], records: list[tuple[str, Path, Path]], approved: set[tuple[Path, Path]]
) -> list[AnswerSurface]:
    """Mask extra runtime names for private inodes, not identical public bytes.

    Canonical frozen public mounts were validated before freezing and may share
    CAS objects with private inputs. Only those exact source/destination pairs
    are exempt; arbitrary runtime aliases do not inherit a public grant.
    """
    private_inodes: set[tuple[int, int]] = set()
    try:
        for _, original, frozen in records:
            for root in {original, frozen}:
                if not root.exists():  # A resumed run may have lost its live source.
                    continue
                for member in (root, *root.rglob("*")) if root.is_dir() else (root,):
                    info = member.stat()
                    if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
                        private_inodes.add((info.st_dev, info.st_ino))
    except OSError as exc:
        raise RuntimeError("private input hardlink identities cannot be verified") from exc
    if not private_inodes:
        return []
    inode_numbers = {inode for _, inode in private_inodes}
    devices = {device for device, _ in private_inodes}
    relevant_mounts = {path for device, path in _host_mount_devices() if device in devices}
    cache: dict[Path, list[Path]] = {}
    surfaces: dict[Path, AnswerSurface] = {}

    def matches(info: os.stat_result) -> bool:
        return stat.S_ISREG(info.st_mode) and (info.st_dev, info.st_ino) in private_inodes

    def scan(root: Path) -> list[Path]:
        hits: list[Path] = []
        info = root.stat()
        if matches(info):
            hits.append(Path("."))
        if not stat.S_ISDIR(info.st_mode):
            return hits
        pending = [root]
        scanned: set[Path] = set()
        while pending:
            directory = pending.pop()
            if directory in scanned:
                continue
            scanned.add(directory)
            if directory.stat().st_dev not in devices:
                pending.extend(path for path in relevant_mounts if directory in path.parents)
                continue
            with os.scandir(directory) as entries:
                for entry in entries:
                    # A kernel directory bind retains symlinks; unlike the input
                    # freezer it does not recursively copy their referents.
                    if entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    elif entry.inode() in inode_numbers and matches(entry.stat(follow_symlinks=False)):
                        hits.append(Path(entry.path).relative_to(root))
        return hits

    for state, raw_source, raw_destination in _mounts(argv):
        if state != "expose":
            continue
        source, destination = Path(raw_source).absolute(), Path(raw_destination).absolute()
        if (source, destination) in approved:
            continue
        try:
            source = source.resolve(strict=True)
        except FileNotFoundError:
            continue  # --bind-try absent roots expose nothing.
        try:
            if source not in cache:
                cache[source] = scan(source)
        except OSError as exc:
            raise RuntimeError("runtime bind hardlink privacy cannot be verified") from exc
        for relative in cache[source]:
            alias = destination / relative
            surfaces[alias] = AnswerSurface("host-only hardlink alias", alias, "file", "hidden")
    return list(surfaces.values())


def host_input_surfaces(
    argv: list[str],
    ws: Path,
    bundle: dict,
    *,
    repo: Path | None = None,
    grant_repo: Path | None = None,
    _policy_test_live_inputs: bool = False,
) -> list[AnswerSurface]:
    """Private surfaces at every destination a composed bind could expose.

    A toolchain can bind a parent or an individual private input under another
    name. Denying only the original path would leave that translated view open.
    Both the live source and the frozen copy retain their private identity.
    ``grant_repo`` relocates only verified repo-relative public destinations;
    external grants keep their original destination and privacy is unchanged.
    """
    if not bundle or (
        not bundle.get("host_inputs") and not bundle.get("private_validation_paths") and _policy_test_live_inputs
    ):
        return []
    repo = (repo or repo_root()).absolute()
    approved: set[tuple[Path, Path]] = set()
    if _policy_test_live_inputs:
        records = [(path, (repo / path).absolute(), (repo / path).absolute()) for path in _host_input_paths(bundle)]
        _validate_host_locations(ws, [(path, source) for path, source, _ in records])
    else:
        manifest, grants = _snapshot_grants(ws, bundle, repo)
        require_snapshot_ownership(manifest)
        records = _snapshot_host_inputs(ws, bundle, repo, manifest)
        # The private inventory is host metadata too. A later broad runtime bind
        # must not expose the unprojected marker even when payloads are masked.
        marker = bundle_snapshot_root(ws) / _SNAPSHOT_COMPLETE
        records.append((_SNAPSHOT_COMPLETE, marker, marker))
        approved = {(source, destination) for _, destination, source in grants}
        if grant_repo is not None:
            relocated = grant_repo.absolute()
            approved.update(
                (source, relocated / destination.relative_to(repo))
                for _, destination, source in grants
                if destination.is_relative_to(repo)
            )
    for path, kind in _private_validation_paths(bundle):
        if path_kind(path) == "missing":
            # A future file cannot be masked reliably while absent. Refuse a
            # composed live bind of its ancestor, including runtime aliases
            # added after the frozen bundle's own public grants.
            for state, source, _destination in _mounts(argv):
                if state != "expose":
                    continue
                host_source = Path(source).absolute()
                if path.is_relative_to(host_source) or path.is_relative_to(host_source.resolve()):
                    raise RuntimeError("runtime bind could expose a future operator-private validation output")
            continue
        if path_kind(path) != kind or path.is_symlink():
            raise RuntimeError("private validation path changed kind or became indirect")
        records.append((str(path), path, path))
    return _private_path_surfaces(argv, records, approved)


def _private_path_surfaces(
    argv: list[str], records: list[tuple[str, Path, Path]], approved: set[tuple[Path, Path]]
) -> list[AnswerSurface]:
    """One alias translation for frozen payloads and exact host provenance files."""
    surfaces: dict[Path, AnswerSurface] = {}
    for _, original, frozen in records:
        kind = path_kind(frozen)
        for private in {original, frozen}:
            surfaces[private] = AnswerSurface("host-only input", private, kind, "hidden")
            for state, raw_source, raw_destination in _mounts(argv):
                if state != "expose":
                    continue
                source = Path(raw_source).absolute()
                for resolved in {source, source.resolve()}:
                    destination = Path(raw_destination)
                    translated = _bind_alias(private, kind, resolved, destination)
                    if translated is None:
                        continue
                    alias, alias_kind = translated
                    if alias_kind == "missing":
                        raise RuntimeError("private input alias cannot be classified for masking")
                    surfaces[alias] = AnswerSurface("host-only input alias", alias, alias_kind, "hidden")
    for surface in _host_hardlink_surfaces(argv, records, approved):
        surfaces[surface.path] = surface
    return list(surfaces.values())


def private_file_surfaces(argv: list[str], paths: list[Path]) -> list[AnswerSurface]:
    """Protect verified host provenance files through composed runtime aliases.

    Callers own provenance verification and select the exact environment/archive
    files; this grants nothing and exposes no caller-defined public exceptions.
    An absent, symlinked or unreadable file cannot silently lose its protection.
    """
    records = []
    for path in paths:
        source = path.absolute()
        try:
            if source.is_symlink() or not stat.S_ISREG(source.stat().st_mode):
                raise ValueError("host provenance must be a regular file")
            resolved = source.resolve(strict=True)
        except (OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError("private provenance file cannot be classified for masking") from exc
        records.append((str(source), source, resolved))
    return _private_path_surfaces(argv, records, set())
