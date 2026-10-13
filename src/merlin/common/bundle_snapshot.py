"""Read and bind original immutable input bytes without an experiment dependency.

This verifies input custody, not public mounts or answer-surface isolation. The
experiment sandbox separately checks those policies before exposing any grant.
Only the original host selects the workspace, archive and repository identities.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import stat
import threading
import weakref
from pathlib import Path

from . import read_lease_digest
from .path_relations import LexicalPathIndex

MARKER = "snapshot.json"


def bundle_snapshot_root(workspace: Path) -> Path:
    return Path(workspace).parent / "bundle_inputs"


def snapshot_path(root: Path, source: Path, repo: Path) -> Path:
    source, repo = Path(source).absolute(), Path(repo).absolute()
    try:
        return root / "repo" / source.relative_to(repo)
    except ValueError:
        return root / "external" / Path(*source.parts[1:])


_ACTIVE_VERIFICATION = contextvars.ContextVar("original_snapshot_verification", default=None)
_ISSUED_VERIFICATIONS = weakref.WeakSet()


class _Verification:
    pass


def _current_verification(issued):
    return (
        type(issued) is _Verification
        and issued in _ISSUED_VERIFICATIONS
        and issued.process == os.getpid()
        and issued.thread is threading.current_thread()
    )


@contextlib.contextmanager
def independent_snapshot_reads():
    """Retain an actual consumer's original fresh reads inside outer batching."""
    previous = _ACTIVE_VERIFICATION.set(None)
    try:
        yield
    finally:
        _ACTIVE_VERIFICATION.reset(previous)


def _policy_bytes(bundle):
    return json.dumps(bundle, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _scanned_members(root):
    """One no-follow scan establishes exact lexical names, types and sizes."""
    if root.resolve() != root:
        raise RuntimeError("bundle input snapshot contains an indirect path")
    pending = [(root, ())]
    members = []
    while pending:
        directory, prefix = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                parts = (*prefix, entry.name)
                metadata = entry.stat(follow_symlinks=False)
                mode = metadata.st_mode
                if stat.S_ISLNK(mode):
                    raise RuntimeError("bundle input snapshot contains an indirect path")
                if stat.S_ISREG(mode):
                    kind, size = "file", metadata.st_size
                elif stat.S_ISDIR(mode):
                    kind, size = "directory", None
                    pending.append((Path(entry.path), parts))
                else:
                    raise RuntimeError("bundle input snapshot contains a special file")
                members.append((parts, Path(entry.path), kind, size))
    members.sort(key=lambda member: member[0])
    return members


def _scanned_roster(members):
    return tuple(("/".join(parts), kind, size) for parts, _, kind, size in members)


def _member_roster(root, members):
    # Retain this internal helper's caller-supplied membership and ordering.
    result = []
    for path in members:
        metadata = path.stat(follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError("bundle input snapshot contains an indirect path")
        if stat.S_ISREG(metadata.st_mode):
            kind, size = "file", metadata.st_size
        elif stat.S_ISDIR(metadata.st_mode):
            kind, size = "directory", None
        else:
            raise RuntimeError("bundle input snapshot contains a special file")
        result.append((path.relative_to(root).as_posix(), kind, size))
    return tuple(result)


@contextlib.contextmanager
def verified_snapshot_scope(workspace: Path, bundle: dict, *, repo: Path):
    """Batch payload reads for one protected operation, never across calls.

    Every ordinary verification still checks mapping, canonical paths and
    membership. Both boundaries independently hash the payload by default.
    With MERLIN_SNAPSHOT_LEASE_CACHE=1, large immutable files may instead reuse
    their SHA-256 under freshly checked Linux read leases; all other files are
    hashed normally. Close this scope before publishing any accepted output.
    """
    workspace, repo = Path(workspace), Path(repo)
    previous = _ACTIVE_VERIFICATION.set(None)
    issued = None
    try:
        original_policy = _policy_bytes(bundle)
        with read_lease_digest.verification() as leases:
            document = verify_bundle_snapshot(workspace, bundle, repo=repo)
        root = bundle_snapshot_root(workspace)
        marker_bytes = (root / MARKER).read_bytes()
        issued = _Verification()
        issued.process, issued.thread = os.getpid(), threading.current_thread()
        issued.root, issued.marker_bytes = root, marker_bytes
        issued.repo, issued.policy_bytes = repo, original_policy
        issued.members = _scanned_roster(_scanned_members(root))
        issued.signature = tuple(document[key] for key in ("content_sha256", "n_files", "n_bytes"))
        issued.leases = leases
        _ISSUED_VERIFICATIONS.add(issued)
        _ACTIVE_VERIFICATION.set(issued)
        try:
            yield document
        finally:
            # Disable ALL outer/nested scope reuse before the mandatory final verification.
            _ACTIVE_VERIFICATION.set(None)
            if _policy_bytes(bundle) != original_policy or (root / MARKER).read_bytes() != marker_bytes:
                raise RuntimeError("original snapshot scope policy or marker changed")
            read_lease_digest.validate(issued.leases)
            final = verify_bundle_snapshot(workspace, bundle, repo=repo)
            if final != document:
                raise RuntimeError("original snapshot scope ownership changed")
    finally:
        if issued is not None:
            _ISSUED_VERIFICATIONS.discard(issued)
        _ACTIVE_VERIFICATION.reset(previous)


def snapshot_content(root: Path) -> tuple[str, int, int]:
    """Same payload digest used by the freezer; marker bytes are bound separately."""
    rows = []
    members = _scanned_members(root)
    issued = _ACTIVE_VERIFICATION.get()
    if _current_verification(issued) and issued.root == root:
        if (root / MARKER).read_bytes() != issued.marker_bytes:
            raise RuntimeError("original snapshot scope marker changed")
        if _scanned_roster(members) != issued.members:
            raise RuntimeError("original snapshot scope member roster changed")
        read_lease_digest.validate(getattr(issued, "leases", None))
        return issued.signature
    with read_lease_digest.verification():
        for parts, path, kind, size in members:
            if kind != "file" or parts == (MARKER,):
                continue
            identity = read_lease_digest.digest(path) if size >= read_lease_digest.MIN_BYTES else None
            if identity is None:
                with path.open("rb") as stream:
                    identity = hashlib.file_digest(stream, "sha256").hexdigest()
            rows.append(("/".join(parts), identity, path.stat().st_size))
    aggregate = hashlib.sha256()
    for relative, identity, size in rows:
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(identity.encode("ascii"))
        aggregate.update(b"\n")
    return aggregate.hexdigest(), len(rows), sum(row[2] for row in rows)


def verify_bundle_snapshot(workspace: Path, bundle: dict, *, repo: Path) -> dict:
    """Verify V4 original input ownership; never determine candidate visibility."""
    workspace, repo = Path(workspace), Path(repo)
    root = bundle_snapshot_root(workspace)
    marker = root / MARKER
    issued = _ACTIVE_VERIFICATION.get()
    if (
        _current_verification(issued)
        and issued.root == root
        and (issued.repo != repo or issued.policy_bytes != _policy_bytes(bundle))
    ):
        raise RuntimeError("original snapshot scope selection changed")
    if (
        not workspace.is_absolute()
        or workspace.resolve() != workspace
        or not repo.is_absolute()
        or repo.resolve() != repo
        or root.resolve() != root
        or marker.is_symlink()
        or not marker.is_file()
    ):
        raise RuntimeError("original snapshot inputs require canonical independent paths")
    document = json.loads(marker.read_bytes())
    if document.get("version") != 4 or document.get("repo") != str(repo):
        raise RuntimeError("original snapshot ownership differs from its selected repository")
    for policy_key, record_key in (("allowed", "grants"), ("host_inputs", "host_records")):
        declared = [row["path"] for row in bundle.get(policy_key, [])]
        label = "allowed" if policy_key == "allowed" else "host_inputs"
        if document.get(label, []) != declared:
            raise RuntimeError("original snapshot input membership differs from its archive")
        records = document.get(record_key)
        if type(records) is not list or len(records) != len(declared):
            raise RuntimeError("original snapshot input records are incomplete")
        for name, record in zip(declared, records, strict=True):
            if type(record) is not dict or record.get("path") != name:
                raise RuntimeError("original snapshot input record ordering differs")
            destination, relative = Path(record["destination"]), Path(record["snapshot"])
            if (
                not destination.is_absolute()
                or ".." in destination.parts
                or relative.is_absolute()
                or ".." in relative.parts
                or root / relative != snapshot_path(root, destination, repo)
            ):
                raise RuntimeError("original snapshot input has an unsafe source mapping")
            if policy_key == "host_inputs" and destination != (repo / Path(name)).absolute():
                raise RuntimeError("original host input changed its canonical declared destination")
            frozen = root / relative
            if not frozen.exists() or frozen.resolve() != frozen:
                raise RuntimeError("original snapshot input source is missing or indirect")
    observed = dict(zip(("content_sha256", "n_files", "n_bytes"), snapshot_content(root), strict=True))
    if observed != {key: document.get(key) for key in observed}:
        raise RuntimeError("original snapshot input content differs from its marker")
    return document


def snapshot_input_paths(workspace: Path, bundle: dict, sources: list[Path], *, repo: Path) -> list[Path]:
    document = verify_bundle_snapshot(workspace, bundle, repo=repo)
    records = [*document["grants"], *document["host_records"]]
    declarations = LexicalPathIndex.from_paths(Path(row["destination"]) for row in records)
    root = bundle_snapshot_root(workspace)
    result = []
    for original in sources:
        original = Path(original).absolute()
        if ".." in original.parts or not declarations.contains(original):
            raise RuntimeError("requested original input was not declared in the frozen archive")
        frozen = snapshot_path(root, original, Path(repo))
        if not frozen.exists() or frozen.resolve() != frozen:
            raise RuntimeError("requested original snapshot input is missing or indirect")
        result.append(frozen)
    return result


def snapshot_marker_digest(workspace: Path) -> str:
    marker = bundle_snapshot_root(workspace) / MARKER
    if marker.is_symlink() or not marker.is_file() or marker.resolve() != marker:
        raise RuntimeError("original snapshot marker is missing or indirect")
    return hashlib.sha256(marker.read_bytes()).hexdigest()
