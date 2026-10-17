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
_PREPARATION_PID = os.getpid()
_PREPARATION_ADMISSION = threading.RLock()


@contextlib.contextmanager
def _snapshot_read_admission():
    """Wait before leasing a CPU for process-local synchronous verification.

    Metadata readers in one Python process cannot reserve the whole pool while
    waiting on interpreter/namespace locks. A caller already holding a genuine
    original CPU slot bypasses admission to avoid a lock/slot dependency cycle.
    No input checks or invocation-boundary verification are skipped or cached.
    """
    from .cpu_pool import current_thread_owns_cpu_lease, shared_compute_enabled

    if current_thread_owns_cpu_lease() or not shared_compute_enabled():
        yield
        return
    global _PREPARATION_PID, _PREPARATION_ADMISSION
    if _PREPARATION_PID != os.getpid():
        _PREPARATION_PID, _PREPARATION_ADMISSION = os.getpid(), threading.RLock()
    with _PREPARATION_ADMISSION:
        yield


class _Verification:
    pass


class _RosterGuard:
    """Kernel-observed names/types/sizes inside one bounded verification scope.

    Every file is watched as well as every directory: writes through an outside
    hard link can resize a file without notifying its parent directory. This
    never caches payload hashes. Entry/exit payload verification stays fresh;
    unsupported watches, any event (including overflow), or mount changes fall
    back to the ordinary complete roster scan.
    """

    def __init__(self, root, members):
        import ctypes
        import sys

        self.fd = None
        self.root = root
        if sys.platform != "linux" or not read_lease_digest.enabled():
            return
        try:
            library = ctypes.CDLL(None, use_errno=True)
            initialize, watch = library.inotify_init1, library.inotify_add_watch
            initialize.argtypes, initialize.restype = (ctypes.c_int,), ctypes.c_int
            watch.argtypes, watch.restype = (ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32), ctypes.c_int
            self.mounts = Path("/proc/self/mountinfo").read_bytes()
            info = root.stat(follow_symlinks=False)
            self.identity = info.st_dev, info.st_ino
            descriptor = initialize(os.O_CLOEXEC | os.O_NONBLOCK)
            if descriptor < 0:
                return
            self.fd = descriptor
            # Linux UAPI: modify/attrib/close-write/moves/create/delete/self,
            # with IN_DONT_FOLLOW. Any delivered event invalidates the shortcut.
            mask = 0x00000FCE | 0x02000000
            directories = [root, *(path for _, path, kind, _ in members if kind == "directory")]
            files = [path for _, path, kind, _ in members if kind == "file"]
            for path in (*directories, *files):
                if watch(descriptor, os.fsencode(path), mask) < 0:
                    self.close()
                    return
            # Close setup races after all parents and individual inodes are watched.
            if _scanned_roster(_scanned_members(root)) != _scanned_roster(members) or not self.unchanged():
                self.close()
        except (AttributeError, OSError):
            self.close()

    def unchanged(self):
        if self.fd is None:
            return False
        try:
            info = self.root.stat(follow_symlinks=False)
            if ((info.st_dev, info.st_ino) != self.identity
                    or Path("/proc/self/mountinfo").read_bytes() != self.mounts):
                self.close()
                return False
            try:
                os.read(self.fd, 65536)
            except BlockingIOError:
                return True
        except OSError:
            pass
        self.close()
        return False

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


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
    from .cpu_pool import shared_compute_pool

    workspace, repo = Path(workspace), Path(repo)
    previous = _ACTIVE_VERIFICATION.set(None)
    issued = None
    try:
        original_policy = _policy_bytes(bundle)
        with read_lease_digest.verification() as leases:
            document = verify_bundle_snapshot(workspace, bundle, repo=repo)
        # The 23k-member roster scan is CPU work too. Acquire only after the
        # verifier's admission gate; end the slot before yielding to a child.
        with shared_compute_pool():
            root = bundle_snapshot_root(workspace)
            marker_bytes = (root / MARKER).read_bytes()
            scanned = _scanned_members(root)
            members = _scanned_roster(scanned)
            roster_guard = _RosterGuard(root, scanned)
        issued = _Verification()
        issued.process, issued.thread = os.getpid(), threading.current_thread()
        issued.root, issued.marker_bytes = root, marker_bytes
        issued.repo, issued.policy_bytes = repo, original_policy
        issued.members = members
        issued.roster_guard = roster_guard
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
            with shared_compute_pool():
                read_lease_digest.validate(issued.leases)
            final = verify_bundle_snapshot(workspace, bundle, repo=repo)
            if final != document:
                raise RuntimeError("original snapshot scope ownership changed")
    finally:
        if issued is not None:
            issued.roster_guard.close()
            _ISSUED_VERIFICATIONS.discard(issued)
        _ACTIVE_VERIFICATION.reset(previous)


def snapshot_content(root: Path) -> tuple[str, int, int]:
    """Same payload digest used by the freezer; marker bytes are bound separately."""
    from .cpu_pool import shared_compute_pool

    with shared_compute_pool():
        return _snapshot_content_in_cpu_scope(root)


def _snapshot_content_in_cpu_scope(root: Path) -> tuple[str, int, int]:
    """Calculate one synchronous payload digest and validate every reused lease."""
    rows = []
    issued = _ACTIVE_VERIFICATION.get()
    if _current_verification(issued) and issued.root == root:
        if (root / MARKER).read_bytes() != issued.marker_bytes:
            raise RuntimeError("original snapshot scope marker changed")
        guard = getattr(issued, "roster_guard", None)
        if (guard is None or not guard.unchanged()) and _scanned_roster(_scanned_members(root)) != issued.members:
            raise RuntimeError("original snapshot scope member roster changed")
        read_lease_digest.validate(getattr(issued, "leases", None))
        return issued.signature
    members = _scanned_members(root)
    ordered = enumerate(members)
    if read_lease_digest.enabled():
        # Spend a constrained descriptor budget on the largest immutable files.
        ordered = sorted(ordered, key=lambda row: -(row[1][3] or 0))
    minimum_lease_bytes = read_lease_digest.minimum_bytes()
    with read_lease_digest.verification():
        for index, (parts, path, kind, size) in ordered:
            if kind != "file" or parts == (MARKER,):
                continue
            identity = read_lease_digest.digest(path) if size >= minimum_lease_bytes else None
            if identity is None:
                with path.open("rb") as stream:
                    identity = hashlib.file_digest(stream, "sha256").hexdigest()
            rows.append((index, "/".join(parts), identity, path.stat().st_size))
    rows.sort(key=lambda row: row[0])
    aggregate = hashlib.sha256()
    for _, relative, identity, size in rows:
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(identity.encode("ascii"))
        aggregate.update(b"\n")
    return aggregate.hexdigest(), len(rows), sum(row[3] for row in rows)


def verify_bundle_snapshot(workspace: Path, bundle: dict, *, repo: Path) -> dict:
    """Verify V4 original input ownership; never determine candidate visibility."""
    from .cpu_pool import shared_compute_pool

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
    # This synchronous metadata/hash stage owns one shared compute CPU, then
    # releases it before returning to any compiler invocation or child wait.
    with _snapshot_read_admission(), shared_compute_pool():
        content = snapshot_content(root)
    observed = dict(zip(("content_sha256", "n_files", "n_bytes"), content, strict=True))
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
