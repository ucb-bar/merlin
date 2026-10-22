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
_FINAL_ROSTER = contextvars.ContextVar("original_snapshot_final_roster", default=None)
_SHARED_ENTRY = contextvars.ContextVar("original_snapshot_shared_entry", default=None)
_ISSUED_VERIFICATIONS = weakref.WeakSet()
_ISSUED_ENTRIES = weakref.WeakSet()
_ENTRY_SCOPES = weakref.WeakSet()
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


class _EntryScope:
    pass


class _RosterGuard:
    """Kernel-observed names/types/sizes inside one bounded verification scope.

    Every file is watched as well as every directory: writes through an outside
    hard link can resize a file without notifying its parent directory. This
    never independently admits a payload hash. A SharedEntry may bind a prior
    full read to unchanged watches for request-owned command boundaries;
    unsupported watches, any event (including overflow), or mount changes fall
    back to the ordinary complete roster scan. Checks are serialized, so one
    guard may serve several threads: an event consumed by one ends it for all.
    """

    def __init__(self, root, members, *, scanner=None):
        import ctypes
        import sys

        self._lock = threading.RLock()
        self.fd = None
        self.resource_errno = None
        self.root = root
        self.scanned = tuple(members)
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
                self._record_resource_failure(ctypes.get_errno())
                return
            self.fd = descriptor
            # Linux UAPI: modify/attrib/close-write/moves/create/delete/self,
            # with IN_DONT_FOLLOW. Any delivered event invalidates the shortcut.
            mask = 0x00000FCE | 0x02000000
            directories = [root, *(path for _, path, kind, _ in members if kind == "directory")]
            files = [path for _, path, kind, _ in members if kind == "file"]
            for path in (*directories, *files):
                if watch(descriptor, os.fsencode(path), mask) < 0:
                    self._record_resource_failure(ctypes.get_errno())
                    self.close()
                    return
            # Close setup races after all parents and individual inodes are watched.
            # The default remains the strict snapshot reader. A trusted alias
            # inventory may supply its lexical no-follow reader; this confers no
            # snapshot/input authority and changes no payload verification.
            if _scanned_roster((scanner or _scanned_members)(root)) != _scanned_roster(members) or not self.unchanged():
                self.close()
        except (AttributeError, OSError, RuntimeError):
            self.close()

    def _record_resource_failure(self, value):
        import errno

        # Admission diagnostics only. Never rearm an existing guard: writes in
        # an unwatched interval require ordinary full payload verification.
        if value in (errno.ENOSPC, errno.EMFILE, errno.ENFILE, errno.ENOMEM):
            self.resource_errno = value

    def unchanged(self):
        with self._lock:
            if self.fd is None:
                return False
            try:
                info = self.root.stat(follow_symlinks=False)
                if (info.st_dev, info.st_ino) != self.identity or Path(
                    "/proc/self/mountinfo"
                ).read_bytes() != self.mounts:
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
        with self._lock:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None


class _GuardView:
    """A scope's non-owning use of a :class:`SharedEntry` guard; closing the scope keeps it."""

    def __init__(self, guard):
        self.unchanged = guard.unchanged
        self.scanned = guard.scanned

    def close(self):
        pass


class SharedEntry:
    """One full entry verification that several protected scopes of one grade start from.

    Built once, before a grade fans out (:func:`shared_entry_verification`). A
    :func:`verified_snapshot_scope` opened with ``entry=`` still checks the mapping,
    canonical paths and membership afresh, but takes the payload digest from this
    entry instead of re-reading the snapshot, and only while this entry's kernel
    roster guard has observed nothing at all: any event, overflow or mount change
    ends reuse for every later scope. Scope finals remain fresh by default;
    explicit original request commands may also use these same unchanged
    watches at publication. No digest survives an unwatched interval.
    """

    def __init__(self, *, root, repo, policy_bytes, marker_bytes, members, document, guard):
        self.process = os.getpid()
        self.root, self.repo, self.policy_bytes = root, repo, policy_bytes
        self.marker_bytes, self.members, self.document, self.guard = marker_bytes, members, document, guard
        self.signature = tuple(document[key] for key in ("content_sha256", "n_files", "n_bytes"))

    def reusable(self, root, repo, policy_bytes):
        return (
            self in _ISSUED_ENTRIES
            and self.process == os.getpid()
            and (self.root, self.repo, self.policy_bytes) == (root, repo, policy_bytes)
            and (root / MARKER).read_bytes() == self.marker_bytes
            and self.guard.unchanged()
        )

    def close(self):
        self.guard.close()


def shared_entry_verification(workspace: Path, bundle: dict, *, repo: Path, diagnostics: dict | None = None) -> SharedEntry | None:
    """Verify the snapshot once for a grade's scopes, or ``None`` when no live guard can protect reuse.

    The guard is armed BEFORE the payload is hashed, so the digest describes bytes the
    kernel has watched since. Requires the operator's read-lease selection, which is
    what enables the roster guard; without it every scope keeps its own full entry.
    """
    from .cpu_pool import shared_compute_pool

    if diagnostics is not None:
        diagnostics.clear()
    if not read_lease_digest.enabled():
        return None
    workspace, repo = Path(workspace), Path(repo)
    previous = _ACTIVE_VERIFICATION.set(None)
    try:
        policy = _policy_bytes(bundle)
        root = bundle_snapshot_root(workspace)
        with _snapshot_read_admission(), shared_compute_pool():
            marker_bytes = (root / MARKER).read_bytes()
            scanned = _scanned_members(root)
            guard = _RosterGuard(root, scanned)
            if diagnostics is not None and guard.resource_errno is not None:
                diagnostics["resource_errno"] = guard.resource_errno
        try:
            document = verify_bundle_snapshot(workspace, bundle, repo=repo)
            if _policy_bytes(bundle) != policy or json.loads(marker_bytes) != document or not guard.unchanged():
                guard.close()
                return None
        except BaseException:
            guard.close()
            raise
    finally:
        _ACTIVE_VERIFICATION.reset(previous)
    entry = SharedEntry(
        root=root,
        repo=repo,
        policy_bytes=policy,
        marker_bytes=marker_bytes,
        members=_scanned_roster(scanned),
        document=document,
        guard=guard,
    )
    _ISSUED_ENTRIES.add(entry)
    return entry


@contextlib.contextmanager
def shared_entry_scope(entry: SharedEntry | None):
    """Select an original guarded entry for nested scopes on this thread only.

    Callers still own exact compiler/profile/workspace selection. Every scope
    matches its own root, repository and closed bundle policy to this entry,
    performs fresh mapping checks and independently verifies its final payload.
    Copies of this context on another thread or after fork confer no reuse.
    """
    if entry is not None and (
        type(entry) is not SharedEntry or entry not in _ISSUED_ENTRIES or entry.process != os.getpid()
    ):
        raise ValueError("shared snapshot entry requires its original live issuer")
    selection = _EntryScope()
    selection.entry, selection.process, selection.thread = entry, os.getpid(), threading.current_thread()
    _ENTRY_SCOPES.add(selection)
    token = _SHARED_ENTRY.set(selection)
    try:
        yield
    finally:
        _ENTRY_SCOPES.discard(selection)
        _SHARED_ENTRY.reset(token)


def _selected_entry():
    selected = _SHARED_ENTRY.get()
    if (
        type(selected) is _EntryScope
        and selected in _ENTRY_SCOPES
        and selected.process == os.getpid()
        and selected.thread is threading.current_thread()
    ):
        return selected.entry
    return None


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
    roster = _FINAL_ROSTER.set(None)
    entry = _SHARED_ENTRY.set(None)
    try:
        yield
    finally:
        _SHARED_ENTRY.reset(entry)
        _FINAL_ROSTER.reset(roster)
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


def _issue(*, root, marker_bytes, repo, policy_bytes, members, roster_guard, document, leases):
    issued = _Verification()
    issued.process, issued.thread = os.getpid(), threading.current_thread()
    issued.root, issued.marker_bytes = root, marker_bytes
    issued.repo, issued.policy_bytes = repo, policy_bytes
    issued.members = members
    issued.roster_guard = roster_guard
    issued.signature = tuple(document[key] for key in ("content_sha256", "n_files", "n_bytes"))
    issued.leases = leases
    _ISSUED_VERIFICATIONS.add(issued)
    return issued


@contextlib.contextmanager
def verified_snapshot_scope(
    workspace: Path, bundle: dict, *, repo: Path,
    entry: SharedEntry | None = None, guarded_exit: bool = False,
):
    """Batch payload reads for one protected operation.

    Each boundary verifies mapping, canonical paths and membership. Nested
    inspections may reuse the exact mapping only while its kernel guard is
    unchanged; marker, policy and borrowed leases are still checked. Both
    boundaries independently hash the payload by default.
    With MERLIN_SNAPSHOT_LEASE_CACHE=1, large immutable files may instead reuse
    their SHA-256 under freshly checked Linux read leases; all other files are
    hashed normally. Close this scope before publishing any accepted output.

    ``entry`` (or the same-thread :func:`shared_entry_scope`) replaces only the
    entry payload read while its guard proves nothing changed. Final payload
    verification remains independent; a live guard may retain only its roster.
    An original request may opt into ``guarded_exit`` with its explicitly issued
    entry. Its unchanged kernel watches then certify the payload at command exit
    as well as entry. Any event, expired entry, path or mount change restores the
    full independent read. This does not cache candidate artifacts or policies.
    Each stage leases one shared CPU behind the process admission gate, so callers
    waiting to verify hold no slot.
    """
    from .cpu_pool import shared_compute_pool

    workspace, repo = Path(workspace), Path(repo)
    previous = _ACTIVE_VERIFICATION.set(None)
    issued = None
    # Only an explicit request-owned admission can select guarded publication;
    # an inherited context alone does not grant that lifetime.
    request_entry = entry if guarded_exit and type(entry) is SharedEntry else None
    reused_entry = False
    try:
        original_policy = _policy_bytes(bundle)
        root = bundle_snapshot_root(workspace)
        if entry is None:
            entry = _selected_entry()
        if type(entry) is SharedEntry and entry.reusable(root, repo, original_policy):
            reused_entry = True
            marker_bytes = entry.marker_bytes
            issued = _issue(
                root=root,
                marker_bytes=marker_bytes,
                repo=repo,
                policy_bytes=original_policy,
                members=entry.members,
                roster_guard=_GuardView(entry.guard),
                document=entry.document,
                leases=None,  # The entry's leases belong to its own thread; its guard and the final read stand.
            )
            _ACTIVE_VERIFICATION.set(issued)
            # Mapping, canonical paths and membership are checked afresh; the payload is the entry's.
            document = verify_bundle_snapshot(workspace, bundle, repo=repo)
            if document != entry.document:
                raise RuntimeError("original snapshot shared entry ownership changed")
        else:
            with read_lease_digest.verification() as leases:
                document = verify_bundle_snapshot(workspace, bundle, repo=repo)
            # The 23k-member roster scan is CPU work too. Acquire only after the
            # verifier's admission gate; end the slot before yielding to a child.
            with _snapshot_read_admission(), shared_compute_pool():
                marker_bytes = (root / MARKER).read_bytes()
                scanned = _scanned_members(root)
                members = _scanned_roster(scanned)
                roster_guard = _RosterGuard(root, scanned)
            issued = _issue(
                root=root,
                marker_bytes=marker_bytes,
                repo=repo,
                policy_bytes=original_policy,
                members=members,
                roster_guard=roster_guard,
                document=document,
                leases=leases,
            )
            _ACTIVE_VERIFICATION.set(issued)
        try:
            yield document
        finally:
            # Never let stale nested payload reuse satisfy a final boundary.
            _ACTIVE_VERIFICATION.set(None)
            if _policy_bytes(bundle) != original_policy or (root / MARKER).read_bytes() != marker_bytes:
                raise RuntimeError("original snapshot scope policy or marker changed")
            final = None
            if (reused_entry and request_entry is entry
                    and workspace.is_absolute() and workspace.resolve() == workspace
                    and repo.is_absolute() and repo.resolve() == repo and root.resolve() == root
                    and entry.reusable(root, repo, original_policy)):
                # The guard was armed before the admitted hash. Watches cover
                # files (including outside hardlink writes), directories and
                # mount topology. They are sticky-invalid across all workers.
                final = json.loads(marker_bytes)
            if final is None:
                with _snapshot_read_admission(), shared_compute_pool():
                    read_lease_digest.validate(issued.leases)
                roster = _FINAL_ROSTER.set(issued)
                try:
                    final = verify_bundle_snapshot(workspace, bundle, repo=repo)
                finally:
                    _FINAL_ROSTER.reset(roster)
            if _policy_bytes(bundle) != original_policy or (root / MARKER).read_bytes() != marker_bytes:
                raise RuntimeError("original snapshot scope policy or marker changed")
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
    issued = _ACTIVE_VERIFICATION.get()
    if _current_verification(issued) and issued.root == root:
        if (root / MARKER).read_bytes() != issued.marker_bytes:
            raise RuntimeError("original snapshot scope marker changed")
        guard = getattr(issued, "roster_guard", None)
        if (guard is None or not guard.unchanged()) and _scanned_roster(_scanned_members(root)) != issued.members:
            raise RuntimeError("original snapshot scope member roster changed")
        read_lease_digest.validate(getattr(issued, "leases", None))
        return issued.signature
    # A final read never uses a previous payload signature. It may avoid a
    # second directory walk only while the kernel still watches every original
    # member and parent. Events during hashing require a complete fresh retry.
    roster = _FINAL_ROSTER.get()
    guard = roster.roster_guard if _current_verification(roster) and roster.root == root else None
    if guard is not None and not guard.unchanged():
        guard = None
    members = guard.scanned if guard is not None else _scanned_members(root)
    result = _payload_content(members)
    if guard is not None and not guard.unchanged():
        result = _payload_content(_scanned_members(root))
    return result


def _payload_content(members):
    rows = []
    ordered = enumerate(members)
    if read_lease_digest.enabled():
        # Spend a constrained descriptor budget on the largest immutable files.
        ordered = sorted(ordered, key=lambda row: -(row[1][3] or 0))
    minimum_lease_bytes = read_lease_digest.minimum_bytes()
    with read_lease_digest.verification():
        # Reopen exact already-admitted paths in one kernel-checked batch.
        # Misses still take the ordinary admission/full-read path. The enclosing
        # verification rechecks every borrowed lease after all slower reads.
        if read_lease_digest.enabled():
            ordered = list(ordered)
            reusable = read_lease_digest.reuse([
                path for _, (parts, path, kind, size) in ordered
                if kind == "file" and parts != (MARKER,) and size >= minimum_lease_bytes
            ]) or {}
        else:
            reusable = {}
        for index, (parts, path, kind, size) in ordered:
            if kind != "file" or parts == (MARKER,):
                continue
            identity = reusable.get(path)
            if identity is None and size >= minimum_lease_bytes:
                identity = read_lease_digest.digest(path)
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


def _require_direct_snapshot_member(root: Path, frozen: Path, checked: set[Path]) -> None:
    """Check every distinct path component once within this verification call.

    A flat toolchain roster can name thousands of siblings. Resolving every
    sibling from the filesystem root repeats all ancestor checks. The caller
    has already checked the canonical snapshot root and lexical mapping; here
    every newly encountered component receives a fresh no-follow stat. Nothing
    is retained across verification calls, and the complete payload/roster
    verification still follows these checks.
    """
    pending = []
    member = frozen
    while member not in checked:
        if not member.is_relative_to(root) or member == root:
            raise RuntimeError("original snapshot input source is outside its canonical root")
        pending.append(member)
        member = member.parent
    try:
        for member in reversed(pending):
            mode = member.lstat().st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise RuntimeError("original snapshot input source is missing or indirect")
            checked.add(member)
    except OSError as exc:
        raise RuntimeError("original snapshot input source is missing or indirect") from exc


def verify_bundle_snapshot(workspace: Path, bundle: dict, *, repo: Path) -> dict:
    """Verify V4 original input ownership; never determine candidate visibility."""
    # Pure nested inspections of this exact issued scope need no second full
    # mapping walk or process-wide admission lock. The kernel guard covers the
    # already verified names/inodes; marker and policy still bind the mapping.
    # Command entry without a shared admission, independent consumers and ALL
    # command final checks clear _ACTIVE_VERIFICATION and use the fresh path.
    issued = _ACTIVE_VERIFICATION.get()
    workspace, repo = Path(workspace), Path(repo)
    root = bundle_snapshot_root(workspace)
    if _current_verification(issued) and issued.root == root:
        if issued.repo != repo or issued.policy_bytes != _policy_bytes(bundle):
            raise RuntimeError("original snapshot scope selection changed")
        if (root / MARKER).read_bytes() != issued.marker_bytes:
            raise RuntimeError("original snapshot scope marker changed")
        guard = getattr(issued, "roster_guard", None)
        if (workspace.is_absolute() and workspace.resolve() == workspace
            and repo.is_absolute() and repo.resolve() == repo and root.resolve() == root
            and guard is not None and guard.unchanged()):
            from .cpu_pool import shared_compute_pool
            with shared_compute_pool():
                read_lease_digest.validate(getattr(issued, "leases", None))
            return json.loads(issued.marker_bytes)
    # Unscoped or invalidated inspections retain complete fresh admission.
    with _snapshot_read_admission():
        return _verify_bundle_snapshot(workspace, bundle, repo=repo)


def _verify_bundle_snapshot(workspace: Path, bundle: dict, *, repo: Path) -> dict:
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
    checked_paths = {root}
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
            _require_direct_snapshot_member(root, frozen, checked_paths)
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
