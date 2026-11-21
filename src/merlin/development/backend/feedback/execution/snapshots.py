"""Private enqueue-time source snapshots shared by promotion producers and consumers.

The existing submission-identity observer is explicit: copy, verify and recovery use
one digest implementation, evaluated at the same points as the original callers.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import yaml

from merlin.common.paths import module_source_path
from merlin.common.storage import bundle_snapshots as B
from merlin.development.targetgen.authoring.tool_profile import verify_tool_profile
from merlin.verification.lean.contract import document_digest

# ---- snapshots ---------------------------------------------------------------------------------


def promotion_snapshot_path(ws, token: str, *, create_store: bool = False) -> Path | None:
    """Resolve a promotion snapshot token inside the driver-private, per-workspace store.

    Requests cross an agent-writable JSON channel, so they must never carry a host path that the broker
    trusts. A token is accepted only when it is a lowercase hex nonce, and the broker derives the
    containing directory itself. The workspace hash prevents simultaneous campaigns from sharing
    snapshots even when their request ids happen to match.
    """
    import hashlib
    import tempfile

    if not isinstance(token, str) or len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
        return None
    base = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
    ws_key = hashlib.sha256(str(Path(ws).resolve()).encode("utf-8")).hexdigest()[:24]
    store = base / f"merlin_promotion_snapshots_{os.getuid()}" / ws_key
    if create_store:
        store.mkdir(parents=True, mode=0o700, exist_ok=True)
        store.chmod(0o700)
    return store / token


def create_promotion_snapshot(
    ws, source_ws, expected_digest: str, *, submission_digest: Callable[[Path], str]
) -> tuple[str | None, str | None]:
    """Freeze and verify the exact source promoted by a loop-tier verdict."""
    import hashlib
    import shutil

    # One immutable source copy per workspace+digest, not one copy per capsule.  A large submission may
    # promote dozens of capsules at once; duplicating a 200 MiB compiler for each queued L3 job would
    # turn correctness isolation into tens of GiB of avoidable storage.
    token = hashlib.sha256(expected_digest.encode("ascii")).hexdigest()[:32]
    final = promotion_snapshot_path(ws, token, create_store=True)
    assert final is not None
    if final.is_dir():
        actual = submission_digest(final)
        if actual == expected_digest:
            return token, None
        shutil.rmtree(final, ignore_errors=True)
    stage = final.with_name(f".{token}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        submission = stage / "submission"
        shutil.copytree(Path(source_ws) / "submission", submission, ignore=shutil.ignore_patterns("__pycache__"))
        actual = submission_digest(stage)
        if actual != expected_digest:
            return None, (f"promotion source moved while snapshotting: requested {expected_digest}, copied {actual}")
        try:
            stage.rename(final)
        except FileExistsError:
            # Another promotion producer froze the same digest concurrently.  Its atomically published
            # copy is equivalent if and only if it verifies against the same digest.
            actual = submission_digest(final)
            if actual != expected_digest:
                return None, (
                    f"concurrent promotion snapshot identity mismatch: requested {expected_digest}, snapshot {actual}"
                )
        return token, None
    except Exception as exc:  # noqa: BLE001 -- promotion is optional; caller logs the concrete cause
        return None, f"promotion snapshot failed: {type(exc).__name__}: {exc}"
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def recover_promotion_snapshot(
    ws: Path,
    expected_digest: str,
    identity: str | None = None,
    snapshot_token: str | None = None,
    *,
    submission_digest: Callable[[Path], str],
):
    """Copy the source a promoted job will grade and verify its enqueue-time identity.

    Promotion is asynchronous: compiling directly from ``ws/submission`` lets an agent edit the tree
    after enqueue but before the child copies it. The resulting executable then cannot resolve the
    pending record (correctly) and the paid-for certificate is discarded. Return
    ``(submission, root, None)`` only for an isolated copy whose digest is exactly the one carried by the
    request; otherwise return ``(None, None, reason)`` and run nothing.
    """
    if not isinstance(expected_digest, str) or not expected_digest:
        return None, None, "promotion request has no enqueue-time submission digest"
    if isinstance(identity, str) and identity.startswith("submission:") and identity != f"submission:{expected_digest}":
        return (
            None,
            None,
            (f"promotion request source identity {identity} does not match its submission digest {expected_digest}"),
        )
    if snapshot_token is not None:
        frozen = promotion_snapshot_path(ws, snapshot_token)
        if frozen is None:
            return None, None, "promotion request has an invalid enqueue-time snapshot token"
        frozen_submission = frozen / "submission"
        if not frozen_submission.is_dir():
            return None, None, "promotion enqueue-time snapshot is missing"
        actual = submission_digest(frozen)
        if actual != expected_digest:
            return None, None, (f"promotion snapshot identity mismatch: requested {expected_digest}, snapshot {actual}")
        # Each child gets a disposable working copy; the content-addressed frozen copy is shared by all
        # capsules promoted from this source version and never exposed as a mutable build directory.
        root = Path(tempfile.mkdtemp(prefix="merlin_promotion_"))
        submission = root / "submission"
        try:
            shutil.copytree(frozen_submission, submission, ignore=shutil.ignore_patterns("__pycache__"))
            copied = submission_digest(root)
            if copied != expected_digest:
                shutil.rmtree(root, ignore_errors=True)
                return (
                    None,
                    None,
                    (f"promotion snapshot moved while preparing launch: requested {expected_digest}, copied {copied}"),
                )
            return submission, root, None
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise

    # Compatibility for requests queued before enqueue-time snapshots existed. Fail closed if the
    # mutable workspace no longer has the requested bytes.
    root = Path(tempfile.mkdtemp(prefix="merlin_promotion_"))
    try:
        submission = root / "submission"
        shutil.copytree(
            Path(ws) / "submission",
            submission,
            # Match ``submission_digests`` exactly: only Python bytecode caches are outside
            # the source identity. A build/ or .git/ path below submission is unusual, but
            # silently dropping it here would make the verified copy a different tree.
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        actual = submission_digest(root)
        if actual != expected_digest:
            shutil.rmtree(root)
            return (
                None,
                None,
                (f"promotion source moved before launch: requested {expected_digest}, available {actual}"),
            )
        return submission, root, None
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


# ---- snapshot_admission ------------------------------------------------------------------------
# Private process-shared admission for two immutable compiler identities.
#
# This limits work, not correctness or access. Every admitted request still runs
# its ordinary source, sandbox, artifact and numerical checks. Equal identities
# may share a slot; this permits nested host consumers without an extra slot.


def identity(service, candidate):
    from merlin.development.backend.optimization.search.retained_revisions import source_inventory
    from merlin.development.targetgen.authoring.library_admission import active_compiler_library

    library = active_compiler_library()
    return hashlib.sha256(
        json.dumps(
            {
                "source": source_inventory(Path(candidate), max_bytes=32 << 20, max_files=8192, exclude_generated=True)[
                    "sha256"
                ],
                "snapshot": service.snapshot_digest,
                "profile": service.profile["sha256"],
                "library": library.record()["contract_sha256"] if library is not None else None,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _open(root, name):
    fd = os.open(root / name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    value = os.fstat(fd)
    if (
        not stat.S_ISREG(value.st_mode)
        or value.st_uid != os.geteuid()
        or value.st_nlink != 1
        or stat.S_IMODE(value.st_mode) != 0o600
    ):
        os.close(fd)
        raise ValueError("snapshot admission needs a private ordinary slot file")
    return fd


@contextmanager
def admitted(root, key, *, timeout_s=1200, stop_reason=None):
    """Hold one of two identity slots without retaining a compute lease.

    The registry lock serializes slot allocation only. Shared slot locks remain
    held for every live borrower and are released by the kernel on process exit.
    No environment variable or candidate-authored declaration bypasses admission.
    """
    from merlin.common.process.cpu_pool import current_thread_owns_cpu_lease

    if current_thread_owns_cpu_lease():
        raise RuntimeError("snapshot admission cannot wait while holding a CPU lease")
    if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("snapshot admission needs its exact native identity")
    if type(timeout_s) not in (int, float) or not 0 < timeout_s <= 86400:
        raise ValueError("snapshot admission timeout must be bounded")
    root = Path(root)
    if not root.is_absolute() or root.resolve() != root or root.is_symlink():
        raise ValueError("snapshot admission root must be canonical")
    root.mkdir(mode=0o700, parents=False, exist_ok=True)
    info = root.stat()
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("snapshot admission root must be private")
    registry = _open(root, "registry.lock")
    slots = []
    held = None
    started = time.monotonic()
    try:
        for i in range(2):
            slots.append(_open(root, f"slot-{i}"))
        while held is None:
            reason = stop_reason() if stop_reason is not None else None
            if reason:
                raise RuntimeError(f"snapshot admission stopped: {reason}")
            fcntl.flock(registry, fcntl.LOCK_EX)
            try:
                # First join an existing equal identity, including another
                # process's nested host consumer of this exact compiler.
                for index, fd in enumerate(slots):
                    try:
                        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    except BlockingIOError:
                        # Contention is expected scheduling state, never admission evidence.
                        if time.monotonic() - started >= timeout_s:
                            raise TimeoutError("snapshot admission slot remained busy") from None
                        continue
                    if os.pread(fd, 128, 0).decode() == key:
                        held = index
                        break
                    fcntl.flock(fd, fcntl.LOCK_UN)
                if held is None:
                    for index, fd in enumerate(slots):
                        try:
                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            # An exclusive owner still holds this slot; retry only within the budget.
                            if time.monotonic() - started >= timeout_s:
                                raise TimeoutError("snapshot admission slot remained busy") from None
                            continue
                        os.ftruncate(fd, 0)
                        os.pwrite(fd, key.encode(), 0)
                        fcntl.flock(fd, fcntl.LOCK_SH)
                        held = index
                        break
            finally:
                fcntl.flock(registry, fcntl.LOCK_UN)
            if held is None:
                left = timeout_s - (time.monotonic() - started)
                if left <= 0:
                    raise TimeoutError("two immutable compiler identities are still active")
                time.sleep(min(0.1, left))
        yield {
            "identity_sha256": key,
            "slot": held,
            "wait_s": time.monotonic() - started,
            "authority": "scheduling_only",
        }
    finally:
        for fd in slots:
            os.close(fd)
        os.close(registry)


# ---- snapshot_inputs ---------------------------------------------------------------------------
# Immutable public-input owner independent of legacy host proof-source pins.


class SnapshotInputs:
    """Trusted evaluator input selection; grants remain the archived bundle's."""

    def __init__(self, *, workspace, bundle_manifest, repo):
        self.workspace, self.bundle_manifest, self.repo = map(Path, (workspace, bundle_manifest, repo))
        if (
            self.bundle_manifest.resolve() != self.bundle_manifest
            or self.bundle_manifest.is_symlink()
            or self.bundle_manifest.is_relative_to(self.workspace)
        ):
            raise ValueError("snapshot input authority must be outside the writable author workspace")
        self.bundle_bytes = self.bundle_manifest.read_bytes()
        self.bundle = yaml.safe_load(self.bundle_bytes)
        self.bundle_identity = document_digest(self.bundle)
        self.profile = verify_tool_profile(self.bundle)
        if self.profile is None:
            raise ValueError("snapshot execution requires an explicit public input profile")
        marker = B.verify_bundle_snapshot(self.workspace, self.bundle, repo=self.repo)
        self.snapshot_digest = marker["content_sha256"]
        self.snapshot_marker_digest = B.snapshot_marker_digest(self.workspace)
        # Record the ACTUAL external checker revision. These are deliberately
        # not presented as matching a historical Lean/tool-profile certificate.
        names = {
            row["module"].removesuffix(".__init__")
            for row in self.profile["automatic_checker_sources"]
            if row.get("module")
        }
        names.update(
            (
                __name__,
                "merlin.development.targetgen.packaging.loading",
                "merlin.development.backend.feedback.execution.snapshot_evaluator",
            )
        )
        self.framework_sources = tuple(self._pin(module_source_path(name)) for name in sorted(names))

    @staticmethod
    def _pin(path):
        path = Path(path).resolve()
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def _verify(self):
        if (
            self.bundle_manifest.read_bytes() != self.bundle_bytes
            or document_digest(self.bundle) != self.bundle_identity
            or verify_tool_profile(self.bundle) != self.profile
        ):
            raise ValueError("snapshot public input declaration changed")
        marker = B.verify_bundle_snapshot(self.workspace, self.bundle, repo=self.repo)
        if (
            marker["content_sha256"] != self.snapshot_digest
            or B.snapshot_marker_digest(self.workspace) != self.snapshot_marker_digest
            or tuple(self._pin(row["path"]) for row in self.framework_sources) != self.framework_sources
        ):
            raise ValueError("snapshot inputs or external evaluator revision changed")

    def record(self):
        return {
            "schema": "merlin.snapshot_external_inputs.v1",
            "workspace": str(self.workspace),
            "repo": str(self.repo),
            "bundle_manifest": self._pin(self.bundle_manifest),
            "snapshot_sha256": self.snapshot_digest,
            "snapshot_marker_sha256": self.snapshot_marker_digest,
            "profile_sha256": self.profile["sha256"],
            "framework_sources": list(self.framework_sources),
        }


# ---- tree_value_cache --------------------------------------------------------------------------
# Reuse a value computed from a directory tree for as long as the tree is unchanged.
#
# Every completed test case of a grade promotes against the same frozen submission snapshot, and
# hashing it again for each one held the grader's lock and interpreter for most of a second. A value
# is keyed by :func:`tree_signature`, so any edit recomputes it; a tree that changed within
# :data:`SETTLED_NS` is recomputed every time.

CACHE: dict[str, tuple] = {}

CACHE_MAX = 32

SETTLED_NS = 2_000_000_000


def tree_signature(root: Path):
    """The stat identity of every entry under ``root``, read without opening a file.

    Any change to a file's bytes or metadata moves its ctime, which an unprivileged process cannot set,
    and adding, removing or renaming an entry moves its directory's; so an equal signature means the
    tree is unchanged. ``None`` when the tree cannot be listed."""
    rows = []
    try:
        for directory, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for name in [".", *sorted(filenames)]:
                path = os.path.join(directory, name) if name != "." else directory
                st = os.lstat(path)
                rows.append(
                    (
                        os.path.relpath(path, root),
                        st.st_dev,
                        st.st_ino,
                        st.st_mode,
                        st.st_size,
                        st.st_mtime_ns,
                        st.st_ctime_ns,
                    )
                )
            for name in dirnames:
                if os.path.islink(os.path.join(directory, name)):
                    st = os.lstat(os.path.join(directory, name))
                    rows.append(
                        (
                            os.path.relpath(os.path.join(directory, name), root),
                            st.st_dev,
                            st.st_ino,
                            st.st_mode,
                            st.st_size,
                            st.st_mtime_ns,
                            st.st_ctime_ns,
                        )
                    )
    except OSError:
        return None
    return tuple(rows)


def cached(root: Path, compute: Callable[[], Any]) -> Any:
    """``compute()``, reused while ``root`` keeps the signature it had when the value was computed."""
    root = Path(root)
    key = str(root.resolve()) if root.exists() else str(root)
    signature = tree_signature(root)
    entry = CACHE.get(key)
    if signature is not None and entry is not None and entry[0] == signature:
        return entry[1]
    value = compute()
    settled = signature is not None and all(
        row[5] < time.time_ns() - SETTLED_NS and row[6] < time.time_ns() - SETTLED_NS for row in signature
    )
    if settled and tree_signature(root) == signature:
        if len(CACHE) >= CACHE_MAX:
            CACHE.pop(next(iter(CACHE)))
        CACHE[key] = (signature, value)
    return value


# ---- snapshot_publication ----------------------------------------------------------------------
# Live feedback belongs to the newest owned snapshot; archived evidence belongs to each.

_OWNER = ContextVar("phase1_snapshot_publication", default=None)


@contextmanager
def _lock(workspace):
    directory = Path(workspace) / "qa"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".snapshot-publication.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield directory / ".snapshot-publication.json"


def claim(workspace, token):
    """Called by the single lifecycle owner before dispatching its newest snapshot."""
    with _lock(workspace) as path:
        temporary = path.with_suffix(".next.json")
        temporary.write_text(json.dumps({"token": str(token)}) + "\n")
        temporary.replace(path)


@contextmanager
def scope(workspace, token):
    selected = _OWNER.set((str(Path(workspace).resolve()), str(token)))
    try:
        yield
    finally:
        _OWNER.reset(selected)


def publish_live(workspace, action, *, default=None):
    """Serialize the ownership check and write; stale workers retain only their archives."""
    owner = _OWNER.get()
    if owner is None:
        return action()
    if owner[0] != str(Path(workspace).resolve()):
        raise ValueError("snapshot publication workspace differs")
    with _lock(workspace) as path:
        if json.loads(path.read_text()).get("token") != owner[1]:
            return default
        return action()


def owns(workspace) -> bool:
    """Whether this context's snapshot is the newest one claimed for live feedback.

    Only an explicit owner proves that; outside a publication scope nothing is proven."""
    owner = _OWNER.get()
    if owner is None:
        return False
    try:
        with _lock(workspace) as path:
            return json.loads(path.read_text()).get("token") == owner[1]
    except (OSError, ValueError):
        return False


def current_token():
    owner = _OWNER.get()
    return owner[1] if owner is not None else None
