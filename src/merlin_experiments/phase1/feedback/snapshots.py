"""Private enqueue-time source snapshots shared by promotion producers and consumers.

The existing submission-identity observer is explicit: copy, verify and recovery use
one digest implementation, evaluated at the same points as the original callers.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path


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
