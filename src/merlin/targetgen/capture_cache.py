"""Capture-cache source observations, locking and atomic attempt publication.

No framework imports or capture execution. Callers own validation and must hold the
slot lock across lookup, capture, identity rechecks and publication. Source observations
are deliberately narrower than complete framework/loader/data provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


def implementation_identity() -> dict:
    """Bind direct Merlin capture/normalization owners without importing their implementations.

    This inventory is not the upstream framework stack or arbitrary loader dependency/data
    closure. Unreadable source refuses cache use through the caller's uncached path.
    """
    from merlin.common.digest import sha256_file
    from merlin.common.paths import module_source_path

    owners = (
        "merlin.targetgen.capsule_source",
        "merlin.targetgen.capture_cache",
        "merlin.targetgen._m2m_capture_worker",
        "merlin.frontends.capture_normalization",
        "merlin.frontends.linalg_mlir",
        "merlin.llvmlower.torchao_affine",
        "merlin.xdsl_dialects._common",
        "merlin.xdsl_dialects.fp8",
        "merlin.common.ir_lock",
        "merlin.common.digest",
    )
    result = {}
    for owner in owners:
        path = module_source_path(owner)
        result[owner] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def observed_sources_match(meta: dict) -> bool:
    """Recheck observed source bytes, without claiming a complete loader dependency closure."""
    from merlin.common.digest import sha256_file

    records = meta.get("loader_dependency_sources")
    if not isinstance(records, list):
        return False
    try:
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                return False
            path = Path(record["path"])
            if not path.is_absolute() or path.is_symlink() or sha256_file(path) != record.get("sha256"):
                return False
    except (OSError, ValueError):
        return False
    return True


@contextmanager
def slot_lock(slot: Path | None):
    """Serialize a shared slot; unavailable locking means no shared cache use.

    Keep the lock beside the slot, never unlink it: replacing a locked inode
    would let another process acquire a different lock for the same capture.
    """
    if slot is None:
        yield None
        return
    stream = None
    try:
        import fcntl

        slot.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(slot.with_name(slot.name + ".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        stream = os.fdopen(descriptor, "a")
        fcntl.flock(stream, fcntl.LOCK_EX)
    except (ImportError, AttributeError, OSError):
        if stream is not None:
            stream.close()
        yield None
        return
    try:
        yield slot
    finally:
        stream.close()


def commit(slot: Path, attempt: Path) -> None:
    """Atomically publish the parent-validated metadata identity, never worker readiness."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=slot, prefix=".complete-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(
                {
                    "version": 1,
                    "attempt": attempt.name,
                    "meta_sha256": hashlib.sha256((attempt / "meta.json").read_bytes()).hexdigest(),
                },
                stream,
            )
        os.replace(temporary, slot / "capture_complete.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def committed_attempt(slot: Path) -> Path | None:
    try:
        document = json.loads((slot / "capture_complete.json").read_text())
        name = document.get("attempt") if isinstance(document, dict) else None
        if not isinstance(name, str) or not name.startswith("attempt-") or Path(name).name != name:
            return None
        attempt = slot / name
        if attempt.is_symlink() or not attempt.is_dir():
            return None
        expected = {
            "version": 1,
            "attempt": name,
            "meta_sha256": hashlib.sha256((attempt / "meta.json").read_bytes()).hexdigest(),
        }
        return attempt if document == expected else None
    except (OSError, ValueError):
        return None
