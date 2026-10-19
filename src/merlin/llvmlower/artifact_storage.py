"""Share a completed artifact only after reopening its full fresh byte identity.

Both producer outputs must be immutable. The fresh producer still runs normally;
this changes storage after completion, not compiler results, namespace semantics,
numeric contracts, executable admission or runtime aliasing.
"""

import hashlib
import os
import tempfile
from pathlib import Path


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def share_identical_artifact(previous, fresh):
    """Atomically replace a fresh immutable file with byte-identical shared storage."""
    previous, fresh = Path(previous), Path(fresh)
    if any(path.is_symlink() or not path.is_file() for path in [previous, fresh]):
        raise ValueError("artifact storage requires regular immutable completed files")
    digest = _sha(fresh)
    size = fresh.stat().st_size
    if previous.stat().st_size != size or _sha(previous) != digest:
        raise ValueError("previous artifact differs from complete fresh bytes")
    already_shared = os.path.samefile(previous, fresh)
    if not already_shared:
        with tempfile.TemporaryDirectory(prefix=".artifact-share-", dir=fresh.parent) as directory:
            link = Path(directory) / "shared"
            os.link(previous, link)
            if _sha(link) != digest or _sha(fresh) != digest:
                raise ValueError("completed artifact changed during storage verification")
            os.replace(link, fresh)
    if not os.path.samefile(previous, fresh) or _sha(fresh) != digest:
        raise ValueError("shared completed artifact changed after storage selection")
    return {
        "schema": "merlin.identical_artifact_storage.v1",
        "method": "fresh_full_byte_check_then_shared_inode",
        "scope": "completed_file_storage_identity_only_not_compilation_or_runtime_permission",
        "previous": str(previous.resolve()),
        "fresh": str(fresh.resolve()),
        "sha256": digest,
        "bytes": size,
        "shared_inode": True,
        "already_shared": already_shared,
    }


def supplemental_object_digest(objects):
    """Identity of actual linked device/matrix bytes, independent of work paths."""
    import hashlib

    digest = hashlib.sha256(b"merlin-supplemental-objects-v1")
    for path in objects:
        content = Path(path).read_bytes()
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.digest()
