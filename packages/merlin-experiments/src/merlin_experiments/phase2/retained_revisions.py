"""Issued, bounded source revisions shared by ordinary component evaluations.

Source bytes have one address; parentage and evaluation options have a separate
revision identity. Nothing in this module grants correctness or promotion.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import shutil
import stat
import threading
from dataclasses import dataclass, field
from pathlib import Path

from merlin.common import storage_lifecycle as storage
from merlin.common.tree_hash import _SKIP

from .contracts import StageGateError, canonical_json, document_sha256
from .edit_authority import FrozenEditAuthority

_ISSUER = object()
_EXECUTING = contextvars.ContextVar("phase2_retained_compilers", default=())


def source_inventory(root, *, max_bytes, max_files, readonly=False, exclude_generated=False):
    """Stream exact source bytes, file modes and directory membership under caps."""
    root = Path(root)
    if root.resolve() != root or root.is_symlink() or not root.is_dir():
        raise StageGateError("retained source requires a canonical real directory")
    digest, files, total, entries = hashlib.sha256(), 0, 0, []

    def walk(directory):
        nonlocal files, total
        with os.scandir(directory) as iterator:
            children = []
            for row in iterator:
                if len(children) >= max_files * 4:
                    raise StageGateError("retained source exceeds its entry budget")
                children.append(row)
            children.sort(key=lambda row: row.name)
        for child in children:
            relative = Path(child.path).relative_to(root).as_posix()
            mode = child.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise StageGateError("retained source contains a linked or special entry")
            if exclude_generated and (child.name in _SKIP or child.name.endswith(".pyc")):
                continue
            if readonly and mode & 0o222:
                raise StageGateError("retained source became writable")
            entries.append((relative, "directory" if stat.S_ISDIR(mode) else "file", bool(mode & 0o111)))
            if len(entries) > max_files * 4:
                raise StageGateError("retained source exceeds its entry budget")
            if stat.S_ISDIR(mode):
                walk(Path(child.path))
                continue
            files += 1
            if files > max_files:
                raise StageGateError("retained source exceeds its file budget")
            digest.update(relative.encode() + b"\0")
            fd = os.open(child.path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise StageGateError("retained source changed during its bounded read")
                while block := stream.read(1024 * 1024):
                    total += len(block)
                    if total > max_bytes:
                        raise StageGateError("retained source exceeds its byte budget")
                    digest.update(block)
            digest.update(b"\0")

    if readonly and root.stat().st_mode & 0o222:
        raise StageGateError("retained source root became writable")
    walk(root)
    if not files:
        raise StageGateError("retained source contains no files")
    return {"sha256": digest.hexdigest(), "n_bytes": total, "n_files": files, "entries": tuple(sorted(entries))}


@dataclass(frozen=True)
class RetainedRevision:
    revision_id: str
    source_sha256: str
    source: Path
    parent_revision: str | None
    options_bytes: bytes
    edit_authority_sha256: str
    inventory_bytes: bytes = field(repr=False, compare=False)
    _store: object = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    @property
    def options(self):
        return json.loads(self.options_bytes)

    @property
    def inventory(self):
        record = json.loads(self.inventory_bytes)
        record["entries"] = tuple(tuple(row) for row in record["entries"])
        return record

    def verify(self, *, authority=None):
        self._store.verify(self, authority=authority)
        return self.source_sha256

    def public_record(self):
        return {
            "revision_id": self.revision_id,
            "source_sha256": self.source_sha256,
            "parent_revision": self.parent_revision,
            "options": self.options,
        }


class RetainedRevisionStore:
    def __init__(self, root, *, edit_authority, max_bytes, max_files, max_total_bytes=None, max_revisions=256):
        if type(edit_authority) is not FrozenEditAuthority or not edit_authority.configured:
            raise StageGateError("retained revisions require frozen controller edit authority")
        if any(type(value) is not int or value < 1 for value in (max_bytes, max_files)):
            raise StageGateError("retained revisions require positive byte and file caps")
        self.root = Path(root)
        if self.root.resolve() != self.root or self.root.is_symlink() or self.root.exists():
            raise StageGateError("retained revision store requires a fresh canonical private root")
        if any(
            self.root.is_relative_to(path) or path.is_relative_to(self.root)
            for path in (edit_authority.initial_source, edit_authority.seed, edit_authority.output)
        ):
            raise StageGateError("retained revisions overlap mutable authoring or immutable edit authority")
        self.authority, self.max_bytes, self.max_files = edit_authority, max_bytes, max_files
        self.max_total_bytes = max_bytes if max_total_bytes is None else max_total_bytes
        if type(self.max_total_bytes) is not int or self.max_total_bytes < max_bytes:
            raise StageGateError("retained revision total budget is smaller than its source budget")
        if type(max_revisions) is not int or not 1 <= max_revisions <= 1024:
            raise StageGateError("retained revision history requires a bounded record cap")
        self.max_revisions = max_revisions
        self.root.mkdir(parents=True, mode=0o700)
        self._revisions, self._sources, self._pins = {}, {}, {}
        self._copy_cleanup_failed = False
        self._lock = threading.RLock()

    def get(self, revision_id):
        with self._lock:
            revision = self._revisions.get(revision_id)
            if revision is None:
                raise StageGateError("retained revision was not issued by this controller")
            self.verify(revision)
            return revision

    def freeze(self, source, *, parent_revision=None, options=None):
        with self._lock:
            if self._copy_cleanup_failed:
                raise StageGateError("retained source rollback UNKNOWN; further freezes refused")
            self.authority.check_integrity()
            if parent_revision is not None:
                self.get(parent_revision)
            options_bytes = canonical_json({} if options is None else options)
            if len(options_bytes) > 16_384 or type(json.loads(options_bytes)) is not dict:
                raise StageGateError("retained revision options exceed their closed mapping budget")
            source = Path(source)
            before = source_inventory(
                source, max_bytes=self.max_bytes, max_files=self.max_files, exclude_generated=True
            )
            source_sha = before["sha256"]
            destination = self.root / "sources" / source_sha
            if source_sha not in self._sources:
                if (
                    sum(row["n_bytes"] for row in self._sources.values())
                    + sum(path.stat().st_size for path in self.root.glob("*.json"))
                    + before["n_bytes"]
                    > self.max_total_bytes
                ):
                    raise StageGateError("retained revisions exceed their aggregate disk grant")
                held = storage.acquire(destination, owner="phase2 retained source freeze")
                created = False
                try:
                    destination.mkdir(parents=True, mode=0o700)
                    created = True
                    copied = 0
                    for name, kind, executable in before["entries"]:
                        target = destination / name
                        if kind == "directory":
                            target.mkdir(parents=True, exist_ok=True)
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        fd = os.open(source / name, os.O_RDONLY | os.O_NOFOLLOW)
                        with os.fdopen(fd, "rb") as reader, target.open("xb") as writer:
                            while block := reader.read(1024 * 1024):
                                copied += len(block)
                                if copied > self.max_bytes:
                                    raise StageGateError("retained source grew beyond its copy budget")
                                writer.write(block)
                        target.chmod(0o555 if executable else 0o444)
                    for path in sorted(destination.rglob("*"), key=lambda path: len(path.parts), reverse=True):
                        if path.is_dir():
                            path.chmod(0o555)
                    destination.chmod(0o555)
                    after = source_inventory(
                        source, max_bytes=self.max_bytes, max_files=self.max_files, exclude_generated=True
                    )
                    actual = source_inventory(
                        destination, max_bytes=self.max_bytes, max_files=self.max_files, readonly=True
                    )
                    if before != after or actual != before:
                        raise StageGateError("retained source changed while freezing")
                    # Validate the immutable bytes actually queued, independently of later live edits.
                    self.authority.validate_candidate(destination)
                    self._pins[source_sha] = storage.pin(destination, reason="retained phase2 source revision")
                    self._sources[source_sha] = actual
                except BaseException:
                    try:
                        if created and destination.exists():
                            # Only this fresh, private copy belongs to the failed
                            # transaction. Restore directory write access first.
                            for directory, _children, _files in os.walk(destination):
                                Path(directory).chmod(0o700)
                            shutil.rmtree(destination)
                        held.close("failed")
                    except BaseException:
                        self._copy_cleanup_failed = True
                        raise
                    raise
                held.close()
            authority_sha = document_sha256(self.authority.binding)
            binding = {
                "schema": "merlin.retained_revision.v1",
                "source_sha256": source_sha,
                "source_inventory_sha256": document_sha256(before),
                "parent_revision": parent_revision,
                "options": json.loads(options_bytes),
                "edit_authority_sha256": authority_sha,
            }
            identity = document_sha256(binding)
            if identity not in self._revisions:
                if len(self._revisions) >= self.max_revisions:
                    raise StageGateError("retained revision history record budget is exhausted")
                revision = RetainedRevision(
                    identity,
                    source_sha,
                    destination,
                    parent_revision,
                    options_bytes,
                    authority_sha,
                    canonical_json(before),
                    self,
                    _ISSUER,
                )
                record = self.root / (identity + ".json")
                if (
                    sum(row["n_bytes"] for row in self._sources.values())
                    + sum(path.stat().st_size for path in self.root.glob("*.json"))
                    + len(canonical_json(binding))
                    > self.max_total_bytes
                ):
                    raise StageGateError("retained source and declarations exceed their aggregate disk grant")
                record.write_bytes(canonical_json(binding))
                record.chmod(0o444)
                self._revisions[identity] = revision
            return self.get(identity)

    def verify(self, revision, *, authority=None):
        with self._lock:
            if (
                type(revision) is not RetainedRevision
                or revision._issuer is not _ISSUER
                or self._revisions.get(revision.revision_id) is not revision
                or authority is not None
                and authority is not self.authority
            ):
                raise StageGateError("retained revision lacks exact issued edit authority")
            self.authority.check_integrity()
            if document_sha256(self.authority.binding) != revision.edit_authority_sha256:
                raise StageGateError("retained revision edit authority changed")
            actual = source_inventory(
                revision.source, max_bytes=self.max_bytes, max_files=self.max_files, readonly=True
            )
            if actual != revision.inventory or actual != self._sources.get(revision.source_sha256):
                raise StageGateError("retained revision source bytes or membership changed")
            record = self.root / (revision.revision_id + ".json")
            if record.is_symlink() or record.stat().st_mode & 0o222:
                raise StageGateError("retained revision declaration is linked or writable")
            binding = json.loads(record.read_bytes())
            expected = {
                "schema": "merlin.retained_revision.v1",
                "source_sha256": revision.source_sha256,
                "source_inventory_sha256": document_sha256(revision.inventory),
                "parent_revision": revision.parent_revision,
                "options": revision.options,
                "edit_authority_sha256": revision.edit_authority_sha256,
            }
            if binding != expected or document_sha256(binding) != revision.revision_id:
                raise StageGateError("retained revision declaration changed")


@contextlib.contextmanager
def executing_revisions(*revisions):
    for revision in revisions:
        revision.verify()
    token = _EXECUTING.set(revisions)
    try:
        yield
    finally:
        _EXECUTING.reset(token)


def executing_revision(compiler):
    """Only the fixed evaluator's live issued context can select retained execution."""
    return next((revision for revision in _EXECUTING.get() if revision.source == compiler), None)
