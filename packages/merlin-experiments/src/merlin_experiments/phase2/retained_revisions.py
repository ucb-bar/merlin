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
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from merlin.common import storage_lifecycle as storage
from merlin.common.tree_hash import _SKIP

from .contracts import StageGateError, canonical_json, document_sha256
from .edit_authority import FrozenEditAuthority

_ISSUER = object()
_EXECUTING = contextvars.ContextVar("phase2_retained_compilers", default=())


def _entry_signature(info):
    return info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _target_records(root_fd, *, max_entries):
    records = {}

    def walk(directory_fd, prefix):
        with os.scandir(directory_fd) as children:
            for child in children:
                name = (prefix / child.name).as_posix()
                info = child.stat(follow_symlinks=False)
                if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                    raise StageGateError("restoration target contains a linked or special entry")
                records[name] = _entry_signature(info)
                if len(records) > max_entries:
                    raise StageGateError("restoration target exceeded its entry budget")
                if stat.S_ISDIR(info.st_mode):
                    fd = os.open(child.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory_fd)
                    try:
                        walk(fd, prefix / child.name)
                    finally:
                        os.close(fd)

    walk(root_fd, Path())
    return records


def _directory_fd(root_fd, parts, *, create=False):
    current = os.dup(root_fd)
    try:
        for part in parts:
            try:
                selected = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(part, 0o755, dir_fd=current)
                selected = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            os.close(current)
            current = selected
        return current
    except BaseException:
        os.close(current)
        raise


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
            entries.append((relative, "directory" if stat.S_ISDIR(mode) else "file", mode & 0o111))
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
        candidate = edit_authority.initial_source
        if candidate is None or candidate.resolve() != candidate or candidate.is_symlink() or not candidate.is_dir():
            raise StageGateError("retained revision authority lacks its exact original candidate directory")
        info = candidate.stat()
        self._candidate_identity = info.st_dev, info.st_ino
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
                        target.chmod(0o444 | executable)
                    directory_modes = {
                        name: executable for name, kind, executable in before["entries"] if kind == "directory"
                    }
                    for path in sorted(destination.rglob("*"), key=lambda path: len(path.parts), reverse=True):
                        if path.is_dir():
                            path.chmod(0o444 | directory_modes[path.relative_to(destination).as_posix()])
                    destination.chmod(0o444 | (source.stat().st_mode & 0o111))
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

    def restore(self, revision_id, candidate):
        """Restore an issued selection in place before the host freezes lineage.

        This proves source/authority agreement, not author process shutdown. The
        caller must arrange its author/subauthor ordering and reject later edits.
        """
        with self._lock:
            revision = self.get(revision_id)
            candidate = Path(candidate)
            if (
                candidate != self.authority.initial_source
                or candidate.resolve() != candidate
                or candidate.is_symlink()
                or not candidate.is_dir()
            ):
                raise StageGateError("restoration requires the exact original candidate directory")
            root_fd = os.open(candidate, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                if _entry_signature(os.fstat(root_fd))[:2] != self._candidate_identity:
                    raise StageGateError("restoration candidate root inode changed")
                before = _target_records(root_fd, max_entries=self.max_files * 4)
                current = source_inventory(
                    candidate, max_bytes=self.max_bytes, max_files=self.max_files, exclude_generated=True
                )
                self.authority.validate_candidate(candidate)
                desired = {name: (kind, executable) for name, kind, executable in revision.inventory["entries"]}
                existing = {name: kind for name, kind, _executable in current["entries"]}
                if any(existing[name] != desired[name][0] for name in existing.keys() & desired.keys()):
                    raise StageGateError("restoration refuses changed source entry types")
                extras = existing.keys() - desired.keys()
                contract = self.authority.contract
                ancillary = tuple(Path(name) for name in contract.get("ancillary_paths", ()))
                helpers = tuple(Path(row["directory"]) for row in contract.get("helper_extensions", ()))
                for name in extras:
                    relative, kind = Path(name), existing[name]
                    if not (
                        any(
                            relative == path
                            or relative.is_relative_to(path)
                            or kind == "directory"
                            and path.is_relative_to(relative)
                            for path in ancillary
                        )
                        or any(relative != path and relative.is_relative_to(path) for path in helpers)
                        and (kind == "directory" or relative.suffix == ".py")
                    ):
                        raise StageGateError("restoration extra source entry lacks frozen edit authority")
                if _target_records(root_fd, max_entries=self.max_files * 4) != before:
                    raise StageGateError("restoration target changed during admission")
                copied = 0
                for name, (kind, executable) in desired.items():
                    relative = Path(name)
                    directory_fd = _directory_fd(root_fd, relative.parts[:-1], create=True)
                    try:
                        if kind == "directory":
                            try:
                                actual = _entry_signature(
                                    os.stat(relative.name, dir_fd=directory_fd, follow_symlinks=False)
                                )
                            except FileNotFoundError:
                                actual = None
                            if actual != before.get(name):
                                raise StageGateError("restoration source directory changed before creation")
                            child_fd = _directory_fd(directory_fd, (relative.name,), create=True)
                            try:
                                mode = stat.S_IMODE(os.fstat(child_fd).st_mode)
                                os.fchmod(child_fd, (mode & ~0o111) | executable)
                            finally:
                                os.close(child_fd)
                            continue
                        temporary = ".restore-" + uuid.uuid4().hex
                        expected = before.get(name)
                        writer_fd = os.open(
                            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd
                        )
                        try:
                            with os.fdopen(writer_fd, "wb") as writer:
                                reader_fd = os.open(revision.source / name, os.O_RDONLY | os.O_NOFOLLOW)
                                with os.fdopen(reader_fd, "rb") as reader:
                                    while block := reader.read(1024 * 1024):
                                        copied += len(block)
                                        if copied > self.max_bytes:
                                            raise StageGateError("restoration exceeded its source byte budget")
                                        writer.write(block)
                                    os.fchmod(writer.fileno(), 0o644 | executable)
                            try:
                                actual = _entry_signature(
                                    os.stat(relative.name, dir_fd=directory_fd, follow_symlinks=False)
                                )
                            except FileNotFoundError:
                                actual = None
                            if actual != expected:
                                raise StageGateError("restoration source entry changed before replacement")
                            os.replace(temporary, relative.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
                        finally:
                            try:
                                os.unlink(temporary, dir_fd=directory_fd)
                            except FileNotFoundError:
                                pass
                    finally:
                        os.close(directory_fd)
                for name in sorted(extras, key=lambda value: len(Path(value).parts), reverse=True):
                    relative = Path(name)
                    directory_fd = _directory_fd(root_fd, relative.parts[:-1])
                    try:
                        actual = _entry_signature(os.stat(relative.name, dir_fd=directory_fd, follow_symlinks=False))
                        expected = before[name]
                        if (actual if existing[name] == "file" else actual[:3]) != (
                            expected if existing[name] == "file" else expected[:3]
                        ):
                            raise StageGateError("restoration extra source entry changed before removal")
                        if existing[name] == "directory":
                            os.rmdir(relative.name, dir_fd=directory_fd)
                        else:
                            os.unlink(relative.name, dir_fd=directory_fd)
                    finally:
                        os.close(directory_fd)
                _target_records(root_fd, max_entries=self.max_files * 4)
                if (candidate.stat().st_dev, candidate.stat().st_ino) != self._candidate_identity:
                    raise StageGateError("restoration changed the original candidate root inode")
                actual = source_inventory(
                    candidate, max_bytes=self.max_bytes, max_files=self.max_files, exclude_generated=True
                )
                if actual != revision.inventory:
                    raise StageGateError("restoration differs from the exact issued source inventory")
                self.authority.validate_candidate(candidate)
                revision.verify(authority=self.authority)
                return revision.public_record()
            finally:
                os.close(root_fd)


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


def executing_candidate_revision(authority):
    """Default verification uses only the fixed worker's first issued source.

    This is verification context, never a compiler-path alias for execution.
    The fixed owner enters candidate first, then its independently frozen seed.
    """
    revisions = _EXECUTING.get()
    if not revisions:
        return None
    candidate = revisions[0]
    candidate.verify(authority=authority)
    return candidate
