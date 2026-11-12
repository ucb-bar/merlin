"""Live parent-published, bounded original native observation archive.

Workers stage signed observations. Only the creating host parent may publish
after its existing guardian's real cleanup. Records never issue seed/replay or
numerical/timing authority; current unspecified-state engines still execute.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import hmac
import json
import os
import secrets
import uuid
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common import storage_lifecycle as storage
from merlin.runtime import _native_files as F
from merlin.runtime.native_runtime import NativeRuntimeView

_ISSUED = {}
_DOCUMENT_BYTES = 1024 * 1024


def _write(path, body):
    text = F.document(body).encode()
    if len(text) > _DOCUMENT_BYTES:
        raise ValueError("native archive metadata exceeds its document budget")
    temporary = path.parent / (".entry-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read(path):
    path = F.canonical(path)
    if path.stat().st_size > _DOCUMENT_BYTES:
        raise ValueError("native archive metadata exceeds its document budget")
    return json.loads(path.read_bytes())


class ArtifactRetestIndex:
    def __init__(self, root, *, context, runtime_owner, max_entries, max_bytes, issuer):
        self.root, self.runtime_owner = root, runtime_owner
        self.context_sha256 = hashlib.sha256(F.document(context).encode()).hexdigest()
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._issuer, self._parent_pid, self._secret = issuer, os.getpid(), secrets.token_bytes(32)
        self._runtime_identity = runtime_owner.verify()
        self._closed = False

    def verify(self, *, runtime_owner=None, context=None):
        if _ISSUED.get(self._issuer) is not self or self._closed:
            raise ValueError("artifact archive needs its exact live original parent owner")
        if runtime_owner is not None and runtime_owner is not self.runtime_owner:
            raise ValueError("artifact archive changed its original native runtime owner")
        if context is not None and hashlib.sha256(F.document(context).encode()).hexdigest() != self.context_sha256:
            raise ValueError("artifact archive changed its original normal admission context")
        if self.runtime_owner.verify() != self._runtime_identity:
            raise ValueError("artifact archive original readonly runtime changed")
        if (
            F.canonical(self.root) != self.root
            or self.root.stat().st_uid != os.getuid()
            or self.root.stat().st_mode & 0o077
        ):
            raise ValueError("artifact archive is no longer a private original output owner")
        return {
            "context_sha256": self.context_sha256,
            "runtime": self._runtime_identity,
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
            "scope": "parent-published original observations; unspecified reset/seed prevents replay",
        }

    @contextlib.contextmanager
    def _locked(self):
        self.verify()
        with (self.root / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _seal(self, body):
        return {
            "body": body,
            "signature": hmac.new(self._secret, F.document(body).encode(), hashlib.sha256).hexdigest(),
        }

    def _reopen(self, path):
        wire = _read(path)
        expected = self._seal(wire["body"])["signature"]
        if not hmac.compare_digest(wire["signature"], expected):
            raise ValueError("native archive entry was not issued in this live host session")
        body = wire["body"]
        return self._verify_body(body)

    def _verify_body(self, body):
        if body["context_sha256"] != self.context_sha256:
            raise ValueError("native archive entry differs from its normal admission")
        for pin in body["pins"]:
            if F.pin(Path(pin["path"]), max_bytes=self.max_bytes) != pin:
                raise ValueError("native archived artifact changed")
        record = Path(body["invocation"]["path"])
        if F.pin(record, max_bytes=_DOCUMENT_BYTES) != body["invocation"]:
            raise ValueError("native original Python dispatch changed")
        observed = I.verify(record)
        if observed["stage"] != "execution" or observed["kind"] != "python_call" or observed["outcome"] != "returned":
            raise ValueError("native archived observation has no actual completed original dispatch")
        native = body["native"]
        console = self.runtime_owner.reopen_observation(native["body"], native["signature"])
        payload = console if type(console) is bytes else console.encode()
        if body["pins"][0] not in observed["inputs"] or Path(observed["stdout"]["path"]).read_bytes() != payload:
            raise ValueError("native archived stream differs from its completed original Python return")
        return body, console

    def _job_bytes(self, job_root):
        size = 0
        for path in F.members(job_root, max_files=self.runtime_owner.max_files):
            if not path.is_symlink() and path.is_file():
                size += path.stat().st_size
            if size > self.max_bytes:
                raise ValueError("native archive retained job exceeds its byte budget")
        return size

    @staticmethod
    def _key(key):
        if type(key) is not str or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("native archive requires its complete SHA256 artifact identity")
        return key

    def stage_observation(self, job_root, *, key, pins, invocation, native):
        self.verify()
        self._key(key)
        job_root = F.canonical(job_root)
        if not job_root.is_relative_to(self.runtime_owner.output_root):
            raise ValueError("native observation job escaped its original output owner")
        record = F.pin(Path(invocation), max_bytes=_DOCUMENT_BYTES)
        # Retaining a job root protects its entire original output directory.
        # Price those bytes, not just the three cited artifact members.
        size = self._job_bytes(job_root)
        body = {
            "context_sha256": self.context_sha256,
            "key": key,
            "pins": pins,
            "invocation": record,
            "native": native,
            "bytes": size,
            "job_root": str(job_root),
        }
        self._verify_body(body)
        name = hashlib.sha256((str(job_root) + "\0" + key).encode()).hexdigest()
        pending = self.root / "pending" / (name + ".json")
        with self._locked():
            if pending.exists():
                self._reopen(pending)
                return  # Preserve the first actual observation, not a new replicate.
            if (
                not pending.exists()
                and len(F.children(self.root / "pending", limit=self.max_entries + 1)) >= self.max_entries
            ):
                raise ValueError("native archive pending observations reached their bounded capacity")
            existing = [
                self._reopen(self.root / "pending" / name)[0]
                for name in F.children(self.root / "pending", limit=self.max_entries + 1)
            ]
            existing += [
                self._reopen(self.root / "entries" / name)[0]
                for name in F.children(self.root / "entries", limit=self.max_entries + 1)
            ]
            if len(existing) >= self.max_entries:
                raise ValueError("native archive observations reached their bounded capacity")
            if sum(row["bytes"] for row in existing) + size > self.max_bytes:
                raise ValueError("native pending archive exceeds its byte budget")
            body["pin_token"] = storage.pin(job_root, reason="pending original native observation; cleanup unpublished")
            try:
                _write(pending, self._seal(body))
            except BaseException:
                storage.unpin(job_root, body["pin_token"])
                raise

    def publish_completed_job(self, job_root):
        """Trusted host seam: call only after actual guardian reaping and release.

        This accepts no status/cleanup JSON. Forked workers and reconstructed
        handles cannot publish. The original pool owns completion verification.
        """
        self.verify()
        if os.getpid() != self._parent_pid:
            raise ValueError("only the creating host parent may publish completed native observations")
        job_root = F.canonical(job_root)
        count = 0
        with self._locked():
            entries = self.root / "entries"
            retained = [
                self._seal(self._reopen(entries / name)[0]) for name in F.children(entries, limit=self.max_entries + 1)
            ]
            for name in F.children(self.root / "pending", limit=self.max_entries + 1):
                path = self.root / "pending" / name
                body, _ = self._reopen(path)
                if not Path(body["job_root"]).is_relative_to(job_root):
                    continue
                # Only this trusted parent seam runs after the original job has
                # closed. Staging-time size cannot price later output growth.
                try:
                    body["bytes"] = self._job_bytes(Path(body["job_root"]))
                except ValueError:
                    storage.unpin(body["job_root"], body["pin_token"])
                    path.unlink()
                    raise
                target = entries / (body["key"] + ".json")
                if target.exists():
                    storage.unpin(body["job_root"], body["pin_token"])
                    path.unlink()
                    continue
                while retained and (
                    len(retained) >= self.max_entries
                    or sum(row["body"]["bytes"] for row in retained) + body["bytes"] > self.max_bytes
                ):
                    old = retained.pop(0)["body"]
                    storage.unpin(old["job_root"], old["pin_token"])
                    (entries / (old["key"] + ".json")).unlink()
                if body["bytes"] > self.max_bytes:
                    raise ValueError("native completed observation exceeds its archive budget")
                wire = self._seal(body)
                _write(target, wire)
                retained.append(wire)
                path.unlink()
                count += 1
        return count

    def discard_cleaned_job(self, job_root):
        """Parent calls after actual failed/cancelled job cleanup and release.

        Pending observations only; unknown cleanup must retain their pins. No
        saved receipt can satisfy this live owner and creating-parent seam.
        """
        self.verify()
        if os.getpid() != self._parent_pid:
            raise ValueError("only the creating host parent may discard cleaned native observations")
        job_root, count = F.canonical(job_root), 0
        with self._locked():
            for name in F.children(self.root / "pending", limit=self.max_entries + 1):
                path = self.root / "pending" / name
                body, _ = self._reopen(path)
                if Path(body["job_root"]).is_relative_to(job_root):
                    storage.unpin(body["job_root"], body["pin_token"])
                    path.unlink()
                    count += 1
        return count

    def original_observation(self, key):
        """Reopen published actual evidence; this never asserts replay eligibility."""
        with self._locked():
            path = self.root / "entries" / (self._key(key) + ".json")
            return None if not path.exists() else self._reopen(path)

    def close(self):
        if os.getpid() != self._parent_pid:
            raise ValueError("native archive closing belongs to its creating host parent")
        with self._locked():
            for name in F.children(self.root / "entries", limit=self.max_entries + 1):
                path = self.root / "entries" / name
                body, _ = self._reopen(path)
                storage.unpin(body["job_root"], body["pin_token"])
                path.unlink()
            for name in F.children(self.root / "pending", limit=self.max_entries + 1):
                path = self.root / "pending" / name
                body, _ = self._reopen(path)
                storage.unpin(body["job_root"], body["pin_token"])
                path.unlink()
        self._closed = True


def prepare_artifact_retest_index(root, *, context, runtime_owner, max_entries, max_bytes):
    if (
        type(runtime_owner) is not NativeRuntimeView
        or type(max_entries) is not int
        or max_entries <= 0
        or type(max_bytes) is not int
        or max_bytes <= 0
    ):
        raise ValueError("native artifact archive requires live runtime authority and explicit bounds")
    runtime_owner.verify()
    root = F.canonical(root, exists=False)
    if root.exists() or not root.is_relative_to(runtime_owner.output_root):
        raise ValueError("native artifact archive requires a fresh original output root")
    root.mkdir(mode=0o700)
    (root / "pending").mkdir(mode=0o700)
    (root / "entries").mkdir(mode=0o700)
    issuer = object()
    index = ArtifactRetestIndex(
        root, context=context, runtime_owner=runtime_owner, max_entries=max_entries, max_bytes=max_bytes, issuer=issuer
    )
    _ISSUED[issuer] = index
    index.verify()
    return index
