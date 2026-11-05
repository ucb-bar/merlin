"""Bounded reuse of original closed compiler products, never compiler calls.

Fresh emitted LLVM inputs are compared inside the exact original build view.
Workers stage authenticated completed outputs; only the creating parent may
publish after its original guardian has reaped and released the job. Unknown
closure executes fresh. Final/calibration callers do not select this owner.
"""

from __future__ import annotations

import contextlib
import contextvars
import fcntl
import hashlib
import hmac
import io
import json
import locale
import os
import secrets
import shutil
import stat
import subprocess
import uuid
from pathlib import Path

from merlin.common import invocation_record as I
from merlin.common import storage_lifecycle as storage
from merlin.runtime import _native_files as F

_ISSUED = {}
_ACTIVE = contextvars.ContextVar("original_build_products", default=None)
_DOCUMENT_BYTES = 1024 * 1024


def _write(path, wire):
    text = F.document(wire).encode()
    if len(text) > _DOCUMENT_BYTES:
        raise ValueError("build product metadata exceeds its byte bound")
    temporary = path.parent / (".staging-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as output:
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read(path):
    path = F.canonical(path)
    if path.stat().st_size > _DOCUMENT_BYTES:
        raise ValueError("build product metadata exceeds its byte bound")
    return json.loads(path.read_bytes())


def _key(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("build products require their complete selected SHA256 identity")
    return value


def _eligible(selection, inputs, outputs):
    """Only original LLVM object compilation with explicit architecture/ABI."""
    from merlin.llvmlower import codegen

    identity = selection["identity"]
    tool = identity["tool"]
    selected = codegen.clang()
    requested = str(selected)
    request = selection["request"]
    environment = request.get("environment", request.get("env"))
    cwd = Path(request["cwd"])
    if Path(requested).is_absolute():
        resolved = Path(requested).resolve()
    elif os.path.dirname(requested):
        resolved = (cwd / requested).resolve()
    else:
        search = os.pathsep.join(str(cwd / entry) for entry in os.get_exec_path(environment))
        found = shutil.which(requested, path=search)
        resolved = Path(found).resolve() if found else None
    if (
        identity["stage"] != "object"
        or len(inputs) != 1
        or Path(inputs[0]).suffix != ".ll"
        or len(outputs) not in {1, 2}
        or Path(outputs[0]).suffix != ".o"
        or requested != tool["requested_path"]
        or resolved != Path(tool["path"])
    ):
        return False
    source = Path(codegen.__file__).resolve()
    if not any(
        row["path"] == str(source) and row["sha256"] == F.digest(source) for row in identity["owner_source_pins"]
    ):
        return False
    argv = identity["guest_argv"]
    if len(argv) < 5 or argv[-2] != "-o" or argv[-4] != "-c":
        return False
    flags = argv[1:-4]
    # Values describe the actual target selection. No target/model names or
    # shapes supply eligibility. Native CPU discovery and unknown options do.
    value_flags = ("--target=", "-march=", "-mabi=", "-mcmodel=")
    plain_flags = {"-ffreestanding", "-fno-builtin", "-fPIC", "-funroll-loops", "-fstack-usage"}
    for flag in flags:
        if flag in plain_flags or flag in {"-O0", "-O1", "-O2", "-O3", "-Os", "-Oz"}:
            continue
        if not any(flag.startswith(prefix) for prefix in value_flags):
            return False
        value = flag.partition("=")[2]
        if (
            value.lower() == "native"
            or not value
            or not all(char.isascii() and (char.isalnum() or char in "_+.-") for char in value)
        ):
            return False
    if not all(any(flag.startswith(prefix) for flag in flags) for prefix in ("--target=", "-march=", "-mabi=")):
        return False
    stack = "-fstack-usage" in flags
    return len(outputs) == (2 if stack else 1) and (
        not stack or Path(outputs[1]) == Path(outputs[0]).with_suffix(".su")
    )


def _restore(source, destination, expected, *, max_bytes):
    """Stream into a new inode; keep the original requested output parent."""
    source = F.canonical(source)
    destination = F.canonical(destination, exists=False)
    parent = destination.parent.stat()
    if destination.exists():
        before = destination.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("build product destination is linked or special")
    else:
        before = None
    temporary = destination.parent / (".reused-" + uuid.uuid4().hex)
    try:
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        digest, size = hashlib.sha256(), 0
        with os.fdopen(descriptor, "rb") as incoming, temporary.open("xb") as outgoing:
            original = os.fstat(incoming.fileno())
            if not stat.S_ISREG(original.st_mode):
                raise ValueError("build product source is special")
            while chunk := incoming.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError("build product exceeds its restored byte bound")
                digest.update(chunk)
                outgoing.write(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
            if F.identity(original) != F.identity(os.fstat(incoming.fileno())):
                raise ValueError("build product changed during restore")
        if digest.hexdigest() != expected["pin"]["sha256"] or size != expected["bytes"]:
            raise ValueError("restored build product differs from its authenticated original")
        after = destination.parent.stat()
        if (parent.st_dev, parent.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError("build product output parent changed")
        if before is not None and (not destination.exists() or F.identity(before) != F.identity(destination.lstat())):
            raise ValueError("build product output changed concurrently")
        if before is None and destination.exists():
            raise ValueError("build product output appeared concurrently")
        temporary.chmod(expected["mode"] & 0o777)
        os.replace(temporary, destination)
        if F.pin(destination, max_bytes=max_bytes)["sha256"] != expected["pin"]["sha256"]:
            raise ValueError("restored build product changed after replacement")
    finally:
        temporary.unlink(missing_ok=True)


class BuildProductIndex:
    """Exact live host owner; saved records and forked workers cannot publish."""

    def __init__(self, root, *, context, build_owner, max_entries, max_bytes, max_files, issuer):
        self.root, self.build_owner = root, build_owner
        self.context_sha256 = hashlib.sha256(F.document(context).encode()).hexdigest()
        self.max_entries, self.max_bytes, self.max_files = max_entries, max_bytes, max_files
        self._issuer, self._parent_pid, self._secret = issuer, os.getpid(), secrets.token_bytes(32)
        self._view_identity = build_owner.verify()
        value = root.stat()
        self._root_inode = (value.st_dev, value.st_ino)
        self._source_pin = F.pin(Path(__file__).resolve(), max_bytes=_DOCUMENT_BYTES)
        self._closed = False

    def verify(self, *, build_owner=None, context=None):
        if _ISSUED.get(self._issuer) is not self or self._closed:
            raise ValueError("build products need their exact live original parent owner")
        if build_owner is not None and build_owner is not self.build_owner:
            raise ValueError("build products changed their original build view owner")
        if context is not None and hashlib.sha256(F.document(context).encode()).hexdigest() != self.context_sha256:
            raise ValueError("build products changed their original admission context")
        if self.build_owner.verify() != self._view_identity:
            raise ValueError("build product original lookup view changed")
        value = F.canonical(self.root).stat()
        if (
            (value.st_dev, value.st_ino) != self._root_inode
            or value.st_uid != os.getuid()
            or value.st_mode & 0o077
            or F.pin(Path(__file__).resolve(), max_bytes=_DOCUMENT_BYTES) != self._source_pin
        ):
            raise ValueError("build product owner root or source changed")
        return {
            "context_sha256": self.context_sha256,
            "build_view": self._view_identity,
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
            "max_files": self.max_files,
            "scope": "original LLVM object/stack products; no new compiler invocation or verdict",
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

    def _document(self, path):
        wire = _read(path)
        if not hmac.compare_digest(wire["signature"], self._seal(wire["body"])["signature"]):
            raise ValueError("build product record was not issued by this original live owner")
        return wire["body"]

    def _signed_entry(self, path):
        body = self._document(path)
        if body["context_sha256"] != self.context_sha256 or not F.canonical(Path(body["job_root"])).is_relative_to(
            self.build_owner.output_root
        ):
            raise ValueError("build product record changed original admission")
        _key(body["key"])
        return body

    def _reopen(self, path):
        body = self._signed_entry(path)
        observed = self.build_owner.reopen_observation(body["observation"])
        if (
            observed["returncode"] != 0
            or observed["selection_key"] != body["key"]
            or observed["identity"] != body["identity"]
        ):
            raise ValueError("build product original process consumed another selected build")
        for item in body["products"]:
            path = Path(item["pin"]["path"])
            value = path.stat()
            if (
                F.pin(path, max_bytes=self.max_bytes) != item["pin"]
                or value.st_nlink != 1
                or value.st_size != item["bytes"]
                or stat.S_IMODE(value.st_mode) != item["mode"]
            ):
                raise ValueError("original build product changed")
        if [item["pin"] for item in body["products"]] != observed["outputs"]:
            raise ValueError("build product record does not contain the complete original output roster")
        return body

    def _job_bytes(self, job_root):
        size = 0
        for path in F.members(job_root, max_files=self.max_files):
            if not path.is_symlink() and path.is_file():
                size += path.stat().st_size
            if size > self.max_bytes:
                raise ValueError("retained original build job exceeds its byte budget")
        return size

    def _parent(self):
        self.verify()
        if os.getpid() != self._parent_pid:
            raise ValueError("only the creating host parent may finalize build products")

    def _request(self, view, selection, inputs, outputs, job_root):
        self.verify(build_owner=view)
        view.reopen_selection(selection)
        _key(selection["key"])
        for role, paths in (("inputs", inputs), ("outputs", outputs)):
            if tuple(map(Path, selection["request"][role])) != tuple(map(Path, paths)):
                raise ValueError("build product paths differ from the exact issued process request")
        job_root = F.canonical(job_root)
        if not job_root.is_relative_to(view.output_root):
            raise ValueError("build products escaped their original job output owner")
        for path in (*inputs, *outputs):
            if not F.canonical(Path(path), exists=False).is_relative_to(job_root):
                raise ValueError("build product input/output escaped its original job")
        if not _eligible(selection, inputs, outputs):
            return None
        return job_root

    def lookup(self, view, selection, *, inputs, outputs, job_root, directory, stage):
        job_root = self._request(view, selection, inputs, outputs, job_root)
        if job_root is None:
            return None
        if Path(selection["request"]["directory"]) != Path(directory) or selection["identity"]["stage"] != stage:
            raise ValueError("build product receipt escaped its original process request")
        with self._locked():
            path = self.root / "entries" / (selection["key"] + ".json")
            if not path.exists():
                return None
            body = self._reopen(path)
            if body["identity"] != selection["identity"] or len(body["products"]) != len(outputs):
                raise ValueError("build products differ from the exact current closed selection")
            # One source-job pin per consuming job, held through its actual
            # parent cleanup even when the cache entry is later evicted.
            name = hashlib.sha256((str(job_root) + "\0" + body["job_root"]).encode()).hexdigest()
            read_path = self.root / "reads" / (name + ".json")
            if not read_path.exists():
                if len(F.children(self.root / "reads", limit=self.max_entries + 1)) >= self.max_entries:
                    raise ValueError("active build product readers exceed their bounded capacity")
                token = storage.pin(
                    body["job_root"], reason="active reused build product; original job cleanup pending"
                )
                try:
                    _write(
                        read_path, self._seal({"job_root": str(job_root), "source": body["job_root"], "token": token})
                    )
                except BaseException:
                    storage.unpin(body["job_root"], token)
                    raise
            else:
                self._document(read_path)
            for destination, product in zip(outputs, body["products"], strict=True):
                _restore(Path(product["pin"]["path"]), Path(destination), product, max_bytes=self.max_bytes)
            view.reopen_selection(selection)
            self._reopen(path)
            original = body["observation"]["body"]
            record = Path(original["record"]["path"])
            receipt = I.record_artifact_reuse(
                directory,
                stage=stage,
                selection_sha256=selection["key"],
                original_invocation=record,
                inputs=inputs,
                outputs=outputs,
            )
            request = selection["request"]
            stdout, stderr = (Path(original[name]["path"]).read_bytes() for name in ("stdout", "stderr"))
            if request.get("text"):
                encoding = request.get("encoding") or locale.getencoding()
                errors = request.get("errors") or "strict"
                stdout, stderr = (
                    io.TextIOWrapper(io.BytesIO(data), encoding=encoding, errors=errors).read()
                    for data in (stdout, stderr)
                )
            result = subprocess.CompletedProcess(request["argv"], 0, stdout, stderr)
            result.artifact_reuse_receipt = str(receipt)
            return result

    def stage_observation(self, view, selection, *, inputs, outputs, observation, job_root):
        job_root = self._request(view, selection, inputs, outputs, job_root)
        if job_root is None:
            return
        observed = view.reopen_observation(observation, selection=selection, outputs=outputs)
        if observed["returncode"] != 0 or observed["selection_key"] != selection["key"]:
            raise ValueError("build products have no actual completed original process")
        if len(observed["outputs"]) != len(outputs):
            raise ValueError("build products lack the complete original output roster")
        products = []
        for output, pin in zip(outputs, observed["outputs"], strict=True):
            output = F.canonical(output)
            value = output.stat()
            if value.st_nlink != 1 or F.pin(output, max_bytes=self.max_bytes) != pin:
                raise ValueError("build product differs from its closed original process")
            products.append({"pin": pin, "bytes": value.st_size, "mode": stat.S_IMODE(value.st_mode)})
        size = self._job_bytes(job_root)
        body = {
            "context_sha256": self.context_sha256,
            "key": selection["key"],
            "identity": selection["identity"],
            "observation": observation,
            "products": products,
            "job_root": str(job_root),
            "bytes": size,
        }
        name = hashlib.sha256((str(job_root) + "\0" + selection["key"]).encode()).hexdigest()
        pending = self.root / "pending" / (name + ".json")
        with self._locked():
            if pending.exists():
                self._reopen(pending)
                return
            retained = [
                self._reopen(self.root / kind / name)
                for kind in ("pending", "entries")
                for name in F.children(self.root / kind, limit=self.max_entries + 1)
            ]
            if len(retained) >= self.max_entries or sum(row["bytes"] for row in retained) + size > self.max_bytes:
                raise ValueError("build product archive exceeds its retained entry/byte budget")
            body["pin_token"] = storage.pin(job_root, reason="pending original compiler products; cleanup unpublished")
            try:
                _write(pending, self._seal(body))
            except BaseException:
                storage.unpin(job_root, body["pin_token"])
                raise

    def _release_reads(self, job_root=None):
        for name in F.children(self.root / "reads", limit=self.max_entries + 1):
            path = self.root / "reads" / name
            body = self._document(path)
            if job_root is None or Path(body["job_root"]).is_relative_to(job_root):
                storage.unpin(body["source"], body["token"])
                path.unlink()

    def publish_completed_job(self, job_root):
        """Parent calls only after actual guardian cleanup and durable release."""
        self._parent()
        job_root, count = F.canonical(job_root), 0
        with self._locked():
            entries = self.root / "entries"
            retained = []
            for name in F.children(entries, limit=self.max_entries + 1):
                path = entries / name
                body = self._signed_entry(path)
                try:
                    retained.append(self._reopen(path))
                except (ValueError, OSError, KeyError, TypeError):
                    # Entry presence was issued only by this parent after its
                    # original job cleanup. Active consumers hold separate pins.
                    storage.unpin(body["job_root"], body["pin_token"])
                    path.unlink()
            for name in F.children(self.root / "pending", limit=self.max_entries + 1):
                path = self.root / "pending" / name
                body = self._signed_entry(path)
                if not Path(body["job_root"]).is_relative_to(job_root):
                    continue
                try:
                    self._reopen(path)
                except (ValueError, OSError, KeyError, TypeError):
                    # Original stack repair may overwrite a staged object.
                    # Its bytes are ineligible; our authenticated retention
                    # token still belongs to this proven-cleaned job.
                    storage.unpin(body["job_root"], body["pin_token"])
                    path.unlink()
                    continue
                target = entries / (body["key"] + ".json")
                try:
                    body["bytes"] = self._job_bytes(Path(body["job_root"]))
                except ValueError:
                    storage.unpin(body["job_root"], body["pin_token"])
                    path.unlink()
                    self._release_reads(job_root)
                    raise
                if target.exists():
                    storage.unpin(body["job_root"], body["pin_token"])
                    path.unlink()
                    continue
                while retained and (
                    len(retained) >= self.max_entries
                    or sum(row["bytes"] for row in retained) + body["bytes"] > self.max_bytes
                ):
                    old = retained.pop(0)
                    storage.unpin(old["job_root"], old["pin_token"])
                    (entries / (old["key"] + ".json")).unlink()
                _write(target, self._seal(body))
                retained.append(body)
                path.unlink()
                count += 1
            self._release_reads(job_root)
        return count

    def discard_cleaned_job(self, job_root):
        """Pending/read pins only; UNKNOWN cleanup must never call this seam."""
        self._parent()
        job_root, count = F.canonical(job_root), 0
        with self._locked():
            for name in F.children(self.root / "pending", limit=self.max_entries + 1):
                path = self.root / "pending" / name
                body = self._signed_entry(path)
                if Path(body["job_root"]).is_relative_to(job_root):
                    storage.unpin(body["job_root"], body["pin_token"])
                    path.unlink()
                    count += 1
            self._release_reads(job_root)
        return count

    def close(self):
        """Creating host closes only after all original job owners proved cleanup."""
        self._parent()
        with self._locked():
            for kind in ("entries", "pending"):
                for name in F.children(self.root / kind, limit=self.max_entries + 1):
                    path = self.root / kind / name
                    body = self._signed_entry(path)
                    storage.unpin(body["job_root"], body["pin_token"])
                    path.unlink()
            self._release_reads()
        self._closed = True


def prepare_build_product_index(root, *, context, build_owner, max_entries, max_bytes, max_files):
    from merlin.runtime.build_runtime import BuildRuntimeView

    if type(build_owner) is not BuildRuntimeView or any(
        type(v) is not int or v <= 0 for v in (max_entries, max_bytes, max_files)
    ):
        raise ValueError("build product archive needs live original authority and explicit positive bounds")
    build_owner.verify()
    root = F.canonical(root, exists=False)
    if root.exists() or not root.is_relative_to(build_owner.output_root):
        raise ValueError("build product archive needs a fresh original output root")
    root.mkdir(mode=0o700)
    for kind in ("pending", "entries", "reads"):
        (root / kind).mkdir(mode=0o700)
    issuer = object()
    index = BuildProductIndex(
        root,
        context=context,
        build_owner=build_owner,
        max_entries=max_entries,
        max_bytes=max_bytes,
        max_files=max_files,
        issuer=issuer,
    )
    _ISSUED[issuer] = index
    index.verify()
    return index


@contextlib.contextmanager
def executing_build_products(index=None, *, job_root=None):
    if index is not None:
        if type(index) is not BuildProductIndex or job_root is None:
            raise ValueError("build products need their exact original index and job output")
        index.verify()
        # The fixed evaluator creates the fresh job directory after entering
        # this context. Actual lookup/publication rechecks its existing owner.
        job_root = F.canonical(job_root, exists=False)
        if not job_root.is_relative_to(index.build_owner.output_root):
            raise ValueError("build product job escaped its original output owner")
    token = _ACTIVE.set(None if index is None else (index, job_root))
    try:
        yield index
    finally:
        _ACTIVE.reset(token)


def lookup_build_products(view, selection, inputs, outputs, directory, stage):
    active = _ACTIVE.get()
    if active is None:
        return None
    index, job_root = active
    try:
        return index.lookup(
            view, selection, inputs=inputs, outputs=outputs, job_root=job_root, directory=directory, stage=stage
        )
    except (ValueError, OSError, KeyError, TypeError):
        return None  # Unknown cache state executes the original selected process freshly.


def stage_build_products(view, selection, inputs, outputs, observation):
    active = _ACTIVE.get()
    if active is not None:
        index, job_root = active
        try:
            index.stage_observation(
                view, selection, inputs=inputs, outputs=outputs, observation=observation, job_root=job_root
            )
        except (ValueError, OSError, KeyError, TypeError):
            pass  # An unavailable cache cannot alter the original completed build.
