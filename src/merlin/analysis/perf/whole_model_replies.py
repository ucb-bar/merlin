"""A compiler package's replies to a whole model's groups: recorded by content, asked in parallel.

Split out of :mod:`merlin.analysis.perf.whole_model_build` (which re-exports these names). A package is a
function of its own files and its input, so a reply recorded against both digests is what the package
would print again (:class:`_ReplyCache`). A model's statement asks one group at a time because each
splice binds against the buffers the previous ones declared; the ASKING does not depend on that, so
:class:`Asker` puts every group's interface to the package the moment the statement writes it, in
parallel, and -- when the target's object cache is on -- compiles each answered group's artifact to its
object in the same worker, so the slowest group's ask and compile start at once rather than after every
other group's. Nothing here names a target.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

#: The recorded-reply cache is bounded, least-recently-used first: it grew ~0.5 GB per 15 min of a live
#: run with no bound at all. A reply is a pure function of its key, so an evicted one is only asked again.
REPLIES_MAX_BYTES_ENV = "MERLIN_PACKAGE_REPLIES_MAX_BYTES"
REPLIES_DEFAULT_MAX_BYTES = 4 << 30


def _replies_max_bytes() -> int:
    raw = (os.environ.get(REPLIES_MAX_BYTES_ENV, "") or "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else REPLIES_DEFAULT_MAX_BYTES


#: Replies live under one folder per package (``<cache>/<package digest[:16]>/<key>``), so a finished
#: package's replies can be dropped as a unit.
REPLY_PACKAGE_PREFIX = 16


def replies_folder(directory: Path, package_digest: str) -> Path:
    return Path(directory) / str(package_digest)[:REPLY_PACKAGE_PREFIX]


def drop_package_replies(directory: Path, package_digest: str) -> int:
    """Remove every recorded reply of one package; returns the bytes freed. A reply is a pure function
    of its key, so a later build of the same package only asks again."""
    folder = replies_folder(directory, package_digest)
    if not folder.is_dir():
        return 0
    freed = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
    shutil.rmtree(folder, ignore_errors=True)
    return freed


def _reply_entries(directory: Path):
    """Every recorded reply entry, in the per-package layout (and any left in the flat one)."""
    for child in Path(directory).iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue  # a staging directory belongs to a writer in flight
        if (child / "reply.json").is_file():
            yield child  # the flat layout this cache used before
            continue
        for entry in child.iterdir():
            if entry.is_dir() and not entry.name.startswith("."):
                yield entry


def prune_replies(directory: Path, *, max_bytes: int | None = None) -> dict[str, Any]:
    """Evict whole reply entries, least recently used (entry mtime; a replay touches it) first, until the
    cache is under ``max_bytes``. Best-effort: a cache that cannot prune still works."""
    limit = _replies_max_bytes() if max_bytes is None else int(max_bytes)
    entries: list[tuple[float, int, Path]] = []
    try:
        for entry in _reply_entries(Path(directory)):
            size = sum(f.stat().st_size for f in entry.iterdir() if f.is_file())
            entries.append((entry.stat().st_mtime, size, entry))
    except OSError:
        return {"evicted": 0, "bytes_evicted": 0}
    total = sum(size for _mtime, size, _entry in entries)
    evicted = freed = 0
    for _mtime, size, entry in sorted(entries):
        if total <= limit:
            break
        shutil.rmtree(entry, ignore_errors=True)
        total -= size
        freed += size
        evicted += 1
    return {"evicted": evicted, "bytes_evicted": freed, "bytes_kept": total, "limit": limit}


class _ReplyCache:
    """A package's replies, keyed by the package's CONTENT, the entrypoint and the interface's bytes.

    A backend is a function of its own files and its input; a reply recorded against both digests is
    that function's value and is what the package would print again. Recorded so a model's seventy
    groups are asked once per package build rather than once per caller, and COUNTED, so a build
    record says how many of its replies were replayed rather than leaving it to be assumed.
    """

    def __init__(self, package_digest: str, directory: Path | None):
        self.package_digest = package_digest
        #: The whole cache (pruned as one); this package's replies live in their own folder under it.
        self.cache_root = directory
        self.directory = replies_folder(directory, package_digest) if directory is not None else None
        self.hits = 0
        self.misses = 0

    def _key(self, name: str, source: Path) -> str:
        digest = hashlib.sha256()
        for part in (self.package_digest, name, Path(source).read_bytes()):
            digest.update(part if isinstance(part, bytes) else str(part).encode())
            digest.update(b"\0")
        return digest.hexdigest()

    def replay(self, name, source, output=None) -> subprocess.CompletedProcess | None:
        """The recorded reply to this question, re-enacted (its output file written), or ``None``."""
        if self.directory is None:
            return None
        entry = self.directory / self._key(name, Path(source))
        record = entry / "reply.json"
        if not record.is_file():
            return None
        reply = json.loads(record.read_text(encoding="utf-8"))
        with contextlib.suppress(OSError):
            os.utime(entry)  # recently used: the bounded cache evicts least-recently-used first
        if output is not None and (entry / "output").is_file():
            Path(output).write_bytes((entry / "output").read_bytes())
        self.hits += 1
        return subprocess.CompletedProcess(reply["argv"], reply["returncode"], reply["stdout"], reply["stderr"])

    def invoke(self, package, name, source, output=None, *, timeout=600):
        from merlin.development.targetgen import package_runtime as OR

        if self.directory is None:
            return OR.run_entrypoint(package, name, source, output, timeout=timeout)
        replayed = self.replay(name, source, output)
        if replayed is not None:
            return replayed
        entry = self.directory / self._key(name, Path(source))
        done = OR.run_entrypoint(package, name, source, output, timeout=timeout)
        self.misses += 1
        staged = entry.with_name(f".{entry.name}.{os.getpid()}.{id(done)}")
        staged.mkdir(parents=True, exist_ok=True)
        if output is not None and Path(output).is_file():
            (staged / "output").write_bytes(Path(output).read_bytes())
        (staged / "reply.json").write_text(
            json.dumps(
                {
                    "argv": [str(a) for a in (done.args or [])],
                    "returncode": done.returncode,
                    "stdout": done.stdout,
                    "stderr": done.stderr,
                }
            ),
            encoding="utf-8",
        )
        try:
            staged.rename(entry)
        except OSError:  # another caller recorded the same reply first; theirs is the same value
            shutil.rmtree(staged, ignore_errors=True)  # and the half-written copy is ours to remove
        return done


class Asker:
    """Every group interface put to the package as soon as it is written, ``jobs`` at a time.

    ``recorded`` is the first statement pass's invoke: it replays a recorded reply, and for a question
    with no recorded answer it starts asking the package for the WHOLE interface in the background and
    refuses for now. ``invoke`` is the second pass's: it waits for a still-running ask of the same
    interface bytes, then replays its recorded replies (asking only what is still missing). The first
    pass therefore runs concurrently with every ask, and the second is a replay.

    ``compile_artifact(text)`` (optional) runs in the same worker once a group's target artifact is
    known -- a whole model's slowest group is usually slowest to compile as well, and that compile no
    longer waits for every other group to be asked first.
    """

    def __init__(self, package, cache: _ReplyCache, work: Path, *, timeout: int, jobs: int, compile_artifact=None):
        self.package, self.cache, self.work, self.timeout = package, cache, Path(work), timeout
        self.compile_artifact = compile_artifact
        self.pool = ThreadPoolExecutor(max_workers=max(1, jobs))
        self.pending: dict[str, Future] = {}
        self.missing: list[str] = []
        self._lock = threading.Lock()

    @staticmethod
    def _identity(source) -> str:
        return hashlib.sha256(Path(source).read_bytes()).hexdigest()

    def _ask(self, interface: Path, scratch: Path) -> None:
        from merlin.development.targetgen import capsule_common as CC

        try:
            _buffer, artifact = CC.lower_interface(
                self.package,
                interface,
                scratch,
                contract=None,
                timeout=self.timeout,
                invoke=self.cache.invoke,
                overlap=True,
            )
        except Exception:  # noqa: BLE001 -- a refusal is the statement's to record, in order
            return
        finally:
            with contextlib.suppress(OSError):
                interface.unlink()
        if self.compile_artifact is not None and artifact:
            with contextlib.suppress(Exception):  # noqa: BLE001 -- an optimization; the build compiles it
                self.compile_artifact(artifact)

    def recorded(self, package, name, source, output=None, *, timeout=600):  # noqa: ARG002
        reply = self.cache.replay(name, source, output)
        if reply is not None:
            return reply
        with self._lock:
            self.missing.append(str(name))
            key = self._identity(source)
            if key not in self.pending:
                # A copy of the bytes: the statement's own work tree is rewritten by its next pass.
                scratch = self.work / "prefetch" / key[:16]
                scratch.mkdir(parents=True, exist_ok=True)
                interface = scratch.with_suffix(".iface.mlir")
                shutil.copyfile(source, interface)
                self.pending[key] = self.pool.submit(self._ask, interface, scratch / "generated")
        return subprocess.CompletedProcess([], 1, "", "no recorded reply yet")

    def invoke(self, package, name, source, output=None, *, timeout=600):
        with self._lock:
            waiting = self.pending.get(self._identity(source))
        if waiting is not None:
            waiting.result()
        return self.cache.invoke(package, name, source, output, timeout=timeout)

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        # The asking's own work trees are scratch: every answer is recorded in the cache, and the
        # statement replays it into its own tree. Measured on SmolVLA (1551 groups) they held 7.9 GB.
        shutil.rmtree(self.work / "prefetch", ignore_errors=True)
