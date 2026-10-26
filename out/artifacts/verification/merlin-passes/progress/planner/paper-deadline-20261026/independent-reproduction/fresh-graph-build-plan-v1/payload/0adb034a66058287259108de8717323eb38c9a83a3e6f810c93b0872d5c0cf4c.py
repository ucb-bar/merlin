"""Bounded SHA-256 reuse while Linux read leases protect exact open files.

Only a dedicated child receives SIGIO. Unsupported files retain ordinary full
hashing; loss of an already issued lease refuses the current verification.
This does not authenticate paths, archives, member rosters or compiler results.
"""

from __future__ import annotations

import array
import atexit
import contextlib
import contextvars
import hashlib
import os
import socket
import stat
import subprocess
import sys
import threading
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows and other non-Linux hosts retain full hashing.
    fcntl = None

MIN_BYTES = 256 << 10
MAX_FILES = 4096
FD_HEADROOM = 64
FD_RECHECK_INTERVAL = 32
_FD_OVERHEAD = 4  # Custodian channel, pending input, hash duplicate and IPC setup.
_LOCK = threading.RLock()
_CHANGED = threading.Condition(_LOCK)
_CACHE = None
_VERIFICATION = contextvars.ContextVar("read_lease_digest_verification", default=None)


def _bounded_setting(name, default, maximum):
    raw = os.environ.get(name)
    if raw is None:
        return default
    if not raw.isascii() or not raw.isdecimal() or not 0 <= int(raw) <= maximum:
        raise ValueError(f"{name} must be an integer between 0 and {maximum}")
    return int(raw)


def minimum_bytes():
    """Operator-selected admission threshold; every hit still needs a live lease."""
    return _bounded_setting("MERLIN_SNAPSHOT_LEASE_MIN_BYTES", MIN_BYTES, 1 << 30)


def maximum_files():
    return _bounded_setting("MERLIN_SNAPSHOT_LEASE_MAX_FILES", MAX_FILES, 1 << 16)


def _identity(metadata):
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_uid, metadata.st_gid


def _descriptor_budget():
    import resource

    try:
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft == resource.RLIM_INFINITY:
            return maximum_files()
        opened = len(os.listdir("/proc/self/fd"))
        # A long-lived grader opens subprocess pipes and concurrent imports after
        # cache creation. Preserve a share of its limit for those ordinary uses.
        reserve = max(FD_HEADROOM, soft // 4)
        return max(0, min(maximum_files(), soft - opened - reserve - _FD_OVERHEAD))
    except OSError:
        return 0


def enabled():
    return (
        os.environ.get("MERLIN_SNAPSHOT_LEASE_CACHE") == "1"
        and sys.platform == "linux"
        and fcntl is not None
        and hasattr(fcntl, "F_SETLEASE")
    )


class _Cache:
    def __init__(self):
        self.owner = os.getpid()
        self.entries, self.paths, self.unsupported = {}, {}, set()
        self.pending = set()
        self.process, self.channel, self.disabled = None, None, False
        self.capacity = _descriptor_budget()
        self.admissions_until_budget_check = 0
        self.minimum_bytes = minimum_bytes()
        self.readers, self.retiring = 0, False

    def close(self, *, inherited=False):
        # Callers hold _LOCK. Retirement forbids new readers before waiting;
        # Condition releases the mutex so borrowed descriptors can drain.
        self.retiring = True
        if not inherited:
            while self.readers:
                _CHANGED.wait()
        else:
            # Other threads do not survive fork; close only inherited copies.
            self.readers = 0
        for entry in self.entries.values():
            os.close(entry[0])
        for descriptor in self.pending:
            os.close(descriptor)
        self.entries.clear()
        self.paths.clear()
        self.pending.clear()
        if self.channel is not None:
            self.channel.close()
            self.channel = None
        if self.process is not None and not inherited:
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1)
        self.process = None

    def refuse(self):
        self.disabled = True
        self.close()
        raise RuntimeError("original snapshot read lease custody changed")

    def start(self):
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-B", str(Path(__file__).resolve()), str(child.fileno())],
                pass_fds=(child.fileno(),),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            parent.settimeout(5)
            self.channel = parent
        except BaseException:
            parent.close()
            raise
        finally:
            child.close()

    def valid(self, path, entry):
        descriptor, _, original = entry
        try:
            if self.process is None or self.process.poll() is not None:
                return False
            if fcntl.fcntl(descriptor, fcntl.F_GETLEASE) != fcntl.F_RDLCK:
                return False
            if not path.is_absolute():
                return False
            # O_PATH pins this namespace lookup without opening a reader or
            # writer. /proc exposes the kernel's resolved ordinary pathname;
            # parent/final symlinks, replacement and deleted names refuse.
            # This replaces a Python realpath walk on every cached member.
            current_fd = os.open(path, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                current = os.fstat(current_fd)
                return (
                    os.readlink(f"/proc/self/fd/{current_fd}") == str(path)
                    and stat.S_ISREG(current.st_mode)
                    and _identity(current) == original == _identity(os.fstat(descriptor))
                    and fcntl.fcntl(descriptor, fcntl.F_GETLEASE) == fcntl.F_RDLCK
                    and self.process.poll() is None
                )
            finally:
                os.close(current_fd)
        except OSError:
            return False

    def valid_batch(self, members):
        """``valid`` for many borrowed entries with one custodian check per batch.

        Each member repeats the O_PATH kernel-resolved name, regular-file and
        identity comparison. Every lease is then read again after all names are
        checked: a lease never returns to an admitted descriptor, so a held lease
        at the end proves no writer opened the file since its original hash.
        """
        try:
            if self.process is None or self.process.poll() is not None:
                return False
            for path, (descriptor, _, original) in members:
                if not path.is_absolute():
                    return False
                current_fd = os.open(path, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC)
                try:
                    current = os.fstat(current_fd)
                    if (
                        os.readlink(f"/proc/self/fd/{current_fd}") != str(path)
                        or not stat.S_ISREG(current.st_mode)
                        or _identity(current) != original
                        or fcntl.fcntl(descriptor, fcntl.F_GETLEASE) != fcntl.F_RDLCK
                    ):
                        return False
                finally:
                    os.close(current_fd)
            return (
                all(fcntl.fcntl(entry[0], fcntl.F_GETLEASE) == fcntl.F_RDLCK for _, entry in members)
                and self.process.poll() is None
            )
        except OSError:
            return False

    def digest(self, path, metadata, minimum_bytes=None):
        """Admit a miss under the creation lock; existing hits borrow outside it."""
        key = metadata.st_dev, metadata.st_ino
        if self.disabled or self.retiring or key in self.unsupported or len(self.entries) >= self.capacity:
            return None
        minimum = self.minimum_bytes if minimum_bytes is None else minimum_bytes
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size < minimum:
            return None
        if not path.is_absolute() or path.resolve() != path:
            raise RuntimeError("original snapshot read lease requires a canonical file")
        # Recheck in bounded batches as other grader work opens descriptors.
        # Existing leases keep their custody; declined new members use full reads.
        if self.admissions_until_budget_check == 0:
            self.capacity = min(self.capacity, len(self.entries) + _descriptor_budget())
            self.admissions_until_budget_check = FD_RECHECK_INTERVAL
        self.admissions_until_budget_check -= 1
        if len(self.entries) >= self.capacity:
            return None
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        self.pending.add(descriptor)
        try:
            if self.process is None:
                self.start()
            elif self.process.poll() is not None:
                self.refuse()
            self.channel.sendmsg([b"add"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [descriptor]))])
            response = self.channel.recv(32)
            if response != b"ok":
                if len(self.unsupported) < maximum_files() * 16:
                    self.unsupported.add(key)
                return None
            entry = descriptor, None, _identity(os.fstat(descriptor))
            if not self.valid(path, entry):
                self.refuse()
            with os.fdopen(os.dup(descriptor), "rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if not self.valid(path, entry):
                self.refuse()
            self.entries[key] = descriptor, digest, entry[2]
            self.paths[path] = key
            self.pending.remove(descriptor)
            return self.entries[key]
        except (OSError, subprocess.SubprocessError):
            if self.entries:
                self.refuse()
            self.disabled = True
            self.close()
            return None
        finally:
            if descriptor in self.pending:
                self.pending.remove(descriptor)
                os.close(descriptor)


def digest(path: Path, *, minimum_bytes: int | None = None):
    """Return a leased SHA-256 or None for the caller's existing full-read path.

    ``minimum_bytes`` lets a caller that re-verifies one closed pin set admit
    smaller files than the operator default; capacity and custody are unchanged.
    """
    global _CACHE
    if not enabled():
        return None
    if minimum_bytes is not None and (type(minimum_bytes) is not int or minimum_bytes < 0):
        raise ValueError("lease admission threshold must be a nonnegative integer")
    path = Path(path)
    metadata = path.stat(follow_symlinks=False)
    key = metadata.st_dev, metadata.st_ino
    with _LOCK:
        if _CACHE is None:
            _CACHE = _Cache()
        if _CACHE.owner != os.getpid():
            _CACHE.close(inherited=True)
            _CACHE = _Cache()
        cache = _CACHE
        if path in cache.paths and cache.paths[path] != key:
            cache.refuse()
        entry = cache.entries.get(key)
        hit = entry is not None
        if hit:
            _borrow(cache)
        else:
            entry = cache.digest(path, metadata, minimum_bytes)
    if hit:
        good = False
        try:
            good = cache.valid(path, entry)
        finally:
            _return(cache, good, path=path, key=key)
    owner = _VERIFICATION.get()
    if entry is not None and owner is not None and owner[:2] == (os.getpid(), threading.current_thread()):
        owner[2][path] = cache, entry
    return None if entry is None else entry[1]


def reuse(paths, *, required: bool = False):
    """Batch the hits of exact paths already admitted by ``digest`` in this process.

    Returns ``{path: sha256}`` only after ``_Cache.valid_batch`` rechecks every
    returned member's name, identity and lease following all lookups. Paths
    never admitted are omitted for the caller's ordinary ``digest``/full read.
    A failed or ownerless member declines with ``None``, so the caller's
    ordinary per-file path reaches the existing refusal. ``required`` refuses
    instead unless every path is still a live hit; use it to recheck reused
    members before accepting a verification that also performed slower reads.
    Nothing is admitted, hashed or cached by mtime here.
    """
    # Keep caller Path objects: rebuilding them discards their parsed form.
    paths = [path if isinstance(path, Path) else Path(path) for path in paths]
    changed = "original snapshot read lease custody changed"
    if not enabled():
        if required:
            raise RuntimeError(changed)
        return None
    with _LOCK:
        cache = _CACHE
        if cache is None or cache.owner != os.getpid() or cache.disabled or cache.retiring:
            if required:
                raise RuntimeError(changed)
            return None
        members = []
        for path in paths:
            key = cache.paths.get(path)
            entry = None if key is None else cache.entries.get(key)
            if entry is not None:
                members.append((path, entry))
            elif required:
                raise RuntimeError(changed)
        if not members:
            return {}
        cache.readers += 1
    good = False
    try:
        good = cache.valid_batch(members)
    finally:
        with _LOCK:
            cache.readers -= 1
            _CHANGED.notify_all()
            good = good and cache is _CACHE and not cache.retiring and not cache.disabled
            if required and not good:
                cache.refuse()
    if not good:
        return None
    owner = _VERIFICATION.get()
    if owner is not None and owner[:2] == (os.getpid(), threading.current_thread()):
        for path, entry in members:
            owner[2][path] = cache, entry
    return {path: entry[1] for path, entry in members}


def _borrow(cache):
    # Only a short registry check/counter update holds the process-wide mutex.
    if cache is not _CACHE or cache.owner != os.getpid():
        raise RuntimeError("original snapshot read lease owner changed")
    if cache.retiring or cache.disabled:
        raise RuntimeError("original snapshot read lease custody changed")
    cache.readers += 1


def _return(cache, good, *, path=None, key=None):
    with _LOCK:
        cache.readers -= 1
        _CHANGED.notify_all()
        if not good or cache.retiring or cache.disabled or cache is not _CACHE:
            cache.refuse()
        if path is not None and len(cache.paths) < maximum_files() * 16:
            cache.paths[path] = key


@contextlib.contextmanager
def verification():
    """Recheck every selected lease before this payload verification returns."""
    owner = os.getpid(), threading.current_thread(), {}
    previous = _VERIFICATION.set(owner)
    try:
        yield owner
    finally:
        try:
            # The surrounding snapshot verifier acquires admission before a
            # compute slot. Lease only this synchronous custody recheck, so a
            # verifier waiting on admission never reserves a CPU.
            from .cpu_pool import shared_compute_pool

            with shared_compute_pool():
                validate(owner)
            parent = previous.old_value
            if parent is not contextvars.Token.MISSING and parent is not None and parent[:2] == owner[:2]:
                parent[2].update(owner[2])
        finally:
            _VERIFICATION.reset(previous)


def validate(owner):
    """A previously used lease remains mandatory even after cache invalidation."""
    if owner is None:
        return
    records = tuple(owner[2].items())
    if not records:
        return
    cache = records[0][1][0]
    with _LOCK:
        if owner[:2] != (os.getpid(), threading.current_thread()) or any(row[1][0] is not cache for row in records):
            raise RuntimeError("original snapshot read lease owner changed")
        _borrow(cache)
    good = False
    try:
        # These are precisely this scope's borrowed entries, not the cache's
        # full roster. The existing batch path checks every namespace/identity
        # and every held lease, then rechecks all leases after the name walk.
        good = cache.valid_batch([(path, entry) for path, (_, entry) in records])
    finally:
        _return(cache, good)


def close():
    """Release this process's cache; never signal another process's custodian."""
    global _CACHE
    with _LOCK:
        if _CACHE is not None:
            _CACHE.close(inherited=_CACHE.owner != os.getpid())
            _CACHE = None


def _after_fork():
    global _CACHE, _LOCK, _CHANGED
    if _CACHE is not None:
        _CACHE.close(inherited=True)
    _CACHE, _LOCK = None, threading.RLock()
    _CHANGED = threading.Condition(_LOCK)


def _custodian(channel_fd):
    import signal

    channel = socket.socket(fileno=channel_fd)
    held = set()
    capacity = _descriptor_budget()

    def release(*_):
        for descriptor in tuple(held):
            held.discard(descriptor)
            try:
                fcntl.fcntl(descriptor, fcntl.F_SETLEASE, fcntl.F_UNLCK)
            except OSError:
                pass
            try:
                os.close(descriptor)
            except OSError:
                pass

    signal.signal(signal.SIGIO, release)
    try:
        while True:
            message, ancillary, _, _ = channel.recvmsg(32, socket.CMSG_SPACE(array.array("i").itemsize))
            if not message:
                break
            descriptors = array.array("i")
            for level, kind, data in ancillary:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    descriptors.frombytes(data[: len(data) - len(data) % descriptors.itemsize])
            if message != b"add" or len(descriptors) != 1 or len(held) >= capacity:
                for descriptor in descriptors:
                    os.close(descriptor)
                channel.send(b"unsupported")
                continue
            descriptor = descriptors[0]
            held.add(descriptor)
            try:
                fcntl.fcntl(descriptor, fcntl.F_SETOWN, os.getpid())
                fcntl.fcntl(descriptor, fcntl.F_SETLEASE, fcntl.F_RDLCK)
            except OSError:
                if descriptor in held:
                    held.remove(descriptor)
                    os.close(descriptor)
                channel.send(b"unsupported")
            else:
                channel.send(b"ok")
    finally:
        release()
        channel.close()


atexit.register(close)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=lambda: _LOCK.acquire(), after_in_parent=lambda: _LOCK.release(), after_in_child=_after_fork
    )
if __name__ == "__main__":
    _custodian(int(sys.argv[1]))
