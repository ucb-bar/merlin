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
_FD_OVERHEAD = 4  # Custodian channel, pending input, hash duplicate and IPC setup.
_LOCK = threading.RLock()
_CHANGED = threading.Condition(_LOCK)
_CACHE = None
_VERIFICATION = contextvars.ContextVar("read_lease_digest_verification", default=None)


def _identity(metadata):
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_uid, metadata.st_gid


def _descriptor_budget():
    import resource

    try:
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft == resource.RLIM_INFINITY:
            return MAX_FILES
        opened = len(os.listdir('/proc/self/fd'))
        return max(0, min(MAX_FILES, soft - opened - FD_HEADROOM - _FD_OVERHEAD))
    except OSError:
        return 0


def enabled():
    return (os.environ.get("MERLIN_SNAPSHOT_LEASE_CACHE") == "1" and sys.platform == "linux"
            and fcntl is not None and hasattr(fcntl, "F_SETLEASE"))


class _Cache:
    def __init__(self):
        self.owner = os.getpid()
        self.entries, self.paths, self.unsupported = {}, {}, set()
        self.pending = set()
        self.process, self.channel, self.disabled = None, None, False
        self.capacity = _descriptor_budget()
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
                pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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
            current = path.stat(follow_symlinks=False)
            return (
                path.is_absolute() and path.resolve() == path
                and stat.S_ISREG(current.st_mode)
                and _identity(current) == original == _identity(os.fstat(descriptor))
                and fcntl.fcntl(descriptor, fcntl.F_GETLEASE) == fcntl.F_RDLCK
                and self.process.poll() is None
            )
        except OSError:
            return False

    def digest(self, path, metadata):
        """Admit a miss under the creation lock; existing hits borrow outside it."""
        key = metadata.st_dev, metadata.st_ino
        if self.disabled or self.retiring or key in self.unsupported or len(self.entries) >= self.capacity:
            return None
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size < MIN_BYTES:
            return None
        if not path.is_absolute() or path.resolve() != path:
            raise RuntimeError("original snapshot read lease requires a canonical file")
        if _descriptor_budget() < 1:
            return None
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        self.pending.add(descriptor)
        try:
            if self.process is None:
                self.start()
            elif self.process.poll() is not None:
                self.refuse()
            self.channel.sendmsg(
                [b"add"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [descriptor]))]
            )
            response = self.channel.recv(32)
            if response != b"ok":
                if len(self.unsupported) < MAX_FILES * 16:
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


def digest(path: Path):
    """Return a leased SHA-256 or None for the caller's existing full-read path."""
    global _CACHE
    if not enabled():
        return None
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
            entry = cache.digest(path, metadata)
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
        if path is not None and len(cache.paths) < MAX_FILES * 16:
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
        good = all(cache.valid(path, entry) for path, (_, entry) in records)
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
                    descriptors.frombytes(data[:len(data) - len(data) % descriptors.itemsize])
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
    os.register_at_fork(before=lambda: _LOCK.acquire(), after_in_parent=lambda: _LOCK.release(),
                        after_in_child=_after_fork)
if __name__ == "__main__":
    _custodian(int(sys.argv[1]))
