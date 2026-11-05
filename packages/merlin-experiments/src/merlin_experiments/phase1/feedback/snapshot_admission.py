"""Private process-shared admission for two immutable compiler identities.

This limits work, not correctness or access. Every admitted request still runs
its ordinary source, sandbox, artifact and numerical checks. Equal identities
may share a slot; this permits nested host consumers without an extra slot.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import time


def identity(service, candidate):
    from merlin.targetgen.compiler_library import active_compiler_library
    from merlin_experiments.phase2.retained_revisions import source_inventory

    library = active_compiler_library()
    return hashlib.sha256(json.dumps({
        'source': source_inventory(Path(candidate), max_bytes=32 << 20,
                                   max_files=8192, exclude_generated=True)['sha256'],
        'snapshot': service.snapshot_digest,
        'profile': service.profile['sha256'],
        'library': library.record()['contract_sha256'] if library is not None else None,
    }, sort_keys=True).encode()).hexdigest()


def _open(root, name):
    fd = os.open(root / name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    value = os.fstat(fd)
    if (not stat.S_ISREG(value.st_mode) or value.st_uid != os.geteuid()
            or value.st_nlink != 1 or stat.S_IMODE(value.st_mode) != 0o600):
        os.close(fd)
        raise ValueError('snapshot admission needs a private ordinary slot file')
    return fd


@contextmanager
def admitted(root, key, *, timeout_s=1200, stop_reason=None):
    """Hold one of two identity slots without retaining a compute lease.

    The registry lock serializes slot allocation only. Shared slot locks remain
    held for every live borrower and are released by the kernel on process exit.
    No environment variable or candidate-authored declaration bypasses admission.
    """
    from merlin.common.cpu_pool import current_thread_owns_cpu_lease

    if current_thread_owns_cpu_lease():
        raise RuntimeError('snapshot admission cannot wait while holding a CPU lease')
    if (not isinstance(key, str) or len(key) != 64
            or any(c not in '0123456789abcdef' for c in key)):
        raise ValueError('snapshot admission needs its exact native identity')
    if type(timeout_s) not in (int, float) or not 0 < timeout_s <= 86400:
        raise ValueError('snapshot admission timeout must be bounded')
    root = Path(root)
    if not root.is_absolute() or root.resolve() != root or root.is_symlink():
        raise ValueError('snapshot admission root must be canonical')
    root.mkdir(mode=0o700, parents=False, exist_ok=True)
    info = root.stat()
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError('snapshot admission root must be private')
    registry = _open(root, 'registry.lock')
    slots = []
    held = None
    started = time.monotonic()
    try:
        for i in range(2):
            slots.append(_open(root, f'slot-{i}'))
        while held is None:
            reason = stop_reason() if stop_reason is not None else None
            if reason:
                raise RuntimeError(f'snapshot admission stopped: {reason}')
            fcntl.flock(registry, fcntl.LOCK_EX)
            try:
                # First join an existing equal identity, including another
                # process's nested host consumer of this exact compiler.
                for index, fd in enumerate(slots):
                    try:
                        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    if os.pread(fd, 128, 0).decode() == key:
                        held = index
                        break
                    fcntl.flock(fd, fcntl.LOCK_UN)
                if held is None:
                    for index, fd in enumerate(slots):
                        try:
                            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            continue
                        os.ftruncate(fd, 0)
                        os.pwrite(fd, key.encode(), 0)
                        fcntl.flock(fd, fcntl.LOCK_SH)
                        held = index
                        break
            finally:
                fcntl.flock(registry, fcntl.LOCK_UN)
            if held is None:
                left = timeout_s - (time.monotonic() - started)
                if left <= 0:
                    raise TimeoutError('two immutable compiler identities are still active')
                time.sleep(min(0.1, left))
        yield {'identity_sha256': key, 'slot': held,
               'wait_s': time.monotonic() - started, 'authority': 'scheduling_only'}
    finally:
        for fd in slots:
            os.close(fd)
        os.close(registry)
