"""Live kernel provenance for explicitly delegated parent qualification.

The parent retains the exact readonly sealed kernel object while its child
redeems qualification. Matching JSON or a separately sealed copy is insufficient.
This proves custody only; original target owners must reopen their source bytes.
"""

from __future__ import annotations

import contextlib
import contextvars
import fcntl
import os
import threading
import weakref
from dataclasses import dataclass

from merlin_experiments.phase2.contracts import StageGateError

_ACTIVE = contextvars.ContextVar("verified_original_parent_pass_proof", default=None)
_ISSUED = weakref.WeakSet()
_KEY = object()


@dataclass(frozen=True, eq=False)
class _ParentProof:
    document: object
    identity: tuple
    process: object
    thread: object
    key: object

    def verify(self):
        from . import compiler_profile_child as C

        if (type(self) is not _ParentProof or self not in _ISSUED or self.key is not _KEY
                or _ACTIVE.get() is not self or threading.current_thread() is not self.thread
                or C._process(os.getpid()) != self.process):
            raise StageGateError("qualification delegation needs its live original sealed-parent proof")
        document = self.document
        if document["parent"] != C._process(os.getppid()):
            raise StageGateError("qualification delegation lost its actual original parent")
        descriptor = document.get("parent_delegation_fd")
        if type(descriptor) is not int or descriptor < 3:
            raise StageGateError("qualification delegation has no exact parent-held descriptor")
        fd = os.open(f"/proc/{os.getppid()}/fd/{descriptor}", os.O_RDONLY | os.O_CLOEXEC)
        try:
            stat = os.fstat(fd)
            if ((stat.st_dev, stat.st_ino) != self.identity
                    or fcntl.fcntl(fd, C._GET_SEALS) & C._SEALS != C._SEALS
                    or C._decode_selection(os.pread(fd, C.MAX_BYTES + 1, 0)) != document):
                raise StageGateError("qualification delegation differs from the inherited sealed kernel object")
            # Opening a proc descriptor readonly does not prove the original
            # parent handle was readonly. Inspect that exact handle's fdinfo.
            info = C.Path(f"/proc/{os.getppid()}/fdinfo/{descriptor}").read_text()
            flags = [line.partition(":")[2].strip() for line in info.splitlines()
                     if line.partition(":")[0] == "flags"]
            if len(flags) != 1 or int(flags[0], 8) & os.O_ACCMODE != os.O_RDONLY:
                raise StageGateError("qualification delegation parent descriptor retains writable authority")
        finally:
            os.close(fd)
        C._verify_parent_delegation_inputs(document)
        return C.copy_record(document["parent_pass_delegation"])


@contextlib.contextmanager
def _verified_parent_scope(document, identity):
    from . import compiler_profile_child as C

    proof = _ParentProof(C.copy_record(document), identity, C._process(os.getpid()),
                         threading.current_thread(), _KEY)
    _ISSUED.add(proof)
    previous = _ACTIVE.set(proof)
    try:
        proof.verify()
        yield proof
        proof.verify()
    finally:
        _ACTIVE.reset(previous)
        _ISSUED.discard(proof)


def verified_parent_pass_delegation(proof):
    if type(proof) is not _ParentProof:
        raise StageGateError("ordinary JSON cannot supply original parent qualification")
    return proof.verify()
