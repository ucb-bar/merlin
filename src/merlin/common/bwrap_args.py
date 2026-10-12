"""Exact large bwrap argv transport over a private sealed read-only descriptor.

No argument file enters a candidate workspace. bwrap consumes and closes its
``--args`` policy descriptor before executing the separate original payload.
The caller supplies that exact boundary; command operands are never interpreted
as policy. Observations retain the
logical argv and transport byte identity; a saved FD number grants no authority.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import invocation_record

_INLINE_BYTES = 64 * 1024
_MAX_BYTES = 32 * 1024 * 1024
_MAX_ARGUMENTS = 262144
# Linux UAPI linux/fcntl.h. Compatible portable CPython builds can omit these
# constants and os.memfd_create; the fallback uses the same enforced kernel API.
_SEALS, _ADD_SEALS, _GET_SEALS = 0x000F, 1033, 1034


def _memfd():
    if sys.platform != "linux":
        raise ValueError("private bwrap argument transport requires Linux sealed descriptors")
    if hasattr(os, "memfd_create"):
        return os.memfd_create("merlin-private-bwrap-args", 0x0001 | 0x0002)
    import ctypes

    function = ctypes.CDLL(None, use_errno=True).memfd_create
    function.argtypes, function.restype = (ctypes.c_char_p, ctypes.c_uint), ctypes.c_int
    descriptor = function(b"merlin-private-bwrap-args", 0x0001 | 0x0002)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "private bwrap sealed descriptor creation failed")
    return descriptor


@dataclass(frozen=True)
class ArgumentTransport:
    logical_argv: tuple[str, ...]
    command: tuple[str, ...]
    descriptor: int | None
    payload_sha256: str
    payload_bytes: int
    policy_end: int

    @property
    def pass_fds(self):
        return () if self.descriptor is None else (self.descriptor,)

    def record(self):
        if self.descriptor is not None:
            if (fcntl.fcntl(self.descriptor, fcntl.F_GETFL) & os.O_ACCMODE) != os.O_RDONLY or (
                fcntl.fcntl(self.descriptor, _GET_SEALS) & _SEALS != _SEALS
            ):
                raise ValueError("private bwrap arguments lost sealed read-only ownership")
            if os.fstat(self.descriptor).st_size != self.payload_bytes:
                raise ValueError("private bwrap argument byte inventory changed")
        return {
            "schema": "merlin.bwrap_argument_transport.v1",
            "transport": "sealed_readonly_fd" if self.descriptor is not None else "inline",
            "payload_sha256": self.payload_sha256,
            "payload_bytes": self.payload_bytes,
            "argument_count": len(self.logical_argv) - 1,
            "encoded_argument_count": self.policy_end - 1,
            "scope": "exact logical argv transport; no source, semantic or execution qualification",
        }


@contextlib.contextmanager
def argument_transport(argv, *, policy_end=None):
    """Keep exact token order and bytes, privately bounded on the original host."""
    logical = tuple(argv)
    if (
        not logical
        or len(logical) > _MAX_ARGUMENTS
        or any(type(token) is not str or "\0" in token for token in logical)
    ):
        raise ValueError("bwrap argument transport requires a bounded original string argv without NUL")
    if not logical[0]:
        raise ValueError("bwrap argument transport has no original executable")
    if policy_end is not None and (type(policy_end) is not int or not 1 <= policy_end < len(logical)):
        raise ValueError("bwrap argument transport requires the original policy/payload boundary")
    end = len(logical) if policy_end is None else policy_end
    encoded, size = [], 0
    for token in logical[1:end]:
        raw = os.fsencode(token)
        size += len(raw) + 1
        if size > _MAX_BYTES:
            raise ValueError("private bwrap arguments exceed the original fixed byte budget")
        encoded.append(raw)
    payload = b"\0".join(encoded) + (b"\0" if encoded else b"")
    identity = hashlib.sha256(payload).hexdigest()
    if size <= _INLINE_BYTES:
        yield ArgumentTransport(logical, logical, None, identity, size, end)
        return
    if policy_end is None:
        raise ValueError("large bwrap argument transport requires the explicit original policy/payload boundary")
    if Path(logical[0]).name != "bwrap":
        raise ValueError("large argument descriptor transport requires the original bwrap executable")
    writable = readonly = None
    try:
        writable = _memfd()
        with os.fdopen(os.dup(writable), "wb") as stream:
            stream.write(payload)
        fcntl.fcntl(writable, _ADD_SEALS, _SEALS)
        readonly = os.open(f"/proc/self/fd/{writable}", os.O_RDONLY | os.O_CLOEXEC)
        os.close(writable)
        writable = None
        selected = ArgumentTransport(
            logical, (logical[0], "--args", str(readonly), *logical[end:]), readonly, identity, size, end
        )
        selected.record()
        yield selected
        selected.record()
    finally:
        for descriptor in (readonly, writable):
            if descriptor is not None:
                os.close(descriptor)


def run(argv, *, policy_end=None, directory=None, stage="bwrap", inputs=(), outputs=(), dependencies=(), **options):
    """Launch the same command and observe logical argv plus original transport."""
    if options.get("pass_fds") or options.get("close_fds") is False:
        raise ValueError("private bwrap transport cannot inherit unrelated original descriptors")
    options["env"] = dict(os.environ if options.get("env") is None else options["env"])
    with argument_transport(argv, policy_end=policy_end) as selected:
        options["pass_fds"] = selected.pass_fds
        observation = (
            contextlib.nullcontext(None)
            if directory is None
            else invocation_record.observe(
                directory,
                stage=stage,
                argv=selected.logical_argv,
                cwd=options.get("cwd"),
                env=options["env"],
                inputs=inputs,
                outputs=outputs,
                dependencies=(*dependencies, Path(__file__)),
            )
        )
        with observation as record:
            if record is not None:
                record.document["argument_transport"] = selected.record()
            actual = subprocess.run(list(selected.command), **options)
            result = subprocess.CompletedProcess(
                list(selected.logical_argv), actual.returncode, actual.stdout, actual.stderr
            )
            if record is not None:
                record.complete(result)
            return result
