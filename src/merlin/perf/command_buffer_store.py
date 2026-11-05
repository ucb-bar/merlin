"""The expanded command buffer a graded result rests on, stored once beside the result.

A capsule result used to embed the whole expanded buffer -- the compiler's commands together with
every canonical input tensor it was graded on -- under ``command_buffer_artifact.command_buffer``.
For a decoder-scale case that is 88 MB of integer lists inside one JSON document (334 MB once
indented), and every reader of the result paid for it: the progress view, the score, the
host-compute veto and each consumer that only wanted the tensor shapes. Measured on one 132-case
scalability self-check, a single progress snapshot re-parsed 1.3 GB of results.

The buffer is now written ONCE, as its canonical bytes (``sort_keys``, compact separators -- the
exact bytes :func:`merlin.perf.work_volume.command_buffer_evidence` digests), to a content-addressed
file next to the result. The result keeps a reference ``{path, name, sha256, bytes, commands}`` and a
small inline summary (tensor shapes and the command count). The file's SHA-256 IS the artifact
digest, so integrity is unchanged: :func:`load_command_buffer` refuses a file whose size or digest
differs from the reference, exactly as an embedded buffer whose recomputed digest differed was
refused.

Every reader goes through :func:`load_command_buffer` (the raw buffer, lazily) or
:func:`command_buffer_tensors` (shapes only, never reading the file). Both accept the legacy inline
form, so results written before this change still read.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "REFERENCE_KEY",
    "SUMMARY_KEY",
    "CommandBufferIntegrityError",
    "canonical_bytes",
    "command_buffer_tensors",
    "has_command_buffer",
    "is_stored_name",
    "load_command_buffer",
    "store_canonical",
    "stored_name",
]

#: Where an artifact names its stored buffer.
REFERENCE_KEY = "command_buffer_file"
#: The small inline fields readers need without opening the stored buffer.
SUMMARY_KEY = "command_buffer_summary"
_PREFIX = "expanded_command_buffer."
_SUFFIX = ".json"
_HEX = frozenset("0123456789abcdef")


class CommandBufferIntegrityError(ValueError):
    """A stored command buffer is missing, or its bytes differ from the reference that names it."""


def canonical_bytes(command_buffer: Mapping[str, Any]) -> bytes:
    """The canonical encoding the artifact digest is taken over (see ``command_buffer_evidence``)."""
    return json.dumps(command_buffer, sort_keys=True, separators=(",", ":")).encode("utf-8")


def stored_name(sha256: str) -> str:
    return f"{_PREFIX}{sha256}{_SUFFIX}"


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def is_stored_name(name: str) -> bool:
    """Whether ``name`` is a stored buffer's file name (retention keeps these beside their result)."""
    if not (name.startswith(_PREFIX) and name.endswith(_SUFFIX)):
        return False
    return _is_sha256(name[len(_PREFIX) : -len(_SUFFIX)])


def _summary(command_buffer: Mapping[str, Any]) -> dict[str, Any]:
    tensors = command_buffer.get("tensors")
    commands = command_buffer.get("commands")
    return {
        "tensors": {
            str(name): {"dtype": spec.get("dtype"), "shape": spec.get("shape"), "role": spec.get("role")}
            for name, spec in sorted(tensors.items())
            if isinstance(spec, Mapping)
        }
        if isinstance(tensors, Mapping)
        else {},
        "commands": len(commands) if isinstance(commands, list) else None,
    }


def _publish(path: Path, payload: bytes) -> None:
    """Write ``payload`` at ``path`` unless identical bytes are already there (content-addressed)."""
    try:
        if path.stat().st_size == len(payload) and _file_sha256(path) == hashlib.sha256(payload).hexdigest():
            return
    except OSError:
        pass
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def store_canonical(
    command_buffer: Mapping[str, Any], payload: bytes, sha256: str, directory: str | Path
) -> dict[str, Any]:
    """Store ``payload`` (``command_buffer``'s canonical bytes) once under ``directory``.

    Returns the ``{REFERENCE_KEY: ..., SUMMARY_KEY: ...}`` fields an artifact carries in place of the
    embedded buffer. ``sha256`` must be the digest of ``payload``; it is checked, never trusted.
    """
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise CommandBufferIntegrityError("command buffer bytes do not match the digest they are stored under")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / stored_name(sha256)
    _publish(path, payload)
    summary = _summary(command_buffer)
    return {
        REFERENCE_KEY: {
            "path": str(path.absolute()),
            "name": path.name,
            "sha256": sha256,
            "bytes": len(payload),
            "commands": summary["commands"],
        },
        SUMMARY_KEY: summary,
    }


def has_command_buffer(artifact: Any) -> bool:
    """Whether ``artifact`` carries a raw buffer, embedded or stored (without reading it)."""
    if not isinstance(artifact, Mapping):
        return False
    return isinstance(artifact.get("command_buffer"), Mapping) or isinstance(artifact.get(REFERENCE_KEY), Mapping)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidates(reference: Mapping[str, Any], base: str | Path | None) -> list[Path]:
    name = reference.get("name")
    out: list[Path] = []
    if base is not None and isinstance(name, str) and is_stored_name(name):
        out.append(Path(base) / name)
    path = reference.get("path")
    if isinstance(path, str) and path:
        out.append(Path(path))
    return out


def load_command_buffer(artifact: Any, *, base: str | Path | None = None) -> Mapping[str, Any] | None:
    """The raw command buffer ``artifact`` carries, or ``None`` when it carries none.

    An embedded buffer (results written before buffers were stored) is returned as it is; its readers
    keep re-digesting it as before. A stored buffer is read from ``base`` (the directory of the result
    that names it, when the result has moved) or from its recorded path, and refused with
    :class:`CommandBufferIntegrityError` unless its size and SHA-256 equal the reference's and the
    reference's digest equals ``artifact_sha256``.
    """
    if not isinstance(artifact, Mapping):
        return None
    inline = artifact.get("command_buffer")
    if isinstance(inline, Mapping):
        return inline
    reference = artifact.get(REFERENCE_KEY)
    if not isinstance(reference, Mapping):
        return None
    sha256, size = reference.get("sha256"), reference.get("bytes")
    if not _is_sha256(sha256) or sha256 != artifact.get("artifact_sha256") or type(size) is not int:
        raise CommandBufferIntegrityError("stored command buffer reference is malformed or names another digest")
    if reference.get("name") != stored_name(sha256):
        raise CommandBufferIntegrityError("stored command buffer name is not its content address")
    for path in _candidates(reference, base):
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != sha256:
            raise CommandBufferIntegrityError(f"stored command buffer {path} changed since it was recorded")
        document = json.loads(payload)
        if not isinstance(document, Mapping):
            raise CommandBufferIntegrityError(f"stored command buffer {path} is not a JSON object")
        return document
    raise CommandBufferIntegrityError(f"stored command buffer {reference.get('name')} is missing")


def command_buffer_tensors(artifact: Any) -> Mapping[str, Any]:
    """``{tensor: {dtype, shape, ...}}`` of ``artifact``'s buffer without reading a stored file."""
    if not isinstance(artifact, Mapping):
        return {}
    inline = artifact.get("command_buffer")
    if isinstance(inline, Mapping):
        tensors = inline.get("tensors")
        return tensors if isinstance(tensors, Mapping) else {}
    summary = artifact.get(SUMMARY_KEY)
    tensors = summary.get("tensors") if isinstance(summary, Mapping) else None
    return tensors if isinstance(tensors, Mapping) else {}
