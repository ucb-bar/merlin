"""Explicit projection of authenticated parameters onto host-required backing.

The caller owns the proof that every removed port is resident-only. This module
does not infer residency from a shape, a provider receipt, or an unused pointer.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from merlin.compiler.lower.c.parameters import ApplicationParameters
from merlin.compiler.lower.c.weights import DTYPE_BYTES


def _projection(original, resident_only):
    if (not isinstance(resident_only, tuple)
            or any(not isinstance(key, str) for key in resident_only)
            or len(set(resident_only)) != len(resident_only)
            or tuple(sorted(resident_only)) != resident_only
            or not set(resident_only) <= set(original.node_ids)):
        raise ValueError("resident-only ports must be an exact canonical parameter subset")
    cursor, spans, offsets = 0, {}, []
    for key, entry in zip(original.node_ids, original.entries, strict=True):
        if key in resident_only:
            offsets.append(None)
            continue
        identity = (entry.offset, entry.nbytes)
        if identity not in spans:
            width = DTYPE_BYTES[entry.dtype]
            cursor = (cursor + width - 1) // width * width
            spans[identity] = cursor
            cursor += entry.nbytes
        offsets.append(spans[identity])
    return tuple(offsets), tuple((start, size, compact) for (start, size), compact in spans.items()), max(1, cursor)


def _stream(original, spans, size, destination=None):
    """Authenticate to a private snapshot before reading any projected span."""
    digest, cursor = sha256(), 0
    with TemporaryDirectory(prefix="merlin-compact-parameters-") as temporary:
        snapshot = Path(temporary) / "source.bin"
        original.write_payload(snapshot)
        with snapshot.open("rb") as source:
            for start, count, compact in spans:
                padding = b"\0" * (compact - cursor)
                digest.update(padding)
                if destination is not None:
                    destination.write(padding)
                source.seek(start)
                remaining = count
                while remaining:
                    chunk = source.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ValueError("compact parameter source span is truncated")
                    digest.update(chunk)
                    if destination is not None:
                        destination.write(chunk)
                    remaining -= len(chunk)
                cursor = compact + count
            if cursor == 0:
                digest.update(b"\0")
                if destination is not None:
                    destination.write(b"\0")
                cursor = 1
    if cursor != size:
        raise ValueError("compact parameter backing size differs")
    return digest.hexdigest()


@dataclass(frozen=True)
class CompactApplicationParameters:
    original: ApplicationParameters
    resident_only: tuple[str, ...]
    compact_offsets: tuple[int | None, ...]
    payload_bytes: int
    payload_sha256: str

    @property
    def source_request_digest(self):
        return self.original.source_request_digest

    @property
    def node_ids(self):
        # Retain every original parameter port, including removed backing.
        return self.original.node_ids

    @property
    def entries(self):
        # Source entries retain their identities; compact_offsets owns backing.
        return self.original.entries

    def check(self, source, *, resident_only=None):
        self.original.check(source)
        if resident_only is not None:
            supplied = tuple(resident_only)
            if (any(not isinstance(key, str) for key in supplied)
                    or len(set(supplied)) != len(supplied)
                    or tuple(sorted(supplied)) != self.resident_only):
                raise ValueError("resident-only proof differs from compact backing")
        offsets, spans, size = _projection(self.original, self.resident_only)
        if (offsets != self.compact_offsets or type(self.payload_bytes) is not int
                or self.payload_bytes != size
                or self.payload_sha256 != _stream(self.original, spans, size)):
            raise ValueError("compact parameter projection or byte identity differs")

    def pointer_expression(self, node, name="parameters"):
        if (not isinstance(name, str) or not name.isascii() or not name.isidentifier()):
            raise ValueError("parameter pointer name must be a C identifier")
        try:
            offset = self.compact_offsets[self.node_ids.index(node)]
        except ValueError as error:
            raise ValueError("unknown compact parameter port") from error
        return "0" if offset is None else f"{name}->data+{offset}u"

    def record(self):
        return {
            "schema": "merlin.compact_application_parameters.v1",
            "source_request_digest": self.source_request_digest,
            "source_parameters": self.original.record(),
            "resident_only": list(self.resident_only),
            "payload_sha256": self.payload_sha256,
            "payload_bytes": self.payload_bytes,
            "path": "handoff/parameters.bin",
            "ports": [{"node_id": key, **vars(entry), "compact_offset": offset}
                      for key, entry, offset in zip(self.node_ids, self.entries, self.compact_offsets, strict=True)],
            "empty_backing": ("one zero sentinel byte" if not any(
                offset is not None and entry.nbytes
                for entry, offset in zip(self.entries, self.compact_offsets, strict=True)) else None),
            "ownership": "caller-owned immutable bytes through completion and recovery",
            "validation": "SHA256 at bind; explicit compiler resident-only proof required",
        }

    def write_payload(self, destination):
        offsets, spans, size = _projection(self.original, self.resident_only)
        if offsets != self.compact_offsets or size != self.payload_bytes:
            raise ValueError("compact parameter projection differs")
        target, created = Path(destination), False
        try:
            with target.open("xb") as stream:
                created = True
                if _stream(self.original, spans, size, stream) != self.payload_sha256:
                    raise ValueError("compact parameter payload changed during publication")
        except BaseException:
            if created:
                target.unlink(missing_ok=True)
            raise

    def emit_binding(self, name):
        return ApplicationParameters.emit_binding(self, name)


def compact_parameters(original, source, *, resident_only):
    """Project backing only for ports explicitly proven resident-only by caller."""
    original.check(source)
    supplied = tuple(resident_only)
    if any(not isinstance(key, str) for key in supplied) or len(set(supplied)) != len(supplied):
        raise ValueError("resident-only ports must be unique parameter node ids")
    proof = tuple(sorted(supplied))
    offsets, spans, size = _projection(original, proof)
    return CompactApplicationParameters(original, proof, offsets, size, _stream(original, spans, size))
