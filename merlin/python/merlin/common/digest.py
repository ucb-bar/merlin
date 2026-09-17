"""SHA-256 helpers: the one definition behind merlin's content hashes and digest-shape checks.

Measured 2026-09-14: about sixty private helpers across compare/, perf/, runtime/, frontends/, targetgen/
and plotting/ re-implemented four operations under five names (``_sha``, ``_sha256``, ``_digest``,
``_is_sha``, ``_is_sha256``), and the NAME did not say which: ``_digest`` hashed a JSON value in most
modules, a file in one, and returned a bool (a digest-SHAPE check) in two others. Where the bodies
overlapped they agreed byte for byte; the hazard was the naming, so each operation now has one name.

``is_sha256`` is strict by default -- a ``str`` of exactly 64 lowercase hex characters -- because a digest
compared as data must be spelled one way. ``allow_upper=True`` exists for the two readers that accepted
either case; nothing else should need it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

HEX_LOWER = frozenset("0123456789abcdef")
SHA256_HEX_LEN = 64
_CHUNK = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    """Lowercase hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Lowercase hex SHA-256 of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: "str | Path") -> str:
    """Lowercase hex SHA-256 of a file's bytes, streamed. Raises ``OSError`` when it cannot be read --
    a digest that silently stood in for a missing file would certify nothing."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def is_sha256(value: object, *, allow_upper: bool = False) -> bool:
    """True when ``value`` is a ``str`` of exactly 64 hex characters (lowercase unless ``allow_upper``)."""
    if not isinstance(value, str) or len(value) != SHA256_HEX_LEN:
        return False
    text = value.lower() if allow_upper else value
    return all(ch in HEX_LOWER for ch in text)
