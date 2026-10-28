"""Reuse results issued by live checks in one gate job; saved receipts grant no authority.

The normal replay path remains mandatory outside this process-local scope. The
checker host registers its actual successful results, including the exact JSON
bytes it wrote; callers cannot authorize reuse by supplying a file's digest.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_ACTIVE: ContextVar[tuple[int, dict[Path, str]] | None] = ContextVar("lean_receipt_session", default=None)


@contextmanager
def reuse_receipts() -> Iterator[None]:
    """Bound reuse to checks executed in this job; discard authority on every exit."""
    session: tuple[int, dict[Path, str]] = (os.getpid(), {})
    token = _ACTIVE.set(session)
    try:
        yield
    finally:
        session[1].clear()
        _ACTIVE.reset(token)


def remember(path: Path, receipt: dict) -> None:
    """Called only after a live successful check finishes and publishes its result.

    Hash the host's result, rather than reading back a possibly replaced file.
    The serialization matches check._atomic_json; any byte change forces replay.
    """
    session = _ACTIVE.get()
    if session is not None and session[0] == os.getpid() and receipt.get("verdict") == "verified":
        encoded = json.dumps(receipt, indent=2, sort_keys=True).encode()
        session[1][path.resolve()] = hashlib.sha256(encoded).hexdigest()


def reusable(path: Path, digest: str) -> bool:
    """Only the unchanged result of an actual check in this process may be reused."""
    session = _ACTIVE.get()
    return session is not None and session[0] == os.getpid() and session[1].get(path.resolve()) == digest
