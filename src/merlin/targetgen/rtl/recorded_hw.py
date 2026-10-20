"""The core HW dialect a selected RTL facts record was extracted from, read back by content."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def recorded_core_hw(target: str) -> Path | None:
    """The core HW dialect the selected facts were EXTRACTED from, when its bytes are still those.

    A facts record names ``inputs.core_hw_path`` with its ``core_hw_sha256``; reading that exact file
    keeps every downstream reader on the bytes the facts describe instead of an unrelated cache. A
    missing file or a digest that no longer matches yields None (the caller then decides), never a
    different dialect.
    """
    from .facts import rtl_facts_path

    try:
        record = json.loads(Path(rtl_facts_path(target)).read_text(encoding="utf-8"))
        inputs = record.get("inputs") or {}
        declared = (record.get("facts") or {}).get("target")
        path, expected = inputs.get("core_hw_path"), inputs.get("core_hw_sha256")
        if declared not in (None, target) or not isinstance(path, str) or not isinstance(expected, str):
            return None
        candidate = Path(path)
        if not candidate.is_file():
            return None
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        return candidate if digest.hexdigest() == expected else None
    except (OSError, ValueError, AttributeError):
        return None
