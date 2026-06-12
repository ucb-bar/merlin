"""Generic single-file ingest fallback.

Used for ad-hoc files and tests when no source-specific adapter applies. Reads one file,
records the given ``source``/``target``, and leaves ``op``/``dtype`` as ``unknown`` unless
caller-provided. Feature extraction still runs against ``raw_text`` via the marker table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from merlin.kernels.types import NormalizedKernel


def ingest_generic(
    path: str,
    source: str = "generic",
    target: str = "generic",
    op: str = "unknown",
    dtype: str = "unknown",
) -> Iterator[NormalizedKernel]:
    """Yield a single NormalizedKernel for ``path``."""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    yield NormalizedKernel(
        source=source, target=target, path=str(p), op=op, dtype=dtype, raw_text=text,
    )
