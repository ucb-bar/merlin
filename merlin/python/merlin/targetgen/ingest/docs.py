"""Discover documentation files referenced by a SourceManifest.

Conservative and deterministic: we list Markdown/reStructuredText/plain-text files under the
manifest's source directories and read short summaries. We do not parse or "understand"
their content here -- that is the human-reviewed synthesis step's concern.
"""
from __future__ import annotations

from pathlib import Path

from ..io import find_by_suffix
from .source_manifest import SourceManifest

DOC_SUFFIXES = ("md", "rst", "txt")


def discover_docs(manifest: SourceManifest) -> list[Path]:
    """Return sorted doc files found under the manifest's source dirs + explicit files."""
    hits: list[Path] = []
    for d in manifest.source_dirs:
        hits.extend(find_by_suffix(d, DOC_SUFFIXES))
    for f in manifest.source_files:
        p = Path(f)
        if p.is_file() and p.suffix.lower().lstrip(".") in DOC_SUFFIXES:
            hits.append(p)
    # de-dup while keeping determinism
    return sorted(set(hits))
