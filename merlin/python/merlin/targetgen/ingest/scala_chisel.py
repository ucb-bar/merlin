"""Discover Scala/Chisel RTL files referenced by a SourceManifest.

The UCB targets (Gemmini, Saturn, Radiance) describe hardware in Scala/Chisel. We only
*locate* these files (so the evidence report can cite them); we make no attempt to interpret
RTL. This deliberately conservative posture is why all non-toy synthesis is flagged
``requires_human_review``.
"""
from __future__ import annotations

from pathlib import Path

from ..io import find_by_suffix
from .source_manifest import SourceManifest

SCALA_SUFFIXES = ("scala",)


def discover_scala(manifest: SourceManifest) -> list[Path]:
    """Return sorted Scala files found under the manifest's scala roots + source dirs."""
    hits: list[Path] = []
    for d in manifest.scala_roots:
        hits.extend(find_by_suffix(d, SCALA_SUFFIXES))
    for d in manifest.source_dirs:
        hits.extend(find_by_suffix(d, SCALA_SUFFIXES))
    return sorted(set(hits))
