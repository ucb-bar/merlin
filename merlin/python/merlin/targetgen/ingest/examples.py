"""Discover example/program files referenced by a SourceManifest.

Lists the kinds of artifacts that demonstrate how a target is programmed: MLIR, C/headers,
assembly, JSON/YAML configs. Deterministic, read-only.
"""
from __future__ import annotations

from pathlib import Path

from ...common.io import find_by_suffix
from .source_manifest import SourceManifest

EXAMPLE_SUFFIXES = ("mlir", "c", "h", "s", "json", "yaml", "yml")


def discover_examples(manifest: SourceManifest) -> list[Path]:
    """Return sorted example files found under the manifest's examples dirs."""
    hits: list[Path] = []
    for d in manifest.examples_dirs:
        hits.extend(find_by_suffix(d, EXAMPLE_SUFFIXES))
    return sorted(set(hits))
