"""Ingest layer: record TargetGen inputs as a SourceManifest.

We only read local paths passed by the caller. Source URLs are recorded for provenance but
never crawled, and large external repos are never vendored.
"""
from __future__ import annotations

from .source_manifest import SourceManifest, build_manifest

__all__ = ["SourceManifest", "build_manifest"]
