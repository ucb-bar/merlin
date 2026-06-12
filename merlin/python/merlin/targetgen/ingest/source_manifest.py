"""The SourceManifest: a record of what TargetGen was pointed at for a target.

Validates against ``merlin/schemas/target_source_manifest.schema.yaml``. Source URLs are
recorded but not fetched; only local directories/files are read downstream.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SourceManifest:
    """Inputs for one TargetGen run.

    All path fields are recorded verbatim (absolute or repo-relative). ``branch``/``commit``/
    ``notes`` are optional provenance.
    """

    target_name: str
    source_dirs: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)
    scala_roots: list[str] = field(default_factory=list)
    examples_dirs: list[str] = field(default_factory=list)
    branch: str | None = None
    commit: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a schema-shaped mapping (matches target_source_manifest schema)."""
        return {
            "target_name": self.target_name,
            "source_dirs": list(self.source_dirs),
            "source_files": list(self.source_files),
            "source_urls": list(self.source_urls),
            "scala_roots": list(self.scala_roots),
            "examples_dirs": list(self.examples_dirs),
            "branch": self.branch,
            "commit": self.commit,
            "notes": self.notes,
        }


def build_manifest(
    target_name: str,
    source_dir: str | None = None,
    examples_dir: str | None = None,
    scala_root: str | None = None,
    source_urls: list[str] | None = None,
    branch: str | None = None,
    commit: str | None = None,
    notes: str | None = None,
) -> SourceManifest:
    """Build a SourceManifest from the CLI's simple flag set.

    ``source_dir`` becomes the single source directory; ``examples_dir``/``scala_root`` are
    added to their respective lists when provided.
    """
    return SourceManifest(
        target_name=target_name,
        source_dirs=[source_dir] if source_dir else [],
        examples_dirs=[examples_dir] if examples_dir else [],
        scala_roots=[scala_root] if scala_root else [],
        source_urls=list(source_urls or []),
        branch=branch,
        commit=commit,
        notes=notes,
    )
