"""Data structures for collected evidence.

``Evidence`` is the in-memory result of the evidence pass; it serializes to the
``evidence_report`` schema (``evidence_index.yaml``) and renders to ``evidence_report.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FileRecord:
    """One discovered source file."""

    path: str            # path relative to its source root (for citations)
    kind: str            # doc | scala | example
    summary: str         # short, filename/first-line derived

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "kind": self.kind, "summary": self.summary}


@dataclass(frozen=True)
class Concept:
    """A keyword-detected concept and the files that support it."""

    concept: str
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"concept": self.concept, "evidence": list(self.evidence)}


@dataclass
class Evidence:
    """Result of the evidence pass for one target."""

    target: str
    sources: dict[str, Any]
    files: list[FileRecord] = field(default_factory=list)
    concepts: list[Concept] = field(default_factory=list)

    def concept_names(self) -> set[str]:
        return {c.concept for c in self.concepts}

    def to_index_dict(self) -> dict[str, Any]:
        """Return the ``evidence_report``-schema mapping for ``evidence_index.yaml`` (validated)."""
        from merlin.common.schemas import validate_or_raise
        d = {
            "target": self.target,
            "sources": self.sources,
            "files": [f.to_dict() for f in self.files],
            "detected_concepts": [c.to_dict() for c in self.concepts],
        }
        validate_or_raise(d, "evidence_report")   # schemas/ rule: if it lives here, code validates it
        return d
