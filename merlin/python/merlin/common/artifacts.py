"""A tiny artifact abstraction shared by TargetGen generators.

An :class:`Artifact` is a (relative path, text payload) pair that knows how to write
itself under a base directory. Generators return lists of artifacts; the pipeline writes
them. Keeping this explicit makes generation deterministic and easy to test (you can
inspect the artifacts without touching the filesystem).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .yaml import dump_yaml


@dataclass(frozen=True)
class Artifact:
    """A single file to emit: ``relpath`` is relative to a generation base directory."""

    relpath: str
    content: str

    def write(self, base: str | Path) -> Path:
        """Write this artifact under ``base``, creating parent directories."""
        out = Path(base) / self.relpath
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.content, encoding="utf-8")
        return out


def yaml_artifact(relpath: str, obj, header: str | None = None) -> Artifact:
    """Build an :class:`Artifact` whose content is deterministic YAML for ``obj``."""
    text = dump_yaml(obj)
    if header:
        text = f"# {header}\n{text}"
    return Artifact(relpath=relpath, content=text)


def write_all(artifacts: list[Artifact], base: str | Path) -> list[Path]:
    """Write every artifact under ``base`` and return the written paths."""
    return [a.write(base) for a in artifacts]
