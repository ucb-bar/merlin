"""Portable, content-bound compilation products; no compiler or runtime imports.

The manifest records production and compatibility claims. It is not independent
numerical verification, hardware qualification, or permission to execute code.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "merlin.compilation.v1"


def digest(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _positive(value: Any, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


@dataclass(frozen=True)
class Deployment:
    """A selected image contract, distinct from a compiler backend identity."""

    system: str
    platform: str
    isa: str
    abi: str
    harts: int
    dram_base: int | None = None
    memory_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in ("system", "platform", "isa", "abi"):
            _text(getattr(self, name), name)
        _positive(self.harts, "harts")
        if (self.dram_base is None) != (self.memory_bytes is None):
            raise ValueError("DRAM origin and memory size must be supplied together")
        if self.dram_base is not None:
            if type(self.dram_base) is not int or self.dram_base < 0:
                raise ValueError("dram_base must be a nonnegative integer")
            _positive(self.memory_bytes, "memory_bytes")


@dataclass(frozen=True)
class Artifact:
    path: Path
    kind: str
    role: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).resolve())
        _text(self.kind, "artifact kind")
        _text(self.role, "artifact role")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.sha256)
        ):
            raise ValueError("artifact sha256 must be a lowercase SHA-256 digest")

    @classmethod
    def from_file(cls, path: str | Path, *, kind: str, role: str) -> Artifact:
        path = Path(path).resolve(strict=True)
        return cls(path, kind, role, digest(path))

    def verify(self) -> None:
        if not self.path.is_file() or digest(self.path) != self.sha256:
            raise ValueError(f"artifact bytes do not match the compilation manifest: {self.path}")


@dataclass(frozen=True)
class Compilation:
    directory: Path
    backend: str
    source_sha256: str
    artifacts: tuple[Artifact, ...]
    deployment: Deployment | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory).resolve())
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        _text(self.backend, "backend")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_sha256)
        ):
            raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
        if not self.artifacts:
            raise ValueError("compilation must contain at least one product")
        paths = [a.path for a in self.artifacts]
        if len(set(paths)) != len(paths):
            raise ValueError("compilation contains duplicate artifact paths")
        for path in paths:
            if not path.is_relative_to(self.directory) or path == self.directory:
                raise ValueError(f"artifact escapes its compilation directory: {path}")
            if path == self.manifest:
                raise ValueError("the compilation manifest cannot be its own artifact")
        # Metadata is evidence supplied by the provider, never an admission result.
        if not isinstance(self.metadata, dict):
            raise ValueError("compilation metadata must be an object")
        json.dumps(self.metadata, allow_nan=False)

    @property
    def manifest(self) -> Path:
        return self.directory / "compilation.json"

    def verify(self) -> None:
        for artifact in self.artifacts:
            # Re-resolve at use time too: a directory symlink may have changed.
            if not artifact.path.resolve().is_relative_to(self.directory):
                raise ValueError("artifact path now escapes the compilation directory")
            artifact.verify()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "backend": self.backend,
            "source_sha256": self.source_sha256,
            "artifacts": [
                {
                    "path": a.path.relative_to(self.directory).as_posix(),
                    "kind": a.kind,
                    "role": a.role,
                    "sha256": a.sha256,
                }
                for a in self.artifacts
            ],
            "deployment": asdict(self.deployment) if self.deployment is not None else None,
            "metadata": self.metadata,
        }

    def write(self) -> Path:
        self.verify()
        self.manifest.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return self.manifest

    @classmethod
    def load(cls, path: str | Path) -> Compilation:
        manifest = Path(path).resolve(strict=True)
        if manifest.is_dir():
            manifest = manifest / "compilation.json"
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        fields = {"schema", "backend", "source_sha256", "artifacts", "deployment", "metadata"}
        if not isinstance(raw, dict) or raw.get("schema") != SCHEMA or set(raw) != fields:
            raise ValueError("unsupported or malformed compilation manifest")
        directory = manifest.parent
        products = []
        for record in raw["artifacts"]:
            if not isinstance(record, dict) or set(record) != {"path", "kind", "role", "sha256"}:
                raise ValueError("malformed artifact record")
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative) or ":" in str(relative):
                raise ValueError("artifact paths must be portable relative paths without traversal")
            products.append(Artifact(directory / relative, record["kind"], record["role"], record["sha256"]))
        contract = Deployment(**raw["deployment"]) if raw["deployment"] is not None else None
        result = cls(directory, raw["backend"], raw["source_sha256"], tuple(products), contract, raw["metadata"])
        result.verify()
        return result
