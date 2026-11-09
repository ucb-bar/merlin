"""Framework-independent input and option records for public compilation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Source:
    text: str
    kind: str = "mlir"
    origin: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("compilation requires nonempty source text")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("source kind must be explicit")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CompileOptions:
    system: str | None = None
    platform: str | None = None
    harts: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("system", "platform"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty string when supplied")
        if self.harts is not None and (type(self.harts) is not int or self.harts < 1):
            raise ValueError("harts must be a positive integer")
        if not isinstance(self.parameters, dict):
            raise ValueError("compiler parameters must be a mapping")
        for key in self.parameters:
            parts = key.split(".") if isinstance(key, str) else []
            valid = (
                len(parts) >= 2
                and all(parts)
                and parts[0][0].isascii()
                and parts[0][0].isalpha()
                and all(c.isascii() and (c.isalnum() or c in "_-") for part in parts for c in part)
            )
            if not valid:
                raise ValueError(f"compiler option must have an explicit namespace: {key!r}")
        json.dumps(self.parameters, allow_nan=False)


@dataclass(frozen=True)
class CompileRequest:
    source: Source
    output: Path
    options: CompileOptions
    preparation: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "output", Path(self.output).resolve())
        if self.preparation is not None:
            from merlin.compiler.frontend.staged import StagedPreparation

            if type(self.preparation) is not StagedPreparation:
                raise ValueError("compilation preparation requires an original checked staged binding")
            self.preparation.revalidate(self.source)
