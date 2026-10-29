"""Typed byte identities and live source-to-snapshot associations for receipt wrappers.

Content association is independent of artifact authenticity or policy admission. A pair of
individually fresh files does not establish that they contain the same submitted contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.verify.lean.check import _file_ref


@dataclass(frozen=True)
class FileReference:
    """A validated file identity, preserving JSON integer/boolean distinctions."""

    path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path or not Path(self.path).is_absolute():
            raise ValueError("file reference must name an absolute file path")
        if type(self.sha256) is not str or len(self.sha256) != 64 or not set(self.sha256) <= set("0123456789abcdef"):
            raise ValueError("file reference must contain a lowercase SHA256 digest")
        if type(self.bytes) is not int or self.bytes < 0:
            raise ValueError("file reference byte length must be a nonnegative JSON integer")

    @classmethod
    def decode(cls, value: Any, label: str) -> FileReference:
        if type(value) is not dict or set(value) != {"path", "sha256", "bytes"}:
            raise ValueError(f"{label} must be an exact file reference")
        try:
            return cls(value["path"], value["sha256"], value["bytes"])
        except ValueError as error:
            raise ValueError(f"{label}: {error}") from error

    def encode(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}

    @property
    def content(self) -> tuple[str, int]:
        return self.sha256, self.bytes

    def assert_current(self, label: str) -> None:
        actual = FileReference.decode(_file_ref(Path(self.path)), label)
        if actual != self:
            raise ValueError(f"{label} file identity changed")


@dataclass(frozen=True)
class SnapshotBinding:
    """Two live file identities that must name exactly the same bytes."""

    original: FileReference
    snapshot: FileReference

    def __post_init__(self) -> None:
        if type(self.original) is not FileReference or type(self.snapshot) is not FileReference:
            raise ValueError("snapshot binding requires validated file references")
        if self.original.content != self.snapshot.content:
            raise ValueError("original and snapshot have different byte identities")

    @classmethod
    def decode(cls, value: Any, label: str) -> SnapshotBinding:
        if type(value) is not dict or set(value) != {"original", "snapshot"}:
            raise ValueError(f"{label} must contain exactly original and snapshot references")
        original = FileReference.decode(value["original"], f"{label} original")
        snapshot = FileReference.decode(value["snapshot"], f"{label} snapshot")
        try:
            return cls(original, snapshot)
        except ValueError as error:
            raise ValueError(f"{label}: {error}") from error

    def encode(self) -> dict[str, Any]:
        return {"original": self.original.encode(), "snapshot": self.snapshot.encode()}

    def assert_current(self, label: str) -> None:
        self.original.assert_current(f"{label} original")
        self.snapshot.assert_current(f"{label} snapshot")


def contract_bindings(report: Any) -> dict[str, SnapshotBinding]:
    """Validate both W8A8 contract kinds without interpreting their numerical policies."""
    if type(report) is not dict:
        raise ValueError("contract binding report must be an object")
    result = {}
    for kind in ("input_contracts", "exponent_contracts"):
        if kind not in report:
            raise ValueError(f"contract binding report omitted {kind}")
        binding = SnapshotBinding.decode(report[kind], kind)
        binding.assert_current(kind)
        result[kind] = binding
    return result
