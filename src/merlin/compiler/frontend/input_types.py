"""Shared archive metadata for input staging and signature validation."""

from dataclasses import dataclass

CONSTANT_FILE = "capture-constants.safetensors"


@dataclass(frozen=True)
class _ArchiveInfo:
    header: dict
    payload_start: int
    file_bytes: int
    checked_sha256: str | None
