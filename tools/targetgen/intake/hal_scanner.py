"""Detect runtime / HAL driver source patterns."""

from __future__ import annotations

import re
from pathlib import Path

from ..model import SourceFinding
from ._common import iter_files, read_text, relative_to

NAME = "hal"
DETECTED_KINDS: tuple[str, ...] = (
    "hal_driver_source",
    "hal_device_source",
    "hal_registration",
    "hal_executable_format",
    "hal_command_buffer",
)

_DRIVER_FILE_NAMES: frozenset[str] = frozenset({"driver.c", "driver.cc"})
_DEVICE_FILE_NAMES: frozenset[str] = frozenset({"device.c", "device.cc"})

_PATTERNS: tuple[tuple[str, str, float, str], ...] = (
    (
        r"iree_hal_[a-z0-9_]+_driver_module_register\b",
        "hal_registration",
        0.9,
        "HAL driver module registration symbol",
    ),
    (
        r"iree_hal_command_buffer_t\b",
        "hal_command_buffer",
        0.6,
        "iree_hal_command_buffer_t reference",
    ),
    (
        r"iree_hal_executable[_t]*\b",
        "hal_executable_format",
        0.5,
        "HAL executable type reference",
    ),
)

_COMPILED = tuple((re.compile(p, re.MULTILINE), kind, conf, evidence) for p, kind, conf, evidence in _PATTERNS)


def scan(root: Path) -> list[SourceFinding]:
    findings: list[SourceFinding] = []
    for path in iter_files(root):
        rel = relative_to(path, root).replace("\\", "/")
        suffix = path.suffix.lower()
        name = path.name
        if suffix in {".c", ".cc", ".cpp", ".h", ".hpp"}:
            text = read_text(path) or ""
            if name in _DRIVER_FILE_NAMES and "iree_hal" in text:
                findings.append(
                    SourceFinding(
                        kind="hal_driver_source",
                        path=rel,
                        symbol=name,
                        evidence="HAL driver translation unit references iree_hal",
                        confidence=0.85,
                    )
                )
            if name in _DEVICE_FILE_NAMES and "iree_hal" in text:
                findings.append(
                    SourceFinding(
                        kind="hal_device_source",
                        path=rel,
                        symbol=name,
                        evidence="HAL device translation unit references iree_hal",
                        confidence=0.8,
                    )
                )
            for regex, kind, confidence, evidence in _COMPILED:
                if regex.search(text):
                    findings.append(
                        SourceFinding(
                            kind=kind,
                            path=rel,
                            symbol=None,
                            evidence=evidence,
                            confidence=confidence,
                        )
                    )
    return findings
