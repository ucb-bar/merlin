"""Detect CMake build-system integration patterns relevant to Merlin."""

from __future__ import annotations

import re
from pathlib import Path

from ..model import SourceFinding
from ._common import iter_files, read_text, relative_to

NAME = "cmake"
DETECTED_KINDS: tuple[str, ...] = (
    "cmake_project",
    "iree_external_hal_driver_registration",
    "iree_compiler_plugin_registration",
    "mlir_dialect_cmake",
    "llvm_cmake",
    "cmake_external_project",
)

_PATTERNS: tuple[tuple[str, str, float], ...] = (
    (
        r"iree_register_external_hal_driver\b",
        "iree_external_hal_driver_registration",
        0.95,
    ),
    (
        r"iree_register_compiler_plugin\b",
        "iree_compiler_plugin_registration",
        0.95,
    ),
    (r"add_mlir_dialect\b", "mlir_dialect_cmake", 0.9),
    (r"mlir_tablegen\b", "mlir_dialect_cmake", 0.7),
    (r"add_mlir_library\b", "mlir_dialect_cmake", 0.5),
    (r"add_llvm_library\b", "llvm_cmake", 0.7),
    (r"add_llvm_target\b", "llvm_cmake", 0.85),
    (r"\bExternalProject_Add\b", "cmake_external_project", 0.6),
    (r"\bFetchContent_Declare\b", "cmake_external_project", 0.5),
)

_COMPILED = tuple((re.compile(p, re.MULTILINE), kind, conf) for p, kind, conf in _PATTERNS)


def scan(root: Path) -> list[SourceFinding]:
    findings: list[SourceFinding] = []
    saw_cmake = False
    for path in iter_files(root):
        if path.name not in {"CMakeLists.txt"} and path.suffix.lower() != ".cmake":
            continue
        saw_cmake = True
        text = read_text(path) or ""
        rel = relative_to(path, root)
        for regex, kind, confidence in _COMPILED:
            for match in regex.finditer(text):
                findings.append(
                    SourceFinding(
                        kind=kind,
                        path=rel,
                        symbol=match.group(0),
                        evidence=f"CMake call: {match.group(0)}",
                        confidence=confidence,
                    )
                )
    if saw_cmake:
        findings.append(
            SourceFinding(
                kind="cmake_project",
                path=".",
                symbol=None,
                evidence="CMakeLists.txt or .cmake files present",
                confidence=0.9,
            )
        )
    return findings
