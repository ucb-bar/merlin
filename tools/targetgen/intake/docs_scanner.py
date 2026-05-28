"""Extract structural facts from Markdown / RST target documentation."""

from __future__ import annotations

import re
from pathlib import Path

from ..model import SourceFinding
from ._common import iter_files, read_text, relative_to

NAME = "docs"
DETECTED_KINDS: tuple[str, ...] = (
    "isa_documentation",
    "memory_model_documentation",
    "synchronization_documentation",
    "runtime_documentation",
    "driver_documentation",
    "simulator_documentation",
    "build_documentation",
)

_HEADING_KIND: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^\s*#+\s*ISA\b", re.IGNORECASE | re.MULTILINE), "isa_documentation"),
    (
        re.compile(r"^\s*#+\s*Instruction\s+Format\b", re.IGNORECASE | re.MULTILINE),
        "isa_documentation",
    ),
    (
        re.compile(r"^\s*#+\s*Memory(?:\s+Model)?\b", re.IGNORECASE | re.MULTILINE),
        "memory_model_documentation",
    ),
    (
        re.compile(r"^\s*#+\s*Synchronization\b", re.IGNORECASE | re.MULTILINE),
        "synchronization_documentation",
    ),
    (
        re.compile(r"^\s*#+\s*Runtime\b", re.IGNORECASE | re.MULTILINE),
        "runtime_documentation",
    ),
    (
        re.compile(r"^\s*#+\s*Driver\b", re.IGNORECASE | re.MULTILINE),
        "driver_documentation",
    ),
    (
        re.compile(r"^\s*#+\s*Simulator\b", re.IGNORECASE | re.MULTILINE),
        "simulator_documentation",
    ),
    (
        re.compile(r"^\s*#+\s*(?:Build|Building)\b", re.IGNORECASE | re.MULTILINE),
        "build_documentation",
    ),
)


def scan(root: Path) -> list[SourceFinding]:
    findings: list[SourceFinding] = []
    for path in iter_files(root):
        suffix = path.suffix.lower()
        if suffix not in {".md", ".rst", ".markdown"}:
            continue
        text = read_text(path) or ""
        rel = relative_to(path, root).replace("\\", "/")
        for regex, kind in _HEADING_KIND:
            match = regex.search(text)
            if match:
                findings.append(
                    SourceFinding(
                        kind=kind,
                        path=rel,
                        symbol=None,
                        evidence=f"Heading match: {match.group(0).strip()}",
                        confidence=0.6,
                    )
                )
    return findings
