"""Detect Chipyard project layout."""

from __future__ import annotations

import re
from pathlib import Path

from ..model import SourceFinding
from ._common import iter_files, read_text, relative_to

NAME = "chipyard"
DETECTED_KINDS: tuple[str, ...] = (
    "chipyard_project",
    "chipyard_config",
    "firesim_collateral",
)

_CONFIG_RE = re.compile(r"class\s+\w+Config\b")
_FIRESIM_RE = re.compile(r"\b(?:firesim|FireSim|FireMarshal)\b")
_LAZY_MODULE_RE = re.compile(r"\bLazyModule\b|\bDiplomacy\b|\bTileLink\b|\bAXI4\b")


def scan(root: Path) -> list[SourceFinding]:
    findings: list[SourceFinding] = []
    saw_build_sbt = False
    saw_generators = False
    saw_firesim_path = False
    for path in iter_files(root):
        rel = relative_to(path, root).replace("\\", "/")
        name = path.name
        if name == "build.sbt":
            saw_build_sbt = True
            findings.append(
                SourceFinding(
                    kind="chipyard_project",
                    path=rel,
                    symbol="build.sbt",
                    evidence="sbt build descriptor",
                    confidence=0.8,
                )
            )
        if "/generators/" in rel or rel.startswith("generators/"):
            saw_generators = True
        if "/firesim" in rel.lower() or rel.lower().startswith("firesim"):
            saw_firesim_path = True
        if path.suffix.lower() == ".scala":
            text = read_text(path) or ""
            if _CONFIG_RE.search(text):
                findings.append(
                    SourceFinding(
                        kind="chipyard_config",
                        path=rel,
                        symbol=None,
                        evidence="Chipyard Config class",
                        confidence=0.6,
                    )
                )
            if _FIRESIM_RE.search(text):
                findings.append(
                    SourceFinding(
                        kind="firesim_collateral",
                        path=rel,
                        symbol=None,
                        evidence="FireSim references",
                        confidence=0.5,
                    )
                )
            if _LAZY_MODULE_RE.search(text):
                findings.append(
                    SourceFinding(
                        kind="chipyard_project",
                        path=rel,
                        symbol=None,
                        evidence="LazyModule / Diplomacy / TileLink / AXI4 reference",
                        confidence=0.55,
                    )
                )
    if saw_firesim_path:
        findings.append(
            SourceFinding(
                kind="firesim_collateral",
                path=".",
                symbol=None,
                evidence="firesim/ directory present",
                confidence=0.7,
            )
        )
    if saw_build_sbt and saw_generators:
        findings.append(
            SourceFinding(
                kind="chipyard_project",
                path=".",
                symbol=None,
                evidence="build.sbt and generators/ tree present",
                confidence=0.85,
            )
        )
    return findings
