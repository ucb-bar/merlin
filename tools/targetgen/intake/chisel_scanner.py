"""Detect Chisel module shape and accelerator attachment kind."""

from __future__ import annotations

import re
from pathlib import Path

from ..model import SourceFinding
from ._common import iter_files, read_text, relative_to

NAME = "chisel"
DETECTED_KINDS: tuple[str, ...] = (
    "chisel_module",
    "chisel_attachment_rocc",
    "chisel_attachment_mmio",
    "chisel_attachment_tilelink",
    "chisel_attachment_axi",
    "chisel_attachment_blackbox",
)

_MODULE_RE = re.compile(r"class\s+\w+\s+extends\s+(?:Lazy)?Module\b")
_ROCC_RE = re.compile(r"\bLazyRoCC\b|\bRoCCCommand\b|\bOpcodeSet\b")
_MMIO_RE = re.compile(r"\bTLRegisterRouter\b|\bRegisterRouter\b|\bRegField\b")
_TILELINK_RE = re.compile(r"\bTLClientNode\b|\bTLManagerNode\b|\bTLBundle\b|\bTileLink\b")
_AXI_RE = re.compile(r"\bAXI4(?:Bundle|Master|Slave|RegisterNode)?\b")
_BLACKBOX_RE = re.compile(r"\bextends\s+BlackBox\b")


def scan(root: Path) -> list[SourceFinding]:
    findings: list[SourceFinding] = []
    for path in iter_files(root):
        if path.suffix.lower() != ".scala":
            continue
        text = read_text(path) or ""
        rel = relative_to(path, root).replace("\\", "/")
        if _MODULE_RE.search(text):
            findings.append(
                SourceFinding(
                    kind="chisel_module",
                    path=rel,
                    symbol=None,
                    evidence="class extends (Lazy)Module",
                    confidence=0.7,
                )
            )
        if _ROCC_RE.search(text):
            findings.append(
                SourceFinding(
                    kind="chisel_attachment_rocc",
                    path=rel,
                    symbol=None,
                    evidence="LazyRoCC / RoCCCommand / OpcodeSet reference",
                    confidence=0.85,
                )
            )
        if _MMIO_RE.search(text):
            findings.append(
                SourceFinding(
                    kind="chisel_attachment_mmio",
                    path=rel,
                    symbol=None,
                    evidence="TLRegisterRouter / RegField MMIO regmap",
                    confidence=0.8,
                )
            )
        if _TILELINK_RE.search(text):
            findings.append(
                SourceFinding(
                    kind="chisel_attachment_tilelink",
                    path=rel,
                    symbol=None,
                    evidence="TileLink node references",
                    confidence=0.6,
                )
            )
        if _AXI_RE.search(text):
            findings.append(
                SourceFinding(
                    kind="chisel_attachment_axi",
                    path=rel,
                    symbol=None,
                    evidence="AXI4 reference",
                    confidence=0.55,
                )
            )
        if _BLACKBOX_RE.search(text):
            findings.append(
                SourceFinding(
                    kind="chisel_attachment_blackbox",
                    path=rel,
                    symbol=None,
                    evidence="extends BlackBox (external Verilog wrapper)",
                    confidence=0.7,
                )
            )
    return findings
