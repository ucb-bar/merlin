"""Detect SystemC model patterns."""

from __future__ import annotations

import re
from pathlib import Path

from ..model import SourceFinding
from ._common import iter_files, read_text, relative_to

NAME = "systemc"
DETECTED_KINDS: tuple[str, ...] = (
    "systemc_module",
    "systemc_tlm",
)

_MODULE_RE = re.compile(r"\bSC_MODULE\s*\(\s*(\w+)\s*\)")
_PORT_RE = re.compile(r"\bsc_(?:in|out|inout|signal|clock|fifo)\s*<")
_TLM_RE = re.compile(r"\btlm[_:]+|\bb_transport\s*\(")


def scan(root: Path) -> list[SourceFinding]:
    findings: list[SourceFinding] = []
    for path in iter_files(root):
        if path.suffix.lower() not in {".cc", ".cpp", ".h", ".hpp", ".cxx"}:
            continue
        text = read_text(path) or ""
        if "systemc" not in text and "SC_MODULE" not in text and "sc_in" not in text:
            continue
        rel = relative_to(path, root).replace("\\", "/")
        for match in _MODULE_RE.finditer(text):
            findings.append(
                SourceFinding(
                    kind="systemc_module",
                    path=rel,
                    symbol=match.group(1),
                    evidence=f"SC_MODULE({match.group(1)})",
                    confidence=0.8,
                )
            )
        if _PORT_RE.search(text):
            findings.append(
                SourceFinding(
                    kind="systemc_module",
                    path=rel,
                    symbol=None,
                    evidence="sc_in/sc_out/sc_signal port declaration",
                    confidence=0.6,
                )
            )
        if _TLM_RE.search(text):
            findings.append(
                SourceFinding(
                    kind="systemc_tlm",
                    path=rel,
                    symbol=None,
                    evidence="TLM b_transport / tlm:: usage",
                    confidence=0.7,
                )
            )
    return findings
