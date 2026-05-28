"""Detect MLIR dialect/op/pass definitions in a source tree."""

from __future__ import annotations

import re
from pathlib import Path

from ..model import SourceFinding
from ._common import iter_files, read_text, relative_to

NAME = "mlir"
DETECTED_KINDS: tuple[str, ...] = (
    "mlir_dialect",
    "mlir_pass",
    "mlir_op_definition",
)

_OPS_TD_RE = re.compile(r"def\s+\w+_Op\s*<", re.MULTILINE)
_DIALECT_TD_RE = re.compile(r"def\s+\w+_Dialect\s*:\s*Dialect", re.MULTILINE)
_PASSES_TD_RE = re.compile(r"def\s+\w+\s*:\s*Pass<", re.MULTILINE)
_DIALECT_NAME_RE = re.compile(r"let\s+name\s*=\s*\"([A-Za-z_][\w]*)\"")
_REGISTER_DIALECT_RE = re.compile(r"register(?:Dialect|Dialects?)\s*\(", re.MULTILINE)


def scan(root: Path) -> list[SourceFinding]:
    findings: list[SourceFinding] = []
    for path in iter_files(root):
        suffix = path.suffix.lower()
        rel = relative_to(path, root)
        name = path.name
        if suffix == ".td":
            text = read_text(path) or ""
            if _DIALECT_TD_RE.search(text):
                m = _DIALECT_NAME_RE.search(text)
                findings.append(
                    SourceFinding(
                        kind="mlir_dialect",
                        path=rel,
                        symbol=m.group(1) if m else None,
                        evidence="MLIR Dialect TableGen definition",
                        confidence=0.95,
                    )
                )
            if _OPS_TD_RE.search(text) or name.endswith("Ops.td"):
                findings.append(
                    SourceFinding(
                        kind="mlir_op_definition",
                        path=rel,
                        symbol=None,
                        evidence="MLIR op TableGen definitions",
                        confidence=0.85,
                    )
                )
            if _PASSES_TD_RE.search(text) or name.endswith("Passes.td"):
                findings.append(
                    SourceFinding(
                        kind="mlir_pass",
                        path=rel,
                        symbol=None,
                        evidence="MLIR pass TableGen definitions",
                        confidence=0.8,
                    )
                )
        elif suffix in {".cpp", ".cc", ".h", ".hpp"}:
            text = read_text(path) or ""
            if _REGISTER_DIALECT_RE.search(text) and "DialectRegistry" in text:
                findings.append(
                    SourceFinding(
                        kind="mlir_dialect",
                        path=rel,
                        symbol=None,
                        evidence="MLIR DialectRegistry registration call",
                        confidence=0.55,
                    )
                )
    return findings
