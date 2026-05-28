"""Detect LLVM backend / TableGen / intrinsic-definition surfaces."""

from __future__ import annotations

import re
from pathlib import Path

from ..model import SourceFinding
from ._common import iter_files, read_text, relative_to

NAME = "llvm"
DETECTED_KINDS: tuple[str, ...] = (
    "llvm_target_backend",
    "llvm_intrinsic_definition",
    "llvm_riscv_features",
    "llvm_inline_asm",
)

_INTRINSICS_RE = re.compile(r"\bIntrinsics?[A-Za-z0-9_]*\.td\b")
_RISCV_FEATURES_RE = re.compile(r"\bRISCVFeatures?\.td\b|\bRISCVInstrInfo\b")
_LLVM_TARGET_PATH_RE = re.compile(r"(^|/)llvm/lib/Target/[A-Za-z0-9_]+/")
_INLINE_ASM_RE = re.compile(r"__asm__\s*(?:volatile)?\s*\(|asm\s*volatile\s*\(", re.MULTILINE)


def scan(root: Path) -> list[SourceFinding]:
    findings: list[SourceFinding] = []
    for path in iter_files(root):
        rel = relative_to(path, root).replace("\\", "/")
        suffix = path.suffix.lower()
        name = path.name
        if suffix == ".td":
            if _INTRINSICS_RE.search(name):
                findings.append(
                    SourceFinding(
                        kind="llvm_intrinsic_definition",
                        path=rel,
                        symbol=name,
                        evidence="LLVM Intrinsics TableGen file",
                        confidence=0.9,
                    )
                )
            if _RISCV_FEATURES_RE.search(name):
                findings.append(
                    SourceFinding(
                        kind="llvm_riscv_features",
                        path=rel,
                        symbol=name,
                        evidence="LLVM RISCV features TableGen",
                        confidence=0.9,
                    )
                )
        if _LLVM_TARGET_PATH_RE.search(rel):
            findings.append(
                SourceFinding(
                    kind="llvm_target_backend",
                    path=rel,
                    symbol=None,
                    evidence="File lives under llvm/lib/Target/<arch>/",
                    confidence=0.8,
                )
            )
        if suffix in {".c", ".cc", ".cpp", ".h", ".hpp"}:
            text = read_text(path) or ""
            if _INLINE_ASM_RE.search(text):
                findings.append(
                    SourceFinding(
                        kind="llvm_inline_asm",
                        path=rel,
                        symbol=None,
                        evidence="Inline asm/__asm__ block",
                        confidence=0.55,
                    )
                )
    return findings
