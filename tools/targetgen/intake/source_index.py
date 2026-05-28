"""Aggregate source-tree scanners into a ``SourceInventory``."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from ..model import SourceFinding, SourceInventory, SourceRepository
from . import (
    chipyard_scanner,
    chisel_scanner,
    cmake_scanner,
    docs_scanner,
    hal_scanner,
    llvm_scanner,
    mlir_scanner,
    rtl_scanner,
    systemc_scanner,
)

ScannerFn = Callable[[Path], list[SourceFinding]]

AVAILABLE_SCANNERS: tuple[tuple[str, ScannerFn, tuple[str, ...]], ...] = (
    (mlir_scanner.NAME, mlir_scanner.scan, mlir_scanner.DETECTED_KINDS),
    (cmake_scanner.NAME, cmake_scanner.scan, cmake_scanner.DETECTED_KINDS),
    (llvm_scanner.NAME, llvm_scanner.scan, llvm_scanner.DETECTED_KINDS),
    (hal_scanner.NAME, hal_scanner.scan, hal_scanner.DETECTED_KINDS),
    (chipyard_scanner.NAME, chipyard_scanner.scan, chipyard_scanner.DETECTED_KINDS),
    (chisel_scanner.NAME, chisel_scanner.scan, chisel_scanner.DETECTED_KINDS),
    (rtl_scanner.NAME, rtl_scanner.scan, rtl_scanner.DETECTED_KINDS),
    (systemc_scanner.NAME, systemc_scanner.scan, systemc_scanner.DETECTED_KINDS),
    (docs_scanner.NAME, docs_scanner.scan, docs_scanner.DETECTED_KINDS),
)


def build_source_inventory(
    target: str,
    sources: Sequence[Path | str],
    *,
    scanners: Iterable[str] | None = None,
) -> SourceInventory:
    """Run scanners over ``sources`` and aggregate findings.

    ``sources`` is one or more directories (or single files); each becomes a
    ``SourceRepository`` entry. ``scanners`` optionally restricts which
    scanners run (defaults to all).
    """
    selected = _select_scanners(scanners)
    repositories: list[SourceRepository] = []
    findings: list[SourceFinding] = []
    detected: set[str] = set()
    missing: list[str] = []

    for raw in sources:
        path = Path(raw).resolve()
        if not path.exists():
            missing.append(f"source path does not exist: {raw}")
            continue
        repositories.append(
            SourceRepository(
                name=path.name,
                url=None,
                local_path=str(path),
                ref=None,
                source_kind_hints=[],
            )
        )
        for _, fn, _ in selected:
            for finding in fn(path):
                findings.append(finding)
                detected.add(finding.kind)

    return SourceInventory(
        target=target,
        repositories=repositories,
        findings=findings,
        detected_source_kinds=sorted(detected),
        missing_information=missing,
    )


def _select_scanners(
    names: Iterable[str] | None,
) -> tuple[tuple[str, ScannerFn, tuple[str, ...]], ...]:
    if names is None:
        return AVAILABLE_SCANNERS
    name_set = {n.lower() for n in names}
    selected = tuple(s for s in AVAILABLE_SCANNERS if s[0] in name_set)
    if not selected:
        raise ValueError(f"No scanners matched {sorted(name_set)}; available: " f"{[s[0] for s in AVAILABLE_SCANNERS]}")
    return selected
