"""Detect Verilog / SystemVerilog RTL surfaces."""

from __future__ import annotations

import re
from pathlib import Path

from ..model import SourceFinding
from ._common import iter_files, read_text, relative_to

NAME = "rtl"
DETECTED_KINDS: tuple[str, ...] = (
    "verilog_module",
    "systemverilog_interface",
    "rtl_axi_port",
    "rtl_tilelink_port",
    "rtl_dpi_c",
    "rtl_mmio_register",
)

_MODULE_RE = re.compile(r"^\s*module\s+(\w+)\b", re.MULTILINE)
_INTERFACE_RE = re.compile(r"^\s*interface\s+(\w+)\b", re.MULTILINE)
_AXI_RE = re.compile(r"\bAXI[34]?\b|\baxi(?:[_-]?(?:lite|stream|4|3|aw|w|b|ar|r))", re.IGNORECASE)
_TILELINK_RE = re.compile(r"\btilelink\b|\btl_[a-z]+_(?:i|o)\b", re.IGNORECASE)
_DPI_RE = re.compile(r"\bDPI[-_]C\b|\bimport\s+\"DPI[-_]C\"\b")
_REGMAP_RE = re.compile(r"\b(?:RegField|RegisterMap|reg_map|REG_MAP)\b")

# F6: Verilog-only MMIO control-plane heuristics.
#
# Pure-Verilog accelerators (NVDLA, Caliptra, Mempress, Snappy/Zstd
# offload, ...) often expose their control plane through one of three
# idioms that don't trigger the Chisel scanner. We detect MMIO when
# *any* of these patterns fire in a single Verilog file:
#
# (a) NVDLA CSB regfile — reg_rd_data / reg_offset / reg_wr_data /
#     reg_wr_en signals co-occur. Almost every NVDLA *_reg.v file has
#     all four; we require at least three.
#
# (b) Vivado AXI-Lite template — slv_reg<N> array + S_AXI_AWADDR-style
#     ports. The slv_reg name is enough on its own, since it's a
#     compile-time idiom no real signal would accidentally match.
#
# (c) Generic AXI-Lite handshake co-occurrence — awaddr + awvalid +
#     wdata + wvalid + bvalid all present, indicating a full
#     write-channel implementation rather than just incidental AXI ports.
_CSB_REG_TOKENS = ("reg_rd_data", "reg_offset", "reg_wr_data", "reg_wr_en")
_SLV_REG_RE = re.compile(r"\bslv_reg\d+\b")
_AXI_LITE_HANDSHAKE_TOKENS = ("awaddr", "awvalid", "wdata", "wvalid", "bvalid")


def _looks_like_mmio_register(text: str) -> tuple[bool, str | None]:
    """Return (is_mmio, evidence_string) for a Verilog file's contents."""
    csb_hits = sum(1 for tok in _CSB_REG_TOKENS if tok in text)
    if csb_hits >= 3:
        return True, f"NVDLA-style CSB regfile signals ({csb_hits}/4 of {_CSB_REG_TOKENS} present)"
    if _SLV_REG_RE.search(text):
        return True, "Vivado AXI-Lite slv_regN register array"
    handshake_hits = sum(1 for tok in _AXI_LITE_HANDSHAKE_TOKENS if tok in text)
    if handshake_hits >= 4:
        return (
            True,
            f"AXI-Lite write-channel handshake ({handshake_hits}/5 of {_AXI_LITE_HANDSHAKE_TOKENS})",
        )
    return False, None


def scan(root: Path) -> list[SourceFinding]:
    findings: list[SourceFinding] = []
    for path in iter_files(root):
        suffix = path.suffix.lower()
        if suffix not in {".v", ".sv", ".svh", ".vh"}:
            continue
        text = read_text(path) or ""
        rel = relative_to(path, root).replace("\\", "/")
        for match in _MODULE_RE.finditer(text):
            findings.append(
                SourceFinding(
                    kind="verilog_module",
                    path=rel,
                    symbol=match.group(1),
                    evidence=f"module {match.group(1)} declaration",
                    confidence=0.75,
                )
            )
        for match in _INTERFACE_RE.finditer(text):
            findings.append(
                SourceFinding(
                    kind="systemverilog_interface",
                    path=rel,
                    symbol=match.group(1),
                    evidence=f"interface {match.group(1)} declaration",
                    confidence=0.75,
                )
            )
        if _AXI_RE.search(text):
            findings.append(
                SourceFinding(
                    kind="rtl_axi_port",
                    path=rel,
                    symbol=None,
                    evidence="AXI naming convention found",
                    confidence=0.55,
                )
            )
        if _TILELINK_RE.search(text):
            findings.append(
                SourceFinding(
                    kind="rtl_tilelink_port",
                    path=rel,
                    symbol=None,
                    evidence="TileLink naming convention found",
                    confidence=0.55,
                )
            )
        if _DPI_RE.search(text):
            findings.append(
                SourceFinding(
                    kind="rtl_dpi_c",
                    path=rel,
                    symbol=None,
                    evidence="DPI-C import declaration",
                    confidence=0.7,
                )
            )
        if _REGMAP_RE.search(text):
            findings.append(
                SourceFinding(
                    kind="rtl_mmio_register",
                    path=rel,
                    symbol=None,
                    evidence="MMIO register-map naming",
                    confidence=0.55,
                )
            )
        # F6: Verilog-only MMIO control-plane idioms.
        is_mmio, evidence = _looks_like_mmio_register(text)
        if is_mmio:
            findings.append(
                SourceFinding(
                    kind="rtl_mmio_register",
                    path=rel,
                    symbol=None,
                    evidence=evidence or "MMIO control-plane signals co-occur",
                    confidence=0.7,
                )
            )
    return findings
