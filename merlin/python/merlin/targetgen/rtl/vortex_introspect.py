"""Deterministic, no-LLM extraction of Vortex hardware facts from the real RTL.

The Vortex analog of :mod:`muon_introspect` (Muon) and :mod:`circt_introspect` (Gemmini) — but it takes
gemmini's road, not muon's. Muon reads elaborated **FIRRTL** out of chipyard's VCS generated-src for
``RadianceMuonConfig``; Vortex is not a chipyard design and has no FIRRTL. What it does have is a CIRCT
**HW-dialect** import produced by its own ``hw/syn/circt`` flow (``Vortex-sv2v.mlir``), which is exactly
the input mlc's target-agnostic ``discover`` machinery consumes. So this module derives facts by walking
that op graph, and holds no Vortex-specific parsing of its own.

WHAT IS DERIVED, AND WHAT DELIBERATELY IS NOT
---------------------------------------------
Two facts come cleanly out of the decoder and are reported:

* ``isa.opcodes`` — the legal 7-bit RISC-V opcode set, from the fan-out of equality comparisons against
  the shared opcode field (mlc ``discover_opcode_set``). This is the fact that matters for this target:
  it contains ``0x0b`` (CUSTOM0, the SIMT extension) and ``0x2b`` (CUSTOM1), so a compiler can be told
  what the silicon actually decodes rather than what a header claims.
* ``isa.cta_csrs`` — the CTA identity/geometry CSR block, from the 12-bit CSR decode fan-out. The whole
  contiguous ``0xcd0``-``0xce1`` range is decoded, including the five the kernel ABI names.

The other four fields ``_simt_fact_bundle`` carries (``simt``, ``registers``, ``shared_memory``,
``fp_datapath``) are reported UNDERIVED, on purpose:

* **geometry is a build parameter, not an RTL fact.** Vortex elaborates from ``VX_CFG_*`` macros, so
  "2 clusters x 2 cores x 8 warps x 8 threads" is an experiment decision that the descriptor freezes
  (``hardware_spec.config``) — not something to re-assert as discovered-from-silicon. Claiming RTL
  provenance for a config choice is precisely the false-provenance pattern this tree strips out.
* **the mesh heuristic actively misfires here.** ``geometry.discover_mesh_dim`` reports
  ``dim=8 (VX_csa_32/FullAdder x64)`` — a 64-cell carry-save adder tree, not an 8x8 systolic mesh. It is
  a systolic-shaped question asked of a SIMT design. Reporting it would be worse than reporting nothing.
* ``recognize_fpu_datapath`` and ``discover_fpnew_formats`` both return None on this graph (Vortex's FPU
  is not an FPnew instance), and no cheap signal replaced them.

Absence here means "not determined", never "not present" — the same rule the SIMT coverage checker uses
for shared-memory classes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# The identity register_simt_introspect keys on (mlc_bridge._SIMT_INTROSPECTS), the muon_introspect
# analog. A target whose arc_target is unset resolves its introspect by this name directly.
TARGET = "vortex"
GENERATOR = "merlin.targetgen.rtl.vortex_introspect"
VERSION = "vortex-introspect-v1"

# The CTA identity/geometry CSRs the kernel ABI names. Used ONLY to annotate which discovered CSR
# addresses are the ones a compiler must emit — the discovered set is whatever the RTL decodes.
CTA_CSRS: dict[int, str] = {
    0xCD3: "thread_id.x", 0xCD6: "block_id.x", 0xCD9: "block_dim.x",
    0xCDF: "local_mem_base", 0xCE1: "cta_entry",
}
CSR_WIDTH = 12          # RISC-V CSR address width
OPCODE_WIDTH = 7        # RISC-V base opcode width


def hw_dialect_path() -> Path | None:
    """The Vortex HW-dialect import, checkout-relative under the descriptor's `rtl.root_env`.

    Honors ``MERLIN_VORTEX_HW_DIALECT`` for an out-of-tree override; otherwise resolves the checkout the
    same way the oracle and the sandbox do (``MERLIN_EXT_VORTEX``, else ``VORTEX_HOME``).
    """
    from merlin.common import paths as _paths
    explicit = _paths.env("MERLIN_VORTEX_HW_DIALECT")
    if explicit:
        return Path(explicit)
    try:
        root = _paths.ext_path("vortex")
    except KeyError:
        home = _paths.env("VORTEX_HOME")
        root = Path(home) if home else None
    if root is None:
        return None
    return root / "build" / "hw" / "syn" / "circt" / "build_Vortex" / "Vortex-sv2v.mlir"


def circt_opt_path() -> Path | None:
    """The ``circt-opt`` that lowers the HW dialect to generic form.

    mlc defaults to a ``third_party/circt/build/bin/circt-opt`` under its own checkout, which is not
    always populated; a sibling CIRCT build is the common case here. Explicit env wins.
    """
    from merlin.common import paths as _paths
    for cand in (_paths.env("MERLIN_CIRCT_OPT"), _paths.env("MERLIN_CIRCT_DIR")):
        if cand:
            p = Path(cand)
            p = p if p.is_file() else p / "build" / "bin" / "circt-opt"
            if p.is_file():
                return p
    try:
        p = _paths.ext_path("circt")
        p = p if p.is_file() else p / "build" / "bin" / "circt-opt"
        if p.is_file():
            return p
    except KeyError:
        pass
    import shutil
    found = shutil.which("circt-opt")
    return Path(found) if found else None


def available() -> bool:
    """True if the inputs for a real extraction are present (never raises)."""
    hw, circt = hw_dialect_path(), circt_opt_path()
    return bool(hw and hw.is_file() and circt and circt.is_file())


def _cache_path() -> Path:
    from merlin.common.paths import artifacts_dir
    return artifacts_dir() / "cache" / "rtl_introspect" / "vortex" / "vortex_facts.json"


def _discover(hw: Path, circt: Path) -> dict[str, Any]:
    """Walk the HW graph once and pull both decoder facts out of it."""
    from mlc.discover.decode import discover_decode_signals, discover_opcode_set
    from mlc.discover.irgraph import load_hw_graph

    graph = load_hw_graph(hw, circt_opt=circt)
    out: dict[str, Any] = {}

    sig = discover_opcode_set(graph, expected_width=OPCODE_WIDTH)
    if sig is not None and sig.width == OPCODE_WIDTH:
        out["opcodes"] = sorted(sig.values)
        out["opcode_module"] = sig.module
        out["opcode_fanout"] = sig.fanout

    # CSR addresses: union every CSR-width sparse decode. min_fanout=2 because the CTA block is split
    # across several comparison groups; the union is the decoded set.
    csrs: set[int] = set()
    for s in discover_decode_signals(graph, min_fanout=2, max_width=CSR_WIDTH):
        if s.width == CSR_WIDTH:
            csrs |= set(s.values)
    if csrs:
        out["csrs"] = sorted(csrs)
        out["cta_csrs"] = {hex(v): n for v, n in sorted(CTA_CSRS.items()) if v in csrs}
        out["cta_csrs_missing"] = {hex(v): n for v, n in sorted(CTA_CSRS.items()) if v not in csrs}
    return out


def build_facts(*, refresh: bool = False) -> dict[str, Any]:
    """The provenance-stamped Vortex fact bundle (same shape as ``muon_introspect.build_facts``).

    Degrades honestly: with no HW dialect or no ``circt-opt`` the facts dict is empty and
    ``inputs.rtl_present`` is False — the caller reports undetermined rather than guessing.
    """
    hw, circt = hw_dialect_path(), circt_opt_path()
    present = bool(hw and hw.is_file() and circt and circt.is_file())

    cache = _cache_path()
    discovered: dict[str, Any] = {}
    if present:
        # The graph walk costs ~a minute on a 1.5 MB import, so cache it. PURGEABLE — regenerate with
        # refresh=True or by deleting the file.
        if cache.is_file() and not refresh:
            try:
                discovered = json.loads(cache.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 — a corrupt cache must never be fatal
                discovered = {}
        if not discovered:
            discovered = _discover(hw, circt)
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(discovered, indent=2), encoding="utf-8")
            except OSError:
                pass

    facts: dict[str, Any] = {}
    if discovered.get("opcodes") or discovered.get("csrs"):
        ops = discovered.get("opcodes") or []
        csrs = discovered.get("csrs") or []
        facts["isa"] = {
            "opcodes": [hex(v) for v in ops],
            "custom0_present": 0x0B in ops,
            "custom1_present": 0x2B in ops,
            "csr_count": len(csrs),
            "cta_csrs": discovered.get("cta_csrs", {}),
            "cta_csrs_missing": discovered.get("cta_csrs_missing", {}),
            "evidence": (
                f"{hw}: comb.icmp eq/ceq fan-out in @{discovered.get('opcode_module', '?')} — "
                f"{discovered.get('opcode_fanout', 0)} distinct {OPCODE_WIDTH}-bit opcode compares "
                f"(CUSTOM0 0x0b {'present' if 0x0B in ops else 'ABSENT'}), and {len(csrs)} distinct "
                f"{CSR_WIDTH}-bit CSR addresses including the CTA identity block"),
        }
    return {
        "schema_version": "1.0",
        "generator": {
            "name": GENERATOR,
            "version": VERSION,
            "method": ("Vortex CIRCT HW-dialect import (hw/syn/circt -> Vortex-sv2v.mlir) walked with "
                       "mlc.discover irgraph+decode; equality fan-out over eq AND four-state ceq"),
        },
        "inputs": {
            "hw_dialect": str(hw) if hw else None,
            "circt_opt": str(circt) if circt else None,
            "rtl_present": present,
        },
        "facts": facts,
        # Stated so a consumer never mistakes silence for absence. See the module docstring for why each
        # is not derived: geometry is a BUILD PARAMETER the descriptor freezes, and the mesh/FPU
        # recognizers are systolic-shaped questions that either misfire or return nothing on this design.
        "undetermined": {
            "simt": "geometry is a VX_CFG_* build parameter, frozen by the experiment descriptor",
            "registers": "no register-file signal recovered from the HW dialect",
            "shared_memory": "not separated from the cache hierarchy by the generic discover pass",
            "fp_datapath": "recognize_fpu_datapath/discover_fpnew_formats return None (not FPnew)",
        },
    }


def _main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Extract Vortex RTL facts from the CIRCT HW dialect.")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and re-walk the graph")
    a = ap.parse_args(argv)
    print(json.dumps(build_facts(refresh=a.refresh), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
