"""Deterministic, no-LLM extraction of Muon (RadianceMuonConfig) hardware facts from the real RTL.

The Muon analog of :mod:`merlin.targetgen.rtl.circt_introspect` (Gemmini). The hardware is the source
of truth: we read the **elaborated FIRRTL + module hierarchy** CIRCT/firtool already produced for
``RadianceMuonConfig`` (under ``$MERLIN_CHIPYARD/sims/vcs/generated-src/...``) plus the cyclotron perf
config and the committed ISA docs, and emit one provenance-stamped ``muon_facts.json`` -- each fact
carrying the exact RTL/config token it came from. Five consumers read that artifact (digest, advisory
checker, ...) instead of re-deriving the geometry by crawling Radiance Scala.

This module is target-gen for Muon ONLY; it imports nothing Gemmini-specific and never runs the agent.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# The single target this SIMT introspect serves. It lives HERE (the target-specific introspect
# module — the Muon analog of circt_introspect for Gemmini), not in the kind-routed core dispatch
# (mlc_bridge), so the core stays target-name-free: the bridge routes by compute-unit KIND and asks
# this module which target it is the introspect for.
TARGET = "muon"
DEFAULT_CHIPYARD = "/path/to/chipyard"
DEFAULT_CONFIG = "/path/to/autocomp/scripts/muon/config_muon.toml"
VCS_CONFIG = "RadianceMuonConfig"

# Muon SIMT FP peak: cores * lanes * 2 flop/FMA (fp32) at 500 MHz. The denominator we report against
# is the cyclotron perf model we actually run (config_muon.toml: 2 cores x 16 lanes -> 64 flop/cycle).
CLOCK_HZ = 500_000_000


def chipyard_root() -> Path:
    return Path(os.environ.get("MERLIN_CHIPYARD", DEFAULT_CHIPYARD))


def config_path() -> Path:
    return Path(os.environ.get("MERLIN_MUON_CONFIG", DEFAULT_CONFIG))


def _gen_src_dir() -> Path:
    return (chipyard_root() / "sims/vcs/generated-src"
            / f"chipyard.harness.TestHarness.{VCS_CONFIG}")


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _parse_config(text: str) -> dict[str, int]:
    """num_lanes / num_warps / num_cores / num_regs from config_muon.toml [muon] (via tomllib)."""
    import tomllib
    try:
        muon = tomllib.loads(text).get("muon", {})
    except tomllib.TOMLDecodeError:
        return {}
    return {k: int(muon[k]) for k in ("num_lanes", "num_warps", "num_cores", "num_regs")
            if isinstance(muon.get(k), int)}


def _hierarchy_counts(gs: Path) -> dict[str, int]:
    """Count Muon module names present in the elaborated module hierarchy JSON (RTL evidence)."""
    txt = _read(gs / "top_module_hierarchy.json")
    # each name appears as a quoted JSON string token; count literal quoted occurrences (no regex).
    return {name: txt.count(f'"{name}"') for name in ("MuonCore", "MuonTile", "RadianceCluster")}


def _smem_evidence(gs: Path) -> str | None:
    """The first shared-memory ``smem mem : UInt<W>[E] [D]`` line in the elaborated FIRRTL, parsed
    structurally (locate the marker, then read the ``<W>`` width and the two ``[N]`` dims) — no regex."""
    fir = _read(gs / f"chipyard.harness.TestHarness.{VCS_CONFIG}.fir")
    marker = "smem mem : UInt<"
    idx = fir.find(marker)
    if idx == -1:
        return None
    rest = fir[idx + len(marker):]
    width, sep, after = rest.partition(">")
    if not sep or not width.strip().isdigit():
        return None
    dims: list[str] = []
    pos = 0
    while len(dims) < 2:
        lb = after.find("[", pos)
        rb = after.find("]", lb) if lb != -1 else -1
        if lb == -1 or rb == -1:
            break
        dims.append(after[lb + 1:rb].strip())
        pos = rb + 1
    if len(dims) < 2 or not (dims[0].isdigit() and dims[1].isdigit()):
        return None
    return f"smem mem : UInt<{width.strip()}>[{dims[0]}] [{dims[1]}]"


def _address_map(gs: Path) -> dict[str, Any] | None:
    """The SoC address map from the elaborated design's ``*.memmap.json``.

    Every MMIO/memory region the design actually exposes, with the device-tree name it was elaborated
    under. This is the ONLY sound source for a device address on this target: a literal would bake in
    one config's map, and the facts block carried no address information at all, which is how the
    device console came to store into an address the SoC does not map.

    Fails closed (``None``) when the elaboration output is absent -- a missing map must read as
    UNKNOWN, never as an empty map that a consumer could mistake for "the SoC has no devices".
    """
    for p in sorted(gs.glob("*.memmap.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        regions = []
        for e in doc.get("mapping") or []:
            names = e.get("names") or []
            base = (e.get("base") or [None])[0]
            size = (e.get("size") or [None])[0]
            if base is None or size is None or not names:
                continue
            regions.append({
                "name": str(names[0]), "base": int(base), "size": int(size),
                "r": bool((e.get("r") or [False])[0]), "w": bool((e.get("w") or [False])[0]),
            })
        if regions:
            return {"regions": sorted(regions, key=lambda r: r["base"]),
                    "config": gs.name.rpartition(".")[2],
                    "evidence": f"elaborated address map {p.name}: {len(regions)} region(s)"}
    return None


def _console_device(gs: Path, amap: dict[str, Any] | None) -> dict[str, Any] | None:
    """The device the elaborated design designates as its console, from the device tree's
    ``stdout-path``.

    Parsed STRUCTURALLY (partition/split, never a regex, per the repo's no-regex rule): ``stdout-path``
    names a label (``&L44``); the label is then defined as ``L44: serial@10020000``. Resolving through
    the label -- rather than assuming a node called "serial" -- is what keeps this target-agnostic.

    Returns ``None`` when the dts is absent or names no console, so a consumer fails closed instead of
    guessing an address.
    """
    label = None
    for p in sorted(gs.glob("*.dts")):
        text = _read(p)
        if not text:
            continue
        for line in text.splitlines():
            if "stdout-path" in line and "&" in line:
                _, _, rest = line.partition("&")
                label = rest.strip().rstrip(";").strip()
                break
        if not label:
            continue
        # the label's own definition line: "<label>: <node>@<hexbase> {"
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith(f"{label}:"):
                continue
            node = stripped.partition(":")[2].strip().partition("{")[0].strip()
            name, _, base_hex = node.partition("@")
            if not base_hex:
                continue
            try:
                base = int(base_hex, 16)
            except ValueError:
                continue
            size = None
            for r in ((amap or {}).get("regions") or []):
                if r["base"] == base:
                    size = r["size"]
                    break
            return {"node": node, "device": name.strip(), "base": base, "size": size,
                    "label": label,
                    "evidence": f"{p.name}: chosen/stdout-path -> &{label} -> {node}"}
    return None


def _isa_block() -> dict[str, Any]:
    """The ISA facts, DERIVED from the mlc ``isa_encoding`` fact (the RTL-decoder field layout + opcode
    table + address-space macros) rather than a hand-copied list. Encoding width, the src/dst operand
    counts, predication, the opcode classes, and the address spaces all come from the derived fact; the
    intrinsic-helper names stay from the HW's own headers. Falls back to the ISA-doc values when the fact is
    not yet derived (mlc/cache absent) — an honest degrade, marked in the evidence."""
    intrinsics = {"smem": ["store_shared", "load32_shared", "load16_shared", "store64_shared"],
                  "sync": ["mu_barrier", "mu_fence_smem", "mu_fence"],
                  "simt": ["vx_thread_id", "vx_warp_id", "vx_core_id", "vx_split", "vx_join", "vx_tmc"]}
    try:
        from merlin.targetgen.rtl import mlc_bridge
        fact = mlc_bridge.isa_encoding_for("muon")
    except Exception:  # noqa: BLE001 — mlc/cache absent
        fact = None
    if fact and fact.get("fields"):
        fields = fact.get("fields", {})
        return {
            "encoding_bits": fact.get("inst_width"),
            "max_src_operands": sum(1 for k in fields if k.startswith("rs")),
            "max_dst_operands": sum(1 for k in fields if k == "rd"),
            "predicated_execution": "pred" in fields,
            "instruction_classes": sorted(fact.get("opcodes", {})),
            "address_spaces": fact.get("address_spaces", {}),
            "intrinsics": intrinsics,
            "evidence": "DERIVED from mlc isa_encoding fact (muon_isa.json): RTL-decoder field layout + "
                        "opcode table + address-space macros; intrinsic names from lib/include/*intrinsics.h",
        }
    return {
        "encoding_bits": 64, "max_src_operands": 4, "max_dst_operands": 1, "predicated_execution": True,
        "instruction_classes": [], "intrinsics": intrinsics,
        "evidence": "isa_encoding fact not yet derived (mlc absent); ISA-doc fallback "
                    "(isa.md 64-bit encoding, 4 src/1 dst, predication)",
    }


def build_facts() -> dict[str, Any]:
    """Merge cyclotron-config geometry + RTL hierarchy/FIRRTL evidence + ISA-doc facts -> facts dict."""
    gs = _gen_src_dir()
    cfg = _parse_config(_read(config_path()))
    hier = _hierarchy_counts(gs)
    smem = _smem_evidence(gs)
    amap = _address_map(gs)
    console = _console_device(gs, amap)

    lanes = cfg.get("num_lanes", 16)
    warps = cfg.get("num_warps", 8)
    cores = cfg.get("num_cores", 2)
    regs = cfg.get("num_regs", 256)
    peak_flops_cycle = cores * lanes * 2
    peak_gflops = peak_flops_cycle * CLOCK_HZ / 1e9

    facts = {
        "schema_version": "1.0",
        "generator": {
            "name": "merlin.targetgen.rtl.muon_introspect",
            "version": "muon-introspect-v1",
            "method": ("config_muon.toml [muon] geometry (cyclotron perf model we report against) + "
                       "RadianceMuonConfig elaborated FIRRTL/module-hierarchy evidence + Radiance ISA docs"),
        },
        "inputs": {
            "config_toml": str(config_path()),
            "rtl_generated_src": str(gs),
            "rtl_present": gs.is_dir(),
        },
        "facts": {
            "simt": {
                "lanes_per_warp": lanes, "warps_per_core": warps, "cores": cores,
                "threads_per_core": lanes * warps,
                "evidence": (f"config_muon.toml: num_lanes={lanes}, num_warps={warps}, "
                             f"num_cores={cores}; RTL hierarchy MuonCore/MuonTile present="
                             f"{bool(hier.get('MuonCore'))}"),
            },
            "registers": {
                "arch_max": 256, "compiler_limit": 128, "config": regs,
                "evidence": "Radiance ISA docs (isa.md: up to 256 arch regs; muon.md: compiler limits to 128)",
            },
            "shared_memory": {
                "bytes_per_cluster": 128 * 1024,
                "rtl_bank_evidence": smem or "smem mem line not found in FIRRTL",
                "evidence": "RadianceBaseConfig TapeoutSmemConfig (128 KiB/cluster); RTL FIRRTL smem mem bank",
            },
            # The SoC address map and the console device it designates. Absent from this facts block
            # until now, which is why the device console had no derived address to target and kept
            # storing to the Vortex IO_COUT_ADDR aperture -- an address this SoC maps no device at, so
            # every OUT/DONE byte was dropped and every RTL grade came back completion-only.
            # UNKNOWN (null) when the elaboration output is absent: fail closed, never guess an address.
            "address_map": amap or {"regions": None,
                                    "evidence": f"no *.memmap.json under {gs} — address map UNKNOWN"},
            "console": console or {"base": None,
                                   "evidence": f"no *.dts stdout-path under {gs} — console UNKNOWN"},
            "fp_datapath": {
                "dtype": "f32", "flop_per_fma": 2,
                "peak_flops_per_cycle": peak_flops_cycle,
                "clock_hz": CLOCK_HZ, "peak_gflops": peak_gflops,
                "evidence": f"{cores} cores x {lanes} lanes x 2 flop/FMA = {peak_flops_cycle} flop/cycle "
                            f"@ {CLOCK_HZ/1e6:g} MHz = {peak_gflops:g} GFLOP/s",
            },
            "isa": _isa_block(),
        },
        "rtl_hierarchy_counts": hier,
    }
    return facts


def default_facts_path() -> Path:
    # muon has no hand-curated merlin/targets/muon; the resolver routes it to artifacts/targets/muon
    # (or $MERLIN_RTL_FACTS). Muon uses a distinct filename (muon_facts.json) next to the pin.
    from merlin.targetgen.rtl.facts import rtl_facts_path
    return rtl_facts_path("muon").with_name("muon_facts.json")


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="extract Muon RTL facts -> muon_facts.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-downgrade", action="store_true",
                    help="permit a regeneration that HOLLOWS OUT facts the existing artifact carries. "
                         "Only for a genuine hardware change that really lost them — a hollowed fact is "
                         "otherwise the signature of a missing extractor (see MERLIN_MLC_DIR).")
    a = ap.parse_args(argv)
    facts = build_facts()
    out = Path(a.out) if a.out else default_facts_path()
    # Guarded write: an extraction that silently lost the isa block looks identical to a good one on the
    # way out, and this is the path a person runs by hand (ensure_facts' _warn_if_degraded does not cover
    # it). Refuse the downgrade here rather than leave it to be caught by eye.
    from merlin.targetgen.rtl.facts import FactsDowngrade, write_facts_guarded
    try:
        write_facts_guarded(out, facts, allow_downgrade=a.allow_downgrade)
    except FactsDowngrade as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    f = facts["facts"]
    print(f"wrote {out}")
    print(f"  simt: {f['simt']['cores']} cores x {f['simt']['warps_per_core']} warps x "
          f"{f['simt']['lanes_per_warp']} lanes; peak {f['fp_datapath']['peak_gflops']:g} GFLOP/s; "
          f"rtl_present={facts['inputs']['rtl_present']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
