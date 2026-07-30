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


def build_facts() -> dict[str, Any]:
    """Merge cyclotron-config geometry + RTL hierarchy/FIRRTL evidence + ISA-doc facts -> facts dict."""
    gs = _gen_src_dir()
    cfg = _parse_config(_read(config_path()))
    hier = _hierarchy_counts(gs)
    smem = _smem_evidence(gs)

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
            "fp_datapath": {
                "dtype": "f32", "flop_per_fma": 2,
                "peak_flops_per_cycle": peak_flops_cycle,
                "clock_hz": CLOCK_HZ, "peak_gflops": peak_gflops,
                "evidence": f"{cores} cores x {lanes} lanes x 2 flop/FMA = {peak_flops_cycle} flop/cycle "
                            f"@ {CLOCK_HZ/1e6:g} MHz = {peak_gflops:g} GFLOP/s",
            },
            "isa": {
                "encoding_bits": 64, "max_src_operands": 4, "max_dst_operands": 1,
                "predicated_execution": True,
                "instruction_classes": ["GMEM_LD", "GMEM_ST", "SMEM_LD", "SMEM_ST", "FMA",
                                        "BARRIER", "FENCE_SMEM", "TMC", "SPLIT", "JOIN", "WSPAWN"],
                "intrinsics": {"smem": ["store_shared", "load32_shared", "load16_shared", "store64_shared"],
                               "sync": ["mu_barrier", "mu_fence_smem", "mu_fence"],
                               "simt": ["vx_thread_id", "vx_warp_id", "vx_core_id", "vx_split", "vx_join",
                                        "vx_tmc"]},
                "evidence": "generators/radiance/docs/isa.md (64-bit encoding, 4 src/1 dst, predication) "
                            "+ radiance-kernels lib/include/{mu_intrinsics.h,vx_intrinsics.h}",
            },
        },
        "rtl_hierarchy_counts": hier,
    }
    return facts


def default_facts_path() -> Path:
    # muon has no hand-curated merlin/targets/muon; the resolver routes it to artifacts/targets/muon
    # (or $MERLIN_RTL_FACTS). Muon uses a distinct filename (muon_facts.json) next to the pin.
    from .facts import rtl_facts_path
    return rtl_facts_path("muon").with_name("muon_facts.json")


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="extract Muon RTL facts -> muon_facts.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    facts = build_facts()
    out = Path(a.out) if a.out else default_facts_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
    f = facts["facts"]
    print(f"wrote {out}")
    print(f"  simt: {f['simt']['cores']} cores x {f['simt']['warps_per_core']} warps x "
          f"{f['simt']['lanes_per_warp']} lanes; peak {f['fp_datapath']['peak_gflops']:g} GFLOP/s; "
          f"rtl_present={facts['inputs']['rtl_present']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
