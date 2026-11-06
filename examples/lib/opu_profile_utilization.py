#!/usr/bin/env python3
"""Attribute an exact profiled Kodiak OPU image without claiming array occupancy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from merlin.runtime.backends import output_identity, zephyr_model

BITSTREAM = "alveo_u250_firesim_kodiak_opu_1core"
PEAK_MAC_PER_CYCLE = 1024
TILE_EDGE = 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def _profile_rows(console: str) -> dict[int, tuple[int, int]]:
    rows: dict[int, tuple[int, int]] = {}
    for line in console.splitlines():
        if not line.startswith("PROF "):
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"malformed PROF line: {line[:120]}")
        try:
            op_id, ticks, hits = map(int, parts[1:])
        except ValueError as exc:
            raise ValueError(f"non-integer PROF line: {line[:120]}") from exc
        if op_id in rows or op_id < 0 or ticks < 0 or hits < 0:
            raise ValueError(f"duplicate or invalid PROF row {op_id}")
        rows[op_id] = (ticks, hits)
    return rows


def _hardware_identity(summary: Path) -> dict:
    from merlin.common import provenance

    facts = {}
    for line in summary.read_text().splitlines():
        key, sep, value = line.partition(":")
        if sep:
            facts[key.strip()] = value.strip()
    if facts.get("RuntimeHWConfig") != BITSTREAM:
        raise ValueError("FireSim did not report the pinned one-core OPU config")
    tar = Path(facts.get("Bitstream tar", "").removeprefix("file://"))
    art = provenance.load_artifacts()["firesim_kodiak_opu_1core"]
    if art.resolve() != tar:
        raise ValueError("FireSim bitstream tar differs from the pinned artifact path")
    verified = provenance.verify_artifact("firesim_kodiak_opu_1core")
    if not verified.ok:
        raise ValueError(f"pinned FireSim bitstream digest failed: {verified.gaps}")
    return {"config": BITSTREAM, "tar": str(tar), "sha256": verified.digest}


def measure(profile_dir: Path, console_path: Path, *, hw_summary: Path | None = None) -> dict:
    """Require exact output, all profile rows, and a SHA-checked routing numerator."""
    profile_dir = profile_dir.resolve()
    if hw_summary and console_path.resolve().parent != hw_summary.resolve().parent:
        raise ValueError("FireSim UART and HW_CFG_SUMMARY must come from the same job directory")
    leg = _json(profile_dir / "leg.json")
    gate = _json(profile_dir / "spike_gate.json")
    if leg["board"] != "firesim_kodiak_opu" or leg["harts"] != 1 or leg["vlen"] != 512 or not leg["profile"]:
        raise ValueError("this report requires the profiled one-core Kodiak OPU image")
    elf = Path(leg["elf"])
    if _sha(elf) != gate["elf_sha256"] or leg["build_hash"] != gate["build_hash"]:
        raise ValueError("profiled ELF differs from its exact Spike gate")
    if gate["spike_status"] != "verified":
        raise ValueError("profiled image did not pass its exact Spike gate")

    console = console_path.read_text(errors="replace")
    if "DONE" not in console.splitlines():
        raise ValueError("profiled image did not reach DONE")
    result = zephyr_model._parse_console(console, 0)
    if result["metrics"].get("build_hash") != leg["build_hash"]:
        raise ValueError("console build hash differs from profiled image")
    if output_identity.output_identity(result) != gate["output_identity"]:
        raise ValueError("console full hash or prefix differs from exact Spike gate")
    cycles = result["metrics"].get("cycles")
    if not isinstance(cycles, int) or cycles <= 0:
        raise ValueError("missing positive target METRIC cycles")
    if result["metrics"].get("iters") != 1:
        raise ValueError("OPU call-window attribution requires exactly one timed inference")

    table_path = profile_dir / "op_profile_table.json"
    pinned = leg.get("sidecar_sha256") or {}
    table_pins = [x for x in (gate.get("profile_table_sha256"), pinned.get("op_profile_table.json")) if x]
    if not table_pins or any(_sha(table_path) != x for x in table_pins):
        raise ValueError("profile table differs from the exact Spike gate receipt")
    table = _json(table_path)
    if not isinstance(table, list) or {row["id"] for row in table} != set(range(len(table))):
        raise ValueError("profile table IDs must be unique and contiguous")
    if result["metrics"].get("prof_marks") != len(table) + 1:
        raise ValueError("profile mark count differs from one complete forward invocation")
    rows = _profile_rows(console)

    signatures_path = profile_dir / "opu_signatures.json"
    signature_pins = [x for x in (gate.get("opu_signatures_sha256"), pinned.get("opu_signatures.json")) if x]
    if not signature_pins or any(_sha(signatures_path) != x for x in signature_pins):
        raise ValueError("OPU signatures differ from the exact Spike gate receipt")
    signatures = _json(signatures_path)
    routed = signatures["routed"]
    call_ids = [
        row["id"]
        for row in table
        if row.get("mlir_op") == "func.call" and str(row.get("callee") or "").startswith("@merlin_opu_")
    ]
    if len(call_ids) != signatures["count"] or len(routed) != signatures["count"]:
        raise ValueError("profiled OPU calls differ from the exact routing sidecar")
    profiled_symbols = Counter(str(table[op_id]["callee"]).removeprefix("@") for op_id in call_ids)
    routed_symbols = Counter(row["symbol"] for row in routed)
    if profiled_symbols != routed_symbols:
        raise ValueError("profiled OPU callee multiset differs from the exact routing sidecar")
    dump_mode = leg.get("profile_dump", "all")
    if dump_mode not in {"all", "opu"}:
        raise ValueError(f"unknown profile dump mode {dump_mode!r}")
    expected_ids = set(call_ids) if dump_mode == "opu" else set(range(len(table)))
    if set(rows) != expected_ids or any(hits == 0 for _, hits in rows.values()):
        raise ValueError("console lacks the exact positive-hit rows selected by the profiled image")
    emitted = result["metrics"].get("prof_dump_selected")
    if (dump_mode == "opu" and emitted != len(call_ids)) or (emitted is not None and emitted != len(rows)):
        raise ValueError("reported profile selector count differs from the emitted rows")
    attributed = result["metrics"].get("prof_total_ticks")
    printed_ticks = sum(ticks for ticks, _ in rows.values())
    if not isinstance(attributed, int) or printed_ticks > attributed or attributed > cycles:
        raise ValueError("profile ticks do not reconcile with the timed inference")
    if dump_mode == "all" and printed_ticks != attributed:
        raise ValueError("full profile rows do not sum to prof_total_ticks")
    valid_macs = sum(row["b"] * row["m"] * row["n"] * row["k"] for row in routed)
    if valid_macs != signatures["macs"]["routed"]:
        raise ValueError("valid MACs differ from exact routing sidecar")
    padded_slots = sum(
        row["b"]
        * ((row["m"] + TILE_EDGE - 1) // TILE_EDGE * TILE_EDGE)
        * ((row["n"] + TILE_EDGE - 1) // TILE_EDGE * TILE_EDGE)
        * row["k"]
        for row in routed
    )
    opu_ticks = sum(rows[op_id][0] for op_id in call_ids)
    if opu_ticks <= 0 or valid_macs > PEAK_MAC_PER_CYCLE * opu_ticks:
        raise ValueError("OPU call window is missing or shorter than the physical MAC lower bound")
    whole_fraction = valid_macs / (PEAK_MAC_PER_CYCLE * cycles)
    window_fraction = valid_macs / (PEAK_MAC_PER_CYCLE * opu_ticks)
    share = opu_ticks / cycles
    if abs(share * window_fraction - whole_fraction) > 1e-12:
        raise ValueError("OPU-window and whole-inference factors do not reconcile")

    hardware = _hardware_identity(hw_summary) if hw_summary else None
    return {
        "schema": "merlin.opu_profile_utilization.v1",
        "source": "firesim" if hardware else "spike",
        "board": leg["board"],
        "bitstream": hardware,
        "elf_sha256": gate["elf_sha256"],
        "console_sha256": _sha(console_path),
        "profile_table_sha256": _sha(table_path),
        "opu_signatures_sha256": _sha(signatures_path),
        "build_hash": leg["build_hash"],
        "full_output_identity": gate["output_identity"],
        "target_inference_cycles": cycles,
        "attributed_op_cycles": attributed,
        "profile_coverage": attributed / cycles,
        "profile_dump": dump_mode,
        "routed_call_sites": len(call_ids),
        "valid_routed_macs": valid_macs,
        "padded_opu_mac_slots": padded_slots,
        "static_tile_fill": valid_macs / padded_slots,
        "physical_peak_mac_per_target_cycle": PEAK_MAC_PER_CYCLE,
        "opu_call_window_cycles": opu_ticks,
        "opu_call_time_share": share,
        "opu_window_effective_peak_fraction": window_fraction,
        "whole_inference_effective_peak_fraction": whole_fraction,
        "array_active_occupancy": None,
        "occupancy_reason": "the pinned bitstream has no active MAC row counter",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_dir", type=Path, help="exact profiled image directory")
    parser.add_argument("--console", type=Path, help="UART log; default: profile_dir/spike_console.log")
    parser.add_argument("--hw-summary", type=Path, help="FireSim HW_CFG_SUMMARY beside the UART log")
    parser.add_argument("--output", type=Path, help="write the report JSON here as well as stdout")
    args = parser.parse_args()
    report = measure(
        args.profile_dir, args.console or args.profile_dir / "spike_console.log", hw_summary=args.hw_summary
    )
    body = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(body)
    print(body, end="")


if __name__ == "__main__":
    main()
