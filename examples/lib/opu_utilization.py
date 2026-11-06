#!/usr/bin/env python3
"""Recompute OPU work and end-to-end throughput from one identity-verified FireSim pair.

This deliberately does not report array-active occupancy: the current bitstream
does not expose MAC-active cycles or valid lanes. Geometry is tied to the named
bitstream, never inferred from its performance result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# The one-core FireSim config in kodiak-mlir/firesim/01-saturn-OPUConfigs.patch
# instantiates Saturn with vLen=512, dLen=256 and 8-bit OPU operands. Add a
# future bitstream only after its own RTL geometry has been pinned and checked.
BITSTREAM_GEOMETRY = {"alveo_u250_firesim_kodiak_opu_1core": {"vlen_bits": 512, "dlen_bits": 256, "operand_bits": 8}}


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def measure(run_dir: Path) -> dict:
    """Use exact-image routing and target mcycle after hardware identity grading."""
    result = _read_json(run_dir / "firesim/results.json")
    leg = _read_json(run_dir / "device/leg.json")
    accepted_pair = result.get("all_verified") and all(
        result.get(part, {}).get("status") == "verified" for part in ("device", "control")
    )
    diagnostic_pair = result.get("diagnostic_only") and result.get("target_pair_hardware_verified")
    if not (accepted_pair or diagnostic_pair):
        raise ValueError("both FireSim legs need accepted grades or diagnostic hardware identity verification")
    if result.get("board") != leg.get("board") or result.get("harts") != leg.get("harts"):
        raise ValueError("FireSim result and exact-image leg disagree on board or hart count")
    if result["harts"] != 1:
        raise ValueError("this calculation requires the one-core bitstream")
    geometry = BITSTREAM_GEOMETRY.get(result["bitstream"])
    if geometry is None:
        raise ValueError("unrecognized bitstream: pin its RTL geometry before calculating utilization")
    dlen_bits, operand_bits = geometry["dlen_bits"], geometry["operand_bits"]
    vlen = leg["vlen"]
    if vlen != geometry["vlen_bits"]:
        raise ValueError("exact-image VLEN differs from the named bitstream's pinned geometry")
    if vlen % dlen_bits or vlen % operand_bits:
        raise ValueError("VLEN must divide into datapath beats and operand lanes")

    sidecar = leg["routing_sidecars"]["opu_signatures.json"]
    signatures_path = run_dir / "device/routing/opu_signatures.json"
    if hashlib.sha256(signatures_path.read_bytes()).hexdigest() != sidecar["sha256"]:
        raise ValueError("OPU signatures differ from the exact-image leg receipt")
    signatures = _read_json(signatures_path)
    edge = vlen // operand_bits
    beats_per_operand = vlen // dlen_bits
    peak_macs_per_cycle = (dlen_bits // operand_bits) ** 2
    if peak_macs_per_cycle * beats_per_operand**2 != edge**2:
        raise ValueError("tile and physical-array geometry disagree")

    routed = signatures["routed"]
    if len(routed) != signatures["count"]:
        raise ValueError("routed contraction count differs from the sidecar")
    valid_macs = 0
    padded_slots = 0
    for row in routed:
        b, m, n, k = (row[field] for field in ("b", "m", "n", "k"))
        if any(not isinstance(x, int) or x <= 0 for x in (b, m, n, k)):
            raise ValueError("each routed contraction needs positive integer b/m/n/k")
        valid_macs += b * m * n * k
        padded_slots += b * ((m + edge - 1) // edge * edge) * ((n + edge - 1) // edge * edge) * k
    if valid_macs != signatures["macs"]["routed"]:
        raise ValueError("calculated valid MACs differ from the routing receipt")
    other_macs = signatures["macs"]["not_routed"]
    device_cycles = result["device"]["cycles"]
    control_cycles = result["control"]["cycles"]
    if any(not isinstance(x, int) or x <= 0 for x in (device_cycles, control_cycles)):
        raise ValueError("verified target METRIC cycles must be positive integers")

    return {
        "diagnostic_only": bool(diagnostic_pair),
        "acceptance": bool(result.get("acceptance", accepted_pair)),
        "framework_fidelity": result.get("framework_fidelity", "not_checked"),
        "board": result["board"],
        "bitstream": result["bitstream"],
        "device_job": result["device"]["queue_job"],
        "control_job": result["control"]["queue_job"],
        "routed_contractions": len(routed),
        "valid_routed_macs": valid_macs,
        "other_contraction_macs": other_macs,
        "padded_opu_mac_slots": padded_slots,
        "routed_mac_share": valid_macs / (valid_macs + other_macs),
        "static_tile_fill": valid_macs / padded_slots,
        "physical_peak_mac_per_target_cycle": peak_macs_per_cycle,
        "device_compute_cycles": device_cycles,
        "control_compute_cycles": control_cycles,
        "valid_mac_per_device_target_cycle": valid_macs / device_cycles,
        "whole_inference_effective_peak_fraction": valid_macs / (peak_macs_per_cycle * device_cycles),
        "control_over_device_speedup": control_cycles / device_cycles,
        "array_active_occupancy": None,
        "occupancy_reason": "the bitstream has no MAC-active-cycle or valid-lane counter",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="immutable FireSim pair directory")
    args = parser.parse_args()
    print(json.dumps(measure(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
