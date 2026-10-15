"""Check a Phase 1 Spike selection against the pinned full FireSim target.

This is an operator preflight. It reads original files and never changes a
release, simulator, bitstream or compiler submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"selected Spike member is absent or indirect: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True,
    )
    return result.stdout.strip()


def _machine_file(row: dict, root: Path) -> Path:
    if not isinstance(row, dict) or row.get("env") != "MERLIN_CHIPYARD_GEMMINI_ROCKET":
        raise ValueError("Phase 2 Spike machine does not select the full Gemmini Chipyard root")
    relative = Path(row.get("join", ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Phase 2 Spike member has an unsafe relative path")
    return root / relative


def audit(contract: Path, chipyard_root: Path, machines: Path, hardware_pins: Path) -> dict:
    """Report exact Spike/tool/ABI agreement with the Phase 2 full U250 machine."""
    selected = yaml.safe_load(contract.read_text())
    machine_doc = yaml.safe_load(machines.read_text())
    registry = yaml.safe_load(hardware_pins.read_text())
    machine = machine_doc["machines"]["full_u250_board"]
    functional = machine_doc["machines"]["functional_model"]
    pins = registry["pins"]
    bitstream_pin = registry["artifacts"]["firesim_gemmini_rocket_30mhz"]
    matches = [
        row for row in registry["artifacts"].values()
        if row.get("role") == "firesim_bitstream" and machine["hw_config"] in row.get("hw_configs", [])
    ]
    if (
        len(matches) != 1
        or matches[0]["digest"] != bitstream_pin["digest"]
        or matches[0]["config"] != "FireSimGemminiRocketConfig"
        or bitstream_pin["built_from"] != ["gemmini_rocket_chipyard", "gemmini_rocket_rtl"]
    ):
        raise ValueError("Phase 2 full board no longer resolves to the selected FireSim pin")
    if not chipyard_root.is_dir() or chipyard_root.is_symlink():
        raise ValueError("selected full Gemmini Chipyard root is absent or indirect")
    command = functional["command"]
    identity_files = functional["identity_files"]
    if len(command) != 2 or command[1] != "--extension=gemmini" or len(identity_files) != 1:
        raise ValueError("Phase 2 functional machine has an unexpected Spike selection")
    expected_spike = _machine_file(command[0], chipyard_root)
    expected_extension = _machine_file(identity_files[0], chipyard_root)
    runner = selected["runner"]
    block = runner["chipyard_rocc"]
    spike = block["engines"]["spike"]["binary"]
    extension = runner["spike_extension"]
    if block["config"] != "GemminiRocketConfig" or extension["extension_name"] != "gemmini":
        raise ValueError("Phase 1 Spike selection names another Gemmini chip or extension")
    selected_spike = Path(spike["path"])
    selected_extension = Path(extension["extlib"])
    observed = {
        "phase1_spike_sha256": _sha256(selected_spike),
        "phase2_spike_sha256": _sha256(expected_spike),
        "phase1_extension_sha256": _sha256(selected_extension),
        "phase2_extension_sha256": _sha256(expected_extension),
        "phase2_chipyard_commit": _commit(chipyard_root),
        "phase2_gemmini_commit": _commit(chipyard_root / "generators/gemmini"),
        "phase2_abi_header_sha256": _sha256(
            chipyard_root / "generators/gemmini/software/gemmini-rocc-tests/include/gemmini_params.h"
        ),
    }
    checks = {
        "phase1_spike_pin_valid": observed["phase1_spike_sha256"] == spike["sha256"],
        "phase1_extension_pin_valid": observed["phase1_extension_sha256"] == extension["sha256"],
        "same_spike_binary": observed["phase1_spike_sha256"] == observed["phase2_spike_sha256"],
        "same_spike_extension": observed["phase1_extension_sha256"] == observed["phase2_extension_sha256"],
        "phase2_chipyard_source_pinned": (
            observed["phase2_chipyard_commit"] == pins["gemmini_rocket_chipyard"]["commit"]
        ),
        "phase2_gemmini_source_pinned": observed["phase2_gemmini_commit"] == pins["gemmini_rocket_rtl"]["commit"],
        "phase2_abi_header_pinned": observed["phase2_abi_header_sha256"] == machine["program_header_sha256"],
    }
    return {
        "schema": "merlin.phase1.spike_alignment.v1",
        "phase1_contract_sha256": _sha256(contract),
        "phase2_bitstream_sha256": bitstream_pin["digest"],
        "phase2_hw_config": machine["hw_config"],
        "checks": checks,
        "observed": observed,
        "spike_aligned": all(checks.values()),
        "rtl_equivalence_proven": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--chipyard-root", required=True, type=Path)
    parser.add_argument("--machines", required=True, type=Path)
    parser.add_argument("--hardware-pins", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit(args.contract, args.chipyard_root, args.machines, args.hardware_pins)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"Spike target alignment: {exc}\n")
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["spike_aligned"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
