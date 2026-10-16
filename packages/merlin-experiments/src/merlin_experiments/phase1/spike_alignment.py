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

GSIM_CONFIG_SOURCE = "generators/firechip/chip/src/main/scala/GemminiGsimFireSimAlignedConfig.scala"
GSIM_CONFIG_SHA256 = "5392120d1ed547b3a70c3786dd5a3d3d697f7817f9ce55e1ffc25b6c2ff6b2ec"


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


def _gitlink(path: Path, member: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD:" + member],
        check=True, text=True, capture_output=True,
    )
    return result.stdout.strip()


def _selected_chipyard_source(root: Path, base: str) -> bool:
    if _sha256(root / GSIM_CONFIG_SOURCE) != GSIM_CONFIG_SHA256:
        return False
    if _commit(root) == base:
        return True
    ancestor = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", base, "HEAD"],
        capture_output=True, check=False,
    )
    changed = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", base, "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return (
        ancestor.returncode == 0
        and changed.stdout.splitlines() == [GSIM_CONFIG_SOURCE]
    )


def _machine_file(row: dict, root: Path) -> Path:
    if not isinstance(row, dict) or row.get("env") != "MERLIN_FIRE_ALIGNED_SPIKE_RUNTIME":
        raise ValueError("Phase 2 Spike machine does not select the FireSim-aligned Spike runtime")
    relative = Path(row.get("join", ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Phase 2 Spike member has an unsafe relative path")
    return root / relative


def audit(
    contract: Path, chipyard_root: Path, spike_runtime: Path, machines: Path, hardware_pins: Path,
    board_catalog: Path, private_model_spec: Path,
) -> dict:
    """Report exact Spike/tool/ABI agreement with the Phase 2 full U250 machine."""
    chipyard_root = chipyard_root.resolve()
    spike_runtime = spike_runtime.resolve()
    selected = yaml.safe_load(contract.read_text())
    machine_doc = yaml.safe_load(machines.read_text())
    registry = yaml.safe_load(hardware_pins.read_text())
    boards = yaml.safe_load(board_catalog.read_text())["boards"]
    model_spec = yaml.safe_load(private_model_spec.read_text())
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
    if not spike_runtime.is_dir() or spike_runtime.is_symlink():
        raise ValueError("selected FireSim-aligned Spike runtime is absent or indirect")
    command = functional["command"]
    identity_files = functional["identity_files"]
    expected_flags = ["--isa=rv64imafdcb_zicsr_zifencei_zihpm_zfh_zba_zbb_zbs", "--priv=m", "-p1",
                      "-m0x80000000:0x400000000"]
    if command[1:] != [*expected_flags, "--extlib=libgemmini.so", "--extension=gemmini"] \
            or len(identity_files) != 2:
        raise ValueError("Phase 2 functional machine has an unexpected Spike selection")
    expected_spike = _machine_file(command[0], spike_runtime)
    expected_extension = _machine_file(identity_files[0], spike_runtime)
    expected_boost = _machine_file(identity_files[1], spike_runtime)
    expected_library_dir = _machine_file(functional["environment"]["LD_LIBRARY_PATH"], spike_runtime)
    deployment = json.loads((spike_runtime / "deployment.json").read_text())
    dts = chipyard_root / (
        "sims/firesim-staging/generated-src/firechip.chip.FireSim.FireSimGemminiRocketConfig/"
        "firechip.chip.FireSim.FireSimGemminiRocketConfig.dts"
    )
    board = boards["firesim_gemmini_rocket_u250_30mhz"]
    if not isinstance(model_spec.get("models"), list) or not model_spec["models"]:
        raise ValueError("private model specification has no selected programs")
    runner = selected["runner"]
    block = runner["chipyard_rocc"]
    spike_engine = block["engines"]["spike"]
    spike = spike_engine["binary"]
    extension = runner["spike_extension"]
    if block["config"] != "GemminiGsimFireSimAlignedConfig" or extension["extension_name"] != "gemmini":
        raise ValueError("Phase 1 Spike selection names another Gemmini chip or extension")
    selected_spike = Path(spike["path"])
    selected_extension = Path(extension["extlib"])
    observed = {
        "phase1_spike_sha256": _sha256(selected_spike),
        "phase2_spike_sha256": _sha256(expected_spike),
        "phase1_extension_sha256": _sha256(selected_extension),
        "phase2_extension_sha256": _sha256(expected_extension),
        "phase2_boost_sha256": _sha256(expected_boost),
        "phase2_chipyard_commit": _commit(chipyard_root),
        "phase2_gemmini_commit": _commit(chipyard_root / "generators/gemmini"),
        "phase2_receiver_gitlink": _gitlink(chipyard_root / "generators/gemmini", "software/libgemmini"),
        "phase2_abi_header_sha256": _sha256(
            chipyard_root / "generators/gemmini/software/gemmini-rocc-tests/include/gemmini_params.h"
        ),
        "phase2_host_dts_sha256": _sha256(dts),
    }
    checks = {
        "phase1_spike_pin_valid": observed["phase1_spike_sha256"] == spike["sha256"],
        "phase1_extension_pin_valid": observed["phase1_extension_sha256"] == extension["sha256"],
        "same_spike_binary": observed["phase1_spike_sha256"] == observed["phase2_spike_sha256"],
        "same_spike_extension": observed["phase1_extension_sha256"] == observed["phase2_extension_sha256"],
        "phase1_launch_matches_16gib_runtime": (
            spike_engine.get("argv") == [*expected_flags, "{elf}"]
            and spike_engine.get("environment", {}).get("LD_LIBRARY_PATH") == str(expected_library_dir)
            and expected_library_dir == spike_runtime / "lib"
        ),
        "phase2_runtime_receipt_pinned": (
            deployment.get("schema") == "merlin.functional_spike_deployment.v1"
            and deployment.get("sources", {}).get("target_profile") == "fire_aligned_30mhz"
            and deployment.get("profile", {}).get("target_profile") == "fire_aligned_30mhz"
            and deployment.get("profile", {}).get("functional_only") is True
            and deployment.get("timing_model_included") is False
            and deployment.get("sources", {}).get("sources", {}).get("spike", {}).get("revision")
                == "9c190a07c6838f6392bafa4ad83acea462c7f759"
            and deployment.get("sources", {}).get("sources", {}).get("libgemmini", {}).get("revision")
                == "5b1254a7dc72f757442280e7f28ef30542c78533"
            and deployment.get("artifacts", {}).get("bin/spike", {}).get("sha256")
                == observed["phase2_spike_sha256"]
            and deployment.get("artifacts", {}).get("lib/libgemmini.so", {}).get("sha256")
                == observed["phase2_extension_sha256"]
            and deployment.get("artifacts", {}).get("lib/" + expected_boost.name, {}).get("sha256")
                == observed["phase2_boost_sha256"]
            and deployment.get("profile", {}).get("hardware_association", {}).get("archive_sha256")
                == bitstream_pin["digest"]
            and deployment.get("profile", {}).get("hardware_association", {}).get("functional_receiver_revision")
                == observed["phase2_receiver_gitlink"]
            and deployment.get("profile", {}).get("hardware_association", {}).get("host_dts_sha256")
                == observed["phase2_host_dts_sha256"]
            and deployment.get("profile", {}).get("hardware_association", {}).get("generated_params_sha256")
                == observed["phase2_abi_header_sha256"]
            and deployment.get("profile", {}).get("launch", {}).get("isa")
                == expected_flags[0].removeprefix("--isa=")
            and deployment.get("profile", {}).get("launch", {}).get("memory_base") == 0x80000000
            and deployment.get("profile", {}).get("launch", {}).get("memory_bytes") == 16 * 1024 ** 3
        ),
        "phase2_chipyard_source_pinned": (
            _selected_chipyard_source(chipyard_root, pins["gemmini_rocket_chipyard"]["commit"])
        ),
        "phase2_gemmini_source_pinned": observed["phase2_gemmini_commit"] == pins["gemmini_rocket_rtl"]["commit"],
        "phase2_receiver_source_pinned": observed["phase2_receiver_gitlink"] == "5b1254a7dc72f757442280e7f28ef30542c78533",
        "phase2_abi_header_pinned": observed["phase2_abi_header_sha256"] == machine["program_header_sha256"],
        "firesim_board_selected": (
            board.get("bitstream") == machine["hw_config"]
            and board.get("rtl_sim_config") == "FireSimGemminiRocketConfig"
            and board.get("dram_bytes") == "16 GiB"
            and board.get("dram_base") == 0x80000000
            and observed["phase2_host_dts_sha256"] == board.get("host_dts_sha256")
        ),
        "private_models_select_firesim_board": all(
            row.get("board") == "firesim_gemmini_rocket_u250_30mhz"
            and row.get("host_dts_sha256") == observed["phase2_host_dts_sha256"]
            and row.get("board_catalog_sha256") == _sha256(board_catalog)
            for row in model_spec["models"]
        ),
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
    parser.add_argument("--spike-runtime", required=True, type=Path)
    parser.add_argument("--machines", required=True, type=Path)
    parser.add_argument("--hardware-pins", required=True, type=Path)
    parser.add_argument("--board-catalog", required=True, type=Path)
    parser.add_argument("--private-model-spec", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit(
            args.contract, args.chipyard_root, args.spike_runtime, args.machines, args.hardware_pins,
            args.board_catalog, args.private_model_spec,
        )
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"Spike target alignment: {exc}\n")
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["spike_aligned"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
