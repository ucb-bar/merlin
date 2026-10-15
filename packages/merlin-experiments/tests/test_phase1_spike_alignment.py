from __future__ import annotations

import hashlib

import yaml
from merlin_experiments.phase1 import spike_alignment as A


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_spike_alignment_requires_firesim_board_dts_and_source(tmp_path, monkeypatch):
    root = tmp_path / "chipyard"
    spike = root / ".conda-env/riscv-tools/bin/spike"
    extension = root / ".conda-env/riscv-tools/lib/libgemmini.so"
    header = root / "generators/gemmini/software/gemmini-rocc-tests/include/gemmini_params.h"
    dts = root / (
        "sims/firesim-staging/generated-src/firechip.chip.FireSim.FireSimGemminiRocketConfig/"
        "firechip.chip.FireSim.FireSimGemminiRocketConfig.dts"
    )
    for path, contents in ((spike, b"spike"), (extension, b"extension"), (header, b"header"), (dts, b"dts")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)

    def write(name, value):
        path = tmp_path / name
        path.write_text(yaml.safe_dump(value))
        return path

    machines = write("machines.yaml", {"machines": {
        "full_u250_board": {"hw_config": "full", "program_header_sha256": _sha(header)},
        "functional_model": {
            "command": [{"env": "MERLIN_CHIPYARD_GEMMINI_ROCKET", "join": spike.relative_to(root).as_posix()},
                        "--extension=gemmini"],
            "identity_files": [{"env": "MERLIN_CHIPYARD_GEMMINI_ROCKET",
                                "join": extension.relative_to(root).as_posix()}],
        },
    }})
    pins = write("pins.yaml", {
        "pins": {"gemmini_rocket_chipyard": {"commit": "chip"}, "gemmini_rocket_rtl": {"commit": "gem"}},
        "artifacts": {
            "firesim_gemmini_rocket_30mhz": {
                "digest": "f" * 64,
                "built_from": ["gemmini_rocket_chipyard", "gemmini_rocket_rtl"],
            },
            "full_bitstream": {"role": "firesim_bitstream", "hw_configs": ["full"],
                               "config": "FireSimGemminiRocketConfig", "digest": "f" * 64},
        },
    })
    catalog = write("boards.yaml", {"boards": {"firesim_gemmini_rocket_u250_30mhz": {
        "bitstream": "full", "rtl_sim_config": "FireSimGemminiRocketConfig", "dram_bytes": "16 GiB",
        "dram_base": 0x80000000, "host_dts_sha256": _sha(dts),
    }}})
    contract = write("contract.yaml", {"runner": {
        "chipyard_rocc": {"config": "GemminiGsimSerialClkConfig", "engines": {
            "spike": {"binary": {"path": str(spike), "sha256": _sha(spike)}}}},
        "spike_extension": {"extension_name": "gemmini", "extlib": str(extension), "sha256": _sha(extension)},
    }})
    row = {"board": "firesim_gemmini_rocket_u250_30mhz", "host_dts_sha256": _sha(dts),
           "board_catalog_sha256": _sha(catalog)}
    spec = write("spec.yaml", {"models": [row]})
    monkeypatch.setattr(A, "_commit", lambda path: "gem" if path.name == "gemmini" else "chip")

    assert A.audit(contract, root, machines, pins, catalog, spec)["spike_aligned"] is True
    row["board"] = "gemmini_rocket_spike_functional"
    spec.write_text(yaml.safe_dump({"models": [row]}))
    report = A.audit(contract, root, machines, pins, catalog, spec)
    assert report["spike_aligned"] is False
    assert report["checks"]["private_models_select_firesim_board"] is False
