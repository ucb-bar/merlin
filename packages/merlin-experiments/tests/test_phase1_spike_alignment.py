from __future__ import annotations

import hashlib
import json

import yaml
from merlin_experiments.phase1 import spike_alignment as A


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_spike_alignment_requires_firesim_board_dts_and_source(tmp_path, monkeypatch):
    root = tmp_path / "chipyard"
    runtime = tmp_path / "runtime"
    spike = runtime / "bin/spike"
    extension = runtime / "lib/libgemmini.so"
    boost = runtime / "lib/libboost_regex.so.1.74.0"
    header = root / "generators/gemmini/software/gemmini-rocc-tests/include/gemmini_params.h"
    dts = root / (
        "sims/firesim-staging/generated-src/firechip.chip.FireSim.FireSimGemminiRocketConfig/"
        "firechip.chip.FireSim.FireSimGemminiRocketConfig.dts"
    )
    for path, contents in ((spike, b"spike"), (extension, b"extension"),
                           (boost, b"boost"), (header, b"header"), (dts, b"dts")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    (runtime / "deployment.json").write_text(json.dumps({
        "schema": "merlin.functional_spike_deployment.v1",
        "timing_model_included": False,
        "sources": {"target_profile": "fire_aligned_30mhz", "sources": {
            "spike": {"revision": "9c190a07c6838f6392bafa4ad83acea462c7f759"},
            "libgemmini": {"revision": "5b1254a7dc72f757442280e7f28ef30542c78533"},
        }},
        "artifacts": {"bin/spike": {"sha256": _sha(spike)},
                      "lib/libgemmini.so": {"sha256": _sha(extension)},
                      "lib/libboost_regex.so.1.74.0": {"sha256": _sha(boost)}},
        "profile": {"target_profile": "fire_aligned_30mhz", "functional_only": True,
                    "hardware_association": {"archive_sha256": "f" * 64,
                                             "functional_receiver_revision": "5b1254a7dc72f757442280e7f28ef30542c78533",
                                             "host_dts_sha256": _sha(dts),
                                             "generated_params_sha256": _sha(header)},
                    "launch": {"isa": "rv64imafdcb_zicsr_zifencei_zihpm_zfh_zba_zbb_zbs",
                               "memory_base": 0x80000000, "memory_bytes": 16 * 1024 ** 3}},
    }))

    def write(name, value):
        path = tmp_path / name
        path.write_text(yaml.safe_dump(value))
        return path

    machines = write("machines.yaml", {"machines": {
        "full_u250_board": {"hw_config": "full", "program_header_sha256": _sha(header)},
        "functional_model": {
            "command": [{"env": "MERLIN_FIRE_ALIGNED_SPIKE_RUNTIME", "join": "bin/spike"},
                        "--isa=rv64imafdcb_zicsr_zifencei_zihpm_zfh_zba_zbb_zbs", "--priv=m", "-p1", "-m0x80000000:0x400000000",
                        "--extlib=libgemmini.so", "--extension=gemmini"],
            "environment": {"LD_LIBRARY_PATH": {"env": "MERLIN_FIRE_ALIGNED_SPIKE_RUNTIME", "join": "lib"}},
            "identity_files": [{"env": "MERLIN_FIRE_ALIGNED_SPIKE_RUNTIME", "join": "lib/libgemmini.so"},
                               {"env": "MERLIN_FIRE_ALIGNED_SPIKE_RUNTIME", "join": "lib/libboost_regex.so.1.74.0"}],
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
        "chipyard_rocc": {"config": "GemminiGsimFireSimAlignedConfig", "engines": {
            "spike": {"binary": {"path": str(spike), "sha256": _sha(spike)},
                      "argv": ["--isa=rv64imafdcb_zicsr_zifencei_zihpm_zfh_zba_zbb_zbs", "--priv=m", "-p1",
                               "-m0x80000000:0x400000000", "{elf}"],
                      "environment": {"LD_LIBRARY_PATH": str(runtime / "lib")}}}},
        "spike_extension": {"extension_name": "gemmini", "extlib": str(extension), "sha256": _sha(extension)},
    }})
    row = {"board": "firesim_gemmini_rocket_u250_30mhz", "host_dts_sha256": _sha(dts),
           "board_catalog_sha256": _sha(catalog)}
    spec = write("spec.yaml", {"models": [row]})
    monkeypatch.setattr(A, "_commit", lambda path: "gem" if path.name == "gemmini" else "chip")
    monkeypatch.setattr(A, "_gitlink", lambda path, member: "5b1254a7dc72f757442280e7f28ef30542c78533")
    monkeypatch.setattr(A, "_selected_chipyard_source", lambda path, base: base == "chip")

    report = A.audit(contract, root, runtime, machines, pins, catalog, spec)
    assert report["spike_aligned"] is True, [key for key, value in report["checks"].items() if not value]
    row["board"] = "gemmini_rocket_spike_functional"
    spec.write_text(yaml.safe_dump({"models": [row]}))
    report = A.audit(contract, root, runtime, machines, pins, catalog, spec)
    assert report["spike_aligned"] is False
    assert report["checks"]["private_models_select_firesim_board"] is False
    row["board"] = "firesim_gemmini_rocket_u250_30mhz"
    spec.write_text(yaml.safe_dump({"models": [row]}))
    selected = yaml.safe_load(contract.read_text())
    selected["runner"]["chipyard_rocc"]["engines"]["spike"]["argv"][3] = "-m2048"
    contract.write_text(yaml.safe_dump(selected))
    report = A.audit(contract, root, runtime, machines, pins, catalog, spec)
    assert report["checks"]["phase1_launch_matches_16gib_runtime"] is False
