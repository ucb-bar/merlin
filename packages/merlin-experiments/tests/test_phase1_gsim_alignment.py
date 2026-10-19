from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase1 import gsim_alignment as A


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gsim_config_source_is_the_audited_patch():
    source = Path(__file__).parents[3] / "build_tools/chipyard/GemminiGsimFireSimAlignedConfig.scala"
    assert _sha(source) == A.CONFIG_SOURCE_SHA256


def test_gsim_alignment_requires_firesim_source_and_exact_receipts(tmp_path, monkeypatch):
    chipyard = tmp_path / "chipyard"
    chipyard.mkdir()
    firesim_dts = chipyard / (
        "sims/firesim-staging/generated-src/firechip.chip.FireSim.FireSimGemminiRocketConfig/"
        "firechip.chip.FireSim.FireSimGemminiRocketConfig.dts"
    )
    firesim_dts.parent.mkdir(parents=True)
    gsim_dts = tmp_path / "gsim.dts"
    dts = """/dts-v1/;
/ {
    #address-cells = <2>;
    #size-cells = <2>;
    cpus { cpu@0 { device_type = "cpu"; riscv,isa = "rv64imafdcbzicsr"; }; };
    memory@80000000 { device_type = "memory"; reg = <0x0 0x80000000 0x4 0x0>; };
};
"""
    firesim_dts.write_text(dts)
    gsim_dts.write_text(dts)
    firrtl = tmp_path / "model.fir"
    firrtl.write_text("circuit TestHarness :\n")
    elaboration = tmp_path / "elaboration.json"
    elaboration.write_text("{}")
    binary = tmp_path / "emulator"
    binary.write_bytes(b"emulator")
    receipt = tmp_path / "build_receipt.json"
    receipt.write_text("{}")

    def write(name, value, *, json_document=False):
        path = tmp_path / name
        path.write_text(json.dumps(value) if json_document else yaml.safe_dump(value))
        return path

    config = A.GSIM_CONFIG
    contract = write(
        "contract.yaml",
        {
            "runtime": {"rtl_sim_config": config},
            "runner": {
                "chipyard_rocc": {
                    "config": config,
                    "engines": {
                        "gsim": {
                            "binary": {"path": str(binary), "sha256": _sha(binary)},
                            "receipt": {"path": str(receipt), "sha256": _sha(receipt)},
                        }
                    },
                }
            },
        },
    )
    facts = write(
        "facts.json",
        {
            "inputs": {"target": "gemmini", "fir_path": str(firrtl), "fir_sha256": _sha(firrtl)},
            "facts": {"source": {"config": config, "fir_sha256": _sha(firrtl)}},
            "source_consistency": {
                "status": "verified",
                "config": config,
                "elaboration": {"status": "reproduced_exact_firrtl", "receipt_sha256": _sha(elaboration)},
                "production": {"elaboration": {"path": str(elaboration), "sha256": _sha(elaboration)}},
            },
        },
        json_document=True,
    )
    pins = write(
        "pins.yaml",
        {
            "pins": {"gemmini_rocket_chipyard": {"commit": "chip"}, "gemmini_rocket_rtl": {"commit": "gem"}},
            "artifacts": {
                "firesim_gemmini_rocket_30mhz": {"built_from": ["gemmini_rocket_chipyard", "gemmini_rocket_rtl"]}
            },
        },
    )
    catalog = write(
        "boards.yaml",
        {
            "boards": {
                "firesim_gemmini_rocket_u250_30mhz": {
                    "dram_base": 0x80000000,
                    "dram_bytes": "16 GiB",
                    "harts": 1,
                    "vector_harts": 0,
                    "host_dts_sha256": _sha(firesim_dts),
                }
            }
        },
    )
    source = {
        "root": str(chipyard),
        "revision": "chip-config",
        "config_file": A.CONFIG_SOURCE,
        "config_sha256": A.CONFIG_SOURCE_SHA256,
        "submodules": {"generators/gemmini": "gem"},
    }
    monkeypatch.setattr(A.elaboration, "verify", lambda *args, **kwargs: {"source": source})
    monkeypatch.setattr(A, "_source_variant", lambda *args: True)
    monkeypatch.setattr(
        A.gsim_emulator,
        "_validate_receipt",
        lambda *args: (
            "bound",
            "valid",
            {
                "schema_version": A.gsim_emulator.STRICT_RECEIPT_SCHEMA,
                "firrtl_sha256": _sha(firrtl),
                "binary_sha256": _sha(binary),
            },
        ),
    )

    assert A.audit(contract, facts, chipyard, pins, catalog, gsim_dts)["gsim_source_aligned"] is True
    source["submodules"]["generators/gemmini"] = "old-gemmini"
    report = A.audit(contract, facts, chipyard, pins, catalog, gsim_dts)
    assert report["gsim_source_aligned"] is False
    assert report["checks"]["matches_firesim_recorded_source_pins"] is False
    source["submodules"]["generators/gemmini"] = "gem"
    old = yaml.safe_load(contract.read_text())
    old["runner"]["chipyard_rocc"]["config"] = "GemminiRocketConfig"
    contract.write_text(yaml.safe_dump(old))
    assert A.audit(contract, facts, chipyard, pins, catalog, gsim_dts)["checks"]["harness_config_selected"] is False
    old["runner"]["chipyard_rocc"]["config"] = config
    contract.write_text(yaml.safe_dump(old))
    gsim_dts.write_text(dts.replace("0x4 0x0", "0x1 0x0"))
    report = A.audit(contract, facts, chipyard, pins, catalog, gsim_dts)
    assert report["gsim_source_aligned"] is False
    assert report["checks"]["host_map_and_isa_match_firesim"] is False


def test_source_variant_refuses_any_other_source_change(tmp_path, monkeypatch):
    source = {
        "revision": "patched",
        "config_file": A.CONFIG_SOURCE,
        "config_sha256": A.CONFIG_SOURCE_SHA256,
        "submodules": {name: name + "-commit" for name in A.BUILD_SUBMODULES},
    }
    changes = [A.CONFIG_SOURCE]

    def run(argv, **kwargs):
        if "merge-base" in argv:
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        if "ls-tree" in argv:
            name = argv[-1]
            return subprocess.CompletedProcess(argv, 0, f"160000 commit {name}-commit\t{name}\n", "")
        return subprocess.CompletedProcess(argv, 0, "\n".join(changes) + "\n", "")

    monkeypatch.setattr(A.subprocess, "run", run)
    assert A._source_variant(tmp_path, "base", source) is True
    del source["submodules"]["generators/saturn"]
    assert A._source_variant(tmp_path, "base", source) is False
    source["submodules"]["generators/saturn"] = "generators/saturn-commit"
    source["submodules"]["sims/firesim"] = "substitute-commit"
    assert A._source_variant(tmp_path, "base", source) is False
    source["submodules"]["sims/firesim"] = "sims/firesim-commit"
    changes.append("generators/gemmini/src/main/scala/gemmini/Configs.scala")
    assert A._source_variant(tmp_path, "base", source) is False


def test_elaboration_refuses_untracked_scala_source(tmp_path):
    root = tmp_path / "chipyard"
    root.mkdir()
    config = root / "SelectedConfig.scala"
    config.write_text("class SelectedConfig\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", config.name], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "-qm",
            "selected source",
        ],
        check=True,
    )
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert A.elaboration._source(root, revision, config.name, "SelectedConfig", {})["revision"] == revision
    (root / "UntrackedHelper.scala").write_text("class UntrackedHelper\n")
    with pytest.raises(ValueError, match="untracked files"):
        A.elaboration._source(root, revision, config.name, "SelectedConfig", {})


def test_dts_identity_uses_root_cells_and_excludes_comments_and_nested_memory(tmp_path):
    path = tmp_path / "host.dts"
    path.write_text("""/dts-v1/;
/ {
    #address-cells = <1>;
    #size-cells = <1>;
    /* memory@0 { device_type = "memory"; reg = <0 1>; }; */
    cpus { #address-cells = <2>; cpu@0 { riscv,isa = "rv64gc"; }; };
    bus { memory@0 { device_type = "memory"; reg = <0 1>; }; };
    memory@40000000 { device_type = "memory"; reg = <0x40000000 0x10000000>; };
};
""")
    assert A._dts_identity(path) == {
        "dram_base": 0x40000000,
        "dram_bytes": 0x10000000,
        "host_isa": "rv64gc",
        "harts": 1,
    }
