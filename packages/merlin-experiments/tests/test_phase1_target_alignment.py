"""Prospective correspondence parsing; synthetic receipts grant no hardware verdict."""

import json
import subprocess
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1 import target_alignment as A

DTS = """/dts-v1/; / { #address-cells = <2>; #size-cells = <2>;
  /* a comment containing memory@bad { reg = <0>; }; */
  cpus { cpu@0 { device_type = "cpu"; riscv,isa = "rv64imafdc"; }; };
  memory@12340000 { device_type = "memory"; reg = <0 0x12340000 0 0x200000>; };
  memory@0 { device_type = "memory"; status = "disabled"; reg = <0 0 0 1>; };
};"""


def test_structural_dts_ignores_comments_disabled_nodes_and_scopes(tmp_path):
    path = tmp_path / "host.dts"
    path.write_text(DTS)
    assert A._dts_identity(path) == {
        "dram_base": 0x12340000,
        "dram_bytes": 0x200000,
        "host_isas": ["rv64imafdc"],
        "harts": 1,
        "vector_harts": 0,
    }
    path.write_text(DTS.replace("reg = <0 0x12340000 0 0x200000>", "reg = <0 -1 0 0x200000>"))
    with pytest.raises(ValueError, match="unsigned"):
        A._dts_identity(path)
    path.write_text(DTS.replace("#size-cells = <2>", "#size-cells = <3>"))
    with pytest.raises(ValueError, match="cell count"):
        A._dts_identity(path)


def policy():
    return {
        "schema": "merlin.phase1.target_alignment.v1",
        "target": "another_accelerator",
        "config": "SelectedHarness",
        "config_source": {"path": "harness/Selected.scala", "sha256": "a" * 64},
        "build_submodules": ["device", "transport"],
        "hardware": {
            "bitstream": "selected_bitstream",
            "board": "selected_board",
            "chipyard_pin": "base",
            "accelerator_pin": "device_pin",
            "accelerator_submodule": "device",
            "firesim_dts": "chip.dts",
        },
    }


def test_exact_source_patch_and_gitlinks_are_selected_data(tmp_path, monkeypatch):
    selected = policy()
    source = {
        "revision": "patched",
        "config_file": selected["config_source"]["path"],
        "config_sha256": "a" * 64,
        "submodules": {"device": "one", "transport": "two"},
    }
    changes = [source["config_file"]]

    def run(argv, **kwargs):
        if "merge-base" in argv:
            return subprocess.CompletedProcess(argv, 0)
        if "ls-tree" in argv:
            return subprocess.CompletedProcess(
                argv, 0, "160000 commit " + source["submodules"][argv[-1]] + "\t" + argv[-1]
            )
        return subprocess.CompletedProcess(argv, 0, "\n".join(changes))

    monkeypatch.setattr(A.subprocess, "run", run)
    assert A._source_variant(tmp_path, "base", source, selected)
    changes.append("device/changed.scala")
    assert not A._source_variant(tmp_path, "base", source, selected)
    path = tmp_path / "selection.yaml"
    path.write_text(yaml.safe_dump(selected))
    assert A._selection(path) == selected
    selected["hardware"]["firesim_dts"] = "../outside.dts"
    path.write_text(yaml.safe_dump(selected))
    with pytest.raises(ValueError, match="contained"):
        A._selection(path)


def test_prospective_audit_joins_selected_target_and_board_without_legacy_constants(tmp_path, monkeypatch):
    import hashlib

    def write(name, document):
        path = tmp_path / name
        path.write_text(yaml.safe_dump(document))
        return path

    def pin(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    selected = policy()
    selection = write("selection.yaml", selected)
    (tmp_path / "chip.dts").write_text(DTS)
    dts = tmp_path / "chip.dts"
    fir, receipt, binary, build = [
        write(name, {}) for name in ("source.fir", "elaboration.json", "emulator", "build.json")
    ]
    config = selected["config"]
    contract = write(
        "contract.yaml",
        {
            "runtime": {"rtl_sim_config": config},
            "runner": {
                "chipyard_rocc": {"config": config, "engines": {"gsim": {"binary": pin(binary), "receipt": pin(build)}}}
            },
        },
    )
    facts = tmp_path / "facts.json"
    facts.write_text(
        json.dumps(
            {
                "inputs": {"target": selected["target"], "fir_path": str(fir), "fir_sha256": pin(fir)["sha256"]},
                "facts": {"source": {"config": config, "fir_sha256": pin(fir)["sha256"]}},
                "source_consistency": {
                    "status": "verified",
                    "config": config,
                    "elaboration": {"status": "reproduced_exact_firrtl", "receipt_sha256": pin(receipt)["sha256"]},
                    "production": {"elaboration": pin(receipt)},
                },
            }
        )
    )
    registry = write(
        "pins.yaml",
        {
            "pins": {"base": {"commit": "base"}, "device_pin": {"commit": "one"}},
            "artifacts": {"selected_bitstream": {"built_from": ["base", "device_pin"]}},
        },
    )
    catalog = write("boards.yaml", {})
    board = SimpleNamespace(
        dram_base=0x12340000, dram_bytes=0x200000, harts=1, n_vector_harts=0, host_dts_sha256=pin(dts)["sha256"]
    )
    monkeypatch.setattr(A, "load_boards", lambda _: {"selected_board": board})
    monkeypatch.setattr(A, "_source_variant", lambda *_: True)
    monkeypatch.setattr(
        A.elaboration, "verify", lambda *_a, **_k: {"source": {"root": str(tmp_path), "submodules": {"device": "one"}}}
    )
    targets = []

    def observed(target, *_):
        targets.append(target)
        return (
            "bound",
            "valid",
            {
                "schema_version": A.gsim_emulator.STRICT_RECEIPT_SCHEMA,
                "firrtl_sha256": pin(fir)["sha256"],
                "binary_sha256": pin(binary)["sha256"],
            },
        )

    monkeypatch.setattr(A.gsim_emulator, "_validate_receipt", observed)
    report = A.audit(contract, facts, tmp_path, registry, catalog, dts, selection=selection)
    assert report["gsim_source_aligned"] and targets == [selected["target"]]
    assert report["chip_equivalence_proven"] is False and report["timing_transfer_proven"] is False
    board.dram_bytes *= 2
    assert not A.audit(contract, facts, tmp_path, registry, catalog, dts, selection=selection)["gsim_source_aligned"]
