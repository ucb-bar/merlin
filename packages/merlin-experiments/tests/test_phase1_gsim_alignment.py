from __future__ import annotations

import hashlib
import json

import yaml
from merlin_experiments.phase1 import gsim_alignment as A


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gsim_alignment_requires_firesim_source_and_exact_receipts(tmp_path, monkeypatch):
    chipyard = tmp_path / "chipyard"
    chipyard.mkdir()
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
    contract = write("contract.yaml", {
        "runtime": {"rtl_sim_config": config},
        "runner": {"chipyard_rocc": {"config": config, "engines": {"gsim": {
            "binary": {"path": str(binary), "sha256": _sha(binary)},
            "receipt": {"path": str(receipt), "sha256": _sha(receipt)},
        }}}},
    })
    facts = write("facts.json", {
        "inputs": {"target": "gemmini", "fir_path": str(firrtl), "fir_sha256": _sha(firrtl)},
        "facts": {"source": {"config": config, "fir_sha256": _sha(firrtl)}},
        "source_consistency": {
            "status": "verified", "config": config,
            "elaboration": {"status": "reproduced_exact_firrtl", "receipt_sha256": _sha(elaboration)},
            "production": {"elaboration": {"path": str(elaboration), "sha256": _sha(elaboration)}},
        },
    }, json_document=True)
    pins = write("pins.yaml", {
        "pins": {"gemmini_rocket_chipyard": {"commit": "chip"},
                 "gemmini_rocket_rtl": {"commit": "gem"}},
        "artifacts": {"firesim_gemmini_rocket_30mhz": {
            "built_from": ["gemmini_rocket_chipyard", "gemmini_rocket_rtl"]}},
    })
    source = {"root": str(chipyard), "revision": "chip",
              "submodules": {"generators/gemmini": "gem"}}
    monkeypatch.setattr(A.elaboration, "verify", lambda *args, **kwargs: {"source": source})
    monkeypatch.setattr(A.gsim_emulator, "_validate_receipt", lambda *args: (
        "bound", "valid", {"schema_version": A.gsim_emulator.STRICT_RECEIPT_SCHEMA,
                           "firrtl_sha256": _sha(firrtl), "binary_sha256": _sha(binary)}
    ))

    assert A.audit(contract, facts, chipyard, pins)["gsim_source_aligned"] is True
    source["submodules"]["generators/gemmini"] = "old-gemmini"
    report = A.audit(contract, facts, chipyard, pins)
    assert report["gsim_source_aligned"] is False
    assert report["checks"]["matches_firesim_recorded_source_pins"] is False
    source["submodules"]["generators/gemmini"] = "gem"
    old = yaml.safe_load(contract.read_text())
    old["runner"]["chipyard_rocc"]["config"] = "GemminiRocketConfig"
    contract.write_text(yaml.safe_dump(old))
    assert A.audit(contract, facts, chipyard, pins)["checks"]["harness_config_selected"] is False
