"""Preparation receipts admit immutable references without claiming model execution."""

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).absolute().parents[1]
for directory in (ROOT / "tools", ROOT / "tools/host_iss"):
    sys.path.insert(0, str(directory))
import kernel_library  # noqa: E402,F401
from atlas_tooling import sha
from model_device import Device


class ResNetPreparationTest(unittest.TestCase):
    def setUp(self):
        self.artifacts = {"program.S": b"ecall\n", "program.bin": b"\x73\0\0\0",
            "initial-dram.bin": bytes(4608), "expected-bf16-tiles.bin": bytes(2048)}
        self.receipt = dict(schema="atlas.resnet_kernel_preparation.v1", status="PREPARED_NOT_EXECUTED",
            numerical_execution_performed=False, source_inputs_sha256={"/frozen/source": "1" * 64},
            artifacts_sha256={name: sha(raw) for name, raw in self.artifacts.items()},
            requested_dma=dict(load_bytes=2048, store_bytes=2048))

    def paths(self, receipt):
        device = Device.__new__(Device)
        device.bundle = type("Fixture", (), {"placement": {"arena_base": 0x90000000}})()
        def read(name):
            basename = name.rsplit("/", 1)[-1]
            return json.dumps(receipt).encode() if basename == "receipt.json" else self.artifacts[basename]
        device.payload = read
        return device.resnet_kernel_paths(0)

    def test_prepared_and_legacy_receipts_keep_their_separate_statuses(self):
        paths = self.paths(self.receipt)
        self.assertEqual(paths["dram_base"], 0x90000000)
        self.assertEqual(paths["output_offset"], 2304)
        self.assertEqual(paths["requested_dma"], self.receipt["requested_dma"])
        self.assertEqual(self.paths(dict(status="passed_model_only", requested_dma={}))['requested_dma'], {})

    def test_changed_artifact_forged_status_and_missing_source_pins_rejected(self):
        mutations = [("status", "PASS"), ("schema", "other"), ("numerical_execution_performed", True),
            ("source_inputs_sha256", {}), ("source_inputs_sha256", {"source": "not-a-hash"}),
            ("artifacts_sha256", {}), ("artifacts_sha256", {name: "0" * 64 for name in self.artifacts})]
        for key, value in mutations:
            receipt = copy.deepcopy(self.receipt)
            receipt[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(AssertionError):
                self.paths(receipt)


if __name__ == "__main__":
    unittest.main()
