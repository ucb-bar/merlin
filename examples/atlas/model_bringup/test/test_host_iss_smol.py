"""CPU-only Smol contract tests; no QEMU or model execution claim."""

import copy
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).absolute().parents[1] / "tools"
sys.path[:0] = [str(TOOLS), str(TOOLS / "host_iss")]
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
from block_bundle import BlockBundle
from elf_image import Image as ELFImage
import smol_contract as smol


class Image:
    def __init__(self):
        self.symbols = {}
        self.instructions_by_name = {
            "atlas_workload_main": [(0x100, 4, 0x10500073), (0x108, 4, 0x10500073)],
            "atlas_block_trap_report": [(0x200, 4, 0x10500073)]}

    def instructions(self, name):
        return self.instructions_by_name[name]


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.image = Image()
        self.memory = {}
        self.payloads = {}
        self.image.symbols = {
            "smol_workload_mailbox": dict(address=0x1000, size=120),
            "tohost": dict(address=0x2000, size=8),
            "smol_selected_state": dict(address=0x3000, size=32)}
        mailbox = bytearray(120)
        struct.pack_into("<6I", mailbox, 0, 0x534d4f4c, 1, 0, 7, 6, 0)
        struct.pack_into("<7I", mailbox, 24, *range(1, 8))
        struct.pack_into("<7I", mailbox, 52, *range(100, 107))
        self.memory[0x1000] = bytes(mailbox)
        self.memory[0x2000] = struct.pack("<Q", 1)
        self.memory[0x3000] = struct.pack("<8I", 1, 7, 6, 0, 1, 1, 0, 0)
        for index, (name, actual, size) in enumerate(smol.FINAL):
            address = 0x10000 + index * 0x100000
            self.image.symbols[actual] = dict(address=address, size=size)
            self.payloads["data/" + name + ".bin"] = bytes([index + 1]) * size
            self.memory[address] = self.payloads["data/" + name + ".bin"]
        self.remote = SimpleNamespace(read=lambda address, size: self.memory[address][:size])
        self.bundle = SimpleNamespace(read=lambda name: self.payloads[name])
        self.device = SimpleNamespace(completed=[dict(index=i, sequence=i + 1, exact=True, drained=True,
                                                      reset_count=1) for i in range(7)])
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)

    def verify(self):
        with patch.object(smol, "validate_bundle"):
            return smol.verify_completion(self.remote, self.image, self.device, self.bundle, self.out)

    def word(self, offset, value):
        raw = bytearray(self.memory[0x1000])
        struct.pack_into("<I", raw, offset, value)
        self.memory[0x1000] = bytes(raw)

    def test_exact_completion_saves_real_buffers(self):
        result = self.verify()
        self.assertTrue(result["all_outputs_exact"] and result["all_caches_exact"])
        self.assertEqual(result["tohost"], 1)
        for _, actual, _ in smol.FINAL:
            address = self.image.symbols[actual]["address"]
            self.assertEqual((self.out / (actual + ".bin")).read_bytes(), self.memory[address])

    def test_each_mailbox_status_field_is_required(self):
        original = self.memory[0x1000]
        for offset in (0, 4, 8, 12, 16, 20, 80, 84, 88, 92, 96, 104, 112):
            with self.subTest(offset=offset):
                self.memory[0x1000] = original
                self.word(offset, int.from_bytes(original[offset:offset+4], "little") ^ 1)
                with self.assertRaises(ValueError):
                    self.verify()

    def test_each_sequence_and_cycle_is_required(self):
        original = self.memory[0x1000]
        for offset in range(24, 80, 4):
            with self.subTest(offset=offset):
                self.memory[0x1000] = original
                self.word(offset, 0)
                with self.assertRaises(ValueError):
                    self.verify()

    def test_tohost_failure_is_rejected(self):
        self.memory[0x2000] = struct.pack("<Q", 511)
        with self.assertRaisesRegex(ValueError, "tohost"):
            self.verify()

    def test_every_block_state_field_is_required(self):
        original = self.memory[0x3000]
        for offset in range(0, 32, 4):
            with self.subTest(offset=offset):
                raw = bytearray(original)
                raw[offset] ^= 1
                self.memory[0x3000] = bytes(raw)
                with self.assertRaisesRegex(ValueError, "block completion"):
                    self.verify()

    def test_live_output_and_both_caches_are_checked(self):
        for _, actual, _ in smol.FINAL:
            with self.subTest(actual=actual):
                address = self.image.symbols[actual]["address"]
                original = self.memory[address]
                self.memory[address] = original[:-1] + bytes([original[-1] ^ 1])
                with self.assertRaisesRegex(ValueError, "live output/cache"):
                    self.verify()
                self.memory[address] = original

    def test_device_ledger_cannot_skip_or_fake_execution(self):
        original = copy.deepcopy(self.device.completed)
        mutations = [dict(index=1), dict(sequence=2), dict(exact=False), dict(drained=False), dict(reset_count=2)]
        for changes in mutations:
            with self.subTest(changes=changes):
                self.device.completed = copy.deepcopy(original)
                self.device.completed[0].update(changes)
                with self.assertRaisesRegex(ValueError, "ledger"):
                    self.verify()
        self.device.completed = original[:-1]
        with self.assertRaisesRegex(ValueError, "launch count"):
            self.verify()

    def test_fixed_live_symbol_extent_is_required(self):
        self.image.symbols["output"]["size"] -= 2
        with self.assertRaisesRegex(ValueError, "symbol extent"):
            self.verify()


class AdmissionTests(unittest.TestCase):
    def bundle(self, value=None):
        value = value or dict(schema="atlas.smol_selected_layer1.epoch3_launch_data.v1")
        raw = json.dumps(value).encode()
        return SimpleNamespace(provenance=dict(workload="smol", contract="host/contract.json",
            contract_sha256=smol.digest(raw)), kernel_count=7, read=lambda name: raw)

    def captured(self):
        root = os.environ.get("SMOL_BLOCK_BUNDLE")
        if not root:
            self.skipTest("SMOL_BLOCK_BUNDLE is unset")
        bundle = BlockBundle(checked(root), "smol")
        return bundle, ELFImage(bundle.read("host/host.elf"))

    def replace_contract(self, bundle, value):
        raw, read = json.dumps(value).encode(), bundle.read
        member = bundle.provenance["contract"]
        bundle.read = lambda name: raw if name == member else read(name)
        bundle.provenance["contract_sha256"] = smol.digest(raw)

    def test_captured_elf_and_source_admission(self):
        bundle, image = self.captured()
        smol.validate_bundle(bundle, image)

    def test_descriptor_and_embedded_payload_mutations_are_rejected(self):
        bundle, image = self.captured()
        original = copy.deepcopy(bundle.kernels[0])
        bundle.kernels[0]["output_offset"] = 1
        with self.assertRaisesRegex(ValueError, "launch descriptor"):
            smol.validate_bundle(bundle, image)
        bundle.kernels[0] = original
        value = smol.contract(bundle)
        projection = value["ordered_projections"][0]
        symbols = (value["assets"]["expected_output"]["ELF_constant_symbol"],
                   projection["ELF_original_program_symbol"], projection["ELF_weight_symbol"])
        read = image.bytes_at
        for symbol in symbols:
            address = image.symbols[symbol]["address"]
            def corrupt(start, size):
                raw = read(start, size)
                return bytes([raw[0] ^ 1]) + raw[1:] if start == address else raw
            with self.subTest(symbol=symbol), patch.object(image, "bytes_at", side_effect=corrupt):
                with self.assertRaises(ValueError):
                    smol.validate_bundle(bundle, image)

    def test_source_payload_geometry_and_role_mutations_are_rejected(self):
        bundle, image = self.captured()
        path, read = bundle.kernels[0]["initial"], bundle.read
        raw = read(path)
        with patch.object(bundle, "read", side_effect=lambda name:
                          bytes([raw[0] ^ 1]) + raw[1:] if name == path else read(name)):
            with self.assertRaisesRegex(ValueError, "projection payload"):
                smol.validate_bundle(bundle, image)
        original = smol.contract(bundle)
        for key, value in (("role", "mlp.down_proj"), ("native_rows", 192), ("output_width", 320)):
            changed = copy.deepcopy(original)
            changed["ordered_projections"][0][key] = value
            self.replace_contract(bundle, changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                smol.validate_bundle(bundle, image)

    def test_source_contract_is_digest_bound(self):
        bundle = self.bundle()
        self.assertIn("schema", smol.contract(bundle))
        bundle.provenance["contract_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest"):
            smol.contract(bundle)

    def test_source_schema_is_required(self):
        with self.assertRaisesRegex(ValueError, "source schema"):
            smol.contract(self.bundle(dict(schema="other")))

    def test_build_only_admission_requires_original_success_and_bookends(self):
        for build in ({}, dict(status="passed_RV64GC_LP64D_ELF_build_only", ELF_executed=True,
                               source_bookends_unchanged=True),
                      dict(status="passed_RV64GC_LP64D_ELF_build_only", ELF_executed=False,
                           source_bookends_unchanged=False)):
            bundle = self.bundle()
            bundle.build = build
            with self.assertRaisesRegex(ValueError, "build status"):
                smol.validate_bundle(bundle, Image())

    def test_stop_sites_include_success_and_trap(self):
        self.assertEqual(smol.stop_sites(Image()), {0x100: 4, 0x108: 4, 0x200: 4})
        for name in ("atlas_workload_main", "atlas_block_trap_report"):
            image = Image()
            image.instructions_by_name[name] = [(0x300, 4, 0x13)]
            with self.assertRaisesRegex(ValueError, "WFI"):
                smol.stop_sites(image)


if __name__ == "__main__":
    unittest.main()
