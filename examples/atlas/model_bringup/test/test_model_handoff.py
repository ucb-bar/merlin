"""Focused coverage of unchanged-ELF packaging and physical memory placement."""

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "tools"))
from atlas_model_handoff import DEFAULT_PLATFORM, describe, package, physical, platform_map


def elf():
    raw = bytearray(0x500)
    ident = b"\x7fELF\x02\x01\x01" + bytes(9)
    struct.pack_into("<16sHHIQQQIHHHHHH", raw, 0, ident, 2, 243, 1,
                     0x80000000, 64, 0x300, 0, 64, 56, 3, 64, 4, 0)
    # Executable file-backed text and high-address NOLOAD DMA/BSS segment.
    struct.pack_into("<IIQQQQQQ", raw, 64, 1, 5, 0x200, 0x80000000,
                     0x80000000, 4, 4, 1)
    struct.pack_into("<IIQQQQQQ", raw, 120, 1, 6, 0x204, 0x80001000,
                     0x80001000, 0, 256, 1)
    struct.pack_into("<IIQQQQQQ", raw, 176, 1, 6, 0x204, 0x100000000,
                     0x100000000, 0, 4096, 1)
    raw[0x200:0x204] = b"\x13\0\0\0"
    strings = b"\0tohost\0mailbox\0fromhost\0"
    raw[0x210:0x210 + len(strings)] = strings
    struct.pack_into("<IBBHQQ", raw, 0x240 + 24, 1, 0x11, 0, 1, 0x80001000, 8)
    struct.pack_into("<IBBHQQ", raw, 0x240 + 48, 8, 0x11, 0, 1, 0x80001040, 64)
    struct.pack_into("<IBBHQQ", raw, 0x240 + 72, 16, 0x11, 0, 1, 0x80001008, 8)
    struct.pack_into("<IIQQQQIIQQ", raw, 0x300 + 64, 0, 1, 3, 0x80001000, 0x204, 256, 0, 0, 8, 0)
    struct.pack_into("<IIQQQQIIQQ", raw, 0x300 + 128, 0, 2, 0, 0, 0x240, 96, 3, 1, 8, 24)
    struct.pack_into("<IIQQQQIIQQ", raw, 0x300 + 192, 0, 3, 0, 0, 0x210, len(strings), 0, 0, 1, 0)
    return bytes(raw)


def memmap():
    def row(base, size, *, cache=True, execute=False):
        return {key: [value] for key, value in dict(base=base, size=size, r=True,
                                                   w=True, x=execute, c=cache).items()}
    return {"mapping": [row(0x20000, 0x20000, cache=False),
                        row(0x40000, 0x1c, cache=False),
                        row(0x80000000, 0x2000, execute=True),
                        row(0x100000000, 4096)]}


class ModelHandoffTest(unittest.TestCase):
    def test_selected_vcs_platform_binds_host_and_high_dma(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.elf"
            source.write_bytes(elf())
            result = package(elf=source, name="selected-vcs", dma_base=None,
                             dma_bytes=4096, out=root / "handoff", platform=DEFAULT_PLATFORM)
            self.assertEqual(result["target_identity"], "EE290GenericIOSimConfig")
            self.assertEqual(result["memory_map_origin"], "selected_platform_contract")
            self.assertTrue(result["footprint"]["memory_map_checked"])
            self.assertEqual(result["footprint"]["dma"]["address"], 0x100000000)
            self.assertEqual(result["platform"]["mapping"][-1]["base"], [0x80000000])
            self.assertEqual((root / "handoff/platform.json").read_bytes(), DEFAULT_PLATFORM.read_bytes())
            self.assertFalse(result["hardware_execution"])
            self.assertIsNone(result["platform"]["bitstream_identity"])

    def test_platform_changes_need_matching_compiled_layout(self):
        profile = json.loads(DEFAULT_PLATFORM.read_bytes())
        profile["mapping"][0]["base"] = [0x60000]
        with self.assertRaisesRegex(ValueError, "rebuild"):
            platform_map(profile)
        profile = json.loads(DEFAULT_PLATFORM.read_bytes())
        profile["dma_arena_base"] = 0x200000000
        with self.assertRaisesRegex(ValueError, "platform contract"):
            platform_map(profile)

    def test_high_dma_zero_fill_and_union_footprint(self):
        value = describe(elf(), 0x100000000, 4096, observe=["mailbox"])
        self.assertEqual(value["resident_union_bytes"], 4356)
        self.assertEqual(value["load_segments"][2]["zero_fill_bytes"], 4096)
        self.assertEqual(value["symbols"]["tohost"]["address"], 0x80001000)
        self.assertFalse(value["memory_map_checked"])

    def test_actual_map_requires_high_ram_and_uncached_mmio(self):
        mapping = memmap()
        self.assertTrue(describe(elf(), 0x100000000, 4096, memmap=mapping)["memory_map_checked"])
        mapping["mapping"][0]["c"] = [True]
        with self.assertRaisesRegex(ValueError, "Atlas IMEM"):
            describe(elf(), 0x100000000, 4096, memmap=mapping)
        mapping = memmap()
        mapping["mapping"][-1]["size"] = [2048]
        with self.assertRaisesRegex(ValueError, "LOAD segment"):
            describe(elf(), 0x100000000, 4096, memmap=mapping)

    def test_dma_cannot_alias_mmio_or_cross_window(self):
        for base, size in [(0x20000, 32), (0x40000, 32), (0x1ffffffe0, 64)]:
            with self.subTest(base=base), self.assertRaises(ValueError):
                describe(elf(), base, size)

    def test_completion_symbol_must_exist(self):
        with self.assertRaisesRegex(ValueError, "missing sized"):
            describe(elf(), 0x100000000, 4096, observe=["absent"])
        raw = bytearray(elf())
        struct.pack_into("<Q", raw, 0x240 + 24 + 8, 0x90000000)
        with self.assertRaisesRegex(ValueError, "writable loaded"):
            describe(bytes(raw), 0x100000000, 4096)

    def test_loader_words_and_code_cannot_alias_dma(self):
        with self.assertRaisesRegex(ValueError, "executable"):
            describe(elf(), 0x80000000, 32)
        with self.assertRaisesRegex(ValueError, "completion"):
            describe(elf(), 0x80001000, 128)
        raw = bytearray(elf())
        struct.pack_into("<Q", raw, 0x240 + 72 + 8, 0x80001000)
        with self.assertRaisesRegex(ValueError, "HTIF words overlap"):
            describe(bytes(raw), 0x100000000, 4096)
        raw = bytearray(elf())
        struct.pack_into("<Q", raw, 0x240 + 72 + 16, 0)
        with self.assertRaisesRegex(ValueError, "fromhost"):
            describe(bytes(raw), 0x100000000, 4096)

    def test_package_is_unchanged_baremetal_and_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, out = root / "input.elf", root / "bundle"
            source.write_bytes(elf())
            result = package(elf=source, name="tiny-full", dma_base=0x100000000,
                             dma_bytes=4096, out=out)
            self.assertEqual((out / "host.elf").read_bytes(), elf())
            self.assertEqual(result["status"], "PREPARATION_ONLY")
            self.assertIsNone(result["numerical_evidence"])
            self.assertFalse(result["cache_coherence_qualified"])
            workload = json.loads((out / "workload.json").read_text())
            self.assertIsNone(workload["common_rootfs"])
            self.assertEqual(workload["common_bootbinary"], "host.elf")
            with self.assertRaises(FileExistsError):
                package(elf=source, name="tiny-full", dma_base=0x100000000,
                        dma_bytes=4096, out=out)

    def test_map_identity_pair_and_exact_map_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, mapping = root / "input.elf", root / "map.json"
            source.write_bytes(elf())
            mapping.write_text(json.dumps(memmap()))
            with self.assertRaisesRegex(ValueError, "both"):
                package(elf=source, name="smol", dma_base=0x100000000,
                        dma_bytes=4096, out=root / "bad", soc_memmap=mapping)
            with self.assertRaisesRegex(ValueError, "both"):
                package(elf=source, name="smol", dma_base=0x100000000,
                        dma_bytes=4096, out=root / "bad-platform", soc_memmap=mapping,
                        platform=DEFAULT_PLATFORM)
            result = package(elf=source, name="smol", dma_base=0x100000000,
                             dma_bytes=4096, out=root / "good", soc_memmap=mapping,
                             target_identity="local-target-id")
            self.assertEqual(result["target_identity"], "local-target-id")
            self.assertEqual((root / "good" / "soc-memmap.json").read_bytes(), mapping.read_bytes())

    def test_restricted_paths_refused_before_io(self):
        with patch("pathlib.Path.lstat", side_effect=AssertionError("restricted path inspected")) as check:
            for value in ("/safe/HaMmEr-artifact/x", "/safe/vLsI-map.json"):
                with self.assertRaisesRegex(ValueError, "restricted"):
                    physical(value)
            check.assert_not_called()
        with patch("atlas_model_handoff.physical") as check:
            with self.assertRaisesRegex(ValueError, "workload name"):
                package(elf="unread", name="vLsI-map", dma_base=0x100000000,
                        dma_bytes=4096, out="unwritten")
            check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
