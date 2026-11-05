"""CPU-only live-layout, immutable-span and exact segment ledger controls."""
import copy
import contextlib
import io
import json
import signal
import tempfile
from types import SimpleNamespace
from pathlib import Path
import struct
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools/host_iss")]
import tiny_segment_contract as segment
from test_host_iss_tiny_full import fixture


def image_fixture():
    data, _ = fixture()
    image = mock.Mock()
    image.segments = [dict(address=0x80000000, memory_bytes=4096, flags=4),
                      dict(address=0x81000000, memory_bytes=128*1024*1024, flags=6)]
    image.symbols = {
        "tiny_full_compiled_live_layout": dict(address=0x80000000, size=64),
        "tiny_full_segment_workspace": dict(address=0x81000000, size=24*1024*1024),
        "tiny_full_keys": dict(address=0x83000000, size=3604480),
        "tiny_full_values": dict(address=0x83400000, size=3604480),
        "tiny_full_segment_phase": dict(address=0x83800000, size=32),
        "tiny_full_segment_control": dict(address=0x83800020, size=64)}
    address = 0x84000000
    for item in data["reference_buffers"]:
        image.symbols[item["actual_symbol"]] = dict(address=address, size=item["bytes"])
        address += item["bytes"]
    values = (1, 17827456, 16254592, 16778880, 524288, 1802240, 32, 48)
    image.bytes_at.return_value = struct.pack("<8Q", *values)
    return image, data


class TinySegmentContractTest(unittest.TestCase):
    def test_reassembled_alternate_program_is_not_a_public_reuse_recipe(self):
        for arena in (0x100000000, 0x90000000):
            item = dict(physical_mkn=[32, 2048, 256], output_offset=32*2048+2048*256+256,
                        rescale=-15, source="selected.S")
            source = segment.relocate_assembly(segment.assembly(*item["physical_mkn"],
                item["output_offset"], -15, traffic_reuse=True), segment.DEFAULT_ARENA, arena).encode()
            bundle = SimpleNamespace(build=dict(placement=dict(arena_base=arena, dram_base=0x80000000,
                dram_bytes=16 << 30)), read=lambda _: source)
            segment.admit_public_source(bundle, item)
            bundle.read = lambda _: source+b"nop\n"
            with self.assertRaisesRegex(ValueError, "public reuse source"):
                segment.admit_public_source(bundle, item)

    def test_all145_regions_derived_from_compiled_abi(self):
        image, data = image_fixture()
        regions, values = segment.layout(image, data)
        self.assertEqual(len(regions), 145)
        self.assertEqual(sum(r["bytes"] for r in regions), 17359988)
        self.assertEqual(regions[4]["address"], 0x81000000 + 16254592)
        self.assertEqual(values[-3:], (1802240, 32, 48))

    def test_missing_kv_alias_bad_layout_or_readonly_actual_rejected(self):
        for kind in ("missing", "alias", "hidden", "abi", "readonly"):
            image, data = image_fixture()
            if kind == "missing":
                image.symbols.pop("tiny_full_keys")
            elif kind == "alias":
                image.symbols["tiny_full_values"]["address"] = image.symbols["tiny_full_keys"]["address"]
            elif kind == "hidden":
                image.symbols["tiny_full_segment_phase"]["address"] = 0x81000000 + 16254592
            elif kind == "abi":
                image.bytes_at.return_value = struct.pack("<8Q", 1, 17827456, 16254592, 16778880,
                                                         524288, 1802240, 40, 48)
            else:
                image.segments[1]["flags"] = 4
            with self.subTest(kind=kind), self.assertRaises((ValueError, KeyError)):
                segment.layout(image, data)

    def test_writable_or_not_file_backed_immutable_span_rejected(self):
        image, _ = image_fixture()
        with self.assertRaises(ValueError):
            segment.span(image, 0x81000000, 16, False)
        image.bytes_at.side_effect = ValueError("not file backed")
        with self.assertRaises(ValueError):
            segment.span(image, 0x80000000, 16, False)

    def test_missing_assembler_rejected_before_full_image_admission(self):
        with mock.patch.object(segment.full, "validate_bundle") as validate:
            with self.assertRaises(ValueError):
                segment.validate_image(mock.Mock(), mock.Mock(), None)
            validate.assert_not_called()

    def test_bad_chain_is_rejected_before_qemu_or_device_creation(self):
        import run_tiny_segment as entry
        import segment_runner as runner
        profile = dict(revision="9e6313bddecb8ba936206eb347f5cd4e5335282c", curve_sha256="a"*64,
                       tracked_source_diff_exit=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = mock.Mock(root=root)
            bundle.read.return_value = b"unadmitted-elf"
            args = SimpleNamespace(out=root/"out", wall_seconds=30, segment_index=24,
                bundle_root=root, qemu=root/"qemu", model_root=root,
                corrupt_region=None, corrupt_offset=1)
            prior = signal.getsignal(signal.SIGTERM)
            try:
                with mock.patch.object(runner, "FullBundle", return_value=bundle), \
                        mock.patch.object(runner, "Image"), \
                        mock.patch.object(runner, "model_binding", return_value=profile), \
                        mock.patch.object(runner, "configure_model", return_value=(None,None,mock.Mock(),None)), \
                        mock.patch.object(segment, "validate_bundle", side_effect=ValueError("bad live inventory")), \
                        mock.patch.object(runner, "Device") as device, \
                        mock.patch.object(runner.subprocess, "Popen") as qemu, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(entry.execute(args), 1)
                    device.assert_not_called()
                    qemu.assert_not_called()
                receipt = json.loads((root/"out/receipt.json").read_text())
                self.assertEqual(receipt["status"], "FAILED")
                self.assertFalse(receipt["host_elf_executed"])
                self.assertFalse(receipt["fresh_device"])
            finally:
                signal.signal(signal.SIGTERM, prior)

    def test_exact_global_index_program_input_output_and_dma_ledger(self):
        item = dict(program=struct.pack("<I", 0x73), initial_sha256="a"*64,
            expected_sha256="b"*64, requested_dma=dict(load_bytes=32, store_bytes=32))
        receipt = dict(index=292, sequence=293, reset_count=1, exact=True, drained=True, cycles=17,
            program_sha256=segment.digest(segment.full.EPILOGUE), host_input_sha256="a"*64,
            actual_output_sha256="b"*64, expected_output_sha256="b"*64,
            observed_dma=dict(load_bytes=32, store_bytes=32, response_beats=2))
        segment.verify_kernel(receipt, item, 292)
        for name, value in (("index", 0), ("sequence", 1), ("cycles", 0), ("drained", False),
                            ("host_input_sha256", "c"*64), ("program_sha256", "d"*64),
                            ("index", True), ("cycles", True)):
            changed = copy.deepcopy(receipt)
            changed[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                segment.verify_kernel(changed, item, 292)


if __name__ == "__main__":
    unittest.main()
