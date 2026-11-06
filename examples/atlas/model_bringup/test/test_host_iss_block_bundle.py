"""CPU-only admission checks for checksum-bound block launch manifests."""

import json
from pathlib import Path
import struct
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

TOOLS = Path(__file__).absolute().parents[1] / "tools"
sys.path[:0] = [str(TOOLS), str(TOOLS / "host_iss")]
from block_bundle import BlockBundle
from bundle import ChecksumBundle
from model_device import Device, launch_bounds
from bundle_payload_fixture import write_bundle_payloads


class BlockBundleTest(unittest.TestCase):
    def fixture(self, mutate=lambda manifest, files: None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        descriptor = dict(source="kernel/program.S", original="kernel/program.bin",
            initial="kernel/initial.bin", expected="kernel/expected.bin", output_offset=16,
            dram_base=0x100000000, requested_dma=dict(load_bytes=16, store_bytes=32))
        manifest = dict(schema="atlas.block_bundle.v1", workload="smol", kernel_count=7,
            kernels=[dict(descriptor) for _ in range(7)])
        files = {"host/host.elf": b"synthetic ELF", "host/build.json": b"{}",
            "kernel/program.S": b"nop\necall\n", "kernel/program.bin": struct.pack("<2I", 0x13, 0x73),
            "kernel/initial.bin": bytes(64), "kernel/expected.bin": bytes(32)}
        mutate(manifest, files)
        files["provenance.json"] = json.dumps(manifest).encode()
        write_bundle_payloads(root, files)
        return root

    def test_valid_and_descriptor_copy(self):
        bundle = BlockBundle(self.fixture(), "smol")
        self.assertEqual(bundle.kernel_count, 7)
        item = bundle.kernel_paths(0)
        item["requested_dma"]["store_bytes"] = 0
        self.assertEqual(bundle.kernel_paths(0)["requested_dma"]["store_bytes"], 32)
        for index in (-1, 7, True):
            with self.subTest(index=index), self.assertRaises(ValueError):
                bundle.kernel_paths(index)

    def test_scope_and_count(self):
        for key, value in (("schema", "other"), ("workload", "tiny"), ("kernel_count", 6), ("kernels", [])):
            root = self.fixture(lambda manifest, files: manifest.update({key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                BlockBundle(root, "smol")

    def test_bad_launch_geometry_and_counters(self):
        cases = (("dram_base", 0x90000000), ("output_offset", -1), ("output_offset", True),
            ("output_offset", 64), ("expected", "missing.bin"),
            ("requested_dma", {"load_bytes": 1}),
            ("requested_dma", {"load_bytes": 1, "store_bytes": 31}),
            ("requested_dma", {"load_bytes": True, "store_bytes": 32}),
            ("requested_dma", {"load_bytes": 1, "store_bytes": 32, "other": 1}))
        for key, value in cases:
            root = self.fixture(lambda manifest, files: manifest["kernels"][0].update({key: value}))
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                BlockBundle(root, "smol")

    def test_program_and_output_payload_extents(self):
        for name, raw in (("kernel/program.bin", b"bad"), ("kernel/program.bin", struct.pack("<I", 0x13)),
            ("kernel/program.bin", bytes(131072) + struct.pack("<I", 0x73)),
            ("kernel/expected.bin", b""), ("kernel/expected.bin", b"a")):
            root = self.fixture(lambda manifest, files: files.update({name: raw}))
            with self.subTest(name=name, size=len(raw)), self.assertRaises(ValueError):
                BlockBundle(root, "smol")

    def test_restricted_roots_refused_without_filesystem_access(self):
        for name in ("a/HAMMER-input.bin", "a/vLsI-data.bin"):
            with self.subTest(root=name), self.assertRaises(ValueError):
                ChecksumBundle(Path("/nonexistent") / name)


class DeviceLaunchTest(unittest.TestCase):
    def test_resnet_read_only_segment_provider(self):
        data = {"kernels/03/model/receipt.json": json.dumps(dict(status="passed_model_only",
                requested_dma=dict(load_bytes=512, store_bytes=256))).encode(),
            "kernels/03/model/initial-dram.bin": bytes(1024),
            "kernels/03/model/expected-bf16-tiles.bin": bytes(256)}

        class ReadOnlyProvider:
            def read(self, name):
                return data[name]

        device = Device.__new__(Device)
        device.bundle, device.pins = ReadOnlyProvider(), {}
        paths = device.resnet_kernel_paths(3)
        self.assertEqual(paths, dict(source="kernels/03/model/program.S", original="kernels/03/model/program.bin",
            initial="kernels/03/model/initial-dram.bin", expected="kernels/03/model/expected-bf16-tiles.bin",
            output_offset=512, dram_base=0x100000000, requested_dma=dict(load_bytes=512, store_bytes=256)))
        self.assertEqual(set(device.pins), set(data))
        data["kernels/03/model/receipt.json"] = b'{"status":"failed"}'
        with self.assertRaises(AssertionError):
            device.resnet_kernel_paths(3)

    def device(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        device = Device.__new__(Device)
        device.out = Path(temporary.name)
        device.deadline = time.monotonic() + 10
        device.cycle_cap = 2
        device.kernel_count = 1
        device.launch_start, device.launch_end, device.launch_count = 0, 1, 1
        device.completed, device.pins = [], {}
        words = struct.pack("<I", 0x73)
        device.imem, device.written = bytearray(words), {0}
        initial, expected = bytes(64), b"x" * 32
        paths = dict(initial="initial", expected="expected", dram_base=0x100000000,
            output_offset=16, requested_dma=dict(load_bytes=0, store_bytes=32))
        device.programs = [(object(), words, paths)]
        device.bundle = Mock(spec=["read"])
        device.bundle.read.side_effect = {"initial": initial, "expected": expected}.__getitem__
        device.remote = Mock()
        device.remote.read.return_value = initial
        device.torch = Mock()
        state = SimpleNamespace(halted=True, _csr_values={0xC10: 0, 0xC11: 1},
            check_flag=lambda channel: False, write_dram=Mock(), halt_reason=None)
        state.read_csrf = state._csr_values.__getitem__
        state.read_dram = Mock()
        state.read_dram.return_value.numpy.return_value.tobytes.return_value = initial[:16] + expected + initial[48:]
        device.core = Mock(arch_state=state, cycle_count=0, exus=[SimpleNamespace(has_in_flight=False)])
        device.core.is_finished.side_effect = lambda: state.halted
        device.dma = SimpleNamespace(last_cycle=SimpleNamespace(a=(0, 0, True), d=None))

        def tick():
            device.core.cycle_count += 1
            state.halted, state.halt_reason = True, "ecall"
            state._csr_values[0xC10] = 1

        device.core.tick.side_effect = tick
        return device, initial, expected

    def test_descriptor_controls_live_buffer_and_output_checks(self):
        device, initial, expected = self.device()
        device.start()
        self.assertEqual(len(device.completed), 1)
        self.assertTrue(device.completed[0]["exact"])
        self.assertEqual(device.completed[0]["observed_dma"]["store_bytes"], 32)
        device.remote.read.assert_called_once_with(0x100000000, 64)
        device.remote.write.assert_called_once_with(0x100000000, initial[:16] + expected + initial[48:])
        self.assertFalse(device.written)
        with self.assertRaises(AssertionError):
            device.start()

    def test_nonzero_segment_uses_actual_global_index_and_sequence(self):
        device, initial, expected = self.device()
        device.kernel_count = 840
        device.launch_start, device.launch_end, device.launch_count = 12, 13, 1
        device.programs = {12: device.programs[0]}
        device.bundle = SimpleNamespace(read=device.bundle.read, initial_dram=Mock(return_value=initial))
        state = device.core.arch_state
        state._csr_values[0xC11] = 13
        original_tick = device.core.tick.side_effect

        def tick():
            original_tick()
            state._csr_values[0xC10] = 13

        device.core.tick.side_effect = tick
        device.start()
        self.assertEqual([item["index"] for item in device.completed], [12])
        self.assertEqual(device.completed[0]["sequence"], 13)
        device.bundle.initial_dram.assert_called_once_with(12)
        self.assertTrue((device.out/"kernel-12.json").is_file())
        self.assertFalse((device.out/"kernel-00.json").exists())
        with self.assertRaisesRegex(AssertionError, "admitted segment range"):
            device.start()
        self.assertEqual(len(device.completed), 1)

    def test_nonzero_segment_rejects_local_instead_of_global_sequence(self):
        device, _, _ = self.device()
        device.kernel_count = 840
        device.launch_start, device.launch_end, device.launch_count = 12, 13, 1
        device.programs = {12: device.programs[0]}
        with self.assertRaises(AssertionError):
            device.start()
        device.remote.read.assert_not_called()
        device.core.tick.assert_not_called()
        self.assertEqual(device.completed, [])

    def test_actual_program_input_output_guards_and_dma_refused(self):
        def bad_imem(device):
            device.imem[0] ^= 1

        def bad_input(device):
            device.remote.read.return_value = b"z" * 64

        def bad_output(device):
            device.core.arch_state.read_dram.return_value.numpy.return_value.tobytes.return_value = bytes(64)

        def bad_guard(device):
            device.core.arch_state.read_dram.return_value.numpy.return_value.tobytes.return_value = b"z" + bytes(15) + b"x" * 32 + bytes(16)

        def bad_dma(device):
            device.programs[0][2]["requested_dma"]["load_bytes"] = 32

        for mutate in (bad_imem, bad_input, bad_output, bad_guard, bad_dma):
            device, _, _ = self.device()
            mutate(device)
            with self.subTest(boundary=mutate.__name__), self.assertRaises(AssertionError):
                device.start()
            self.assertFalse(device.completed)
            device.remote.write.assert_not_called()


class DeviceRangeTest(unittest.TestCase):
    def test_default_and_global_intervals(self):
        self.assertEqual(launch_bounds(840), (0, 840))
        self.assertEqual(launch_bounds(840, (12, 24)), (12, 24))
        self.assertEqual(launch_bounds(2715, [2714, 2715]), (2714, 2715))

    def test_invalid_interval_fails_before_model_configuration(self):
        with tempfile.TemporaryDirectory() as temporary, patch("model_device.configure_model") as configure:
            for selected in ((-1, 1), (0, 841), (12, 12), (24, 12), (True, 12), (0, True),
                             (0,), "12:24", (0.0, 12), (0, 12, 24)):
                with self.subTest(selected=selected), self.assertRaises(ValueError):
                    Device(None, Path(temporary), 0, SimpleNamespace(kernel_count=840),
                           Path(temporary), launch_range=selected)
            configure.assert_not_called()
        for count in (0, -1, True, 1.0, 1 << 32):
            with self.subTest(count=count), self.assertRaises(ValueError):
                launch_bounds(count)


if __name__ == "__main__":
    unittest.main()
