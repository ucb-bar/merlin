"""Focused full-model image admission and live staging boundary controls."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock

TOOLS = Path(__file__).absolute().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "host_iss"))
from full_bundle import FullBundle, KernelAssets
from full_bundle_fixture import full_bundle_fixture
from model_device import Device
from run import arguments, mmio_access_cap


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class FullBundleTest(unittest.TestCase):
    def fixture(self, mutate=None, *, kernel_only=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        return full_bundle_fixture(Path(temporary.name), mutate, kernel_only=kernel_only)

    def test_kernel_assets_are_preparation_without_host_and_cannot_admit_full_host(self):
        root, initial = self.fixture(kernel_only=True)
        assets = KernelAssets(root, "tiny-full")
        self.assertTrue(assets.preparation_only)
        self.assertFalse(hasattr(assets, "build"))
        self.assertEqual(assets.initial_dram(1), initial)
        self.assertEqual(assets.metadata_name, "kernels.json")
        with self.assertRaises(ValueError):
            FullBundle(root, "tiny-full")

    def test_kernel_assets_require_listed_metadata_and_ordered_typed_indices(self):
        for mutate in (lambda p: p["kernels"][1].update(index=0),
                       lambda p: p["kernels"][0].update(index=True),
                       lambda p: p.update(workload="smol-full")):
            root, _ = self.fixture(mutate, kernel_only=True)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                KernelAssets(root, "tiny-full")
        root, _ = self.fixture(kernel_only=True)
        sums = root / "SHA256SUMS"
        sums.write_text(''.join(line + '\n' for line in sums.read_text().splitlines()
                                if not line.endswith('  kernels.json')))
        with self.assertRaisesRegex(ValueError, "unlisted"):
            KernelAssets(root, "tiny-full")

    def test_shared_weights_reconstruct_each_exact_image(self):
        root, initial = self.fixture()
        bundle = FullBundle(root, "tiny-full")
        self.assertEqual(bundle.initial_dram(0), initial)
        self.assertEqual(bundle.initial_dram(1), initial)
        paths = bundle.kernel_paths(0)
        paths["physical_mkn"][0] = 64
        paths["requested_dma"]["store_bytes"] = 0
        self.assertEqual(bundle.kernel_paths(0)["physical_mkn"], [32, 32, 32])
        self.assertEqual(bundle.kernel_paths(0)["requested_dma"]["store_bytes"], 2048)
        self.assertEqual(len([name for name in bundle.sums if name.endswith("w.bin")]), 1)

    def test_changed_shared_weight_and_image_hash_rejected(self):
        root, _ = self.fixture()
        bundle = FullBundle(root, "tiny-full")
        (root / "assets/w.bin").write_bytes(b"x" * 1024)
        with self.assertRaisesRegex(ValueError, "changed bundle payload"):
            bundle.initial_dram(1)
        root, _ = self.fixture(lambda value: value["kernels"][0].update(initial_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "initial image"):
            FullBundle(root, "tiny-full")

    def test_geometry_and_counts_rejected_before_image_allocation(self):
        mutations = [lambda p: p.update(kernel_count=20001),
                     lambda p: p.update(kernel_count=True),
                     lambda p: p.update(contract_sha256="0" * 64),
                     lambda p: p["kernels"][0].update(physical_mkn=[True, 32, 32]),
                     lambda p: p["kernels"][0].update(physical_mkn=[32, 31, 32]),
                     lambda p: p["kernels"][0].update(total_dram_bytes=64 * 1024 * 1024 + 32),
                     lambda p: p["kernels"][0].update(output_offset=2048),
                     lambda p: p["kernels"][0].update(weights="unlisted.bin"),
                     lambda p: p["kernels"][0].update(requested_dma=dict(load_bytes=1, store_bytes=1))]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                root, _ = self.fixture(mutate)
                with self.assertRaises(ValueError):
                    FullBundle(root, "tiny-full")

    def test_device_compares_real_guest_staging_to_reconstructed_image(self):
        root, initial = self.fixture()
        bundle = FullBundle(root, "tiny-full")
        device = Device.__new__(Device)
        device.bundle, device.completed, device.kernel_count, device.pins = bundle, [], 2, {}
        device.launch_start, device.launch_end, device.launch_count = 0, 2, 2
        device.core = Mock()
        device.core.exus = []
        device.core.arch_state.halted = True
        device.core.arch_state.check_flag.return_value = False
        device.core.arch_state.read_csrf.side_effect = lambda address: int(address == 0xC11)
        device.programs = [(None, bytes(4), bundle.kernel_paths(0))]
        device.imem, device.written = bytearray(4), {0}
        device.remote = Mock()
        device.remote.read.return_value = b"x" + initial[1:]
        # This must fail before ticking any device or substituting captured data.
        with self.assertRaisesRegex(AssertionError, "host staging differs"):
            device.start()
        device.remote.read.assert_called_once_with(0x100000000, len(initial))
        device.core.tick.assert_not_called()
        self.assertEqual(device.completed, [])


class FullBudgetTest(unittest.TestCase):
    def cli(self, *extra):
        return ["--out", "out", "--elf", "elf", "--bundle-root", "bundle",
                "--model-root", "model", "--qemu", "qemu", *extra]

    def test_full_budget_explicit_and_bounded_old_budget_unchanged(self):
        self.assertEqual(arguments(self.cli()).wall_seconds, 21600)
        self.assertEqual(arguments(self.cli("--workload", "tiny-full", "--wall-seconds", "1209600")).wall_seconds,
                         1209600)
        self.assertEqual(arguments(self.cli("--workload", "smol-full", "--wall-seconds", "86400")).workload,
                         "smol-full")
        for extra in [("--workload", "tiny-full"),
                      ("--workload", "tiny", "--wall-seconds", "86401"),
                      ("--workload", "tiny-full", "--wall-seconds", "2592001"),
                      ("--workload", "smol-full"),
                      ("--wall-seconds", "0"), ("--kernel-cycle-cap", "0")]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(self.cli(*extra))

    def test_full_mmio_budget_uses_actual_admitted_program_sizes(self):
        device = Mock()
        device.programs = {index: (None, bytes(20000), {}) for index in range(840)}
        self.assertEqual(mmio_access_cap(device, "tiny-full"), 100000 + 840 * 5064)
        self.assertEqual(mmio_access_cap(device, "tiny"), 100000)
        device.programs = {index: (None, bytes(20000), {}) for index in range(12, 24)}
        self.assertEqual(mmio_access_cap(device, "tiny-full"), 100000 + 12 * 5064)


if __name__ == "__main__":
    unittest.main()
