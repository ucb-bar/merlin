"""Resource and provenance controls for the separate workload producer."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    import run_workload_contraction as workload
    import kernel_library  # noqa: E402,F401
    from atlas_tile_arithmetic import output_halves
    from run_model_tile import validate_arrays


@unittest.skipUnless(np is not None, "NumPy required for workload producer")
class WorkloadAdmissionTest(unittest.TestCase):
    def arrays(self, m=32, k=32, n=32):
        return {"A": np.zeros((m, k), dtype=np.float32), "B": np.zeros((k, n), dtype=np.float32),
            "native_C": np.zeros((m, n), dtype=np.float32)}

    def test_separate_k4608_admission_preserves_old_limit(self):
        arrays = self.arrays(k=4608)
        self.assertEqual(workload.validate_workload_arrays(arrays, "float32")[0].shape, (32, 4608))
        with self.assertRaises(ValueError):
            validate_arrays(arrays, "float32", matrix=True)
        plan = workload.resource_plan(32, 4608, 32)
        self.assertTrue(plan["resident_b"])
        self.assertEqual(plan["resident_bank1_bytes"], 147456)
        self.assertEqual(plan["dram_bytes"], 297472)

    def test_c1_exact_memory_ledger(self):
        plan = workload.resource_plan(3136, 64, 64)
        self.assertEqual(plan["dram_bytes"], 606720)
        self.assertEqual(plan["resident_bank1_byte_range"], [0x40000, 0x41000])
        self.assertEqual(plan["streamed_scratch_bank0_byte_ranges"], [[0x2000, 0x2400], [0x2400, 0x2800], [0x2800, 0x3000]])

    def test_resident_a_and_streamed_choices(self):
        plan = workload.resource_plan(32, 4096, 96)
        self.assertFalse(plan["resident_b"])
        self.assertTrue(plan["resident_a"])
        self.assertEqual(plan["resident_bank1_byte_range"], [0x40000, 0x60000])
        plan = workload.resource_plan(96, 4096, 96)
        self.assertFalse(plan["resident_b"] or plan["resident_a"])
        self.assertEqual(plan["resident_bank1_bytes"], 0)

    def test_shape_and_storage_refusals(self):
        for shape in ((0, 32, 32), (31, 32, 32), (32, 8224, 32), (1056, 32, 1024), (8192, 8192, 32)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                workload.resource_plan(*shape)
        for field in ("A", "B", "native_C"):
            arrays = self.arrays()
            arrays[field][0, 0] = np.nan
            with self.subTest(field=field), self.assertRaises(ValueError):
                workload.validate_workload_arrays(arrays, "float32")
        arrays = self.arrays()
        arrays["A"] = arrays["A"].astype(np.float64)
        with self.assertRaises(ValueError):
            workload.validate_workload_arrays(arrays, "float32")
        arrays = self.arrays()
        arrays["B"] = np.zeros((64, 32), dtype=np.float32)
        with self.assertRaises(ValueError):
            workload.validate_workload_arrays(arrays, "float32")
        with self.assertRaises(ValueError):
            workload.validate_workload_arrays(self.arrays(), "bfloat16")
        arrays = self.arrays()
        arrays["A_bf16"] = np.zeros((32, 32), dtype=np.uint16)
        with self.assertRaises(ValueError):
            workload.validate_workload_arrays(arrays, "float32")

    def test_only_one_terminal_ecall_with_conservative_imem_bound(self):
        workload.admit_program([0x13] * 8191 + [0x73])
        for words in ([], [0x13], [0x73, 0x73], [0x00100073, 0x73], [0x13] * 8192 + [0x73]):
            with self.subTest(length=len(words)), self.assertRaises(ValueError):
                workload.admit_program(words)

    def test_physical_payload_matches_existing_tile_layout(self):
        bits = np.arange(32 * 64, dtype=np.uint16).reshape(32, 64)
        expected = output_halves(bits[:, :32]) + output_halves(bits[:, 32:])
        self.assertEqual(workload.physical_payload(bits), expected)

    def test_all_five_producer_sources_are_hashed(self):
        sources = workload.code_sources()
        self.assertEqual(set(sources),
            {"run_workload_contraction.py", "atlas_program.py", "run_model_tile.py", "atlas_tile_arithmetic.py",
            "atlas_vector_arithmetic.py"})
        paths = {name: kernel_library.source_path(name) for name in sources}
        self.assertEqual(paths["run_workload_contraction.py"], Path(workload.__file__).absolute())
        self.assertEqual(paths["run_model_tile.py"], Path(workload.__file__).absolute().with_name("run_model_tile.py"))
        for name in ("atlas_program.py", "atlas_tile_arithmetic.py", "atlas_vector_arithmetic.py"):
            self.assertEqual(paths[name], kernel_library.module_path(name))
        for name, digest in sources.items():
            self.assertEqual(digest, workload.sha256(paths[name]))

    def test_requested_dma_counts_tile_reuse(self):
        from atlas_program import dma_load_bytes

        # One tile loads A and B once in every mode.
        for flags in ({}, {"resident_a": True}, {"resident_b": True}):
            with self.subTest(flags=flags):
                self.assertEqual(dma_load_bytes(32, 32, 32, **flags), 2048)
        # A 2x4 output grid reduces three K tiles per output tile: 48 loads.
        self.assertEqual(dma_load_bytes(64, 96, 128), 48 * 1024)
        # Resident A: 6 A tiles once, 12 B tiles twice.
        self.assertEqual(dma_load_bytes(64, 96, 128, resident_a=True), 30 * 1024)
        # Resident B: 12 B tiles once, 6 A tiles four times.
        self.assertEqual(dma_load_bytes(64, 96, 128, resident_b=True), 36 * 1024)
        with self.assertRaises(ValueError):
            dma_load_bytes(32, 32, 32, resident_a=True, resident_b=True)


if __name__ == "__main__":
    unittest.main()
