"""Focused SmolVLA tail and connector recipe arithmetic controls."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import kernel_library  # noqa: E402,F401
from atlas_tile_arithmetic import mxu0_dot, rescale_bf16
from smol_policy_kernel import prepare


class SmolPolicyKernelTest(unittest.TestCase):
    def test_unaligned_tails_and_parallel_order_match_scalar(self):
        rng = np.random.default_rng(8)
        # Include signed and flushed-zero values, with a nontrivial K tail.
        codes = np.array([0, 0x80, 0x08, 0x38, 0x40, 0xb8, 0xc0], np.uint8)
        a, b = rng.choice(codes, (3, 35)), rng.choice(codes, (35, 7))
        scalar = np.array([[rescale_bf16(mxu0_dot(a[r], b[:, c]), -2)
                            for c in range(7)] for r in range(3)], np.uint16)
        for workers in (1, 4):
            result = prepare(a, b, -2, workers=workers)
            np.testing.assert_array_equal(result["logical_output"], scalar)
            self.assertEqual(result["physical_mkn"], [32, 64, 32])
            self.assertFalse(np.any(result["A"][3:]))
            self.assertFalse(np.any(result["B"][35:]))
            self.assertFalse(np.any(result["B"][:, 7:]))

    def test_connector_reduction_is_single_ordered_program(self):
        result = prepare(np.zeros((1, 12288), np.uint8), np.zeros((12288, 1), np.uint8), 0)
        self.assertEqual(result["source"].count("vmatmul.mxu0 "), 1)
        self.assertEqual(result["source"].count("vmatmul.acc.mxu0 "), 383)
        self.assertEqual(result["source"].count("vmatpop.bf16.acc.mxu0"), 1)
        self.assertTrue(result["source"].endswith("ecall\n"))

    def test_invalid_values_and_resource_caps(self):
        good = np.zeros((32, 32), np.uint8)
        for a, b, exponent, workers in ((good.astype(np.int16), good, 0, 1),
                                      (good, good, 128, 1), (good, good, 0, 17),
                                      (np.full((32, 32), 127, np.uint8), good, 0, 1),
                                      (np.zeros((1, 12289), np.uint8), np.zeros((12289, 1), np.uint8), 0, 1)):
            with self.assertRaises(ValueError):
                prepare(a, b, exponent, workers=workers)

    def test_dram_admission_precedes_padding_and_arithmetic(self):
        a = np.zeros((1, 12288), np.uint8)
        b = np.broadcast_to(np.zeros((1, 8192), np.uint8), (12288, 8192))
        with patch('smol_policy_kernel.np.pad') as pad, patch('smol_policy_kernel.ordered_matmul') as oracle:
            with self.assertRaises(ValueError):
                prepare(a, b, 0)
            pad.assert_not_called()
            oracle.assert_not_called()


if __name__ == "__main__":
    unittest.main()
