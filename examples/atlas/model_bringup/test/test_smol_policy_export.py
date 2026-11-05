"""Focused packed initial-image and descriptor export controls."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from export_smol_policy_kernels import split_initial, require_finite_trajectory


class SmolPolicyExportTest(unittest.TestCase):
    def test_raw_bf16_and_float_nonfinite_trajectory_refused(self):
        require_finite_trajectory({'cache': np.array([0x3f80], np.uint16), 'state': np.ones(1, np.float32)})
        for word in (0x7f80, 0xff80, 0x7fc1):
            with self.assertRaises(ValueError):
                require_finite_trajectory({'cache': np.array([word], np.uint16)})
        with self.assertRaises(ValueError):
            require_finite_trajectory({'state': np.array([np.nan], np.float32)})

    def setUp(self):
        self.a, self.w, self.expected = b'A'*1024, b'W'*1024, b'\0'*2048
        self.initial = self.a + self.w + b'\xa5'*256 + b'\xcd'*2048 + b'Z'*256
        self.row = {'physical_mkn': [32, 32, 32], 'output_offset': 2304,
                    'dram_bytes': 4608, 'resident_a': False, 'resident_b': True,
                    'requested_dma': {'load_bytes': 2048, 'store_bytes': 2048}}

    def test_exact_packed_operands_retained(self):
        self.assertEqual(split_initial(self.initial, self.expected, self.row), (self.a, self.w))

    def test_guard_geometry_and_dma_mutations_refused(self):
        for offset in (2048, 2304, 4607):
            changed = bytearray(self.initial)
            changed[offset] ^= 1
            with self.assertRaises(ValueError):
                split_initial(bytes(changed), self.expected, self.row)
        for key, value in (('output_offset', 2305), ('dram_bytes', 4609),
                           ('physical_mkn', [32, 33, 32]),
                           ('requested_dma', {'load_bytes': 2049, 'store_bytes': 2048})):
            with self.assertRaises(ValueError):
                split_initial(self.initial, self.expected, {**self.row, key: value})



if __name__ == '__main__':
    unittest.main()
