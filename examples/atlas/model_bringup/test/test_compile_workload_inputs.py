"""Source-only packing controls, with assembler and codec isolated from execution."""

from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compile_workload_inputs as producer


class WorkloadPreparationTest(unittest.TestCase):
    def setUp(self):
        self.arrays = {"A": np.zeros((64, 64), dtype=np.float32), "B": np.zeros((64, 64), dtype=np.float32),
            "native_C": np.zeros((64, 64), dtype=np.float32)}
        r, c = np.indices((64, 64))
        self.encoded = {"A": ((r * 37 + c * 13) % 251).astype(np.uint8), "B": ((r * 11 + c * 29) % 239).astype(np.uint8)}
        self.conversion = {"A": {"scale_exponent": -1}, "B": {"scale_exponent": 1}}
        self.assembler = mock.Mock(return_value=mock.Mock(assemble=lambda: [0x13, 0x73]))

    def test_independent_tile_permutations_and_guard_regions(self):
        with mock.patch.object(producer, "encode_operands", return_value=(self.encoded, {}, self.conversion)):
            prepared = producer.prepare(self.arrays, "float32", self.assembler)
        # Grid-axis permutations provide an independent layout specification.
        expected_a = self.encoded["A"].reshape(2, 32, 2, 32).transpose(0, 2, 1, 3).tobytes()
        expected_b = self.encoded["B"].reshape(2, 32, 2, 32).transpose(2, 0, 1, 3).tobytes()
        self.assertEqual(prepared["activation"], expected_a)
        self.assertEqual(prepared["weights"], expected_b)
        self.assertEqual(prepared["initial"], expected_a + expected_b + b"\xa5" * 256 + b"\xcd" * 8192 + b"\x5a" * 256)
        self.assertEqual(prepared["program"], b"\x13\0\0\0\x73\0\0\0")
        self.assertEqual(prepared["rescale_exponent"], 0)
        self.assertEqual(prepared["requested_dma"], {"load_bytes": 12288, "store_bytes": 8192})
        self.assertEqual(prepared["memory"]["output_offset"], 8448)

    def test_source_admission_precedes_codec_and_assembler(self):
        self.arrays["A"][0, 0] = np.nan
        with mock.patch.object(producer, "encode_operands") as codec:
            with self.assertRaises(ValueError):
                producer.prepare(self.arrays, "float32", self.assembler)
        codec.assert_not_called()
        self.assembler.assert_not_called()

    def test_invalid_terminal_and_rescale_refused_without_device(self):
        for words in ([0x13], [0x73, 0x73], [0x00100073, 0x73]):
            self.assembler.return_value.assemble = lambda: words
            with mock.patch.object(producer, "encode_operands", return_value=(self.encoded, {}, self.conversion)):
                with self.assertRaises(ValueError):
                    producer.prepare(self.arrays, "float32", self.assembler)
        self.conversion["A"]["scale_exponent"] = 127
        with mock.patch.object(producer, "encode_operands", return_value=(self.encoded, {}, self.conversion)):
            with self.assertRaisesRegex(ValueError, "normal finite"):
                producer.prepare(self.arrays, "float32", self.assembler)

    def test_output_halves_and_physical_dtype_shape_admission(self):
        values = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
        expected = values.reshape(2, 32, 2, 2, 16).transpose(0, 2, 3, 1, 4).astype("<u2").tobytes()
        self.assertEqual(producer.pack_output(values), expected)
        for helper, valid in ((producer.pack_activation, self.encoded["A"]), (producer.pack_weights, self.encoded["B"]),
            (producer.pack_output, values)):
            for invalid in (valid.astype(np.float32), valid[:31], valid.reshape(-1), valid[:0]):
                with self.subTest(helper=helper.__name__, shape=invalid.shape), self.assertRaises(ValueError):
                    helper(invalid)
