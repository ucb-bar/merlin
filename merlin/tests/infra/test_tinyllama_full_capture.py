"""Focused full Tiny greedy feedback and physical operand controls."""
import unittest
import sys

from merlin.common.paths import repo_root

sys.path.insert(0, str(repo_root() / "examples/atlas/comparisons"))

import numpy as np

from capture_tinyllama_full import SITES, greedy_token, padded_words


class FullCaptureTest(unittest.TestCase):
    def test_all_layers_and_head_in_original_order(self):
        self.assertEqual(len(SITES), 155)
        self.assertEqual(SITES[:2], ("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj"))
        self.assertEqual(SITES[-2:], ("model.layers.21.mlp.down_proj", "lm_head"))
        self.assertEqual(len(set(SITES)), len(SITES))

    def test_greedy_maximum_tie_and_signed_zero(self):
        values = np.full(32000, -4, np.float32)
        values[23] = values[42] = 3
        self.assertEqual(greedy_token(values), 23)
        values[:] = -0.0
        values[17] = 0.0
        self.assertEqual(greedy_token(values), 0)

    def test_logits_admission(self):
        for value in (np.zeros(31999, np.float32), np.zeros(32000, np.float64),
                      np.full(32000, np.inf, np.float32), np.full(32000, np.nan, np.float32)):
            with self.subTest(shape=value.shape, dtype=value.dtype), self.assertRaises(ValueError):
                greedy_token(value)

    def test_decode_padding_is_positive_zero_and_preserves_words(self):
        words = np.array([[0x8000, 0x3f80, 0xbf80]], np.uint16)
        physical = padded_words(words, 1)
        self.assertEqual(physical.shape, (32, 3))
        self.assertEqual(physical[0].tobytes(), words[0].tobytes())
        self.assertTrue(np.all(physical[1:] == 0))
        self.assertEqual(padded_words(np.ones((128, 32), np.uint16), 128).shape, (128, 32))

    def test_bad_physical_geometry_rejected(self):
        for words, rows in ((np.zeros((2, 32), np.uint16), 2),
                            (np.zeros((1, 32), np.float32), 1),
                            (np.zeros((128, 32), np.uint16), 1)):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                padded_words(words, rows)


if __name__ == "__main__":
    unittest.main()
