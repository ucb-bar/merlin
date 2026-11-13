"""Directed policy checks use scalar double products and struct f32 rounding."""
import struct
import unittest
import numpy as np
import whole_reference as ref


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def scalar_reference(a, b):
    total = 0.0
    for left, right in zip(a, b):
        lv = struct.unpack('<f', struct.pack('<I', int(left) << 16))[0]
        rv = struct.unpack('<f', struct.pack('<I', int(right) << 16))[0]
        total = f32(total + lv * rv)
    bits = struct.unpack('<I', struct.pack('<f', total))[0]
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff


class HeadTests(unittest.TestCase):
    def test_exact_products_and_cancellation(self):
        a = np.array([[0x3f81, 0xbf80]], np.uint16)
        b = np.array([[0x3f81, 0x3f82]], np.uint16)
        self.assertEqual(int(ref.bf16_linear(a, b)[0, 0]), 0x3880)  # 2**-14

    def test_sequential_not_tree_or_tile_rounding(self):
        a = np.full((2, 65), 0x3f80, np.uint16)
        b = np.full((3, 65), 0x3f80, np.uint16)
        b[:, 0], b[:, -1] = 0x4c80, 0xcc80  # 2**26 + small - 2**26 -> 0
        np.testing.assert_array_equal(ref.bf16_linear(a, b, columns=2), np.zeros((2, 3), np.uint16))

    def test_ties_even_once(self):
        a = np.array([[0x3f80, 0x3b80]], np.uint16)
        b = np.array([[0x3f80, 0x3f80], [0x3f81, 0x3f80]], np.uint16)
        np.testing.assert_array_equal(ref.bf16_linear(a, b), [[0x3f80, 0x3f82]])

    def test_random_scalar_and_column_invariance(self):
        rng = np.random.default_rng(4)
        a = (rng.integers(0x3c00, 0x4200, (2, 67), dtype=np.uint16) |
             (rng.integers(0, 2, (2, 67), dtype=np.uint16) << 15))
        b = (rng.integers(0x3c00, 0x4200, (5, 67), dtype=np.uint16) |
             (rng.integers(0, 2, (5, 67), dtype=np.uint16) << 15))
        expected = np.array([[scalar_reference(row, w) for w in b] for row in a], np.uint16)
        for columns in (1, 3, 256):
            np.testing.assert_array_equal(ref.bf16_linear(a, b, columns=columns), expected)

    def test_tiny_products_follow_f32_rne_underflow(self):
        np.testing.assert_array_equal(ref.bf16_linear(np.array([[0x0080]], np.uint16),
                                                     np.array([[0x3f80]], np.uint16)), [[0x0080]])
        np.testing.assert_array_equal(ref.bf16_linear(np.array([[0x0001]], np.uint16),
                                                     np.array([[0x0001]], np.uint16)), [[0]])

    def test_final_bf16_overflow_refused(self):
        with self.assertRaisesRegex(ValueError, 'final result must be finite'):
            ref.bf16_linear(np.array([[0x7f7f, 0x7b00]], np.uint16),
                            np.array([[0x3f80, 0x3f80]], np.uint16))


if __name__ == '__main__':
    unittest.main()
