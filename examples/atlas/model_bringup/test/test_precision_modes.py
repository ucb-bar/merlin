"""Precision-mode arithmetic against the scalar oracle and exact Fraction references, without model files.

default_cells.json holds one full-K activation row and weight column per model from the PR #26 run artifacts (BF16
words and the promoted FP32 output in little-endian hex), with the scales of the original full tensors.
"""

from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import kernel_library  # noqa: E402,F401
import atlas_tile_arithmetic as oracle  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from precision_modes import arithmetic


def bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0] >> 16


def reference(a, b, ea, eb, chunk=0, low=None):
    """Fraction products and BF16 stores per chain, then one exact sum of the scaled chains and one rounding."""
    result = []
    for row in range(a.shape[0]):
        for col in range(b.shape[1]):
            total = Fraction(0)
            for start in range(0, a.shape[1], chunk or a.shape[1]):
                for weight, exponent in [(b, eb)] + ([low] if low else []):
                    accumulator = 0
                    for k in range(start, min(start + (chunk or a.shape[1]), a.shape[1])):
                        product = Fraction(float(a[row, k])) * Fraction(float(weight[k, col]))
                        accumulator = oracle.bf16_round(oracle.bf16_value(accumulator) + product)
                    e = torch.broadcast_to(ea + exponent, (a.shape[0], b.shape[1]))[row, col]
                    total += oracle.bf16_value(accumulator) * oracle.power2(int(e))
            result.append(oracle.bf16_round(total))
    return result


@unittest.skipUnless(HAS_TORCH and shutil.which("c++"), "precision tests require optional Torch and C++")
class PrecisionModesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        library = Path(cls.temporary.name) / "ordered_mac.so"
        subprocess.run(["c++", "-O3", "-shared", "-fPIC", str(ROOT / "tools/precision_modes/ordered_mac.cpp"), "-o",
                        str(library)], check=True)
        os.environ["ATLAS_PRECISION_MAC_LIBRARY"] = str(library)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def assert_bits(self, actual, expected):
        self.assertEqual([bits(x) for x in actual.flatten()], expected)

    def test_codec_and_scales_match_scalar_oracle(self):
        generator = torch.Generator().manual_seed(947)
        words = [0, 0x8000, 1, 0x7f, 0x80, 0x3f88, 0x3f98, 0x43e0, 0x43ef, 0x7f7f]
        words += torch.randint(0, 0x7f80, (256,), generator=generator).tolist()
        front = torch.tensor(words, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        for exponent in (-20, 0, 20):
            expected = [float(oracle.fp8_value(oracle.pack_bf16(w, exponent + 127))) for w in words]
            self.assertEqual(arithmetic.quantize(front, torch.tensor(exponent)).tolist(), expected)
        x = torch.randn((3, 40), generator=generator) * torch.tensor([[1e-3], [1.0], [3e3]])
        x = torch.cat((x, torch.tensor([[448.0], [480.0], [0.0]]).expand(3, 40)))
        words = x.bfloat16().view(torch.int16).int() & 65535
        self.assertEqual(int(arithmetic.scale(x)), oracle.tensor_scale_exponent(words.flatten().tolist()))
        self.assertEqual(arithmetic.scale(x, "row").flatten().tolist(),
                         [oracle.tensor_scale_exponent(row) for row in words.tolist()])

    def test_recipes_match_fraction_reference(self):
        generator = torch.Generator().manual_seed(733)
        left, right = torch.randn((2, 65), generator=generator), torch.randn((65, 3), generator=generator)
        left[1] *= 2**-12
        right[:, 0] *= 2**10
        for a_axis, b_axis, chunk, two in (("tensor", "tensor", 0, False), ("row", "column", 0, False),
                                           ("tensor", "tensor", 32, False), ("tensor", "tensor", 0, True),
                                           ("row", "column", 32, True)):
            with self.subTest(a_axis=a_axis, b_axis=b_axis, chunk=chunk, two=two):
                a, ea = arithmetic.pack(left, a_axis)
                (b, eb), low = arithmetic.two_term(right, b_axis) if two else (arithmetic.pack(right, b_axis), None)
                self.assert_bits(arithmetic.contraction(a, b, ea, eb, chunk, low), reference(a, b, ea, eb, chunk, low))
        # Ascending K with a BF16 store per MAC discriminates a BLAS sum, and a K split changes that order.
        a, b = torch.ones((1, 65), dtype=torch.float64), torch.ones((65, 1), dtype=torch.float64)
        b[0] = 256
        self.assertEqual(float(arithmetic.ordered_mm(a[:, :3], b[:3])), 256.0)
        self.assertEqual(float(a[:, :3] @ b[:3]), 258.0)
        zero = torch.tensor(0)
        self.assertNotEqual(float(arithmetic.contraction(a, b, zero, zero, 32)),
                            float(arithmetic.contraction(a, b, zero, zero)))

    def test_direct_bf16_avoids_float32_double_round(self):
        values = [1 + 2**-8 + 2**-40, 1 + 3 * 2**-8 - 2**-40]
        source = torch.tensor(values + [-x for x in values], dtype=torch.float64)
        actual = arithmetic.direct_bf16(source)
        self.assert_bits(actual, [oracle.bf16_round(Fraction(float(x))) for x in source])
        self.assertFalse(torch.equal(actual, source.float().bfloat16().float()))
        self.assertEqual(bits(arithmetic.direct_bf16(torch.tensor([-0.0]))), 0)
        # Chains of 1, 2^-8 (second K chunk), and 2^-40 (low term) sum above a BF16 midpoint that float32 rounds onto.
        a, b, low = torch.zeros((1, 64)), torch.zeros((64, 1)), torch.zeros((64, 1))
        a[0, 0], a[0, 32], b[0], b[32], low[0] = 1, 0.25, 1, 2**-6, 1
        zero, low = torch.tensor(0), (low, torch.tensor(-40))
        self.assert_bits(arithmetic.contraction(a, b, zero, zero, 32, low), [0x3f81])

    def test_site_selection(self):
        args = SimpleNamespace(finer_scales=None, k_split=0, two_term=False, sites=["*"])
        self.assertEqual(arithmetic.settings(args, "lm_head"), ("tensor", "tensor", 0, False))
        args.two_term, args.sites = True, ["layer1_0_conv1"]
        self.assertEqual([arithmetic.settings(args, s)[3] for s in ("layer1_0_conv1", "fc")], [True, False])
        args.two_term, args.k_split, args.finer_scales, args.sites = False, 128, "both", ["*self_attn.*", "lm_head"]
        self.assertEqual(arithmetic.settings(args, "lm_head"), ("row", "column", 128, False))
        self.assertEqual(arithmetic.settings(args, "model.layers.0.mlp.down_proj"), ("row", "column", 0, False))
        args.finer_scales = "activation"
        self.assertEqual(arithmetic.settings(args, "lm_head")[:2], ("row", "tensor"))

    def test_frozen_default_cells(self):
        for item in json.loads((ROOT / "test/fixtures/precision_modes/default_cells.json").read_text()):
            with self.subTest(model=item["model"]):
                def front(key, shape):
                    words = torch.frombuffer(bytearray.fromhex(item[key]), dtype=torch.int16)
                    return words.view(torch.bfloat16).reshape(shape).float()
                ea, eb = torch.tensor(item["ea"]), torch.tensor(item["eb"])
                a = arithmetic.quantize(front("a_bf16_le_hex", (1, item["k"])), ea)
                b = arithmetic.quantize(front("b_bf16_le_hex", (item["k"], 1)), eb)
                self.assertEqual(struct.pack("<f", float(arithmetic.contraction(a, b, ea, eb))),
                                 bytes.fromhex(item["expected_f32_le_hex"]))


if __name__ == "__main__":
    unittest.main()
