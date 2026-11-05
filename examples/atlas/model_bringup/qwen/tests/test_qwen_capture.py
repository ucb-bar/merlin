"""PyTorch semantics and fresh capture regression for Qwen subgraphs."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[5]


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("torch_mlir"),
    "requires the capture extra",
)
class QwenCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "examples/atlas/model_bringup/qwen"))
        import torch
        import kernels

        cls.torch, cls.kernels = torch, kernels
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(ROOT / "examples/atlas/model_bringup/qwen"))

    def test_attention_decomposition_preserves_all_128_channels(self):
        t = self.torch
        t.manual_seed(19)
        pieces = tuple(t.randn(32, 32) for _ in range(12))
        actual = t.cat(self.kernels.AttentionTile()(*pieces), dim=-1)
        q, k = t.cat(pieces[:4], -1), t.cat(pieces[4:8], -1)
        v = t.cat([x.T for x in pieces[8:]], -1)
        expected = t.softmax(q @ k.T / (128**0.5), dim=-1) @ v
        t.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_swiglu_partition_preserves_silu(self):
        t = self.torch
        t.manual_seed(5)
        x, g, u, d = (t.randn(32, 32) * 0.125 for _ in range(4))
        expected = (t.nn.functional.silu(x @ g.T) * (x @ u.T)) @ d.T
        t.testing.assert_close(self.kernels.SwiGLUTile()(x, g, u, d), expected)

    def test_fresh_attention_and_sigmoid_capture_normalizes(self):
        from atlas_mlir.capture import capture, load
        from atlas_mlir.normalize import normalize_linalg

        with tempfile.TemporaryDirectory() as tmp:
            # The loader contains a dataclass; loading must register its module.
            model, inputs = load(ROOT / "examples/atlas/model_bringup/qwen/kernels.py")
            attention = capture(model, inputs, Path(tmp) / "attention")
            source = attention.linalg.read_text()
            self.assertEqual(source.count("linalg.matmul"), 8)
            self.assertEqual(source.count("linalg.reduce"), 2)
            self.assertNotIn("arith.truncf", source)
            self.assertNotIn("tensor<32xi64>", source)
            self.assertEqual(normalize_linalg(source), source)
            swiglu = capture(
                self.kernels.SwiGLUTile(), inputs[:4], Path(tmp) / "swiglu"
            )
            self.assertIn("math.exp", swiglu.linalg.read_text())
            self.assertIn("arith.negf", swiglu.linalg.read_text())
            self.assertTrue((swiglu.linalg.parent / "raw-linalg.mlir").exists())

    def test_live_argmax_indices_are_preserved(self):
        from atlas_mlir.capture import capture

        t = self.torch

        class MaxWithIndices(t.nn.Module):
            def forward(self, x):
                return x.max(dim=-1)

        with tempfile.TemporaryDirectory() as tmp:
            result = capture(MaxWithIndices(), (t.rand(32, 32),), Path(tmp) / "capture")
            source = result.linalg.read_text()
            self.assertIn("tensor<32xi64>", source)
            self.assertIn("arith.index_cast", source)


if __name__ == "__main__":
    unittest.main()
