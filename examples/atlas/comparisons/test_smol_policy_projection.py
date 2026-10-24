"""Focused target projection conversion, bias and convolution order controls."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from smol_policy_projection import AtlasProjection
from tiny_fp8_pack import encode

if not os.environ.get('SMOL_ATLAS_TOOLS'):
    raise unittest.SkipTest('Optional Atlas tools directory is not configured')
TOOLS = Path(os.environ['SMOL_ATLAS_TOOLS'])
sys.path.insert(0, str(TOOLS))
from atlas_tile_arithmetic import mxu0_dot, rescale_bf16


class ProjectionTest(unittest.TestCase):
    def test_foreign_cached_helper_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {'atlas_program': SimpleNamespace(__file__='/other/atlas_program.py')}):
                with self.assertRaises(ValueError):
                    AtlasProjection(TOOLS, Path(directory) / 'kernels', 1)

    def test_float32_bias_and_logical_tails_match_scalar(self):
        module = torch.nn.Linear(35, 7).requires_grad_(False)
        module.weight.copy_(torch.arange(245).reshape(7, 35).float() / 128 - 1)
        module.bias.copy_(torch.arange(7).float() / 16)
        incoming = (torch.arange(105).reshape(3, 35).float() / 32 - 1).unsqueeze(0)
        a, sa = encode(incoming.to(torch.bfloat16).view(torch.uint16).numpy()[0].copy())
        b, sb = encode(module.weight.to(torch.bfloat16).t().contiguous().view(torch.uint16).numpy().copy())
        words = np.array([[rescale_bf16(mxu0_dot(a[r], b[:, c]), sa + sb) for c in range(7)] for r in range(3)], np.uint16)
        expected = torch.from_numpy(words).view(torch.bfloat16).float().unsqueeze(0) + module.bias
        with tempfile.TemporaryDirectory() as directory:
            provider = AtlasProjection(TOOLS, Path(directory) / 'kernels', 2)
            actual = provider.apply('linear', module, incoming, torch.float32)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertEqual(actual.shape, incoming.shape[:-1] + (7,))
            self.assertEqual(actual.numpy().tobytes(), expected.numpy().tobytes())
            self.assertEqual(provider.kernels[0]['physical_mkn'], [32, 64, 32])
            self.assertFalse(provider.finish()['fresh_model_executed'])

    def test_patch_convolution_preserves_channel_row_column_order(self):
        module = torch.nn.Conv2d(3, 2, 16, stride=16, padding='valid', bias=False).to(torch.bfloat16).requires_grad_(False)
        module.weight.zero_()
        module.weight[0, 0, 0, 0] = 1
        module.weight[1, 2, 15, 15] = 1
        incoming = torch.zeros((1, 3, 16, 32), dtype=torch.bfloat16)
        incoming[0, 0, 0, 0], incoming[0, 0, 0, 16] = 2, 4
        incoming[0, 2, 15, 15], incoming[0, 2, 15, 31] = 8, 16
        with tempfile.TemporaryDirectory() as directory:
            provider = AtlasProjection(TOOLS, Path(directory) / 'kernels', 1)
            actual = provider.apply('patch', module, incoming, torch.bfloat16)
            expected = torch.tensor([[[[2, 4]], [[8, 16]]]], dtype=torch.bfloat16)
            self.assertEqual(actual.view(torch.uint16).numpy().tobytes(), expected.view(torch.uint16).numpy().tobytes())

    def test_unsupported_patch_geometry_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = AtlasProjection(TOOLS, Path(directory) / 'kernels', 1)
            with self.assertRaises(ValueError):
                provider.apply('bad', torch.nn.Conv2d(3, 2, 3), torch.zeros(1, 3, 16, 16), torch.float32)
            self.assertFalse(provider.kernels)


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unittest.main()
