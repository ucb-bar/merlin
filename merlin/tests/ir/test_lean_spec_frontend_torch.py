"""Independent NumPy transcription of proposed frontend intent vs pinned PyTorch."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import repo_root


def test_frontend_intent_torch_oracle():
    interpreter = os.environ.get("MERLIN_TORCH_PYTHON")
    if not interpreter or not Path(interpreter).is_file():
        if os.environ.get("MERLIN_LEAN_REQUIRED") == "1":
            pytest.fail("MERLIN_TORCH_PYTHON must identify the pinned PyTorch oracle")
        pytest.skip("MERLIN_TORCH_PYTHON is not configured")
    # Source specification: positive one/zero, magnitude != zero, independent
    # leading rows with weight[output_channel, contraction_coordinate].
    program = r"""
import numpy as np
import torch
assert torch.__version__ == "2.10.0+cpu", torch.__version__
b = np.array([False, True], dtype=np.bool_)
np.testing.assert_array_equal(torch.from_numpy(b).to(torch.float32).numpy(),
                              np.where(b, np.float32(1), np.float32(0)))
bits = np.array([0, 0x80000000, 0x3f000000, 0xbf000000, 1, 0x80000001,
                 0x7f800000, 0xff800000, 0x7fc00000, 0xffc00001,
                 0x7f800001, 0xff800001, 0x00800000, 0x80800000,
                 0x7f7fffff, 0xff7fffff, 0x3f800000, 0xbf800000], dtype=np.uint32)
x = bits.view(np.float32)
expected = (bits.astype(np.uint64) % (2**31)) != 0
np.testing.assert_array_equal(torch.from_numpy(x).to(torch.bool).numpy(), expected)
for leading in [(2, 3), (1, 2, 3), (0, 3)]:
    x = (np.arange(np.prod(leading) * 4, dtype=np.float32) % 5).reshape(*leading, 4)
    weight = np.arange(20, dtype=np.float32).reshape(5, 4) % 3
    expected = np.einsum("...k,nk->...n", x, weight)
    actual = torch.nn.functional.linear(torch.from_numpy(x), torch.from_numpy(weight))
    assert tuple(actual.shape) == (*leading, 5)
    np.testing.assert_array_equal(actual.numpy(), expected)
"""
    result = subprocess.run([interpreter, "-c", program], cwd=repo_root(), text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
