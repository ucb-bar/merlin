"""Frozen outputs from the independent PyTorch 2.16 AVX2 CPU wheel."""

import numpy as np
import pytest
from torch_cpu_cascade_ref import avx2_f32_row_mean


@pytest.mark.parametrize("count", [32, 64, 256, 2048])
def test_frozen_torch_avx2_means(count):
    # Integer-based input generation is stable across NumPy versions and
    # exercises positive, signed, and cancellation-heavy float additions.
    indices = np.arange(3 * count, dtype=np.int64).reshape(3, count)
    positive = ((indices[0] * 1664525 + 1013904223) % 32749).astype(np.float32) / np.float32(8192)
    signed = ((indices[1] * 22695477 + 1) % 65521).astype(np.float32) / np.float32(4096) - np.float32(8)
    cancel = np.where(indices[2] % 2 == 0, np.float32(1e4), np.float32(-1e4))
    cancel += ((indices[2] * 1103515245 + 12345) % 97).astype(np.float32) / np.float32(64)
    rows = np.stack((positive, signed, cancel))
    # Filled by PyTorch's torch.from_numpy(rows).mean(dim=-1).numpy().view('u4').
    expected_words = {
        32: (1073924336, 3205633280, 1061068800),
        64: (1073815512, 3198128256, 1061339136),
        256: (1073752050, 1048071424, 1061072896),
        2048: (1073723866, 3155253376, 1061150720),
    }[count]
    np.testing.assert_array_equal(avx2_f32_row_mean(rows).view(np.uint32), expected_words)


def test_refuses_unqualified_source_arithmetic():
    with pytest.raises(ValueError, match="power-of-two"):
        avx2_f32_row_mean(np.ones((1, 33), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        avx2_f32_row_mean(np.full((1, 32), np.nan, dtype=np.float32))
