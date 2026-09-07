"""Never enter native code with a bundle ABI that differs from its prepared module."""
import sys

import numpy as np
import pytest

from merlin.common.paths import merlin_dir

sys.path.insert(0, str(merlin_dir() / "experiments/gemmini_perf_bench/scripts"))
from host_codegen_ab import validate_buffers


def test_valid_buffers():
    validate_buffers([([2, 3], "f32"), ([3], "i8")],
                     [np.zeros((2, 3), dtype=np.float32), np.zeros(3, dtype=np.int8)])


@pytest.mark.parametrize("signature,arrays", [
    ([([2, 3], "f32")], []),
    ([], [np.zeros((2, 3), dtype=np.float32)]),
    ([([3, 2], "f32")], [np.zeros((2, 3), dtype=np.float32)]),
    ([([2, 3], "i32")], [np.zeros((2, 3), dtype=np.float32)]),
    ([([2, 3], "unknown")], [np.zeros((2, 3), dtype=np.float32)]),
    ([([3, 2], "f32")], [np.zeros((2, 3), dtype=np.float32).T]),
])
def test_mismatched_buffers_refuse(signature, arrays):
    with pytest.raises(ValueError):
        validate_buffers(signature, arrays)
