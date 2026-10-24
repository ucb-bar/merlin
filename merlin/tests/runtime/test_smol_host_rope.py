"""Source-native half-split rotary tables use runtime positions with checked bounds."""
import ctypes
import subprocess

import numpy as np
import pytest

from merlin.common.paths import repo_root

pytestmark = pytest.mark.target("atlas")


@pytest.fixture(scope="module")
def rope(tmp_path_factory):
    source = repo_root() / "examples/atlas/comparisons/smol_host_rope.c"
    library = tmp_path_factory.mktemp("rope") / "rope.so"
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    "-ffp-contract=off", "-fno-fast-math", "-fPIC", "-shared", str(source),
                    "-o", str(library)], check=True, capture_output=True, text=True)
    owner = ctypes.CDLL(str(library))
    function = owner.smol_host_rope_bf16
    function.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] + [ctypes.c_size_t] * 4 + [ctypes.c_void_p]
    return owner, function


def pointer(array):
    return array.ctypes.data_as(ctypes.c_void_p)


def words(array):
    return (np.asarray(array, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def test_runtime_positions_half_split_heads_and_negative_lower_bound(rope):
    _, function = rope
    values = words([[[1, 2, 3, 4], [5, 6, 7, 8]], [[1, 2, 3, 4], [5, 6, 7, 8]]])
    positions = np.array([-1, 0], np.int64)
    cosine = np.array([[1, 1], [0, 0]], np.float32)
    sine = np.array([[0, 0], [1, 1]], np.float32)
    cosine_before, sine_before = cosine.copy(), sine.copy()
    output = np.empty_like(values)
    assert function(pointer(values), pointer(positions), pointer(cosine), pointer(sine), -1, 2, 2, 2, 4, pointer(output)) == 0
    expected = words([[[1, 2, 3, 4], [5, 6, 7, 8]], [[-3, -4, 1, 2], [-7, -8, 5, 6]]])
    assert np.array_equal(output, expected)
    assert np.array_equal(cosine, cosine_before) and np.array_equal(sine, sine_before)


@pytest.mark.parametrize("invalid_position", [-2, 1])
def test_every_position_validated_before_first_output_write(rope, invalid_position):
    _, function = rope
    values = words(np.ones((2, 1, 4), np.float32))
    positions = np.array([-1, invalid_position], np.int64)
    cosine, sine = np.ones((2, 2), np.float32), np.zeros((2, 2), np.float32)
    output = np.full_like(values, 0x1234)
    assert function(pointer(values), pointer(positions), pointer(cosine), pointer(sine), -1, 2, 2, 1, 4, pointer(output)) == -1
    assert np.all(output == 0x1234)


def test_exact_input_output_alias_is_supported(rope):
    _, function = rope
    values = words([[[1, 2, 3, 4]]])
    positions = np.array([0], np.int64)
    cosine, sine = np.zeros((1, 2), np.float32), np.ones((1, 2), np.float32)
    assert function(pointer(values), pointer(positions), pointer(cosine), pointer(sine), 0, 1, 1, 1, 4, pointer(values)) == 0
    assert np.array_equal(values, words([[[-3, -4, 1, 2]]]))


@pytest.mark.parametrize("rows,heads,dimension", [(0, 1, 4), (1, 0, 4), (1, 1, 0), (1, 1, 3), (2**63, 1, 4)])
def test_invalid_or_overflowing_extents_refused(rope, rows, heads, dimension):
    _, function = rope
    values = words([[[1, 2, 3, 4]]])
    positions = np.array([0], np.int64)
    cosine, sine = np.ones((1, 2), np.float32), np.zeros((1, 2), np.float32)
    output = np.full_like(values, 0x1234)
    assert function(pointer(values), pointer(positions), pointer(cosine), pointer(sine), 0, 1, rows, heads, dimension, pointer(
        output)) == -1
    assert np.all(output == 0x1234)
