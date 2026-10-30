"""GQA head ownership, finite masks and checked portable host attention buffers."""
import ctypes

import numpy as np
import pytest

from smol_host_helpers import load_host_library, pointer, words

pytestmark = pytest.mark.target("atlas")


@pytest.fixture(scope="module")
def attention(tmp_path_factory):
    owner = load_host_library(tmp_path_factory.mktemp("attention"), "attention",
                              "smol_host_attention.c", "smol_host_softmax.c")
    function = owner.smol_host_attention_cpu16_bf16
    function.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_size_t] * 5 + [ctypes.c_void_p] * 4
    return owner, function


def buffers(rows=2, keys=2, heads=4, kv=2, dimension=2):
    return [np.zeros((rows, heads, dimension), np.uint16),
            np.zeros((keys, kv, dimension), np.uint16),
            np.zeros((keys, kv, dimension), np.uint16),
            np.ones((rows, keys), np.uint8),
            np.full(keys, 123, np.float32),
            np.full((heads, rows, keys), 123, np.float32),
            np.full((heads, rows, keys), 0x1234, np.uint16),
            np.full((rows, heads, dimension), 0x1234, np.uint16)]


def invoke(function, arrays, extents=(2, 2, 4, 2, 2)):
    return function(*[pointer(a) for a in arrays[:4]], *extents,
                    *[pointer(a) for a in arrays[4:]])


def test_grouped_heads_and_fully_masked_row_remains_uniform(attention):
    _, function = attention
    arrays = buffers()
    arrays[2][:] = words([[[1, 2], [8, 16]], [[3, 4], [16, 32]]])
    arrays[3][:] = [[1, 0], [0, 0]]
    before = [a.copy() for a in arrays[:4]]
    assert invoke(function, arrays) == 0
    expected = words([[[1, 2], [1, 2], [8, 16], [8, 16]],
                      [[2, 3], [2, 3], [12, 24], [12, 24]]])
    assert np.array_equal(arrays[-1], expected)
    assert np.all(arrays[-2][:, 0] == [0x3F80, 0])
    assert np.all(arrays[-2][:, 1] == [0x3F00, 0x3F00])
    assert all(np.array_equal(a, b) for a, b in zip(arrays[:4], before))


def test_fixed_native_emulation_pair_order_rounding_witness(attention):
    _, function = attention
    arrays = buffers(1, 4, 1, 1, 1)
    arrays[2][:] = words([[[2**26]], [[4]], [[-2**26]], [[4]]])
    assert invoke(function, arrays, (1, 4, 1, 1, 1)) == 0
    # Uniform probabilities are 1/4. Upper-before-lower pair accumulation
    # produces +0; left-to-right accumulation would produce 1.
    assert arrays[-1].item() == 0


@pytest.mark.parametrize("index,value", [(4, 0), (6, 3), (4, 2**63), (0, None)])
def test_invalid_null_or_overflowing_arguments_refused_before_write(attention, index, value):
    _, function = attention
    arrays = buffers()
    before = [a.copy() for a in arrays[4:]]
    arguments = [*[pointer(a) for a in arrays[:4]], 2, 2, 4, 2, 2,
                 *[pointer(a) for a in arrays[4:]]]
    arguments[index] = value
    assert function(*arguments) == -1
    assert all(np.array_equal(a, b) for a, b in zip(arrays[4:], before))


@pytest.mark.parametrize("source,invalid", [(0, 0x7F80), (1, 0x7FC0), (3, 2)])
def test_all_input_values_validated_before_first_scratch_or_output_write(attention, source, invalid):
    _, function = attention
    arrays = buffers()
    arrays[source].flat[-1] = invalid
    before = [a.copy() for a in arrays[4:]]
    assert invoke(function, arrays) == -1
    assert all(np.array_equal(a,b) for a,b in zip(arrays[4:],before))


def test_overflowing_QK_refused_before_probability_or_output_write(attention):
    _, function = attention
    arrays = buffers()
    arrays[0][:] = 0x7F7F
    arrays[1][:] = 0x7F7F
    assert invoke(function, arrays) == -1
    assert np.all(arrays[-1] == 0x1234) and np.all(arrays[-2] == 0x1234)
