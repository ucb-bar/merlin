"""Source-native prefix/suffix mask and position ownership contracts."""
import ctypes

import numpy as np
import pytest

from smol_host_helpers import load_host_library, pointer

pytestmark = pytest.mark.target("atlas")


@pytest.fixture(scope = "module")
def controls(tmp_path_factory):
    owner = load_host_library(tmp_path_factory.mktemp("controls"), "controls",
                              "smol_host_controls.c", strict_fp=False, math=False)
    prefix = owner.smol_host_prefix_controls
    prefix.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    suffix = owner.smol_host_suffix_controls
    suffix.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    return owner, prefix, suffix


def outputs(rows, keys):
    return np.full(rows, 123, np.int64), np.full((rows, keys), 7, np.uint8), np.full(rows, 123, np.int64)


def test_prefix_padding_group_boundaries_and_negative_position(controls):
    _, prefix, _ = controls
    pad, attention = np.array([0, 1, 1, 0], np.uint8), np.array([0, 0, 1, 0], np.uint8)
    before = pad.copy(), attention.copy()
    groups, mask, positions = outputs(4, 4)
    assert prefix(pointer(pad), pointer(attention), 4, pointer(groups), pointer(mask), pointer(positions)) == 0
    assert np.array_equal(groups, [0, 0, 1, 1])
    assert np.array_equal(positions, [-1, 0, 1, 1])
    assert np.array_equal(mask, [[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
    assert np.array_equal(pad, before[0]) and np.array_equal(attention, before[1])


def test_suffix_cached_prefix_ownership_for_padded_queries(controls):
    _, _, suffix = controls
    prefix_pad = np.array([1, 0, 1], np.uint8)
    pad, attention = np.array([0, 1, 1], np.uint8), np.array([1, 0, 1], np.uint8)
    groups, mask, positions = outputs(3, 6)
    assert suffix(pointer(prefix_pad), 3, pointer(pad), pointer(attention), 3,
                  pointer(groups), pointer(mask), pointer(positions)) == 0
    assert np.array_equal(positions, [1, 2, 3])
    assert np.array_equal(mask, [[1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1]])


def test_zero_length_prefix_accepts_null_prefix_storage(controls):
    _, _, suffix = controls
    pad, attention = np.array([0, 1, 1], np.uint8), np.ones(3, np.uint8)
    groups, mask, positions = outputs(3, 3)
    assert suffix(None, 0, pointer(pad), pointer(attention), 3, pointer(groups), pointer(mask), pointer(positions)) == 0
    assert np.array_equal(positions, [-1, 0, 1])
    assert np.array_equal(mask, [[0, 0, 0], [0, 1, 0], [0, 1, 1]])


def test_suffix_BF16_group_integer_domain_refused_before_writes(controls):
    _, _, suffix = controls
    data = np.ones(257, np.uint8)
    group, mask, position = outputs(257, 257)
    assert suffix(None, 0, pointer(data), pointer(data), 257, pointer(group), pointer(mask), pointer(position)) == -1
    assert np.all(group == 123) and np.all(position == 123) and np.all(mask == 7)


@pytest.mark.parametrize("rows,invalid", [(0, None), (2**32, None), (2**63, None), (3, "boolean"), (3, "null")])
def test_prefix_invalid_arguments_refused_before_writes(controls, rows, invalid):
    _, prefix, _ = controls
    pad, attention = np.ones(3, np.uint8), np.ones(3, np.uint8)
    group, mask, position = outputs(3, 3)
    if invalid == "boolean":
        attention[-1] = 2
    source = None if invalid == "null" else pointer(pad)
    assert prefix(source, pointer(attention), rows, pointer(group), pointer(mask), pointer(position)) == -1
    assert np.all(group == 123) and np.all(position == 123) and np.all(mask == 7)


@pytest.mark.parametrize("prefix_rows,suffix_rows,invalid", [
    (1, 0, None), (2**64 - 1, 1, None), (2**63, 1, None), (2**32, 2**32, None),
    (3, 3, "boolean"), (3, 3, "null"),
])
def test_suffix_invalid_arguments_refused_before_writes(controls, prefix_rows, suffix_rows, invalid):
    _, _, suffix = controls
    pad, attention = np.ones(3, np.uint8), np.ones(3, np.uint8)
    group, mask, position = outputs(3, 6)
    if invalid == "boolean":
        attention[-1] = 2
    source = None if invalid == "null" else pointer(pad)
    assert suffix(source, prefix_rows, pointer(pad), pointer(attention), suffix_rows,
                  pointer(group), pointer(mask), pointer(position)) == -1
    assert np.all(group == 123) and np.all(position == 123) and np.all(mask == 7)
