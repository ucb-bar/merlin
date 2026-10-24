"""Source-native prefix/suffix mask and position ownership contracts."""
import ctypes
import subprocess

import numpy as np
import pytest

from merlin.common.paths import repo_root

pytestmark = pytest.mark.target("atlas")


@pytest.fixture(scope = "module")
def controls(tmp_path_factory):
    source = repo_root() / "examples/atlas/comparisons/smol_host_controls.c"
    library = tmp_path_factory.mktemp("controls") / "controls.so"
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    "-fPIC", "-shared", str(source), "-o", str(library)],
                   check = True, capture_output = True, text = True)
    owner = ctypes.CDLL(str(library))
    prefix = owner.smol_host_prefix_controls
    prefix.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    suffix = owner.smol_host_suffix_controls
    suffix.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    return owner, prefix, suffix


def pointer(array):
    return array.ctypes.data_as(ctypes.c_void_p)


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


@pytest.mark.parametrize("rows", [0, 2 ** 32, 2 ** 63])
def test_prefix_invalid_extents_refused_before_read(controls, rows):
    _, prefix, _ = controls
    data = np.ones(1, np.uint8)
    group, mask, position = outputs(1, 1)
    assert prefix(pointer(data), pointer(data), rows, pointer(group), pointer(mask), pointer(position)) == -1
    assert group[0] == position[0] == 123 and mask[0, 0] == 7


@pytest.mark.parametrize("prefix_rows,suffix_rows", [(1, 0), (2 ** 64 - 1, 1), (2 ** 63, 1), (1, 2 ** 63), (2 ** 32, 2 ** 32)])
def test_suffix_invalid_extents_refused_before_read(controls, prefix_rows, suffix_rows):
    _, _, suffix = controls
    data = np.ones(1, np.uint8)
    group, mask, position = outputs(1, 1)
    assert suffix(pointer(data), prefix_rows, pointer(data), pointer(data), suffix_rows, pointer(group), pointer(mask), pointer(
        position)) == -1
    assert group[0] == position[0] == 123 and mask[0, 0] == 7


@pytest.mark.parametrize("invalid_input", [0, 1])
def test_prefix_all_boolean_inputs_checked_before_writes(controls, invalid_input):
    _, prefix, _ = controls
    inputs = [np.ones(3, np.uint8), np.ones(3, np.uint8)]
    inputs[invalid_input][-1] = 2
    group, mask, position = outputs(3, 3)
    assert prefix(pointer(inputs[0]), pointer(inputs[1]), 3, pointer(group), pointer(mask), pointer(position)) == -1
    assert np.all(group == 123) and np.all(position == 123) and np.all(mask == 7)


@pytest.mark.parametrize("invalid_input", [0, 1, 2])
def test_suffix_all_boolean_inputs_checked_before_writes(controls, invalid_input):
    _, _, suffix = controls
    inputs = [np.ones(3, np.uint8) for _ in range(3)]
    inputs[invalid_input][-1] = 2
    group, mask, position = outputs(3, 6)
    assert suffix(pointer(inputs[0]), 3, pointer(inputs[1]), pointer(inputs[2]), 3, pointer(group), pointer(mask), pointer(
        position)) == -1
    assert np.all(group == 123) and np.all(position == 123) and np.all(mask == 7)


@pytest.mark.parametrize("missing", [0, 1, 3, 4, 5])
def test_prefix_null_pointer_refused(controls, missing):
    _, prefix, _ = controls
    data = np.ones(1, np.uint8)
    group, mask, position = outputs(1, 1)
    arguments = [pointer(data), pointer(data), 1, pointer(group), pointer(mask), pointer(position)]
    arguments[missing] = None
    assert prefix(*arguments) == -1
    assert group[0] == position[0] == 123 and mask[0, 0] == 7


@pytest.mark.parametrize("missing", [0, 2, 3, 5, 6, 7])
def test_suffix_null_pointer_refused(controls, missing):
    _, _, suffix = controls
    data = np.ones(1, np.uint8)
    group, mask, position = outputs(1, 2)
    arguments = [pointer(data), 1, pointer(data), pointer(data), 1, pointer(group), pointer(mask), pointer(position)]
    arguments[missing] = None
    assert suffix(*arguments) == -1
    assert group[0] == position[0] == 123 and np.all(mask == 7)
