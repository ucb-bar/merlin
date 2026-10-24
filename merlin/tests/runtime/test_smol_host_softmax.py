"""Finite full-axis softmax contracts and refusal-before-write controls."""
import ctypes
import subprocess

import numpy as np
import pytest

from merlin.common.paths import repo_root

pytestmark = pytest.mark.target("atlas")


@pytest.fixture(scope="module")
def softmax(tmp_path_factory):
    source = repo_root() / "examples/atlas/comparisons/smol_host_softmax.c"
    directory = tmp_path_factory.mktemp("softmax")
    rounding = directory / "rounding.c"
    rounding.write_text("#include <fenv.h>\nint test_down(void) { return fesetround(FE_DOWNWARD); }\n"
                        "int test_nearest(void) { return fesetround(FE_TONEAREST); }\n")
    library = directory / "softmax.so"
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    "-ffp-contract=off", "-fno-fast-math", "-fPIC", "-shared", str(source),
                    str(rounding), "-lm", "-o", str(library)],
                   check=True, capture_output=True, text=True)
    owner = ctypes.CDLL(str(library))
    function = owner.smol_host_softmax_cpu16_bf16
    function.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                         ctypes.c_void_p, ctypes.c_void_p]
    return owner, function


def pointer(array):
    return array.ctypes.data_as(ctypes.c_void_p)


def test_full_axis_uniform_including_fully_masked_finite_row(softmax):
    _, function = softmax
    scores = np.array([[0] * 16, [-np.finfo(np.float32).max] * 16], np.float32)
    before = scores.copy()
    workspace = np.empty(16, np.float32)
    guarded = np.full((2 * 16 + 2,), 0x1234, np.uint16)
    assert function(pointer(scores), 2, 16, pointer(workspace), pointer(guarded[1:])) == 0
    assert np.all(guarded[1:-1] == 0x3D80)  # Exactly 1/16 in BF16.
    assert guarded[0] == 0x1234 and guarded[-1] == 0x1234
    assert np.array_equal(scores, before)


def test_small_axis_and_finite_subtraction_overflow(softmax):
    _, function = softmax
    largest = np.finfo(np.float32).max
    scores = np.array([[largest, -largest], [-104, 0], [7, 7]], np.float32)
    workspace = np.empty(2, np.float32)
    output = np.empty((3, 2), np.uint16)
    assert function(pointer(scores), 3, 2, pointer(workspace), pointer(output)) == 0
    assert np.array_equal(output, [[0x3F80, 0], [0, 0x3F80], [0x3F00, 0x3F00]])


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_all_scores_validated_before_any_output_or_workspace_write(softmax, invalid):
    _, function = softmax
    scores = np.array([[1, 2], [3, invalid]], np.float32)
    workspace = np.full(2, 123, np.float32)
    output = np.full((2, 2), 0x1234, np.uint16)
    assert function(pointer(scores), 2, 2, pointer(workspace), pointer(output)) == -1
    assert np.all(output == 0x1234) and np.all(workspace == 123)


@pytest.mark.parametrize("rows,keys", [(0, 1), (1, 0), (2**63, 1), (1, 2**63), (2**62, 2)])
def test_invalid_or_overflowing_extents_refused_before_read(softmax, rows, keys):
    _, function = softmax
    scores, workspace = np.ones(1, np.float32), np.full(1, 123, np.float32)
    output = np.full(1, 0x1234, np.uint16)
    assert function(pointer(scores), rows, keys, pointer(workspace), pointer(output)) == -1
    assert output[0] == 0x1234 and workspace[0] == 123


@pytest.mark.parametrize("missing", [0, 3, 4])
def test_null_buffer_refused(softmax, missing):
    _, function = softmax
    scores, workspace = np.ones(1, np.float32), np.full(1, 123, np.float32)
    output = np.full(1, 0x1234, np.uint16)
    arguments = [pointer(scores), 1, 1, pointer(workspace), pointer(output)]
    arguments[missing] = None
    assert function(*arguments) == -1
    assert output[0] == 0x1234 and workspace[0] == 123


def test_other_rounding_mode_refused_before_output_write(softmax):
    owner, function = softmax
    scores, workspace = np.ones(1, np.float32), np.full(1, 123, np.float32)
    output = np.full(1, 0x1234, np.uint16)
    try:
        assert owner.test_down() == 0
        assert function(pointer(scores), 1, 1, pointer(workspace), pointer(output)) == -1
    finally:
        assert owner.test_nearest() == 0
    assert output[0] == 0x1234 and workspace[0] == 123
