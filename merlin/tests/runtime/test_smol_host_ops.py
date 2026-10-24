"""Portable C host primitive contracts and arithmetic rounding boundaries."""
import ctypes
from pathlib import Path
import subprocess

import numpy as np
import pytest

from merlin.common.paths import repo_root

pytestmark = pytest.mark.target("atlas")


@pytest.fixture(scope="module")
def host_ops(tmp_path_factory):
    source = repo_root() / "examples/atlas/comparisons"
    output = tmp_path_factory.mktemp("smol-host") / "host.so"
    command = ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
               "-ffp-contract=off", "-fno-fast-math", "-fPIC", "-shared",
               str(source / "smol_host_ops.c"), "-lm", "-o", str(output)]
    subprocess.run(command, capture_output=True, text=True, check=True)
    library = ctypes.CDLL(str(output))
    library.smol_host_euler_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float,
                                          ctypes.c_void_p, ctypes.c_size_t]
    for name in ("smol_host_residual_bf16", "smol_host_residual_bf16_f32", "smol_host_silu_mul_bf16"):
        getattr(library, name).argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t]
    for name in ("smol_host_rmsnorm_bf16", "smol_host_rmsnorm_f32_to_bf16"):
        getattr(library, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float,
                                         ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    return library


def pointer(array):
    return array.ctypes.data_as(ctypes.c_void_p)


def test_bounded_configuration_omits_only_mixed_residual(tmp_path, host_ops):
    source = repo_root() / "examples/atlas/comparisons/smol_host_ops.c"
    output = tmp_path / "bounded.so"
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                    "-ffp-contract=off", "-fno-fast-math", "-fPIC", "-shared",
                    "-DSMOL_HOST_EXCLUDE_MIXED_RESIDUAL", str(source), "-lm", "-o", str(output)],
                   capture_output=True, text=True, check=True)
    bounded = ctypes.CDLL(str(output))
    with pytest.raises(AttributeError):
        bounded.smol_host_residual_bf16_f32
    values = np.array([0x3f80, 0xbf80, 0x8000], np.uint16)
    right = np.array([0x3b80, 0x3f80, 0x8000], np.uint16)
    result, reference = np.empty_like(values), np.empty_like(values)
    bounded.smol_host_residual_bf16.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t]
    assert bounded.smol_host_residual_bf16(pointer(values), pointer(right), pointer(result), 3) == 0
    assert host_ops.smol_host_residual_bf16(pointer(values), pointer(right), pointer(reference), 3) == 0
    assert result.tolist() == reference.tolist() == [0x3f80, 0, 0x8000]


def test_euler_keeps_separate_FP32_rounding_and_allows_output_alias(host_ops):
    state = np.array([-1.0], np.float32)
    velocity = np.array([1.0 + 2.0**-23], np.float32)
    dt = ctypes.c_float(1.0 - 2.0**-23)
    assert host_ops.smol_host_euler_f32(pointer(state), pointer(velocity), dt, pointer(state), 1) == 0
    # A fused multiply-add would retain the negative 2**-46 product error.
    assert state.view(np.uint32)[0] == 0


def test_residual_RNE_ties_and_exact_cancellation(host_ops):
    left = np.array([0x3f80, 0x3f80, 0x3f80, 0xbf80, 0x8000], np.uint16)
    right = np.array([0x3b80, 0x3c40, 0xbf80, 0x3f80, 0], np.uint16)
    expected = np.array([0x3f80, 0x3f82, 0, 0, 0], np.uint16)
    actual = np.empty_like(expected)
    assert host_ops.smol_host_residual_bf16(pointer(left), pointer(right), pointer(actual), actual.size) == 0
    assert np.array_equal(actual, expected)


def test_silu_multiply_preserves_source_signed_zero(host_ops):
    gate = np.array([0, 0x8000, 0], np.uint16)
    up = np.array([0x3f80, 0x3f80, 0xbf80], np.uint16)
    output = np.empty_like(gate)
    assert host_ops.smol_host_silu_mul_bf16(pointer(gate), pointer(up), pointer(output), output.size) == 0
    assert np.array_equal(output, np.array([0, 0x8000, 0x8000], np.uint16))


def test_mixed_residual_keeps_FP32_operand_until_final_BF16_cast(host_ops):
    left = np.array([0x3f80], np.uint16)
    right = np.array([2.0**-8 + 2.0**-22], np.float32)
    output = np.empty_like(left)
    assert host_ops.smol_host_residual_bf16_f32(pointer(left), pointer(right), pointer(output), 1) == 0
    # Casting the right operand to BF16 first would produce a tie rounded to 1.
    assert output[0] == 0x3f81


@pytest.mark.parametrize("width", [1, 3, 5, 7, 8, 9, 31, 32, 33, 63, 64, 65, 127, 128, 129, 720, 960, 2048, 2560, 10000])
@pytest.mark.parametrize("bf16", [False, True])
def test_norm_widths_and_tails_keep_zero_unit_and_weight_rounding(host_ops, width, bf16):
    weight = np.resize(np.array([0x3f80, 0x4000, 0x3f00], np.uint16), width)
    if bf16:
        values = np.full((2, width), 0x3f80, np.uint16)
        values[0] = 0
        function = host_ops.smol_host_rmsnorm_bf16
    else:
        values = np.ones((2, width), np.float32)
        values[0] = 0
        function = host_ops.smol_host_rmsnorm_f32_to_bf16
    output = np.full((2, width), 0xffff, np.uint16)
    assert function(pointer(values), pointer(weight), ctypes.c_float(1e-5), pointer(output), 2, width) == 0
    assert not output[0].any()
    assert np.array_equal(output[1], weight)


def test_invalid_norm_extent_or_epsilon_refused_without_output_write(host_ops):
    values = np.ones((2, 2), np.float32)
    weight = np.full(2, 0x3f80, np.uint16)
    output = np.full((2, 2), 0x1234, np.uint16)
    function = host_ops.smol_host_rmsnorm_f32_to_bf16
    for rows, width, epsilon in ((0, 2, 1e-5), (2, 0, 1e-5), (2, 2, 0.0), (2, 2, float("nan"))):
        assert function(pointer(values), pointer(weight), ctypes.c_float(epsilon), pointer(output), rows, width) == -1
        assert np.all(output == 0x1234)
    assert function(None, pointer(weight), ctypes.c_float(1e-5), pointer(output), 2, 2) == -1
