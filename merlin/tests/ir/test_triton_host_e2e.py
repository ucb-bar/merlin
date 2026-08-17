"""A Triton kernel compiled and EXECUTED, end to end, with the numbers checked.

Everything before this point could be true of a translator that produces plausible IR. These tests
run the result and compare it to an independent NumPy computation, which is the first evidence that
the bridge preserved meaning rather than shape.

The size sweep is the substance. Vector add is compiled at fifteen extents spanning "smaller than one
block", "exactly one block", "one past a block", and several tails, because the failure this whole
design is exposed to — a mask dropped during pointer re-raising — is invisible at every size where
the block divides the extent, and only those sizes get tested by hand.
"""
from __future__ import annotations

import numpy as np
import pytest
import triton_kernels as K

from merlin.triton import source
from merlin.triton.bridge import to_linalg

pytestmark = pytest.mark.skipif(not K.HAS_TRITON, reason="the `triton` optional extra is not installed")

SIZES = [1, 15, 16, 17, 63, 64, 65, 255, 256, 257, 1000, 1023, 1024, 1025, 4099]


def compile_and_load(spec, workdir):
    """Bridge the kernel and lower it to a host shared library, ready to call."""
    from merlin.llvmlower.kernel_backend import compile_host, extract_kernel

    result = to_linalg(source.make_ttir(spec), spec)
    kernel = extract_kernel(result.module, spec.name)
    return compile_host(kernel, workdir), result


def call(model, inputs: list[np.ndarray], outputs: list[np.ndarray]) -> None:
    model([(a.ctypes.data, a.shape) for a in inputs] + [(o.ctypes.data, o.shape) for o in outputs])


@pytest.mark.parametrize("n", SIZES)
def test_vector_add_matches_numpy_at_every_extent(n, tmp_path):
    """The masked tail is the point: n=1000 exercises a partial final program, n=1024 does not."""
    model, _ = compile_and_load(K.vector_add_spec(n=n), tmp_path)
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    out = np.full(n, np.nan, np.float32)
    call(model, [x, y], [out])
    assert np.array_equal(out, x + y), f"n={n}: max abs error {np.abs(out - (x + y)).max()}"


def test_the_int8_matmul_matches_an_integer_reference(tmp_path):
    """i8 x i8 -> i32 with no rounding anywhere, so the comparison is exact, not approximate."""
    model, _ = compile_and_load(K.matmul_one_tile_spec(), tmp_path)
    rng = np.random.default_rng(7)
    a = rng.integers(-8, 8, size=(K.TILE_M, K.TILE_K), dtype=np.int8)
    b = rng.integers(-8, 8, size=(K.TILE_K, K.TILE_N), dtype=np.int8)
    out = np.zeros((K.TILE_M, K.TILE_N), np.int32)
    call(model, [a, b], [out])
    assert np.array_equal(out, a.astype(np.int32) @ b.astype(np.int32))


def test_the_float_matmul_matches_within_float_tolerance(tmp_path):
    model, _ = compile_and_load(K.matmul_one_tile_spec(dtype="fp32", acc_dtype="fp32"), tmp_path)
    rng = np.random.default_rng(11)
    a = rng.standard_normal((K.TILE_M, K.TILE_K)).astype(np.float32)
    b = rng.standard_normal((K.TILE_K, K.TILE_N)).astype(np.float32)
    out = np.zeros((K.TILE_M, K.TILE_N), np.float32)
    call(model, [a, b], [out])
    # Only reassociation separates the two; the reference is computed in the same precision.
    assert np.allclose(out, a @ b, rtol=1e-5, atol=1e-5)


def test_a_kernel_reading_uninitialized_accumulator_would_be_caught(tmp_path):
    """The matmul's accumulator is zeroed by an explicit fill, so a second call repeats itself.

    linalg.matmul accumulates into its `outs` operand. Had the bridge passed a bare `tensor.empty`
    — which is what the repo's own synthetic workload does — the result would depend on whatever
    that buffer happened to hold, and calling twice could give two answers.
    """
    model, _ = compile_and_load(K.matmul_one_tile_spec(), tmp_path)
    rng = np.random.default_rng(3)
    a = rng.integers(-8, 8, size=(K.TILE_M, K.TILE_K), dtype=np.int8)
    b = rng.integers(-8, 8, size=(K.TILE_K, K.TILE_N), dtype=np.int8)
    first = np.zeros((K.TILE_M, K.TILE_N), np.int32)
    second = np.full((K.TILE_M, K.TILE_N), 12345, np.int32)
    call(model, [a, b], [first])
    call(model, [a, b], [second])
    assert np.array_equal(first, second)
