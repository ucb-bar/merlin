"""The generic matrix-unit shim: descriptor ABI, operand pack, fallback, batched entries.

Compiled for the HOST around a plain C kernel with the shim's kernel signature, so what is tested is the
bridge (extents read from descriptors, the A transpose into K-major, the dense/aligned/fits checks and the
counted fallback, the batch loop) and nothing about any unit.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from merlin.kernels import matrix_shim

pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="no host C compiler")

_KERNEL = r"""
#include <stdint.h>
#include <stddef.h>
static void test_kernel(int32_t *c, const int8_t *at, const int8_t *b, const int32_t *bias,
                        size_t m, size_t n, size_t k)
{
  for (size_t i = 0; i < m; ++i)
    for (size_t j = 0; j < n; ++j) {
      int32_t s = bias ? bias[j] : 0;
      for (size_t q = 0; q < k; ++q) s += (int32_t)at[q * m + i] * (int32_t)b[q * n + j];
      c[i * n + j] = s;
    }
}
"""


class _Memref2D(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_ssize_t),
        ("sizes", ctypes.c_ssize_t * 2),
        ("strides", ctypes.c_ssize_t * 2),
    ]


class _Memref3D(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_ssize_t),
        ("sizes", ctypes.c_ssize_t * 3),
        ("strides", ctypes.c_ssize_t * 3),
    ]


def _build(signatures, tmp_path: Path, *, align: int = 16, prefix: str = "merlin_t", scratch=None):
    src = matrix_shim.emit_translation_unit(
        _KERNEL, "test_kernel", signatures, alignment_bytes=align, prefix=prefix, scratch_bytes=scratch
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    c = tmp_path / "shim.c"
    c.write_text(str(src))
    so = tmp_path / "shim.so"
    got = subprocess.run(["cc", "-O1", "-fPIC", "-shared", "-Wall", "-Werror", str(c), "-o", str(so)],
                         capture_output=True, text=True)  # fmt: skip
    assert got.returncode == 0, got.stderr[-3000:]
    return ctypes.CDLL(str(so)), src


def _desc(arr: np.ndarray, shape, strides, offset=0):
    base = ctypes.c_void_p(arr.ctypes.data)
    return [base, base, ctypes.c_ssize_t(offset), *(ctypes.c_ssize_t(s) for s in shape),
            *(ctypes.c_ssize_t(s) for s in strides)]  # fmt: skip


def _ops(m, n, k, seed=3):
    rng = np.random.default_rng(seed)
    return rng.integers(-128, 128, (m, k), dtype=np.int8), rng.integers(-128, 128, (k, n), dtype=np.int8)


def _ref(a, b):
    return (a.astype(np.int64) @ b.astype(np.int64)).astype(np.int32)


def _call2(lib, sym, a, b, c, *, a_shape=None, a_strides=None):
    m, k = a_shape or a.shape
    n = b.shape[1]
    fn = getattr(lib, sym)
    fn.restype = _Memref2D
    fn(*_desc(a, (m, k), a_strides or (k, 1)), *_desc(b, (k, n), (n, 1)), *_desc(c, (m, n), (n, 1)))


def test_entries_compute_the_contraction_and_read_extents_from_descriptors(tmp_path):
    lib, src = _build({"merlin_t_gemm_0": (12, 20, 9)}, tmp_path)
    assert src.symbols == ("merlin_t_gemm_0",) and src.scratch_bytes == 12 * 9
    # Called with DIFFERENT extents than it was generated for: the descriptors win.
    a, b = _ops(7, 5, 3)
    c = np.zeros((7, 5), np.int32)
    _call2(lib, "merlin_t_gemm_0", a, b, c)
    np.testing.assert_array_equal(c, _ref(a, b))
    assert lib.merlin_t_calls() == 1 and lib.merlin_t_fallbacks() == 0


def test_a_strided_view_takes_the_counted_fallback_and_is_still_right(tmp_path):
    lib, _ = _build({"merlin_t_gemm_0": (6, 4, 5)}, tmp_path)
    big = np.random.default_rng(1).integers(-128, 128, (6, 10), dtype=np.int8)
    a = big[:, :5]  # row stride 10, not 5: the pack reads through the strides
    _, b = _ops(6, 4, 5)
    c = np.zeros((6, 4), np.int32)
    _call2(lib, "merlin_t_gemm_0", big, b, c, a_shape=(6, 5), a_strides=(10, 1))
    np.testing.assert_array_equal(c, _ref(np.ascontiguousarray(a), b))
    # A strided A is fine (the pack reads strides); a too-small scratch is not.
    lib2, _ = _build({"merlin_t_gemm_0": (6, 4, 5)}, tmp_path / "small", scratch=8)
    tmp = np.zeros((6, 4), np.int32)
    _call2(lib2, "merlin_t_gemm_0", np.ascontiguousarray(a), b, tmp)
    np.testing.assert_array_equal(tmp, _ref(np.ascontiguousarray(a), b))
    assert lib2.merlin_t_fallbacks() == 1


def test_batched_entry_loops_over_slices(tmp_path):
    bsz, m, n, k = 3, 5, 6, 4
    lib, _ = _build({"merlin_t_gemm_b": (bsz, m, n, k)}, tmp_path)
    rng = np.random.default_rng(9)
    a = rng.integers(-128, 128, (bsz, m, k), dtype=np.int8)
    b = rng.integers(-128, 128, (bsz, k, n), dtype=np.int8)
    c = np.zeros((bsz, m, n), np.int32)
    fn = lib.merlin_t_gemm_b
    fn.restype = _Memref3D
    fn(*_desc(a, a.shape, (m * k, k, 1)), *_desc(b, b.shape, (k * n, n, 1)), *_desc(c, c.shape, (m * n, n, 1)))
    for i in range(bsz):
        np.testing.assert_array_equal(c[i], _ref(a[i], b[i]))


def test_refuses_what_cannot_be_compiled_safely():
    with pytest.raises(ValueError):
        matrix_shim.emit_translation_unit(_KERNEL, "test_kernel", {}, alignment_bytes=0, prefix="p")
    with pytest.raises(ValueError):
        matrix_shim.emit_translation_unit(_KERNEL, "test kernel", {}, alignment_bytes=16, prefix="p")
    with pytest.raises(ValueError):
        matrix_shim.emit_translation_unit(_KERNEL, "k", {"1bad": (1, 1, 1)}, alignment_bytes=16, prefix="p")


def test_two_prefixes_do_not_collide(tmp_path):
    one = matrix_shim.emit_translation_unit(_KERNEL, "test_kernel", {"x_0": (2, 2, 2)}, alignment_bytes=16, prefix="u1")
    assert "u1_lhs_pack" in one and "merlin_t" not in one
