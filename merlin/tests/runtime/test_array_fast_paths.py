"""The numpy fast paths for stimulus fill and integer matmul give exactly the pure-Python values."""

from __future__ import annotations

import random

import pytest

from merlin.common import stimulus as S
from merlin.runtime import tensor as T

pytest.importorskip("numpy")


def _scalar_fill(name, shape, lo, hi):
    rows, cols = S.grid_shape(shape)
    salt = S.det_seed(name)
    return [lo + (S.mix(r, c, salt) % (hi - lo + 1)) for r in range(rows) for c in range(cols)]


def _loop_matmul(a, b, m, k, n):
    out = [0] * (m * n)
    for i in range(m):
        for p in range(k):
            for j in range(n):
                out[i * n + j] += a[i * k + p] * b[p * n + j]
    return out


@pytest.mark.parametrize("shape", [(128, 128), (3, 70000), (70000, 1), (2, 3, 4096), (16384,)])
@pytest.mark.parametrize("lo, hi", [(0, 3), (-3, 3), (-128, 127), (0, 1 << 20)])
def test_array_fill_matches_the_scalar_hash(shape, lo, hi):
    for name in ("W", "attn.q", "x" * 300):
        assert S.fill(name, shape, lo, hi) == _scalar_fill(name, shape, lo, hi)


@pytest.mark.parametrize(
    "m, k, n, lo, hi",
    [(32, 64, 48, -128, 127), (1, 4096, 9, -128, 127), (17, 333, 29, -(2**31), 2**31 - 1), (8, 8, 1024, -(2**40), 2**40)],
)
def test_matmul_is_exact_including_the_overflow_fallback(m, k, n, lo, hi):
    rng = random.Random(m * k * n)
    a = [rng.randint(lo, hi) for _ in range(m * k)]
    b = [rng.randint(lo, hi) for _ in range(k * n)]
    got = T.Tensor((m, k), a).matmul(T.Tensor((k, n), b)).data
    assert got == _loop_matmul(a, b, m, k, n)
    assert all(type(value) is int for value in got)
