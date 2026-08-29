"""Tiling a mesh layer must change which extents RUN, never what they compute.

A backend that emits DxD tiles refuses a layer whose on-chip working set (K*N + M*K) is too large. The
execution path answers by splitting the layer — along N when N is splittable, along M otherwise — and
both splits are algebraically exact: C[:, a:b] is A @ W[:, a:b], and C[a:b, :] is A[a:b, :] @ W.

These tests stand in a fake backend with a declared working-set ceiling, so the arithmetic is checked
against numpy without a simulator. That matters: the real path costs a cycle-accurate run per tile, so
without this the exactness of the split was only ever observed indirectly, through a whole-model grade.

The extents the backend SEES are rounded up to the mesh tile edge first: a generated package is entitled
to reject a sub-tile extent, and a real model is full of them, so the execution path pads to the tile edge
and slices the result back. The ceilings below are therefore stated in terms of the PADDED working set --
they are the arithmetic of the shape the backend is actually offered, not of the caller's raw extent.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin import compile_cli as CC


@pytest.fixture(autouse=True)
def _clean_caches():
    CC._MESH_NTILE_WIDTH.clear()
    yield
    CC._MESH_NTILE_WIDTH.clear()


def _fake_backend(monkeypatch, *, ceiling: int, seen: list | None = None):
    """A backend that computes A@W exactly, but refuses any extent whose working set exceeds `ceiling`."""
    # `layer_id` rides along so each layer gets its own artifact directory; the double accepts and
    # ignores it, the same as any other caller-side detail it does not model.
    def _run(target, mlir, A, W, *, package, timeout, observed=None, layer_id=None):
        M, K, N = len(A), len(A[0]), len(W[0])
        if seen is not None:
            seen.append((M, K, N))
        if K * N + M * K > ceiling:
            return None
        return (np.asarray(A, dtype=np.float64) @ np.asarray(W, dtype=np.float64)).tolist()

    # `run_matmul_on_mesh` imports these from capsule_runner / corpus_spec INSIDE the function, so they
    # must be patched at their source module — patching the names on compile_cli would miss entirely.
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen import corpus_spec as CS

    monkeypatch.setattr(CC, "_matmul_via_bespoke_sim", _run)
    monkeypatch.setattr(CR, "_bespoke_sim_via", lambda t: "fake")
    monkeypatch.setattr(CR, "_endpoint_of", lambda t: ("external_backend", None))
    monkeypatch.setitem(CR._SIM_ORACLES, "fake",
                        type("SO", (), {"exclusive": True, "adapters": staticmethod(lambda t: {}),
                                        "available": staticmethod(lambda t: (True, ""))})())
    # The double must carry `cap_dtype` too: the real binding is a corpus_spec.CorpusBinding, and the
    # per-layer artifact id content-addresses on the CANONICAL dtype spelling. Delegating to the same
    # dtype_info the real binding uses keeps the double honest -- a stubbed constant here would let the
    # id collapse two different dtypes onto one directory, which is the collision that id exists to stop.
    from merlin.targetgen.corpus_spec import dtype_info as _dtype_info
    monkeypatch.setattr(CC, "_mesh_tile_binding",
                        lambda t, o, a, **k: type("B", (), {
                            "tile_dim": 16, "operand_dtype": "fp32", "accum_dtype": "f32",
                            "integer": False,
                            "cap_dtype": staticmethod(lambda token: _dtype_info(token)[0]),
                        })())
    monkeypatch.setattr(CS, "build", lambda entry, binding: (None, "<mlir>"))


def _check(monkeypatch, M, K, N, ceiling, seen=None):
    _fake_backend(monkeypatch, ceiling=ceiling, seen=seen)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K))
    W = rng.standard_normal((K, N))
    out = CC.run_matmul_on_mesh("t", A.tolist(), W.tolist(), operand_dtype="fp32", accum_dtype="f32")
    assert out is not None, "the layer should have been carried by tiling"
    got = np.asarray(out, dtype=np.float64)
    assert got.shape == (M, N)
    np.testing.assert_allclose(got, A @ W, rtol=1e-12, atol=1e-12)
    return got


def test_a_layer_that_fits_runs_untiled(monkeypatch):
    seen: list = []
    _check(monkeypatch, 8, 128, 128, ceiling=10**9, seen=seen)
    # One attempt, nothing split. M is rounded 8 -> 16 by the tile-edge padding before the backend sees
    # it; K and N are already tile-aligned.
    assert seen == [(16, 128, 128)], "nothing to split — one attempt at the tile-aligned extent"


def test_an_oversized_layer_is_split_along_n_and_stays_exact(monkeypatch):
    """K*N dominates, so narrowing N is what shrinks the working set."""
    seen: list = []
    _check(monkeypatch, 8, 128, 256, ceiling=8 * 128 + 128 * 64, seen=seen)
    widths = {n for _, _, n in seen}
    assert widths != {256}, "the raw extent alone cannot have carried it"
    assert all(n <= 256 for _, _, n in seen)


def test_a_layer_with_no_room_along_n_is_split_along_m(monkeypatch):
    """N is already below the tile dim, so only the M*K term can shrink — the 32x2048x2 case."""
    seen: list = []
    # N pads 2 -> 16 (the tile edge), so K*N is fixed at 2048*16 and only the M*K term can move: the
    # ceiling admits a 16-row block and refuses the full 32.
    _check(monkeypatch, 32, 2048, 2, ceiling=2048 * 16 + 16 * 2048, seen=seen)
    heights = {m for m, _, _ in seen}
    assert heights != {32}, "M must have been split; N was never splittable"


def test_the_result_is_assembled_in_the_right_order(monkeypatch):
    """A transposed or mis-ordered reassembly still has the right shape and would pass a shape check —
    so compare against the true product with a strict tolerance, on a non-symmetric matrix."""
    # N must exceed the tile dim for N-splitting to be exercised at all, and the ceiling must admit a
    # NARROWER N while refusing the whole layer. At the padded extent (M 2 -> 16, K 4 -> 16) the whole
    # layer is 16*64 + 16*16 = 1280 and a width-32 tile is 16*32 + 16*16 = 768.
    _fake_backend(monkeypatch, ceiling=16 * 32 + 16 * 16)
    A = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    W = [[float(r * 10 + c) for c in range(64)] for r in range(4)]
    out = CC.run_matmul_on_mesh("t", A, W, operand_dtype="fp32", accum_dtype="f32")
    np.testing.assert_allclose(np.asarray(out), np.asarray(A) @ np.asarray(W), rtol=1e-12)


def test_a_backend_that_refuses_everything_still_fails_closed(monkeypatch):
    """Tiling must not turn an unrunnable layer into a fabricated answer."""
    _fake_backend(monkeypatch, ceiling=0)
    A = [[1.0, 2.0], [3.0, 4.0]]
    W = [[1.0, 0.0], [0.0, 1.0]]
    assert CC.run_matmul_on_mesh("t", A, W, operand_dtype="fp32", accum_dtype="f32") is None
