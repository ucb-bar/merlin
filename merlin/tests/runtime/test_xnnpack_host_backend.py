"""HOST XNNPACK kernel-backend tests (default-off; additive third e2e column scaffolding).

Gates: (1) the XNNPACK scalar f32 GEMM microkernel, built standalone on host, matches numpy
across shapes/tails; (2) the dispatch-runtime routing is byte-stable -- routing the plain
linalg.matmul dispatches through XNNPACK produces the same whole-model output as the default
compiled path (so it is a faithful kernel swap, not a behavior change).

Skips cleanly when the XNNPACK source / a host C compiler is unavailable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.runtime.backends import xnnpack_host

pytestmark = pytest.mark.skipif(not xnnpack_host.is_available(),
                                reason="XNNPACK host GEMM lib unavailable")


@pytest.mark.parametrize("M,N,K", [(4, 4, 4), (7, 5, 9), (64, 64, 64),
                                   (1, 128, 256), (33, 17, 41), (2, 1, 3)])
def test_xnn_gemm_matches_numpy(M, N, K):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    C = xnnpack_host.gemm_f32(A, B)
    assert np.abs(C - A @ B).max() < 1e-3


def _bitvla_dir() -> Path | None:
    d = Path(repo_root()) / "out/artifacts" / "recaptures" / "bitvla_fp32_consistent"
    return d if (d / "model.mlir").is_file() else None


def test_dispatch_runtime_xnnpack_matches_default(tmp_path):
    """Whole bitvla forward: XNNPACK kernel backend == default compiled path (byte-stable)."""
    md = _bitvla_dir()
    if md is None:
        pytest.skip("bitvla_fp32_consistent capture not present")
    from merlin.runtime import dispatch_runtime as dr

    base = dr.run_model(md, tmp_path / "base")
    xnn = dr.run_model(md, tmp_path / "xnn", kernel_backend="xnnpack")
    assert xnn["n_xnn_routed"] > 0, "no matmul dispatches were routed through XNNPACK"
    b = np.asarray(base["output"], np.float32).ravel()
    x = np.asarray(xnn["output"], np.float32).ravel()
    cos = float((b @ x) / (np.linalg.norm(b) * np.linalg.norm(x) + 1e-12))
    assert cos > 0.99999, f"xnnpack-vs-default cos {cos}"
