"""Pressure-extraction parity: real model2MLIR region vs synthetic region of same shape.

Gated on xDSL being importable and a model2MLIR fp32 export being present (a sibling repo).
Validates that the design-pressure pass extracts the same per-invocation facts from real VLA
model MLIR as from the synthetic builder.
"""
import os

import pytest

from merlin.common import paths
from merlin.design_pressure import synthesize as S
from merlin.design_pressure.pressure_vector import compute_rpv
from merlin.design_pressure.workloads.vla_action_chunk_decode import build_region

from merlin.design_pressure.ingest import mlir_m2m

_CANDIDATES = [
    paths.repo_root().parent / "model2MLIR" / "workloads" / "openvla" / "openvla.mlir",
    paths.repo_root().parent / "model2MLIR" / "workloads" / "xr0" / "xr0.mlir",
]
_MLIR = next((str(p) for p in _CANDIDATES if p.is_file()), None)

pytestmark = pytest.mark.skipif(
    not mlir_m2m.available() or _MLIR is None,
    reason="xdsl not installed or no model2MLIR fp32 export present",
)


def test_extraction_parity_with_synthetic():
    region = mlir_m2m.region_from_mlir(_MLIR, H=8)
    rpv = compute_rpv(region)
    M, K, N = rpv["metrics"]["M"], rpv["metrics"]["K"], rpv["metrics"]["N"]
    assert M and K and N

    syn = compute_rpv(build_region(H=8, K=K, M=M, N=N, dtype="f32",
                                   epilogue=rpv["facts"]["has_epilogue"]))
    for key in ("op", "rhs_reuse_count", "rhs_mutable", "K", "has_epilogue"):
        assert rpv["facts"][key] == syn["facts"][key], key

    pol = S.load_policies()
    assert S.recommended_features(rpv, pol) == S.recommended_features(syn, pol)
