"""Broadened kernel-ceiling workloads (vbinary / reduce / clamp / transpose) + the work-weighted
model op census. Hermetic: generation + golden correctness + structured MLIR checks + the census
weighting math on a synthetic module. No board, no m2m toolchain.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from merlin.common import mlir_query as mq
from merlin.common.paths import repo_root
from merlin.rvvgen import workloads


def _golden(bundle):
    return np.load(bundle / "golden.npy")


def _inputs(bundle):
    return dict(np.load(bundle / "inputs.npz"))


@pytest.mark.parametrize("op,np_op", [("mul", np.multiply), ("add", np.add), ("sub", np.subtract)])
def test_gen_binary_golden_and_ir(tmp_path, op, np_op):
    b = workloads.gen_binary_f32(tmp_path, op=op, N=1024)
    ins, g = _inputs(b), _golden(b)
    np.testing.assert_allclose(g, np_op(ins["in0"], ins["in1"]), rtol=1e-5, atol=1e-5)
    m = mq.parse(b / "model.mlir")
    # exactly one elementwise generic (all-parallel, no reduction)
    gens = [o for o in m.walk() if mq.op_name(o) == "linalg.generic"]
    assert len(gens) == 1
    assert "reduction" not in str(gens[0].properties.get("iterator_types", ""))


@pytest.mark.parametrize("op,np_red", [("sum", lambda x: x.sum(1)),
                                       ("max", lambda x: x.max(1)),
                                       ("min", lambda x: x.min(1))])
def test_gen_reduce_golden_and_ir(tmp_path, op, np_red):
    b = workloads.gen_reduce_f32(tmp_path, op=op, M=8, N=64)
    ins, g = _inputs(b), _golden(b)
    np.testing.assert_allclose(g, np_red(ins["in0"]), rtol=1e-4, atol=1e-4)
    assert g.shape == (8,)
    m = mq.parse(b / "model.mlir")
    gens = [o for o in m.walk() if mq.op_name(o) == "linalg.generic"]
    assert len(gens) == 1
    # the reduction axis must actually be present (else it degenerates to a copy)
    assert "reduction" in str(gens[0].properties.get("iterator_types", ""))


def test_gen_relu_clamps_both_ends(tmp_path):
    b = workloads.gen_relu_f32(tmp_path, N=4096, lo=0.0, hi=6.0)
    ins, g = _inputs(b), _golden(b)
    np.testing.assert_allclose(g, np.minimum(np.maximum(ins["in0"], 0.0), 6.0), atol=1e-6)
    assert g.min() >= 0.0 and g.max() <= 6.0
    # a real input distribution (~N(0,3)) must exercise BOTH clamp ends, else the shape is degenerate
    assert (ins["in0"] < 0.0).any() and (ins["in0"] > 6.0).any()


def test_gen_transpose_nonsquare_moves_data(tmp_path):
    b = workloads.gen_transpose_f32(tmp_path, R=16, C=8)
    ins, g = _inputs(b), _golden(b)
    assert g.shape == (8, 16)                      # (C,R)
    np.testing.assert_array_equal(g, ins["in0"].T)
    m = mq.parse(b / "model.mlir")
    gen = next(o for o in m.walk() if mq.op_name(o) == "linalg.generic")
    # the input map must be a permutation (d1,d0), not identity, or it is not a transpose
    maps = gen.properties["indexing_maps"].data
    assert str(maps[0].data) == "(d0, d1) -> (d1, d0)"


def _load_census():
    path = repo_root() / "build_tools" / "scripts" / "model_op_census.py"
    spec = importlib.util.spec_from_file_location("model_op_census", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_census_weights_matmul_by_full_iteration_space(tmp_path):
    """A matmul's work must include its K reduction dim (M*N*K*2), not just the M*N output — the
    whole point of weighting by iteration space rather than op/output count."""
    census = _load_census()
    b = workloads.gen_matmul_f32(tmp_path, M=8, N=4, K=16)
    fams = census.census_bundle(b / "model.mlir")
    assert "matmul" in fams
    # named linalg.matmul: iters = M*N*K, body = 2 (mul+add) -> 8*4*16*2 = 1024
    assert fams["matmul"]["work"] == 8 * 4 * 16 * 2


def test_census_reduce_counts_reduction_axis(tmp_path):
    """A row reduction's work is M*N (every element visited), not M (the output). The synthetic
    workload carries no prov.op, so an unstamped generic falls back to the `generic` family — which
    also exercises that fallback path."""
    census = _load_census()
    b = workloads.gen_reduce_f32(tmp_path, op="sum", M=8, N=64)
    fams = census.census_bundle(b / "model.mlir")
    assert fams["generic"]["work"] == 8 * 64            # M*N visited, 1-op body
    assert fams["generic"]["linalg_ops"] == {"linalg.generic": 1}
