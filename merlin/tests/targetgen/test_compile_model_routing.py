"""Mixed-dialect whole-model routing: each op is split across a target's compute units — matmul/systolic
tiles execute on the mesh, norms/activations/elementwise fall to the vector/scalar (RVV) lane. The split is
derived structurally from the captured model linalg (prov.op/prov.family, no regex) and is an honest,
data-driven decision (an op no unit supports is a scalar/RVV fallback, never a silent drop).

Target-agnostic: the target is a parameter; this edge names one as data under test."""
from __future__ import annotations

import pytest

from merlin.targetgen import capsule_source as CSrc
from merlin.targetgen import routing as R

_LINALG = (
    "builtin.module {\n"
    "  func.func @forward(%0: tensor<16x16xf32>, %1: tensor<16x16xf32>) -> tensor<16x16xf32> {\n"
    '    %f = linalg.fill {prov.op = "fill", prov.family = "fill"} ...\n'
    '    %2 = linalg.matmul {prov.op = "matmul", prov.family = "contraction"} ... -> tensor<16x16xf32>\n'
    '    %3 = linalg.generic {prov.op = "softmax", prov.family = "normalization"} ... -> tensor<16x16xf32>\n'
    "    return %3 : tensor<16x16xf32>\n  }\n}\n")


def test_model_op_demands_structural():
    """Contraction ops carry a weight format; normalization/elementwise are unary; fill is skipped."""
    dem = CSrc.model_op_demands(_LINALG, "int8")
    by = {d.op: d for d in dem}
    assert "fill" not in by                                   # init op, not routable
    assert by["matmul"].weight_fmt == "int8"                  # contraction -> weighted
    assert by["softmax"].weight_fmt is None                   # normalization -> unary


def _gemmini_available():
    try:
        from merlin.targetgen import target_registry as tr
        tr.load_contract("gemmini")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _gemmini_available(), reason="gemmini contract not resolvable in this env")
def test_route_plan_splits_mesh_vs_scalar():
    """matmul -> the systolic mesh; softmax (no accelerator unit) -> the scalar/RVV lane, honestly."""
    dem = [R.OpDemand("matmul", "int8", "int8", "mm"), R.OpDemand("softmax", "int8", None, "sm")]
    plan = R.route_plan(dem, "gemmini")
    assert [r.demand.op for r in plan["mesh"]] == ["matmul"]
    assert [r.demand.op for r in plan["scalar_rvv"]] == ["softmax"]


def test_summarize_route_plan_shape():
    from merlin.compile_cli import _summarize_route_plan
    plan = {"mesh": [R.RouteResult(R.OpDemand("matmul", "int8", "int8"), "systolic_mesh", None, None)],
            "fallback": [],
            "scalar_rvv": [R.RouteResult(R.OpDemand("softmax", "int8", None), None, None, "gap")]}
    s = _summarize_route_plan(plan)
    assert s["on_mesh"] == {"matmul": 1} and s["scalar_rvv_lane"] == {"softmax": 1}
    assert s["n_mesh_ops"] == 1 and s["n_scalar_ops"] == 1
