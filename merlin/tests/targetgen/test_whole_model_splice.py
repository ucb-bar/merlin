"""Whole-model image splice: one co-scheduled program that walks a routing plan and dispatches each op
to its lane (matmuls -> the accelerator MESH, norms/activations/elementwise -> the scalar/RVV lane),
wiring every op's output tensor into the next op's input.

These tests prove the splice ORCHESTRATION on the engine — ordering, activation handoff between steps,
and mesh/scalar co-scheduling — by running a FULLY-MODELED model (a matmul chain and a vector block:
``combine(relu(A@W1), A@W2)``) through the spliced per-op path and asserting the final tensor equals the
single-module ``lower_module`` result NUMERICALLY EXACT (both paths run matmuls on the same engine). The
single-binary arena runtime that fuses these kernels into one image is a later slice.

Target-agnostic: the lane of each op is READ from the routing plan (whether the router placed the op on a
mesh unit), never assumed from an op name. The routing units here are a synthetic f32 mesh + vector
contract passed as data, so the test binds to no specific target.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

# A synthetic, target-agnostic contract: an f32 systolic mesh (matmuls land here) + an f32 vector lane
# (relu/add/mul land here). Passed as data so the splice is proven without depending on any one target's
# dtype table.
_F32_UNITS = {
    "compute_units": [
        {"name": "mesh", "kind": "systolic", "dtypes": ["f32", "fp32"], "ops": ["matmul"],
         "accumulate": [{"in": "f32", "weight": "f32", "acc": "f32"}]},
        {"name": "vec", "kind": "vector", "dtypes": ["f32", "fp32"],
         "ops": ["relu", "add", "mul", "elementwise"], "accumulate": []},
    ]
}

# The engine reference lowers through a real in-tree reference target (used only as the whole-module
# compiler edge — the routing/lane decisions come entirely from _F32_UNITS above).
_REF_TARGET = "toy_npu"


def _units():
    from merlin.targetgen import compute_units as cu
    return cu.compute_units(_F32_UNITS)


def _chain(dims):
    from merlin.xdsl_dialects.lowering.input_workload import build_matmul_chain
    return build_matmul_chain(dims=dims, elem="f32")


def _vecblock(combine, relu):
    from merlin.xdsl_dialects.lowering.input_workload import build_vector_block
    return build_vector_block(m=8, k=16, elem="f32", combine=combine, relu=relu)


# --------------------------------------------------------------------------- the core proof


@pytest.mark.parametrize("dims", [(4, 5, 3), (8, 16, 12, 6), (4, 8, 6, 5, 3)])
def test_matmul_chain_splice_matches_single_module(dims):
    """A feed-forward matmul chain routes entirely onto the mesh; the spliced per-layer execution (each
    layer's output handed to the next as its LHS) reproduces the single-module engine result exactly."""
    from merlin.targetgen import mesh_program_run as mp

    r = mp.verify_whole_model_program(_chain(dims), target=_REF_TARGET, in_fmt="f32", units=_units())
    assert r["ref_correct"] is True
    assert r["exact"] is True                        # bit-for-bit == the single-module lower_module result
    n_layers = len(dims) - 2                          # dims = [m, k_1, ..., k_L] -> L-1 matmuls
    assert r["n_steps"] == n_layers and r["n_mesh"] == n_layers and r["n_scalar"] == 0


@pytest.mark.parametrize("combine", ["add", "mul"])
@pytest.mark.parametrize("relu", [True, False])
def test_vector_block_splice_matches_single_module(combine, relu):
    """A mixed model — two matmuls (mesh) plus a relu and an elementwise combine (scalar lane) — splices
    correctly: the mesh outputs hand off to the scalar ops and back, final == single-module, exact."""
    from merlin.targetgen import mesh_program_run as mp

    r = mp.verify_whole_model_program(_vecblock(combine, relu), target=_REF_TARGET, in_fmt="f32",
                                      units=_units())
    assert r["ref_correct"] is True
    assert r["exact"] is True
    assert r["n_mesh"] == 2                           # A@W1 and A@W2 on the mesh
    assert r["n_scalar"] == (2 if relu else 1)        # relu + combine (or just combine)


def test_splice_final_equals_manual_numpy():
    """Belt-and-suspenders: the spliced final also equals an independent numpy recomputation of the
    whole model, so the exactness is real correctness, not two matching bugs."""
    from merlin.targetgen import mesh_program_run as mp

    r = mp.verify_whole_model_program(_vecblock("add", True), target=_REF_TARGET, in_fmt="f32",
                                      units=_units())
    assert np.allclose(r["ref_final"], r["spliced_final"], rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------- program STRUCTURE


def test_program_lane_tagging_extents_and_handoff():
    """The built program tags each op's lane from the plan, carries the matmul extents, and threads the
    tensor ids so activations hand off between steps (t0 -> relu -> t1; {t1,t2} -> add -> t3)."""
    from merlin.targetgen import mesh_program_run as mp
    from merlin.targetgen import routing as rt

    mod = _vecblock("add", True)
    demands = mp.demands_from_module(mod, "f32")
    plan = rt.route_plan_on(demands, _units())
    prog = mp.build_whole_model_program(plan, _REF_TARGET, mod)

    fams = [(s.family, s.lane) for s in prog.steps]
    assert fams == [("matmul", "mesh"), ("relu", "scalar"), ("matmul", "mesh"), ("add", "scalar")]

    # tensor-id handoff: layer-1 matmul -> relu -> combine; layer-2 matmul -> combine.
    s0, s1, s2, s3 = prog.steps
    assert s0.inputs == ("L0", "L1") and s0.output == "t0"          # A @ W1
    assert s1.inputs == ("t0",) and s1.output == "t1"              # relu(A@W1)
    assert s2.inputs == ("L0", "L2") and s2.output == "t2"          # A @ W2 (reuses activation leaf)
    assert s3.inputs == ("t1", "t2") and s3.output == "t3"          # combine(...)
    assert prog.output == "t3"

    # real extents ride on the mesh matmul steps (8x16x16 for this vector block).
    assert (s0.m, s0.k, s0.n) == (8, 16, 16)
    assert (s2.m, s2.k, s2.n) == (8, 16, 16)

    # leaves carry roles: the shared activation L0 and the two resident weights L1/L2.
    assert prog.leaves["L0"]["role"] == "activation"
    assert prog.leaves["L1"]["role"] == "weight" and prog.leaves["L2"]["role"] == "weight"


def test_demands_from_module_are_ordered_and_carry_matmul_extents():
    """``demands_from_module`` reads the compute ops structurally, in program order, with real M/K/N on
    every contraction and no extents on the unary ops."""
    from merlin.targetgen import mesh_program_run as mp

    demands = mp.demands_from_module(_chain((4, 8, 6, 5, 3)), "f32")
    assert [d.op for d in demands] == ["matmul", "matmul", "matmul"]
    assert [(d.m, d.k, d.n) for d in demands] == [(4, 8, 6), (4, 6, 5), (4, 5, 3)]
    assert all(d.weight_fmt == "f32" for d in demands)


def test_plan_result_count_must_match_module():
    """A plan whose op count does not match the module's compute ops is rejected (the two walks must be
    aligned, else the lane/extent/handoff mapping would be silently wrong)."""
    from merlin.targetgen import mesh_program_run as mp
    from merlin.targetgen import routing as rt

    mod = _chain((4, 8, 6, 5))
    demands = mp.demands_from_module(mod, "f32")
    plan = rt.route_plan_on(demands[:-1], _units())   # drop one op -> misaligned
    with pytest.raises(ValueError, match="must be routed from this module"):
        mp.build_whole_model_program(plan, _REF_TARGET, mod)


def test_route_plan_on_matches_route_plan_split():
    """``route_plan_on`` (units given directly) produces the same lane split as ``route_plan`` would for
    the same units, and preserves op order in results."""
    from merlin.targetgen import routing as rt

    demands = [rt.OpDemand("matmul", "f32", "f32", m=4, k=4, n=4),
               rt.OpDemand("relu", "f32"),
               rt.OpDemand("add", "f32")]
    plan = rt.route_plan_on(demands, _units())
    assert [r.demand.op for r in plan["results"]] == ["matmul", "relu", "add"]
    assert len(plan["mesh"]) == 1 and plan["mesh"][0].demand.op == "matmul"
    # relu/add land on the in-contract vector unit (not the mesh).
    assert {r.demand.op for r in plan["fallback"]} == {"relu", "add"}
