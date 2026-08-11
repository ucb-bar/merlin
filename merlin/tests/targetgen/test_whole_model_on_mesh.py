"""Whole-model run co-scheduled across the mesh + scalar lanes with the mesh lane on the REAL oracle.

The engine splice (``test_whole_model_splice``) proves the ORCHESTRATION on the engine. These tests prove
the next slice: each mesh matmul LAYER executes on the target's real oracle (operands injected, kernel
emitted by the generated OOT package, output read back), the scalar lane runs inline, activations hand off
between lanes, and the whole-model final is gated against the target-agnostic engine reference.

Two levels:
- fast plumbing (no oracle): drive ``run_whole_model_program`` with an INJECTED mesh executor that returns
  the exact matmul, so the co-scheduling (lane dispatch + activation handoff) and the fail-closed path are
  proven without a toolchain;
- slow on-hardware: ``run_whole_model_on_mesh`` runs the matmul layers on the real gemmini oracle (spike),
  bit-exact vs the engine reference (small-integer operands make the integer mesh reproduce it exactly).
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

# Same synthetic f32 mesh + vector contract as the engine splice test: target-agnostic lane routing.
_F32_UNITS = {
    "compute_units": [
        {"name": "mesh", "kind": "systolic", "dtypes": ["f32", "fp32"], "ops": ["matmul"],
         "accumulate": [{"in": "f32", "weight": "f32", "acc": "f32"}]},
        {"name": "vec", "kind": "vector", "dtypes": ["f32", "fp32"],
         "ops": ["relu", "add", "mul", "elementwise"], "accumulate": []},
    ]
}
_REF_TARGET = "toy_npu"


def _units():
    from merlin.targetgen import compute_units as cu
    return cu.compute_units(_F32_UNITS)


def _vecblock(combine="add", relu=True, m=4, k=4):
    from merlin.xdsl_dialects.lowering.input_workload import build_vector_block
    return build_vector_block(m=m, k=k, elem="f32", combine=combine, relu=relu)


def _program(mod):
    from merlin.targetgen import mesh_program_run as mp
    from merlin.targetgen import routing as rt
    demands = mp.demands_from_module(mod, "f32")
    plan = rt.route_plan_on(demands, _units())
    return mp.build_whole_model_program(plan, _REF_TARGET, mod)


def _seed_leaves(prog, seed=0):
    rng = np.random.default_rng(seed)
    return {lid: rng.standard_normal(tuple(meta["shape"])).astype(np.float32)
            for lid, meta in prog.leaves.items()}


# --------------------------------------------------------------------------- fast plumbing


def test_injected_mesh_executor_threads_layer_outputs_between_lanes():
    """With a mesh executor injected (standing in for the real oracle), the two matmuls dispatch through it
    and their outputs hand off to the scalar relu/add — final == an independent numpy recompute."""
    from merlin.targetgen import mesh_program_run as mp

    prog = _program(_vecblock("add", True))
    leaves = _seed_leaves(prog)

    calls: list = []

    def fake_oracle(lhs, rhs, step):
        calls.append(step.index)                 # record which steps used the "oracle"
        return np.asarray(lhs) @ np.asarray(rhs)  # an exact-matmul stand-in for the device

    run = mp.run_whole_model_program(prog, leaves, mesh_exec=fake_oracle)

    A, W1, W2 = leaves["L0"], leaves["L1"], leaves["L2"]
    expected = np.maximum(A @ W1, 0.0) + (A @ W2)
    assert np.allclose(run["outputs"][prog.output], expected, rtol=1e-5, atol=1e-5)
    assert calls == [0, 2]                        # only the two matmul steps hit the mesh executor


def test_mesh_executor_none_fails_closed():
    """A mesh executor that cannot run a layer (returns None) raises MeshLayerUnavailable — the whole run
    fails closed rather than fabricating the layer's output."""
    from merlin.targetgen import mesh_program_run as mp

    prog = _program(_vecblock("add", True))
    leaves = _seed_leaves(prog)

    with pytest.raises(mp.MeshLayerUnavailable):
        mp.run_whole_model_program(prog, leaves, mesh_exec=lambda lhs, rhs, step: None)


def test_default_mesh_lane_still_runs_on_engine():
    """Omitting mesh_exec keeps the engine path (backward compatible with the pure-orchestration proof)."""
    from merlin.targetgen import mesh_program_run as mp

    r = mp.verify_whole_model_program(_vecblock("add", True), target=_REF_TARGET, in_fmt="f32",
                                      units=_units())
    assert r["exact"] is True and r["n_mesh"] == 2


# --------------------------------------------------------------------------- on real hardware (spike)


def test_int8_chain_reference_is_deterministic_and_saturating():
    """The host int8-chain reference (i32 matmul, round-half-even acc_scale, i8 saturate) is deterministic
    and stays in i8 range — a no-oracle sanity check on the golden the on-mesh chain is gated against."""
    from merlin import compile_cli

    rng = np.random.default_rng(0)
    A0 = np.rint(rng.standard_normal((4, 4)) * 4).clip(-8, 7).astype(int).tolist()
    Ws = [np.rint(rng.standard_normal((4, 4)) * 4).clip(-8, 7).astype(int).tolist() for _ in range(3)]
    r1 = compile_cli._int8_chain_reference(A0, Ws, 0.25)
    r2 = compile_cli._int8_chain_reference(A0, Ws, 0.25)
    assert np.array_equal(r1, r2)
    assert r1.min() >= -128 and r1.max() <= 127        # requant kept every layer in i8 range


@pytest.mark.slow
def test_int8_chain_on_gemmini_mesh_bit_exact():
    """A 3-layer int8 matmul CHAIN runs end-to-end on the real gemmini mesh with the per-layer acc_scale
    requant handoff — each layer's i8 output feeds the next mesh layer — and matches the host int8-chain
    reference bit-exact at EVERY layer. This is the inter-layer int8 handoff a real quantized model needs
    (a single independent matmul does not exercise it). Skips honestly if the mesh oracle is unavailable."""
    from merlin import compile_cli

    rng = np.random.default_rng(7)
    A0 = np.rint(rng.standard_normal((4, 4)) * 4).clip(-8, 7).astype(int).tolist()
    Ws = [np.rint(rng.standard_normal((4, 4)) * 4).clip(-8, 7).astype(int).tolist() for _ in range(3)]

    res = compile_cli.run_int8_chain_on_mesh("gemmini", A0, Ws, acc_scale=0.25,
                                             operand_dtype="i8", accum_dtype="i32",
                                             simulator="spike", timeout=900)
    if res["status"] == "oracle_unavailable":
        pytest.skip(res.get("reason", "mesh oracle unavailable"))
    assert res["status"] == "pass", res
    assert res["exact"] is True and res["n_layers"] == 3
    assert all(layer["matches_ref"] for layer in res["per_layer"])


@pytest.mark.slow
def test_whole_model_matmuls_on_gemmini_mesh_bit_exact():
    """The whole model — two matmuls on the real gemmini mesh + a scalar relu and add inline — runs
    co-scheduled on spike and reproduces the engine reference bit-exact (small-integer operands keep the
    int8 mesh == f32 reference). Skips honestly if the mesh oracle is unavailable in this env."""
    from merlin import compile_cli

    res = compile_cli.run_whole_model_on_mesh(
        "gemmini", _vecblock("add", True), in_fmt="int8", weight_fmt="int8",
        operand_dtype="i8", accum_dtype="i32", simulator="spike", ref_target=_REF_TARGET,
        seed=0, timeout=900)

    if res["status"] == "oracle_unavailable":
        pytest.skip(res.get("reason", "mesh oracle unavailable"))
    assert res["status"] == "pass", res
    assert res["exact"] is True                       # int mesh reproduces the f32 reference bit-for-bit
    assert res["n_mesh"] == 2 and res["n_scalar"] == 2
    assert all(layer["oracle"] == "ok" for layer in res["per_layer"])
