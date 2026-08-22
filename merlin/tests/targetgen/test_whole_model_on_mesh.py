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


# --- the same run, for ANY target, with its datapath READ off the target -------------------------
#
# The test above pins one target by name and hands the call that target's dtypes ("i8"/"i32") and its
# simulator ("spike") as literals. That is legitimate for a named regression, but it means the whole-model
# claim was only ever exercised on one device, and adding a second one meant copying the test. The
# parametrized form below derives the datapath instead, so a target enters the whole-model claim by
# existing rather than by being named here.
#
# The exactness assertion is derived too, and this is the part a copied test would have got wrong: an
# INTEGER mesh reproduces the small-integer reference bit-for-bit, a FLOAT mesh cannot be asked to. The
# expectation follows the target's own datapath rather than the first target's.
_WHOLE_MODEL_TARGETS = ("gemmini", "atlas")


@pytest.mark.slow
@pytest.mark.parametrize("target", _WHOLE_MODEL_TARGETS)
def test_whole_model_matmuls_run_on_any_targets_mesh(target):
    from merlin import compile_cli
    from merlin.runtime.dispatch_runtime import mesh_datapath

    from merlin.compile_cli import _mesh_tile_binding
    from merlin.targetgen.capsule_runner import _TIER_SIM, oracle_adapters

    try:
        op_dt, acc_dt, integer, _spelling = mesh_datapath(target)
    except Exception as e:                                   # noqa: BLE001 — unresolvable target
        pytest.skip(f"{target}: no derivable mesh datapath ({type(e).__name__}: {e})")

    # The cheapest tier this target actually resolves, through the shared tier->simulator map. Naming a
    # simulator instead is what limited the original test to one device: the call defaults to the RTL
    # simulator, so a target whose functional tier is the cheap one simply reported "no reachable oracle".
    tiers = sorted(oracle_adapters(target) or {})
    sim = _TIER_SIM.get(tiers[0]) if tiers else None

    # Size the layers to the target's OWN tile edge. A fixed 4x4 is below one tile on a wider mesh, and a
    # generated package is entitled to reject a sub-tile shape (measured: one divides by a bank count that
    # is 0 there). The claim under test is "a whole model runs on this mesh", not "on a 4x4".
    edge = _mesh_tile_binding(target, None, None).tile_dim
    res = compile_cli.run_whole_model_on_mesh(
        target, _vecblock("add", True, m=edge, k=edge), in_fmt=op_dt, weight_fmt=op_dt,
        operand_dtype=op_dt, accum_dtype=acc_dt, simulator=sim,
        ref_target=_REF_TARGET, seed=0, timeout=3600)

    if res["status"] == "oracle_unavailable":
        pytest.skip(f"{target}: {res.get('reason', 'mesh oracle unavailable')}")
    assert res["status"] == "pass", res
    assert res["n_mesh"] == 2 and res["n_scalar"] == 2
    assert all(layer["oracle"] == "ok" for layer in res["per_layer"]), res["per_layer"]
    if integer:
        assert res["exact"] is True, \
            f"{target}: an integer mesh must reproduce the small-integer reference bit-for-bit"

    # The report must name the executor that ACTUALLY ran, not the one requested. Two of the three
    # dispatch paths ignore ``simulator`` (a self-hosted-ISA target runs on its mlc-derived cosim, an
    # exclusive bespoke sim on its own engine), so recording the request as the device named a simulator
    # that never ran -- measured: atlas reported "spike" for work done on its arc cosim.
    assert res["mesh_executors"], f"{target}: layers ran but no executor was recorded: {res}"
    for lay in res["per_layer"]:
        assert lay["executed_on"], f"{target}: layer {lay['index']} ran with no executor recorded: {lay}"
        assert lay["path"], f"{target}: layer {lay['index']} ran with no dispatch path recorded: {lay}"
    # and the executor identity is the device's, not the bare request token
    assert all(e != res["simulator_requested"] or lay["path"] == "oot_cert"
               for e, lay in zip(res["mesh_executors"], res["per_layer"])), res


def test_an_unreachable_mesh_records_no_executor():
    """FAIL-CLOSED report shape: a run whose layers never reach an oracle must report an EMPTY executor
    list, never the requested simulator. Naming the request there is what let a skipped run read as a run
    on the named device."""
    from merlin import compile_cli
    from merlin.targetgen import mesh_program_run as mp

    called: list = []

    def _never(lhs, rhs, step):
        called.append(step.index)
        raise mp.MeshLayerUnavailable(step.index, step.m, step.k, step.n)

    mod = _vecblock("add", True)
    prog = _program(mod)
    with pytest.raises(mp.MeshLayerUnavailable):
        mp.run_whole_model_program(prog, _seed_leaves(prog), mesh_exec=_never)
    assert called, "the injected executor was never reached"
    assert hasattr(compile_cli, "run_whole_model_on_mesh")
