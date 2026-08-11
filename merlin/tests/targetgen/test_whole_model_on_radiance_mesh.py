"""Whole-model compile + co-schedule for the ``radiance`` SIMT target, gated against an engine reference.

This is the radiance analog of ``test_whole_model_on_mesh`` (which is gemmini-specific). It proves the two
halves of the claim "we can compile whole models for radiance" that DO NOT need the paid experiment or a
live oracle, so they can gate a radiance run the moment it finishes:

- ROUTING (fast): a whole model's ops route across radiance's REAL contract compute units — matmuls onto
  the SIMT mesh (``simt_cluster``), norms/activations/elementwise onto the scalar/vector lane;
- SPLICE PLUMBING (fast): the co-scheduled whole-model program built from that plan threads each layer's
  output into the op that consumes it, dispatching mesh matmuls through an INJECTED executor (standing in
  for the cyclotron oracle) and running the scalar lane inline — final == an independent numpy recompute,
  and a mesh executor that cannot run a layer fails CLOSED (never fabricates the layer's output).

The third half — executing the mesh matmul layers on the REAL cyclotron oracle with the per-layer
activation handoff, gated bit-exact — is covered by the ``@pytest.mark.slow`` test below, which SKIPS
honestly until ``run_matmul_on_mesh`` grows a cyclotron dispatch (see the module docstring of
``merlin.compile_cli`` / the accompanying implementation spec). It is target-name-legitimate to name
radiance here: this is a test that is ABOUT radiance, not shared library code.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

_TARGET = "radiance"
_REF_TARGET = "toy_npu"


def _units():
    from merlin.targetgen import compute_units as cu
    from merlin.targetgen import target_registry as tr

    return cu.compute_units(tr.load_contract(_TARGET))


def _mesh_dtype_token(units):
    """The fp32-family operand token radiance's SIMT mesh actually DECLARES (e.g. ``fp32``), read from the
    contract rather than assumed — the router matches on the contract's own token, so the caller must feed
    that token (``f32`` and ``fp32`` are distinct tokens to the router; this derives whichever the mesh
    uses)."""
    for u in units:
        if u.kind == "simt":
            for d in u.dtypes:
                if d.startswith("f") and "32" in d:
                    return d
    raise AssertionError("radiance contract declares no fp32-family SIMT mesh dtype")


def _vecblock():
    from merlin.xdsl_dialects.lowering.input_workload import build_vector_block

    return build_vector_block(m=4, k=4, elem="f32", combine="add", relu=True)


def _program(mod, units):
    from merlin.targetgen import mesh_program_run as mp
    from merlin.targetgen import routing as rt

    demands = mp.demands_from_module(mod, _mesh_dtype_token(units))
    plan = rt.route_plan_on(demands, units)
    return plan, mp.build_whole_model_program(plan, _TARGET, mod)


# --------------------------------------------------------------------------- routing (contract)


def test_radiance_contract_routes_matmuls_to_simt_mesh():
    """The radiance contract has a SIMT mesh unit that accepts fp32 matmuls, and the router places a vector
    block's two matmuls on that mesh while relu/add fall to the scalar/vector lane."""
    units = _units()
    assert any(u.kind == "simt" for u in units), "radiance contract has no SIMT mesh unit"

    plan, prog = _program(_vecblock(), units)
    assert len(plan["mesh"]) == 2, plan            # both A@W1 and A@W2 route onto the mesh
    assert all(r.demand.op == "matmul" for r in plan["mesh"])
    # relu + add are not mesh ops -> scalar/vector lane (honest, expected for a matmul mesh).
    scalar_ops = {r.demand.op for r in plan["scalar_rvv"]} | {r.demand.op for r in plan["fallback"]}
    assert scalar_ops == {"relu", "add"}, scalar_ops
    assert [(s.family, s.lane) for s in prog.steps] == [
        ("matmul", "mesh"), ("relu", "scalar"), ("matmul", "mesh"), ("add", "scalar")]


# --------------------------------------------------------------------------- splice plumbing (no oracle)


def test_radiance_splice_threads_layer_outputs_with_injected_mesh_executor():
    """With a mesh executor injected (standing in for the cyclotron oracle), the two radiance-routed
    matmuls dispatch through it and hand their outputs to the scalar relu/add — the whole-model final ==
    an independent numpy recompute. Proves the co-scheduling + activation handoff for a RADIANCE plan."""
    from merlin.targetgen import mesh_program_run as mp

    units = _units()
    _, prog = _program(_vecblock(), units)
    rng = np.random.default_rng(0)
    leaves = {lid: rng.standard_normal(tuple(meta["shape"])).astype(np.float32)
              for lid, meta in prog.leaves.items()}

    calls: list = []

    def fake_oracle(lhs, rhs, step):
        calls.append(step.index)
        return np.asarray(lhs) @ np.asarray(rhs)   # exact-matmul stand-in for the device

    run = mp.run_whole_model_program(prog, leaves, mesh_exec=fake_oracle)
    A, W1, W2 = leaves["L0"], leaves["L1"], leaves["L2"]
    expected = np.maximum(A @ W1, 0.0) + (A @ W2)
    assert np.allclose(run["outputs"][prog.output], expected, rtol=1e-5, atol=1e-5)
    assert calls == [0, 2]                          # only the two matmul steps hit the mesh executor


def test_radiance_splice_fails_closed_when_mesh_layer_unavailable():
    """A mesh executor that cannot run a radiance matmul layer (returns None) raises MeshLayerUnavailable —
    the run fails closed rather than fabricating the layer's output. This is the fail-closed contract a
    real cyclotron dispatch must honor when the oracle is absent."""
    from merlin.targetgen import mesh_program_run as mp

    _, prog = _program(_vecblock(), _units())
    leaves = {lid: np.zeros(tuple(meta["shape"]), dtype=np.float32)
              for lid, meta in prog.leaves.items()}
    with pytest.raises(mp.MeshLayerUnavailable):
        mp.run_whole_model_program(prog, leaves, mesh_exec=lambda lhs, rhs, step: None)


# --------------------------------------------------------------------------- on real hardware (cyclotron)


@pytest.mark.slow
def test_whole_model_matmuls_on_radiance_mesh_bit_exact():
    """The whole model — two matmuls on the real radiance cyclotron mesh + a scalar relu and add inline —
    runs co-scheduled and reproduces the engine reference bit-exact (small-integer operands keep the mesh
    == f32 reference).

    ``run_matmul_on_mesh`` now has a cyclotron dispatch for radiance: the exclusive bespoke-sim executor
    emits the matmul kernel from radiance's generated OOT package, INJECTS the real per-layer operands onto
    the command buffer, runs the emitted kernel on the real cyclotron oracle, and reads the named output
    back. This is a LIVE bit-exact gate. It skips honestly ONLY when the environment genuinely lacks the
    prerequisite (no radiance OOT package materialized, or the cyclotron oracle / MERLIN_MUON_* toolchain
    is absent) — never to paper over a real failure."""
    from merlin import compile_cli
    from merlin.targetgen import capsule_runner as CR

    if compile_cli._default_oot_package(_TARGET) is None:
        pytest.skip(f"no OOT package materialized for {_TARGET!r} — cannot emit the mesh kernel")
    so = CR._SIM_ORACLES.get(CR._bespoke_sim_via(_TARGET))
    if so is None or not so.exclusive:
        pytest.skip(f"{_TARGET!r} declares no exclusive bespoke-sim oracle")
    ok, reason = so.available(_TARGET)
    if not ok:
        pytest.skip(f"cyclotron oracle unavailable for {_TARGET!r}: {reason}")

    units = _units()
    tok = _mesh_dtype_token(units)
    res = compile_cli.run_whole_model_on_mesh(
        _TARGET, _vecblock(), in_fmt=tok, weight_fmt=tok,
        operand_dtype=tok, accum_dtype=tok, ref_target=_REF_TARGET, seed=0, timeout=900)

    if res.get("status") == "oracle_unavailable":
        pytest.skip(res.get("reason", "radiance mesh oracle unavailable at run time"))
    assert res["status"] == "pass", res
    assert res["exact"] is True
    assert res["n_mesh"] == 2 and res["n_scalar"] == 2
    assert all(layer["oracle"] == "ok" for layer in res["per_layer"])
