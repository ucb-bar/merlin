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


# --------------------------------------------------------------------------- on-mesh tile execution
# `compile_model(mesh_verify=True)` goes past the PLAN and actually EXECUTES each mesh matmul as a single
# systolic tile through the OOT `certify` accelerator path. These tests exercise the machinery — real tile
# synthesis (corpus_spec.build_matmul over the target's derived binding) + aggregation + fail-closed
# accounting — with the oracle stubbed, so they are deterministic without a spike/arc sim in the env.

def _two_matmul_plan():
    """A route plan with two mesh matmuls + one scalar/RVV op, via the real gemmini routing."""
    dem = [R.OpDemand("matmul", "int8", "int8", "l0.mm"),
           R.OpDemand("matmul", "int8", "int8", "l1.mm"),
           R.OpDemand("softmax", "int8", None, "l0.sm")]
    return R.route_plan(dem, "gemmini")


@pytest.mark.skipif(not _gemmini_available(), reason="gemmini contract not resolvable in this env")
def test_mesh_verify_synthesizes_and_passes(monkeypatch):
    """Each mesh matmul is synthesized as a real DxD merlin_iface tile and run through certify; a passing
    oracle aggregates to status=verified with per-tile evidence (dims, dtype, oracle kind)."""
    import merlin.compile_cli as CC
    from merlin.targetgen import oot_runner

    seen = []

    def fake_certify(pkg_dir, iface, **kw):
        txt = iface.read_text(encoding="utf-8")
        seen.append(txt)
        assert "merlin_iface.matmul" in txt          # real synthesized tile, not a stub string
        assert kw.get("target") == "gemmini"          # target threaded to the oracle
        return {"status": "pass",
                "oracle": {"kind": "spike_gemmini_functional", "result": "pass", "cycles": 47}}

    monkeypatch.setattr(oot_runner, "build_package", lambda pkg, timeout=1800: None)
    monkeypatch.setattr(oot_runner, "load_package", lambda p, contract=None: object())
    monkeypatch.setattr(oot_runner, "certify", fake_certify)

    res = CC._mesh_verify(_two_matmul_plan(), target="gemmini", package="/pkg", timeout=60)
    assert res["status"] == "verified"
    assert res["n_tiles"] == 2 and res["n_passed"] == 2 and res["n_unavailable"] == 0
    assert len(seen) == 2                                          # only the 2 mesh matmuls executed
    t0 = res["per_tile"][0]
    assert t0["status"] == "pass" and t0["operand_dtype"] == "i8" and t0["output_dtype"] == "i32"
    assert t0["M"] == t0["K"] == t0["N"] >= 1 and t0["cycles"] == 47


@pytest.mark.skipif(not _gemmini_available(), reason="gemmini contract not resolvable in this env")
def test_mesh_verify_unavailable_is_fail_closed(monkeypatch):
    """An unavailable oracle is recorded honestly (oracle_unavailable), never a silent pass."""
    import merlin.compile_cli as CC
    from merlin.targetgen import oot_runner

    def fake_certify(pkg_dir, iface, **kw):
        return {"status": "fail",
                "oracle": {"kind": "spike_unavailable", "result": "skipped", "cycles": None},
                "failure": {"detail": "spike sim unavailable in this env"}}

    monkeypatch.setattr(oot_runner, "build_package", lambda pkg, timeout=1800: None)
    monkeypatch.setattr(oot_runner, "load_package", lambda p, contract=None: object())
    monkeypatch.setattr(oot_runner, "certify", fake_certify)

    res = CC._mesh_verify(_two_matmul_plan(), target="gemmini", package="/pkg", timeout=60)
    assert res["status"] == "oracle_unavailable"
    assert res["n_unavailable"] == 2 and res["n_passed"] == 0
    assert all(t["status"] == "oracle_unavailable" for t in res["per_tile"])


def test_mesh_verify_no_default_package_is_not_run(monkeypatch):
    """No default OOT backend + no override -> honest not_run (never a fabricated mesh result)."""
    import merlin.compile_cli as CC
    monkeypatch.setattr(CC, "_default_oot_package", lambda t: None)
    plan = {"mesh": [R.RouteResult(R.OpDemand("matmul", "int8", "int8"), "systolic_mesh", None, None)]}
    res = CC._mesh_verify(plan, target="gemmini", package=None, timeout=60)
    assert res["status"] == "not_run" and res["n_tiles"] == 0 and "package" in res["reason"]


@pytest.mark.skipif(not _gemmini_available(), reason="gemmini contract not resolvable in this env")
def test_mesh_verify_compiles_layer_at_real_extent(monkeypatch):
    """A mesh matmul carrying real (M,K,N) extents is synthesized at that shape (rounded up to the mesh
    dim), not a fixed tile — so a whole-model matmul LAYER runs at its true shape."""
    import merlin.compile_cli as CC
    from merlin.targetgen import oot_runner
    seen = {}

    def fake_certify(pkg_dir, iface, **kw):
        seen["mlir"] = iface.read_text(encoding="utf-8")
        return {"status": "pass", "oracle": {"kind": "verilator_gemmini", "result": "pass", "cycles": 512}}

    monkeypatch.setattr(oot_runner, "build_package", lambda pkg, timeout=1800: None)
    monkeypatch.setattr(oot_runner, "load_package", lambda p, contract=None: object())
    monkeypatch.setattr(oot_runner, "certify", fake_certify)
    # a layer with K=64 (> mesh dim) forces multi-tile; N=32 rectangular
    plan = {"mesh": [R.RouteResult(R.OpDemand("matmul", "int8", "int8", "layer", m=16, k=64, n=32),
                                   "systolic_mesh", None, None)]}
    res = CC._mesh_verify(plan, target="gemmini", package="/pkg", timeout=60)
    assert res["status"] == "verified" and res["n_passed"] == 1
    t = res["per_tile"][0]
    assert (t["M"], t["K"], t["N"]) == (16, 64, 32)          # real extent, rounded to the mesh dim
    assert "16x64" in seen["mlir"]                            # the interface carries the true layer shape


@pytest.mark.skipif(not _gemmini_available(), reason="gemmini contract not resolvable in this env")
def test_mesh_verify_unsynthesizable_op_is_honest(monkeypatch):
    """A mesh op with no single-tile synthesizer is recorded, never counted as executed or passed."""
    import merlin.compile_cli as CC
    from merlin.targetgen import oot_runner
    monkeypatch.setattr(oot_runner, "build_package", lambda pkg, timeout=1800: None)
    monkeypatch.setattr(oot_runner, "load_package", lambda p, contract=None: object())
    # a fabricated mesh op with no corpus_spec builder
    plan = {"mesh": [R.RouteResult(R.OpDemand("mystery_op", "int8", "int8", "x"),
                                   "systolic_mesh", None, None)]}
    res = CC._mesh_verify(plan, target="gemmini", package="/pkg", timeout=60)
    assert res["n_tiles"] == 0 and res["n_unsynthesizable"] == 1
    assert res["per_tile"][0]["status"] == "no_tile_synthesizer"
