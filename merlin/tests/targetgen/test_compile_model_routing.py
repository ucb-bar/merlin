"""Mixed-dialect whole-model routing: each op is split across a target's compute units — matmul/systolic
tiles execute on the mesh, norms/activations/elementwise fall to the vector/scalar (RVV) lane. The split is
derived structurally from the captured model linalg (prov.op/prov.family, no regex) and is an honest,
data-driven decision (an op no unit supports is a scalar/RVV fallback, never a silent drop).

Target-agnostic: the target is a parameter; this edge names one as data under test."""
from __future__ import annotations

import os
from pathlib import Path

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


def test_summarize_route_plan_surfaces_matmul_extents():
    """The summary reports each mesh matmul's REAL M x K x N extent (threaded from the linalg), and a
    unary mesh op with no extents contributes none — so the plan carries true layer shapes, not just
    the op family."""
    from merlin.compile_cli import _summarize_route_plan
    plan = {"mesh": [R.RouteResult(R.OpDemand("matmul", "int8", "int8", "l0", m=8, k=2048, n=2048),
                                   "systolic_mesh", None, None),
                     R.RouteResult(R.OpDemand("matmul", "int8", "int8", "l1", m=8, k=2048, n=256),
                                   "systolic_mesh", None, None)],
            "fallback": [],
            "scalar_rvv": [R.RouteResult(R.OpDemand("softmax", "int8", None), None, None, "gap")]}
    s = _summarize_route_plan(plan)
    ext = s["mesh_matmul_extents"]
    assert [(e["m"], e["k"], e["n"]) for e in ext] == [(8, 2048, 2048), (8, 2048, 256)]
    assert [e["site"] for e in ext] == ["l0", "l1"]


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
def test_run_matmul_on_mesh_injects_real_operands(monkeypatch):
    """run_matmul_on_mesh builds the matmul interface at the operands' real shape and INJECTS A/W as the
    certify inputs (not name-materialized), returning the mesh's actual output. A stubbed oracle lets us
    assert the injection wiring without a live sim."""
    import merlin.compile_cli as CC
    from merlin.targetgen import oot_runner
    seen = {}

    def fake_certify(pkg, iface, **kw):
        seen["inputs"] = kw.get("inputs")
        seen["mlir"] = iface.read_text(encoding="utf-8")
        return {"status": "pass", "oracle_outputs": {"Y0": [[42, 0], [0, 42]]}}

    monkeypatch.setattr(CC, "_default_oot_package", lambda t: "/pkg")
    monkeypatch.setattr(oot_runner, "certify", fake_certify)
    A = [[1, 2], [3, 4]]
    W = [[5, 6], [7, 8]]
    D = CC._mesh_tile_binding("gemmini", "int8", "i32").tile_dim
    # the oracle answers at the PADDED extent; the caller must still get its own 2x2 back
    padded_out = [[42 if r == c else 0 for c in range(D)] for r in range(D)]
    seen.clear()

    def fake_certify_padded(pkg, iface, **kw):
        seen["inputs"] = kw.get("inputs")
        seen["mlir"] = iface.read_text(encoding="utf-8")
        return {"status": "pass", "oracle_outputs": {"Y0": padded_out}}

    monkeypatch.setattr(oot_runner, "certify", fake_certify_padded)
    out = CC.run_matmul_on_mesh("gemmini", A, W, operand_dtype="int8", accum_dtype="i32")

    # PADDED TO THE TILE EDGE, then sliced back. A package is entitled to reject a sub-tile extent, and a
    # real model is full of them (every matmul layer of an 8-token sequence has M=8 against a 16- or
    # 32-wide mesh), so building at the operands' raw shape meant the mesh refused the layer and the
    # runtime silently fell back to the host. Zero-padding is exact for a contraction.
    assert out == [[42, 0], [0, 42]]                          # sliced back to the caller's extent
    assert seen["inputs"]["A0"][0][:2] == [1.0, 2.0]          # real operands in the top-left...
    assert seen["inputs"]["A0"][1][:2] == [3.0, 4.0]
    assert seen["inputs"]["W"][0][:2] == [5.0, 6.0]
    assert len(seen["inputs"]["A0"]) == D and len(seen["inputs"]["A0"][0]) == D
    assert all(v == 0.0 for v in seen["inputs"]["A0"][0][2:])  # ...zeros everywhere else
    assert all(v == 0.0 for v in seen["inputs"]["A0"][2])
    assert f"{D}x{D}" in seen["mlir"]                         # built at the padded, tile-aligned extent


def test_run_matmul_on_mesh_none_without_package(monkeypatch):
    """No OOT backend package -> None (never a fabricated result)."""
    import merlin.compile_cli as CC
    monkeypatch.setattr(CC, "_default_oot_package", lambda t: None)
    assert CC.run_matmul_on_mesh("gemmini", [[1]], [[1]]) is None


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


# --------------------------------------------------------------------------- real whole-model routing
# End-to-end over a REAL model2MLIR whole-model int8 linalg: the per-op demands must carry each matmul
# LAYER's true (M,K,N) extents (not just the op family), and routing them onto the target mesh must
# preserve those extents so a whole-model matmul layer is compiled at its real shape. Gated on the m2m
# checkout being resolvable (mirrors merlin/tests/ir/test_whole_model_compilability.py).
def _tiny_llama_int8():
    for var in ("MERLIN_M2M_DIR", "MERLIN_MODEL2MLIR"):
        base = os.environ.get(var)
        if base:
            p = Path(base) / "workloads" / "tiny_llama" / "tiny_llama_int8.mlir"
            if p.is_file():
                return p
    return None


@pytest.mark.skipif(_tiny_llama_int8() is None,
                    reason="model2MLIR checkout not resolvable (set MERLIN_M2M_DIR)")
def test_real_tiny_llama_demands_carry_matmul_extents():
    """model_op_demands over the real int8 tiny_llama linalg attaches each of the 15 matmul layers' real
    2D (M,K,N) extents, threaded structurally from the linalg.matmul ins-operand tensor shapes."""
    linalg = _tiny_llama_int8().read_text(encoding="utf-8")
    dem = CSrc.model_op_demands(linalg, "int8")
    mm = [d for d in dem if d.op == "matmul"]
    assert len(mm) == 15                                       # the int8 linear backbone
    for d in mm:
        assert d.m and d.k and d.n                             # real extents, not None/0
        assert d.weight_fmt == "int8"                          # contraction -> weighted
    # the leading layers' real shapes (attention/mlp projections of an 8-token, 2048-dim model)
    assert (mm[0].m, mm[0].k, mm[0].n) == (8, 2048, 2048)
    assert (mm[1].m, mm[1].k, mm[1].n) == (8, 2048, 256)


@pytest.mark.skipif(_tiny_llama_int8() is None,
                    reason="model2MLIR checkout not resolvable (set MERLIN_M2M_DIR)")
@pytest.mark.skipif(not _gemmini_available(), reason="gemmini contract not resolvable in this env")
def test_real_tiny_llama_routes_matmuls_onto_mesh_with_extents():
    """Routing the real tiny_llama demands onto the target: all 15 matmul layers land on the systolic
    mesh and each mesh entry preserves its real (M,K,N) extent; the route summary reports them so the
    plan carries true layer shapes instead of only the op family."""
    from merlin.compile_cli import _summarize_route_plan
    linalg = _tiny_llama_int8().read_text(encoding="utf-8")
    dem = CSrc.model_op_demands(linalg, "int8")
    plan = R.route_plan(dem, "gemmini")

    mesh_mm = [r for r in plan["mesh"] if r.demand.op == "matmul"]
    assert len(mesh_mm) == 15
    extents = [(r.demand.m, r.demand.k, r.demand.n) for r in mesh_mm]
    assert all(m and k and n for (m, k, n) in extents)         # every mesh matmul carries real extents
    assert extents[0] == (8, 2048, 2048) and extents[1] == (8, 2048, 256)

    summary = _summarize_route_plan(plan)
    assert summary["on_mesh"].get("matmul") == 15
    summ_ext = [(e["m"], e["k"], e["n"]) for e in summary["mesh_matmul_extents"]
                if e["op"] == "matmul"]
    assert summ_ext == extents                                 # the summary reports every layer's shape


# -- capacity-fit mesh tiling: a whole layer's weight+activation working set may exceed the target's
# on-chip scratchpad, so a mesh matmul is tiled to the largest capacity-fit unit (derived from the RTL
# memory fact) and n_subtiles reports how many tile the layer. Pure-arithmetic tests run everywhere;
# the fact-derivation test skips when the target's fact bundle is unavailable. --

def test_dtype_bytes():
    from merlin.compile_cli import _dtype_bytes
    assert _dtype_bytes("i8") == 1
    assert _dtype_bytes("i32") == 4
    assert _dtype_bytes("f16") == 2
    assert _dtype_bytes("f32") == 4
    assert _dtype_bytes(None) == 1


def test_capacity_fit_tile_shrinks_to_fit():
    from merlin.compile_cli import _capacity_fit_tile
    cap = 262144  # gemmini int8 scratchpad elements
    # A tile that already fits is returned whole, one subtile.
    assert _capacity_fit_tile(16, 256, 256, 16, cap) == (16, 256, 256, 1)
    # The real tiny_llama layer (rounded to the mesh dim) overflows and is tiled; every returned tile
    # fits the capacity and the count covers the layer.
    mt, kt, nt, n = _capacity_fit_tile(16, 2048, 2048, 16, cap)
    assert kt * nt + mt * kt <= cap
    assert mt % 16 == 0 and kt % 16 == 0 and nt % 16 == 0
    import math
    assert n == math.ceil(2048 / kt) * math.ceil(2048 / nt) * math.ceil(16 / mt)
    assert n > 1


def test_scratchpad_capacity_is_derived_not_hardcoded():
    """The on-chip capacity comes from the target's RTL memory fact; skip when facts are unavailable."""
    from merlin.compile_cli import _scratchpad_capacity_elems
    try:
        from merlin.targetgen.rtl import facts as _facts
        mems = (_facts.load_facts("gemmini").get("facts") or {}).get("memories") or []
    except Exception:
        pytest.skip("gemmini fact bundle unavailable")
    sp = next((m for m in mems if m.get("name") == "scratchpad"), None)
    if not (sp and sp.get("bytes")):
        pytest.skip("no scratchpad memory fact for gemmini")
    assert _scratchpad_capacity_elems("gemmini", 1) == int(sp["bytes"])
    assert _scratchpad_capacity_elems("gemmini", 4) == int(sp["bytes"]) // 4
