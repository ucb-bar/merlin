"""Tests for the rvv + gemmini_mx capability manifests and cross-target routing."""
from __future__ import annotations

from merlin.targetgen import capability_manifests as cm
from merlin.targetgen import compute_units as cu
from merlin.targetgen import routing as rt


def test_manifests_are_schema_valid():
    for name in cm.MANIFESTS:
        cm.validate(cm.MANIFESTS[name]())   # raises on any problem


def _units(name):
    return cu.compute_units(cm.MANIFESTS[name]())


def test_rvv_accepts_regular_formats_rejects_low_bit():
    units = _units("rvv")
    ok = rt.route([
        rt.OpDemand("matmul", "int8", "int8"),
        rt.OpDemand("matmul", "fp16", "fp16"),
        rt.OpDemand("matmul", "bf16", "bf16"),
    ], units)
    assert rt.is_fully_routed(ok)
    # RVV has no fp4/fp6/native-fp8 datapath -> honest gaps.
    for fmt in ("mxfp4", "mxfp6", "fp4_e2m1", "fp8_e4m3"):
        res = rt.route([rt.OpDemand("matmul", fmt, fmt)], units)
        assert res[0].gap is not None, fmt


def test_gemmini_mx_accepts_low_bit_and_mixed():
    units = _units("gemmini_mx")
    ok = rt.route([
        rt.OpDemand("matmul", "mxfp4", "mxfp4"),
        rt.OpDemand("matmul", "mxfp6", "mxfp6"),
        rt.OpDemand("matmul", "mxfp8", "mxfp8"),
        rt.OpDemand("matmul", "int8", "int8"),
    ], units)
    assert rt.is_fully_routed(ok)


def test_cross_target_contrast():
    # The same fp4 matmul: gap on RVV, routed on gemmini_mx — the whole point.
    d = [rt.OpDemand("matmul", "mxfp4", "mxfp4")]
    assert rt.route(d, _units("rvv"))[0].gap is not None
    assert rt.route(d, _units("gemmini_mx"))[0].unit == "mx_pe"


def test_write_and_route_target(tmp_path):
    # Writing to a temp base and resolving via a plugged-in path proves the end-to-end plumbing.
    import os
    from merlin.targetgen import target_registry as tr

    cm.write_all(base_root=tmp_path)
    os.environ["MERLIN_TARGET_PATH"] = os.pathsep.join(str(tmp_path / n) for n in cm.MANIFESTS)
    try:
        assert tr.resolve("gemmini_mx").kind == "external"
        res = rt.route_target([rt.OpDemand("matmul", "mxfp6", "mxfp6")], "gemmini_mx")
        assert res[0].unit == "mx_pe" and res[0].acc == "f32"
    finally:
        os.environ.pop("MERLIN_TARGET_PATH", None)


def test_radiance_composes_gemmini_mx():
    # radiance's SIMT cluster CONTAINS the gemmini-mx PE: effective dtypes = regular floats + MX.
    units = cu.compute_units(cm.radiance_manifest())
    simt = next(u for u in units if u.name == "simt_cluster")
    eff = cu.effective(simt, units)
    assert {"fp16", "bf16", "fp32"} <= set(eff.dtypes)          # SIMT regular floats
    assert {"mxfp4", "mxfp6", "mxfp8"} <= set(eff.dtypes)       # via the contained gemmini-mx PE
    # the contained unit is the exact gemmini-mx PE (standalone OR embedded)
    assert cm.gemmini_mx_manifest()["compute_units"][0]["name"] == "mx_pe"


def test_radiance_oot_package_discovers_and_routes(tmp_path, monkeypatch):
    from merlin.targetgen import target_registry as tr

    root = cm.write_oot_target("radiance", tmp_path / "radiance")
    assert (root / "contracts" / "target_contract.yaml").is_file()
    assert (root / "contracts" / "dialect_plan.yaml").is_file()
    assert (root / "AGENT.md").is_file()

    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    info = tr.resolve("radiance")
    assert info.kind == "external"
    # plugin block points at the OOT dialect + lowering (Merlin reads, never executes)
    assert info.plugin()["dialect_module"] == "radiance_mlir.dialect"
    # routes both regular floats (SIMT) and low-bit MX (contained gemmini-mx PE)
    assert rt.route_target([rt.OpDemand("matmul", "fp16", "fp16")], "radiance")[0].unit == "simt_cluster"
    assert rt.route_target([rt.OpDemand("matmul", "mxfp6", "mxfp6")], "radiance")[0].unit == "simt_cluster"


def test_dialect_plan_derived_from_units():
    plan = cm.dialect_plan_from_manifest(cm.radiance_manifest())
    assert plan["target"] == "radiance" and plan["dialect_name"] == "radiance"
    assert {t["name"] for t in plan["types"]} == {"simt_cluster_tensor", "mx_pe_tensor"}
    assert {o["name"] for o in plan["ops"]} >= {"matmul", "elementwise"}
