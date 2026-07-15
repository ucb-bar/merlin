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
