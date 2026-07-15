"""Tests for datatype -> compute-unit routing (merlin.targetgen.routing)."""
from __future__ import annotations

from merlin.targetgen import compute_units as cu
from merlin.targetgen import routing as rt


def _rvv_units():
    # An RVV-like vector unit: regular floats + int8, no fp4/fp6.
    return cu.compute_units({
        "compute_units": [
            {"name": "vector", "kind": "vector", "dtypes": ["fp32", "fp16", "bf16", "int8"],
             "ops": ["matmul", "elementwise"],
             "accumulate": [
                 {"in": "fp16", "weight": "fp16", "acc": "f32"},
                 {"in": "int8", "weight": "int8", "acc": "i32"},
             ]},
        ]
    })


def _gemmini_mx_units():
    return cu.compute_units({
        "compute_units": [
            {"name": "mx_pe", "kind": "systolic", "dtypes": ["mxfp4", "mxfp6", "mxfp8", "int8", "fp16"],
             "ops": ["matmul"],
             "accumulate": [
                 {"in": "mxfp4", "weight": "mxfp4", "acc": "f32"},
                 {"in": "mxfp6", "weight": "mxfp6", "acc": "f32"},
                 {"in": "fp16", "weight": "mxfp4", "acc": "f32"},   # mixed act/weight
             ]},
        ]
    })


def test_rvv_routes_supported_and_gaps_low_bit():
    units = _rvv_units()
    results = rt.route([
        rt.OpDemand("matmul", "int8", "int8", site="layer0"),
        rt.OpDemand("matmul", "fp16", "fp16", site="layer1"),
        rt.OpDemand("matmul", "mxfp4", "mxfp4", site="mlp0"),   # RVV has no fp4 datapath
    ], units)
    assert results[0].unit == "vector" and results[0].acc == "i32"
    assert results[1].unit == "vector" and results[1].acc == "f32"
    assert results[2].unit is None and "no compute unit" in results[2].gap
    assert not rt.is_fully_routed(results)
    assert [g.demand.site for g in rt.gaps(results)] == ["mlp0"]


def test_gemmini_mx_routes_low_bit_and_mixed():
    units = _gemmini_mx_units()
    results = rt.route([
        rt.OpDemand("matmul", "mxfp4", "mxfp4"),
        rt.OpDemand("matmul", "mxfp6", "mxfp6"),
        rt.OpDemand("matmul", "fp16", "mxfp4"),      # mixed activation/weight precision
    ], units)
    assert rt.is_fully_routed(results)
    assert all(r.unit == "mx_pe" and r.acc == "f32" for r in results)


def test_combo_not_in_accumulate_matrix_gaps():
    # int8 is a listed dtype but there is no int8xint8 accumulate rule -> illegal mode -> gap.
    units = _gemmini_mx_units()
    results = rt.route([rt.OpDemand("matmul", "int8", "int8")], units)
    assert results[0].gap is not None


def test_route_target_reads_contract():
    # gemmini's contract compute_units accept int8 matmul.
    results = rt.route_target([rt.OpDemand("matmul", "int8", "int8")], "gemmini")
    assert results[0].unit == "systolic_mesh" and results[0].acc == "i32"
    # ... but not fp4.
    assert rt.route_target([rt.OpDemand("matmul", "fp4_e2m1", "fp4_e2m1")], "gemmini")[0].gap is not None
