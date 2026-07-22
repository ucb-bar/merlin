"""Workstream A: the Gemmini CCA lifter (RoCC trace -> SpatialFacet) + micro-kernel resolver.

These are backend-specific glue that registers into the agnostic core via the gemmini plugin. The lifter
is what lets arm 3 diff its authored dialect's behavior against a reference (cca_compare.compare); the
resolver makes the micro-kernel granularity expressible for gemmini (microkernel.resolve).
"""
from __future__ import annotations

from merlin.targetgen import gemmini_cca as G
from merlin.kernels import cca, cca_compare, microkernel
from merlin.kernels.microkernel import MicrokernelSpec


def _trace(hist, insns=()):
    return {"instructions": list(insns), "summary": {"class_histogram": hist}}


def test_op_counts_accumulator_resident_from_compute_accumulate():
    counts = G.op_counts_from_trace(_trace({"COMPUTE_ACCUMULATE": 3, "MVOUT": 1, "CONFIG_EX": 1}))
    assert counts["acc_resident"] is True and counts["widening"] is True
    assert counts["dataflow"] == "ws"


def test_op_counts_accumulator_resident_from_preload_bit():
    insns = [{"class": "PRELOAD", "accumulate": True}, {"class": "COMPUTE_PRELOADED"},
             {"class": "MVOUT", "readout": "i32"}]
    counts = G.op_counts_from_trace(_trace({"COMPUTE_PRELOADED": 1, "PRELOAD": 1, "MVOUT": 1}, insns))
    assert counts["acc_resident"] is True and counts["acc_dtype"] == "i32"


def test_op_counts_non_resident_when_no_accumulate():
    insns = [{"class": "PRELOAD", "accumulate": False}, {"class": "COMPUTE_PRELOADED"},
             {"class": "MVOUT", "readout": "i8"}]
    counts = G.op_counts_from_trace(_trace({"COMPUTE_PRELOADED": 1, "PRELOAD": 1, "MVOUT": 1}, insns))
    assert counts["acc_resident"] is False and counts["acc_dtype"] == "i8"
    assert counts["dataflow"] is None          # no CONFIG_EX -> honestly unknown, not guessed


def test_lift_from_trace_produces_spatial_cca_and_diffs():
    """The lifter yields a gemmini CCA whose SpatialFacet a cca_compare diff can act on."""
    resident = G.lift_from_trace(_trace({"COMPUTE_ACCUMULATE": 2, "CONFIG_EX": 1}), pe_dim=16)
    assert resident.backend == ["gemmini"]
    assert resident.spatial.accumulator_resident is True and resident.spatial.pe_rows == 16
    nonres = G.lift_from_trace(
        _trace({"COMPUTE_PRELOADED": 1, "PRELOAD": 1},
               [{"class": "PRELOAD", "accumulate": False}, {"class": "COMPUTE_PRELOADED"}]), pe_dim=16)
    divs = cca_compare.compare(resident, nonres)
    axes = {d.axis for d in divs}
    assert "spatial.accumulator_resident" in axes   # the diff surfaces the residency divergence


def test_microkernel_resolver_registered_and_clamps_to_dim():
    from merlin.targetgen import gemmini_plugin
    gemmini_plugin.register()
    assert "gemmini" in microkernel.registered_targets()
    real = microkernel.resolve("gemmini", MicrokernelSpec(MR=64, NR=64, KC=8, k_block=True))
    assert real["tile_rows"] == 16 and real["tile_cols"] == 16   # clamped to the mesh DIM
    assert real["k_tile"] == 8
    assert real["opts"].accumulator_resident is True             # k_block -> accumulator-resident
