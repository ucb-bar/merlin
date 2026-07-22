"""Generic derivation-driven backend: the target profile + implied levers come from RTL discovery, for
ANY target — no hand facts, no per-target code. (See the derive-dont-overfit rule.)"""
from __future__ import annotations

import pytest

from merlin.targetgen import rtl_backend as RB
from merlin.targetgen.rtl import mlc_bridge as B

_MLC_OK = B.mlc_available()[0]


def test_derived_levers_come_from_discovered_structure():
    """The levers are IMPLIED by the discovered hardware, not hand-listed — pure function, no mlc."""
    mesh_acc = RB.TargetProfile("x", legal_opcodes=(0, 1, 2), memory_map={"accum_mem": "acc"}, dim=16)
    assert RB.derived_levers(mesh_acc) == ["spatial.dataflow", "spatial.accumulator_resident"]
    scalar = RB.TargetProfile("y", legal_opcodes=(0, 1), memory_map={}, dim=None)
    assert RB.derived_levers(scalar) == []                 # no mesh/accumulator => no spatial levers


def test_profile_degrades_honestly_without_mlc(monkeypatch):
    monkeypatch.setattr(B, "mlc_available", lambda: (False, "unavailable"))
    monkeypatch.setattr(B, "discovered_memory_map", lambda t: None)
    monkeypatch.setattr(B, "discovered_dim", lambda t: None)
    prof = RB.target_profile("any_target")
    assert prof.legal_opcodes is None and prof.dim is None and not prof.has_mesh


@pytest.mark.skipif(not _MLC_OK or B.core_hw_mlir("gemmini") is None,
                    reason="mlc / prebuilt core HW dialect not available for the example target")
def test_profile_derived_from_rtl_for_example_target():
    """End-to-end derivation for one example target (gemmini as an ARGUMENT): the ISA, memory map, DIM
    and thus the levers all come from mlc RTL discovery. The identical call works for any target."""
    prof = RB.target_profile("gemmini")
    assert prof.legal_opcodes and 126 in prof.legal_opcodes and 25 not in prof.legal_opcodes
    assert prof.dim == 16 and prof.has_mesh and prof.has_accumulator
    assert set(RB.derived_levers(prof)) == {"spatial.dataflow", "spatial.accumulator_resident"}
