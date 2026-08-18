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


_MESH_ACC = RB.TargetProfile("t", legal_opcodes=(0, 1), memory_map={"accum_mem": "acc"}, dim=16)


def test_codegen_opts_apply_only_derived_levers():
    assert RB.apply_codegen_opts({}, frozenset(), _MESH_ACC) == {}          # baseline unchanged
    on = RB.apply_codegen_opts({}, frozenset({"spatial.dataflow"}), _MESH_ACC)
    assert on["spatial.dataflow"] == "os"
    with pytest.raises(KeyError):                                           # a non-derived lever is rejected
        RB.apply_codegen_opts({}, frozenset({"spatial.dataflow"}),
                              RB.TargetProfile("t", legal_opcodes=(0,), memory_map={}, dim=None))


def test_resolver_clamps_to_discovered_dim():
    class Spec:  # a stand-in MicrokernelSpec
        MR, NR, KC, k_block = 64, 64, 8, True
    r = RB._resolver(Spec(), _MESH_ACC)
    assert r["tile_rows"] == 16 and r["tile_cols"] == 16 and r["k_tile"] == 8
    assert r["opts"]["spatial.accumulator_resident"] is True                # k_block + accumulator -> resident


def test_lift_cca_from_trace_is_generic():
    trace = {"instructions": [{"class": "PRELOAD", "accumulate": True}, {"class": "COMPUTE_PRELOADED"}],
             "summary": {"class_histogram": {"COMPUTE_PRELOADED": 1, "PRELOAD": 1}}}
    cca = RB.lift_cca_from_trace(trace, _MESH_ACC)
    assert cca.backend == ["t"] and cca.spatial.accumulator_resident is True and cca.spatial.pe_rows == 16


@pytest.mark.skipif(not _MLC_OK or B.core_hw_mlir("gemmini") is None,
                    reason="mlc / prebuilt core HW dialect not available for the example target")
def test_profile_derived_from_rtl_for_example_target():
    """End-to-end derivation for one example target (gemmini as an ARGUMENT): the ISA, memory map, DIM
    and thus the levers all come from mlc RTL discovery. The identical call works for any target."""
    prof = RB.target_profile("gemmini")
    assert prof.legal_opcodes and 126 in prof.legal_opcodes and 25 not in prof.legal_opcodes
    assert prof.dim == 16 and prof.has_mesh and prof.has_accumulator
    assert set(RB.derived_levers(prof)) == {"spatial.dataflow", "spatial.accumulator_resident"}


@pytest.mark.skipif(not _MLC_OK or B.core_hw_mlir("gemmini") is None,
                    reason="mlc / prebuilt core HW dialect not available for the example target")
def test_register_derives_routes_into_agnostic_core():
    """register(target) derives the routes from discovery and registers them into the agnostic core: the
    bijection closes and a spatial divergence routes to the target's OOT codegen seam."""
    from merlin.kernels import cca_contract, action_catalog, microkernel
    from merlin.kernels.cca_compare import Divergence
    RB.register("gemmini")
    assert cca_contract.check_bijection("gemmini").unexpected().clean
    assert "gemmini" in microkernel.registered_targets()
    a = action_catalog.route(Divergence(axis="spatial.dataflow", expert="os", ours="ws",
                                        backend="gemmini", evidence=[]))
    loc = action_catalog.seam_location(a.target_seam, backend="gemmini", oot_package="/pkg")
    assert loc["seam_file"].startswith("/pkg/")   # OOT-relative, agent's own middle-end


class TestAnEmptyLeverListIsNotAFinding:
    """`derived_levers() == []` has two causes and the list cannot tell them apart."""

    def _profile(self, **kw):
        from merlin.targetgen.rtl_backend import TargetProfile
        base = dict(target="t", legal_opcodes=None, memory_map=None, dim=None)
        base.update(kw)
        return TargetProfile(**base)

    def test_nothing_discovered_is_reported_as_unknown(self):
        # MEASURED on the spatial tensor tile: every field comes back None and derived_levers() returns
        # [], which reads as "this accelerator exposes no structural levers" when in fact nothing was
        # read. A caller acting on that reports a bare target instead of a missing capability.
        from merlin.targetgen import rtl_backend as RB
        prof = self._profile()
        assert RB.derived_levers(prof) == []
        assert prof.discovered_nothing
        gaps = RB.lever_derivation_gaps(prof)
        assert gaps and "UNKNOWN" in gaps[0]

    def test_a_read_target_with_no_such_structure_is_not_an_unknown(self):
        # Discovery ran (opcodes were grounded) and found no mesh and no accumulator. That is a fact
        # about the hardware, and must not be reported the same way as silence.
        from merlin.targetgen import rtl_backend as RB
        prof = self._profile(legal_opcodes=(0x7B,))
        assert not prof.discovered_nothing
        gaps = RB.lever_derivation_gaps(prof)
        assert not any("UNKNOWN" in g for g in gaps)

    def test_a_discovered_mesh_yields_a_dataflow_lever_and_no_mesh_gap(self):
        from merlin.targetgen import rtl_backend as RB
        prof = self._profile(dim=16)
        assert "spatial.dataflow" in RB.derived_levers(prof)
        assert not any("mesh dimension" in g for g in RB.lever_derivation_gaps(prof))

    def test_a_discovered_accumulator_yields_a_residency_lever(self):
        from merlin.targetgen import rtl_backend as RB
        prof = self._profile(memory_map={"accum_mem": "h"})
        assert "spatial.accumulator_resident" in RB.derived_levers(prof)
        assert not any("accumulator memory" in g for g in RB.lever_derivation_gaps(prof))
