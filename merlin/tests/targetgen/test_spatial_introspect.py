"""T6 Stage 0/1 — the SPATIAL (tensor-tile) fact family and its first instance, the Saturn OPU.

Two things are proven here:

* the ADDITIVE routing: a new ``spatial`` compute-unit kind with a ``fact_extractor`` that
  :func:`merlin.targetgen.rtl.mlc_bridge.fact_bundle_for` dispatches on (systolic/vector/scalar ->
  the CIRCT static bundle, simt -> muon, spatial -> the OPU state-manifest introspect), and
* the STRUCTURAL facts extracted from the OuterProductUnit state manifest — the ``cluster x cell`` tile
  geometry read from the cell count (NOT ``discover_mesh_dim``, which mis-derives DIM=4), the MRF depth,
  the int8/fp8 element widths + datapaths, and the FP8 FMA latency.

The systolic path must be provably untouched: ``fact_bundle_for("gemmini")`` routes to the unchanged
``target_fact_bundle`` and (when mlc is available) is byte-identical to it.
"""
from __future__ import annotations

import json

import pytest

from merlin.targetgen import families as F
from merlin.targetgen.compute_units import KINDS
from merlin.targetgen.rtl import mlc_bridge as B
from merlin.targetgen.rtl import spatial_introspect as SI

_OPU_TARGET = "saturn_opu_mxv256d128"


def _opu_available(target: str = _OPU_TARGET) -> bool:
    paths = B.opu_artifact_paths(target)
    return bool(paths and paths.get("man") and paths["man"].is_file())


# --------------------------------------------------------------------------- Stage 0: family routing
def test_spatial_is_a_canonical_kind_with_a_profile():
    # spatial joins the taxonomy additively; the registry stays exactly aligned with KINDS.
    assert "spatial" in KINDS
    assert set(F.known_kinds()) == set(KINDS)
    p = F.family_profile("spatial")
    assert p.kind == "spatial" and p.endpoint_kind_default in F.ENDPOINT_KINDS
    # a spatial tile is a command-ISA MXU, but NOT RoCC -> no rocc_insn trace gate.
    assert p.trace_gate is None


def test_fact_extractor_routing_table():
    # every kind names the fact-extraction family fact_bundle_for dispatches on — never a target name.
    assert F.family_profile("systolic").fact_extractor == "circt_static"
    assert F.family_profile("vector").fact_extractor == "circt_static"
    assert F.family_profile("scalar").fact_extractor == "circt_static"
    assert F.family_profile("simt").fact_extractor == "simt_config"
    assert F.family_profile("spatial").fact_extractor == "opu"


def test_fact_bundle_for_routes_by_kind(monkeypatch):
    """The dispatch sends each kind to its extractor — proven without mlc by monkeypatching the leaves."""
    monkeypatch.setattr(B, "target_fact_bundle", lambda t: {"via": "circt_static", "target": t})
    monkeypatch.setattr(B, "spatial_fact_bundle", lambda t: {"via": "opu", "target": t})
    monkeypatch.setattr(B, "_simt_fact_bundle", lambda t: {"via": "muon", "target": t})

    monkeypatch.setattr(B, "_resolve_kind", lambda t: "systolic")
    assert B.fact_bundle_for("gemmini")["via"] == "circt_static"
    monkeypatch.setattr(B, "_resolve_kind", lambda t: "spatial")
    assert B.fact_bundle_for(_OPU_TARGET)["via"] == "opu"
    monkeypatch.setattr(B, "_resolve_kind", lambda t: "simt")
    assert B.fact_bundle_for("muon")["via"] == "muon"
    # an unresolved kind degrades to the systolic static path (pre-existing behavior).
    monkeypatch.setattr(B, "_resolve_kind", lambda t: None)
    assert B.fact_bundle_for("brand_new")["via"] == "circt_static"


def test_resolve_kind_maps_opu_targets_without_a_manifest():
    # the OPU arc targets carry no in-tree capability manifest yet -> resolved via the arc-target map.
    assert B._resolve_kind("saturn_opu_mxv256d128") == "spatial"
    assert B._resolve_kind("saturn_opu_v128d64") == "spatial"


# ------------------------------------------------------------ Stage 1: pure structural parse (no mlc)
def test_grid_and_geometry_from_operand_ports():
    # a synthetic 2x2-cluster x 2x2-cell (=4x4 tile) manifest name set.
    names = [f"io_op_in_l_{g}_{i}" for g in range(2) for i in range(2)]
    names += [f"io_op_in_t_{g}_{j}" for g in range(2) for j in range(2)]
    names += [f"clusters_{cy}_{cx}/cells_{r}_{c}/regs_{m}"
              for cy in range(2) for cx in range(2) for r in range(2) for c in range(2)
              for m in range(4)]
    names += ["io_op_macc_0", "io_op_macc_1", "io_op_mvin_0", "io_op_shift"]
    grid = SI._grid_from_operand_ports(names)
    assert grid["rows"] == 4 and grid["cols"] == 4
    assert grid["l_groups"] == 2 and grid["l_cells"] == 2
    assert SI._cell_count(names) == 16          # 4 clusters x 4 cells, one regs_0 each
    assert SI._mrf_depth(names) == 4            # regs_0..regs_3
    assert SI._op_categories(names) == ["macc", "mvin", "shift"]
    # not an OPU manifest -> None, honestly.
    assert SI._grid_from_operand_ports(["io_foo", "io_bar"]) is None


def test_cell_port_widths_and_fp8_fma_parse_via_shared_helpers():
    hw = ("hw.module private @OuterProductCell(in %clock : i1, in %reset : i1, in %io_in_l : i8, "
          "in %io_in_t : i8, in %io_mrf_idx : i4, in %io_fp8 : i1, in %io_mvin_data : i32, "
          "out io_out : i32) {\n}\n"
          "hw.module private @MulAddRecFNPipe_l1_e8_s24(in %clock : i1) {\n}\n")
    w = SI._cell_port_widths(hw)
    assert w == {"operand_bits": 8, "accumulator_bits": 32, "mrf_idx_bits": 4}
    fma = SI._fp8_fma(hw)
    assert fma == {"latency_cycles": 1, "exp_bits": 8, "sig_bits": 24}
    # an int8-only tile has no FMA pipe module -> None.
    assert SI._fp8_fma("hw.module private @OuterProductCell(in %clock : i1) {}") is None


def test_unavailable_bundle_is_honest(monkeypatch):
    monkeypatch.setattr(B, "opu_artifact_paths", lambda t: None)
    b = SI.build_fact_bundle(_OPU_TARGET)
    assert b["kind"] == "spatial" and b["n_derived"] == 0
    assert all(f["derived"] is False and f["value"] is None for f in b["fields"].values())


# -------------------------------------------------------- Stage 1: real RTL extraction (mlc-gated)
@pytest.mark.skipif(not _opu_available(), reason="mlc / OuterProductUnit arc artifacts not present")
def test_opu_facts_from_real_state_manifest():
    """The decisive Stage-1 result: the 16x16 = 256-cell tile geometry read from the OPU state manifest,
    NOT the DIM=4 that discover_mesh_dim mis-derives from the cluster layer."""
    b = B.fact_bundle_for(_OPU_TARGET)
    assert b["kind"] == "spatial" and b["n_derived"] == len(b["fields"])
    td = b["fields"]["tile_dim"]
    assert td["value"]["rows"] == 16 and td["value"]["cols"] == 16 and td["value"]["cells"] == 256
    assert td["value"]["clusters"] == {"rows": 4, "cols": 4}
    assert "NOT discover_mesh_dim" in td["source"]
    # the tile geometry is NOT the mesh-DIM discovery (which mis-derives 4).
    assert td["value"]["rows"] != (B.discovered_dim(_OPU_TARGET) or -1)

    assert b["fields"]["mrf_depth"]["value"] == 16
    assert b["fields"]["element_widths"]["value"] == {"operand_bits": 8, "accumulator_bits": 32}
    names = [d["name"] for d in b["fields"]["dtypes"]["value"]]
    assert names == ["int8", "fp8_e4m3", "fp8_e5m2"]
    assert b["fields"]["fma_latency"]["value"] == {"int8_mac_cycles": 0, "fp8_fma_cycles": 1}
    assert b["fields"]["op_categories"]["value"] == ["macc", "mvin", "shift"]
    assert b["fields"]["accum_kind"]["value"] == {"int8": "i32", "fp8": "f32"}


@pytest.mark.skipif(not _opu_available("saturn_opu_v128d64"),
                    reason="mlc / v128d64 arc artifacts not present")
def test_opu_int8_only_config_has_no_fp8_datapath():
    b = B.fact_bundle_for("saturn_opu_v128d64")
    td = b["fields"]["tile_dim"]["value"]
    assert td["rows"] == 8 and td["cols"] == 8 and td["cells"] == 64
    assert [d["name"] for d in b["fields"]["dtypes"]["value"]] == ["int8"]
    assert b["fields"]["fma_latency"]["value"]["fp8_fma_cycles"] is None
    assert b["fields"]["accum_kind"]["value"] == {"int8": "i32"}


@pytest.mark.skipif(not B.mlc_available()[0], reason="mlc not available")
def test_gemmini_systolic_bundle_byte_identical_through_dispatch():
    """The additivity invariant: routing gemmini through fact_bundle_for yields the UNCHANGED systolic
    static bundle, byte-for-byte identical to target_fact_bundle."""
    via_dispatch = json.dumps(B.fact_bundle_for("gemmini"), sort_keys=True)
    direct = json.dumps(B.target_fact_bundle("gemmini"), sort_keys=True)
    assert via_dispatch == direct
