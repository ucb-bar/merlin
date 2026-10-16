"""T6 Stage 0/1 — the SPATIAL (tensor-tile) fact family and its first instance, the Saturn OPU.

Two things are proven here:

* the ADDITIVE routing: a new ``spatial`` compute-unit kind with a ``fact_extractor`` that
  :func:`merlin.targetgen.rtl.mlc_bridge.fact_bundle_for` dispatches on (systolic/vector/scalar ->
  the CIRCT static bundle, simt -> muon, spatial -> the OPU state-manifest introspect), and
* the STRUCTURAL facts extracted from the OuterProductUnit state manifest — the ``cluster x cell`` tile
  geometry read from the cell count (NOT ``discover_mesh_dim``, which mis-derives DIM=4), MRF depth
  and storage widths. Arithmetic and latency remain unknown without a dataflow observation.

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

pytestmark = pytest.mark.target("gemmini", "muon", "saturn_opu_mxv256d128")

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
    """The dispatch sends each kind to its extractor — proven without mlc by monkeypatching the leaves.

    Dispatch reads the SET of kinds now, not a primary one: routing on the primary made a SIMT cluster
    embedding a systolic mesh run the config introspect and never look at the mesh, so its geometry,
    capacities and legal opcodes were absent for hardware that has all three.

    fact_bundle_for is memoized process-wide, so clear the memo first (else a real bundle cached by an
    earlier test shadows the monkeypatched leaf) and again in teardown."""
    B.clear_fact_bundle_cache()
    monkeypatch.setattr(
        B,
        "target_fact_bundle",
        lambda t: {"via": "circt_static", "target": t, "fields": {"mesh_dim": {"value": 16, "derived": True}}},
    )
    monkeypatch.setattr(B, "spatial_fact_bundle", lambda t: {"via": "opu", "target": t, "fields": {}})
    monkeypatch.setattr(
        B,
        "_simt_fact_bundle",
        lambda t: {"via": "muon", "target": t, "fields": {"simt": {"value": 1, "derived": True}}},
    )
    try:
        monkeypatch.setattr(B, "_resolve_kinds", lambda t: ("systolic",))
        assert B.fact_bundle_for("gemmini")["via"] == "circt_static"
        B.clear_fact_bundle_cache()
        monkeypatch.setattr(B, "_resolve_kinds", lambda t: ("spatial",))
        assert B.fact_bundle_for(_OPU_TARGET)["via"] == "opu"
        B.clear_fact_bundle_cache()
        monkeypatch.setattr(B, "_resolve_kinds", lambda t: ("simt",))
        assert B.fact_bundle_for("muon")["via"] == "muon"
        # a HYBRID runs BOTH and merges: the mesh half must survive, which is the whole fix.
        B.clear_fact_bundle_cache()
        monkeypatch.setattr(B, "_resolve_kinds", lambda t: ("simt", "systolic"))
        merged = B.fact_bundle_for("hybrid")
        assert merged["fields"]["simt"]["derived"] and merged["fields"]["mesh_dim"]["value"] == 16
        assert merged["extractors"] == ["simt_config", "circt_static"]
        # an unresolved kind degrades to the systolic static path (pre-existing behavior).
        B.clear_fact_bundle_cache()
        monkeypatch.setattr(B, "_resolve_kinds", lambda t: ())
        assert B.fact_bundle_for("brand_new")["via"] == "circt_static"
    finally:
        B.clear_fact_bundle_cache()


def test_resolve_kind_maps_opu_targets_without_a_manifest():
    # the OPU arc targets carry no in-tree capability manifest yet -> resolved via the arc-model registry.
    assert B._resolve_kind("saturn_opu_mxv256d128") == "spatial"
    assert B._resolve_kind("saturn_opu_v128d64") == "spatial"


# ------------------------------------------------------------ Stage 1: pure structural parse (no mlc)
def test_grid_and_geometry_from_operand_ports():
    # a synthetic 2x2-cluster x 2x2-cell (=4x4 tile) manifest name set.
    names = [f"io_op_in_l_{g}_{i}" for g in range(2) for i in range(2)]
    names += [f"io_op_in_t_{g}_{j}" for g in range(2) for j in range(2)]
    names += [
        f"clusters_{cy}_{cx}/cells_{r}_{c}/regs_{m}"
        for cy in range(2)
        for cx in range(2)
        for r in range(2)
        for c in range(2)
        for m in range(4)
    ]
    names += ["io_op_macc_0", "io_op_macc_1", "io_op_mvin_0", "io_op_shift"]
    grid = SI._grid_from_operand_ports(names)
    assert grid["rows"] == 4 and grid["cols"] == 4
    assert grid["l_groups"] == 2 and grid["l_cells"] == 2
    assert SI._cell_count(names) == 16  # 4 clusters x 4 cells, one regs_0 each
    assert SI._mrf_depth(names) == 4  # regs_0..regs_3
    assert SI._op_categories(names) == ["macc", "mvin", "shift"]
    # not an OPU manifest -> None, honestly.
    assert SI._grid_from_operand_ports(["io_foo", "io_bar"]) is None


def test_cell_port_widths_and_fp8_fma_parse_via_shared_helpers():
    hw = (
        "hw.module private @OuterProductCell(in %clock : i1, in %reset : i1, in %io_in_l : i8, "
        "in %io_in_t : i8, in %io_mrf_idx : i4, in %io_fp8 : i1, in %io_mvin_data : i32, "
        "out io_out : i32) {\n}\n"
        "hw.module private @MulAddRecFNPipe_l1_e8_s24(in %clock : i1) {\n}\n"
    )
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


def _synthetic_spatial_inputs(tmp_path, monkeypatch, *, operand_bits, accumulator_bits, pipe=None):
    names = [f"io_op_in_{side}_0_{i}" for side in ("l", "t") for i in range(2)]
    states = [{"name": name, **({"numBits": operand_bits} if operand_bits is not None else {})} for name in names]
    states += [
        {"name": f"clusters_0_0/cells_{r}_{c}/regs_{bank}",
         **({"numBits": accumulator_bits} if accumulator_bits is not None else {})}
        for r in range(2) for c in range(2) for bank in range(2)
    ]
    states += [{"name": "io_op_macc_0", "numBits": 1}]
    manifest = tmp_path / "states.json"
    manifest.write_text(json.dumps({"name": "Variant", "states": states}))
    hw = tmp_path / "model.hw.mlir"
    hw.write_text("" if pipe is None else f"hw.module private @MulAddRecFNPipe_l{pipe}_e8_s24() {{}}\n")
    monkeypatch.setattr(B, "opu_artifact_paths", lambda _target: {"man": manifest, "hw": hw})


@pytest.mark.parametrize("operand_bits,accumulator_bits", [(8, 32), (16, 48), (None, 32), (16, None)])
def test_arithmetic_facts_never_come_from_width_defaults(tmp_path, monkeypatch, operand_bits, accumulator_bits):
    _synthetic_spatial_inputs(tmp_path, monkeypatch, operand_bits=operand_bits, accumulator_bits=accumulator_bits)
    bundle = SI.build_fact_bundle("synthetic_spatial")
    widths = bundle["fields"]["element_widths"]
    assert widths["value"] == {"operand_bits": operand_bits, "accumulator_bits": accumulator_bits}
    assert widths["derived"] is (operand_bits is not None and accumulator_bits is not None)
    for name in ("dtypes", "accum_kind", "fma_latency"):
        assert bundle["fields"][name]["value"] is None and bundle["fields"][name]["derived"] is False
    artifact = SI.spatial_facts("synthetic_spatial")
    assert "datapaths" not in artifact["facts"] and "timing" not in artifact["facts"]
    assert {"dtypes", "accum_kind", "fma_latency"} <= set(artifact["unknown"])


@pytest.mark.parametrize("pipe", [None, 0, 3])
def test_module_names_do_not_establish_mac_latency(tmp_path, monkeypatch, pipe):
    _synthetic_spatial_inputs(tmp_path, monkeypatch, operand_bits=16, accumulator_bits=48, pipe=pipe)
    latency = SI.build_fact_bundle("synthetic_spatial")["fields"]["fma_latency"]
    assert latency["value"] is None and latency["derived"] is False
    assert "register boundaries" in latency["evidence"]


# -------------------------------------------------------- Stage 1: real RTL extraction (mlc-gated)
@pytest.mark.skipif(not _opu_available(), reason="mlc / OuterProductUnit arc artifacts not present")
def test_opu_facts_from_real_state_manifest():
    """The decisive Stage-1 result: the 16x16 = 256-cell tile geometry read from the OPU state manifest,
    NOT the DIM=4 that discover_mesh_dim mis-derives from the cluster layer."""
    b = B.fact_bundle_for(_OPU_TARGET)
    assert b["kind"] == "spatial"
    assert b["n_derived"] == len(b["fields"]) - 3
    td = b["fields"]["tile_dim"]
    assert td["value"]["rows"] == 16 and td["value"]["cols"] == 16 and td["value"]["cells"] == 256
    assert td["value"]["clusters"] == {"rows": 4, "cols": 4}
    assert "NOT discover_mesh_dim" in td["source"]
    # the tile geometry is NOT the mesh-DIM discovery (which mis-derives 4).
    assert td["value"]["rows"] != (B.discovered_dim(_OPU_TARGET) or -1)

    assert b["fields"]["mrf_depth"]["value"] == 16
    assert b["fields"]["element_widths"]["value"] == {"operand_bits": 8, "accumulator_bits": 32}
    for name in ("dtypes", "fma_latency", "accum_kind"):
        assert b["fields"][name]["derived"] is False
        assert b["fields"][name]["value"] is None
    assert b["fields"]["op_categories"]["value"] == ["macc", "mvin", "shift"]


@pytest.mark.skipif(not _opu_available("saturn_opu_v128d64"), reason="mlc / v128d64 arc artifacts not present")
def test_second_spatial_config_keeps_arithmetic_unknown():
    b = B.fact_bundle_for("saturn_opu_v128d64")
    td = b["fields"]["tile_dim"]["value"]
    assert td["rows"] == 8 and td["cols"] == 8 and td["cells"] == 64
    for name in ("dtypes", "fma_latency", "accum_kind"):
        assert b["fields"][name]["derived"] is False
        assert b["fields"][name]["value"] is None


@pytest.mark.skipif(not B.mlc_available()[0], reason="mlc not available")
def test_gemmini_systolic_bundle_byte_identical_through_dispatch():
    """The additivity invariant: routing gemmini through fact_bundle_for yields the UNCHANGED systolic
    static bundle, byte-for-byte identical to target_fact_bundle."""
    via_dispatch = json.dumps(B.fact_bundle_for("gemmini"), sort_keys=True)
    direct = json.dumps(B.target_fact_bundle("gemmini"), sort_keys=True)
    assert via_dispatch == direct
