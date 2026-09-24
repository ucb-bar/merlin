"""The derived probe set has to be able to CROSS a tile boundary, on each axis separately.

Two defects, both of which let a shape-keyed backend pass everything the harness could ask it:

1. ``_TILE`` was the literal ``16``, defended by a comment claiming the corners were "structural, not a
   specific mesh dim". A corner only means anything relative to the tile it brackets: on a device whose
   mesh edge is 32, ``tile_plus_1`` is 17 -- still inside tile 0 -- so no probe crossed a boundary in any
   dimension. 16 was one target's mesh row, silently applied to every target.
2. There was no multi-tile corner at all. Nothing in the derived set asked for two tiles in anything.

Together those made the probe suite structurally incapable of detecting a backend that lowers exactly one
tile, which is the backend that shipped. The corners are per-axis because a lowering that loops over K and
N but not M passes two of three, and naming the axis is the difference between "add a loop over M" and
"the compiler does not generalize".
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from merlin.targetgen import capability_probes as CP
from merlin.targetgen.compute_units import SemanticCapability
from merlin.targetgen.target_registry import TargetContractMissing

pytestmark = pytest.mark.target("atlas", "gemmini")


def _cap(family="contraction", **kw):
    return SemanticCapability(
        family=family, dtypes=kw.pop("dtypes", ("int8",)), layouts=(), transpose=False, batch=False, **kw
    )


def _tile_or_skip(target):
    try:
        return CP.tile_edge(target)
    except TargetContractMissing as exc:
        pytest.skip(f"target package not generated: {exc}")
    except ValueError as exc:
        if "no derived tile geometry" not in str(exc):
            raise
        pytest.skip(f"fixed-array probe needs RTL geometry: {exc}")


def test_the_tile_edge_is_derived_per_target_not_a_literal():
    """The two targets have DIFFERENT mesh edges; one literal cannot be right for both."""
    assert _tile_or_skip("atlas") == 32, "atlas's discovered mesh is 32x32"
    assert _tile_or_skip("gemmini") == 16, "gemmini's discovered mesh is 16x16"


def test_an_unspecified_target_uses_a_software_tile_but_an_unknown_named_target_refuses():
    assert CP.tile_edge(None) == CP._FALLBACK_TILE
    with pytest.raises(TargetContractMissing, match="not-a-target"):
        CP.tile_edge("not-a-target")


def test_fixed_hardware_without_geometry_cannot_emit_probes(monkeypatch):
    from merlin.targetgen import target_experiment
    from merlin.targetgen.rtl import facts

    fixed = {"compute_units": [{"kind": "systolic", "dtypes": ["int8"]}]}
    software = {"compute_units": [{"kind": "simt", "dtypes": ["fp32"]}]}
    hybrid = {"compute_units": [software["compute_units"][0], fixed["compute_units"][0]]}
    contracts = {"fixed": fixed, "software": software, "hybrid": hybrid}
    monkeypatch.setattr(
        target_experiment, "load_capability_manifest", lambda target: SimpleNamespace(contract=contracts[target])
    )
    monkeypatch.setattr(facts, "load_facts", lambda _target: {"facts": {"arrays": []}})

    with pytest.raises(ValueError, match="no derived tile geometry"):
        CP.synthesize({"contraction": _cap()}, target="fixed")
    assert CP.tile_edge("software", operand="f32") == CP._FALLBACK_TILE
    assert CP.tile_edge("hybrid", operand="f32") == CP._FALLBACK_TILE
    with pytest.raises(ValueError, match="no derived tile geometry"):
        CP.synthesize({"contraction": _cap(dtypes=("f32", "int8"))}, target="hybrid")

    hybrid["capabilities"] = {"mesh": {"rows": 32}}
    assert CP.tile_edge("hybrid", operand="f32") == CP._FALLBACK_TILE
    assert CP.tile_edge("hybrid", operand="int8") == 32


def test_every_corner_is_measured_against_the_given_tile():
    """The bug in one line: at tile 32 the old corners (built from 16) never left tile 0."""
    c32 = dict((n, s) for n, s, _ in CP.shape_corners(32))
    assert c32["tile"] == (32, 32, 32)
    assert c32["tile_plus_1"] == (33, 33, 33), "must sit just PAST the real tile, not at 17"
    c16 = dict((n, s) for n, s, _ in CP.shape_corners(16))
    assert c16["tile_plus_1"] == (17, 17, 17)


def test_each_axis_gets_its_own_multi_tile_corner():
    """Per-axis, because K/N generalizing while M does not is a real and common shape of failure."""
    c = dict((n, s) for n, s, _ in CP.shape_corners(32))
    assert c["m_2tiles"] == (64, 32, 32)
    assert c["k_2tiles"] == (32, 64, 32)
    assert c["n_2tiles"] == (32, 32, 64)


def test_additional_dtype_probe_uses_that_datapaths_tile():
    probes = CP.probes_for_family(
        "contraction", _cap(dtypes=("int8", "fp16")), tile=32, dtype_tiles={"int8": 32, "fp16": 8}
    )
    dtype_probe = next(p.descriptor for p in probes if p.name == "contraction.dtype_fp16")
    assert (dtype_probe.m, dtype_probe.k, dtype_probe.n) == (8, 8, 8)


def test_the_probe_set_actually_crosses_a_boundary_on_the_wide_mesh():
    """End to end: synthesize for the wide-mesh target and confirm a >1-tile probe exists per axis."""
    probes = CP.probes_for_family("contraction", _cap(), tile=_tile_or_skip("atlas"))
    by_name = {p.name.rpartition(".")[2]: p.descriptor for p in probes}
    assert by_name["m_2tiles"].m == 64 and by_name["m_2tiles"].k == 32
    assert by_name["k_2tiles"].k == 64 and by_name["k_2tiles"].m == 32
    assert by_name["n_2tiles"].n == 64 and by_name["n_2tiles"].m == 32


def test_synthesize_threads_the_target_through():
    """A caller that forgets ``target=`` gets the fallback, which is why every call site passes it."""
    _tile_or_skip("atlas")
    _tile_or_skip("gemmini")
    wide = {p.name: p.descriptor for p in CP.synthesize({"contraction": _cap(dtypes=("fp8_e4m3",))}, target="atlas")}
    narrow = {p.name: p.descriptor for p in CP.synthesize({"contraction": _cap()}, target="gemmini")}
    assert wide["contraction.m_2tiles"].m == 64
    assert narrow["contraction.m_2tiles"].m == 32


def test_a_unary_family_gets_no_k_or_n_multi_tile_probe():
    """It has no K or N to tile over; emitting those would be an unanswerable probe, not a strict one."""
    names = {p.name.rpartition(".")[2] for p in CP.probes_for_family("normalization", _cap("normalization"), tile=32)}
    assert "m_2tiles" in names
    assert "k_2tiles" not in names and "n_2tiles" not in names


def test_probes_stay_deterministic():
    """Same contract -> same probe set; the measurement has to be comparable across rounds."""
    a = [p.name for p in CP.probes_for_family("contraction", _cap(), tile=32)]
    b = [p.name for p in CP.probes_for_family("contraction", _cap(), tile=32)]
    assert a == b
