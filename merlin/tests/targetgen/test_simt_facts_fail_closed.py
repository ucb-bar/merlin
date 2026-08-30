"""A fact the extractor could not read must be recorded as ABSENT, never substituted.

The SIMT introspect reads its geometry from the target's ``[muon]`` config. ``_read`` swallows OSError
and returns ``""``, so an unreadable config used to leave the parse empty and the four geometry values
fell through to the literals 16/8/2/256 -- which were then emitted under the evidence string
``"config_muon.toml: num_lanes=16, ..."``, naming a file that was never opened. The artifact committed
for this target is in exactly that state (``inputs.config_toml`` is the ``/path/to/...`` placeholder),
so its ``simt`` and ``fp_datapath`` groups are code constants wearing a derived label. They match this
particular cyclotron config, which is why nothing caught it, and they would be silently wrong for any
other configuration.

This is the repo's cardinal rule at its sharpest: when a value cannot be derived, fail closed and
surface it -- never substitute a baked default. ``address_map`` and ``console`` in the same builder
already did that; these tests hold the geometry to it, and prove by MUTATION that the numbers come
from the config rather than from the source.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def introspect():
    base = pytest.importorskip("merlin.runtime.backends.base")
    try:
        return base.get_backend("muon").muon_introspect
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"muon backend package not loadable here: {type(e).__name__}: {e}")


def _config(tmp_path, **kw):
    body = "\n".join(f"{k} = {v}" for k, v in kw.items())
    p = tmp_path / "config_muon.toml"
    p.write_text(f"[muon]\n{body}\n")
    return p


def test_an_unreadable_config_yields_UNKNOWN_not_a_default(introspect, monkeypatch, tmp_path):
    monkeypatch.setenv("MERLIN_MUON_CONFIG", str(tmp_path / "does_not_exist.toml"))
    f = introspect.build_facts()
    simt = f["facts"]["simt"]
    for k in ("lanes_per_warp", "warps_per_core", "cores", "threads_per_core"):
        assert simt[k] is None, f"{k} was substituted ({simt[k]!r}) instead of recorded absent"
    assert f["inputs"]["config_read"] is False


def test_the_evidence_does_not_name_a_file_that_was_never_read(introspect, monkeypatch, tmp_path):
    monkeypatch.setenv("MERLIN_MUON_CONFIG", str(tmp_path / "does_not_exist.toml"))
    ev = introspect.build_facts()["facts"]["simt"]["evidence"]
    assert "UNKNOWN" in ev, f"evidence should say the geometry is unknown, got: {ev}"
    assert "num_lanes=16" not in ev, "evidence still asserts config values that were never read"


def test_the_derived_peak_is_absent_when_the_geometry_it_multiplies_is_absent(
        introspect, monkeypatch, tmp_path):
    monkeypatch.setenv("MERLIN_MUON_CONFIG", str(tmp_path / "does_not_exist.toml"))
    fp = introspect.build_facts()["facts"]["fp_datapath"]
    assert fp["peak_flops_per_cycle"] is None and fp["peak_gflops"] is None
    assert "UNKNOWN" in fp["evidence"]


def test_the_geometry_comes_from_the_config_and_not_from_the_source(
        introspect, monkeypatch, tmp_path):
    """Mutation test. These values are deliberately NOT the old literals (16/8/2/256): if any of them
    were still hardcoded, the assertion below reads the source constant instead of the config."""
    monkeypatch.setenv("MERLIN_MUON_CONFIG", str(_config(
        tmp_path, num_lanes=4, num_warps=2, num_cores=1, num_regs=64)))
    f = introspect.build_facts()
    simt = f["facts"]["simt"]
    assert (simt["lanes_per_warp"], simt["warps_per_core"], simt["cores"]) == (4, 2, 1)
    assert simt["threads_per_core"] == 8
    assert f["facts"]["registers"]["config"] == 64
    assert f["facts"]["fp_datapath"]["peak_flops_per_cycle"] == 1 * 4 * 2
    assert f["inputs"]["config_read"] is True


def test_a_malformed_config_is_also_absent_rather_than_defaulted(introspect, monkeypatch, tmp_path):
    p = tmp_path / "config_muon.toml"
    p.write_text("[muon\nnum_lanes = ")            # not valid TOML
    monkeypatch.setenv("MERLIN_MUON_CONFIG", str(p))
    simt = introspect.build_facts()["facts"]["simt"]
    assert simt["lanes_per_warp"] is None and simt["cores"] is None
