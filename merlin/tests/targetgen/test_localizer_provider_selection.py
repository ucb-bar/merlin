"""Advisory geometry must not borrow an unselected support provider's facts."""

import json

import pytest

from merlin.targetgen import divergence_localizer as localizer
from merlin.targetgen.rtl import facts


@pytest.fixture
def selected(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path / "repo"))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.delenv("MERLIN_RTL_FACTS", raising=False)
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    provider = tmp_path / "support"
    contracts = provider / "contracts"
    contracts.mkdir(parents=True)
    (provider / "provider.yaml").write_text(
        "schema: merlin.provider.v1\nid: selected\ntarget: synthetic\nrole: support\ncontract: contracts/custom.yaml\n"
    )
    (contracts / "custom.yaml").write_text(
        "name: synthetic\ncapabilities:\n  tile:\n    cols: 7\nplugin:\n  module: must_never_be_imported\n"
    )
    native = tmp_path / "repo/merlin/targets/synthetic/contracts"
    native.mkdir(parents=True)
    (native / "target_contract.yaml").write_text("name: synthetic\ncapabilities:\n  tile:\n    cols: 99\n")
    cached = facts.rtl_facts_path("synthetic")
    cached.parent.mkdir(parents=True)
    cached.write_text(json.dumps({"facts": {"arrays": [{"name": "stale", "rows": 99, "cols": 99}]}}))
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(provider))
    monkeypatch.setattr(facts, "_dump_facts_for_kind", lambda *args: pytest.fail("advisory extracted RTL"))
    facts.clear_resolution_cache()
    yield contracts
    facts.clear_resolution_cache()


def test_advisory_reads_selected_facts_not_same_name_cache(selected):
    pin = selected / "rtl_facts/facts.json"
    pin.parent.mkdir()
    pin.write_text(
        json.dumps({"facts": {"target": "synthetic", "arrays": [{"name": "selected", "rows": 5, "cols": 5}]}})
    )
    edge, basis = localizer.tile_cols("synthetic")
    assert edge == 5
    assert "selected" in basis


def test_missing_selected_facts_use_selected_custom_contract(selected):
    edge, basis = localizer.tile_cols("synthetic")
    assert edge == 7
    assert "capabilities.tile.cols" in basis


@pytest.mark.parametrize("bad", ["absent", "malformed"])
def test_invalid_selected_support_has_no_borrowed_geometry(selected, bad):
    path = selected / "custom.yaml"
    if bad == "absent":
        path.unlink()
    else:
        path.write_text("invalid: [")
    edge, reason = localizer.tile_cols("synthetic")
    assert edge is None
    assert reason
