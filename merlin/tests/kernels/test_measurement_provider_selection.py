"""A selected support package cannot inherit another provider's measurement authority.

A same-named reference contract is the semantic view a selected support provider must implement,
so a provider whose measurement block differs from that namesake is refused, never merged with it.
"""

import pytest
import yaml

from merlin.kernels.measurement import authority_for


@pytest.fixture
def selected(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path / "repo"))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    native = tmp_path / "repo/merlin/targets/synthetic/contracts"
    native.mkdir(parents=True)
    (native / "target_contract.yaml").write_text("name: synthetic\nmeasurement: {cycles_from: native}\n")
    provider = tmp_path / "provider"
    (provider / "contracts").mkdir(parents=True)
    (provider / "provider.yaml").write_text(
        "schema: merlin.provider.v1\nid: selected\ntarget: synthetic\nrole: support\n"
    )
    contract = provider / "contracts/target_contract.yaml"
    contract.write_text("name: synthetic\nmeasurement: {cycles_from: selected}\n")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(provider))
    return contract


def _without_namesake(monkeypatch, tmp_path):
    monkeypatch.setenv("MERLIN_TARGETS_DIR", str(tmp_path / "absent-native"))


def test_selected_authority_beats_native_even_without_compute_family(selected, tmp_path):
    # The namesake must agree with the selected support; the authority is still read through
    # the selected provider, never through the native tracked-contract fallback.
    native = tmp_path / "repo/merlin/targets/synthetic/contracts/target_contract.yaml"
    native.write_text(selected.read_text())
    authority = authority_for("synthetic")
    assert authority.declared and authority.cycles_from == "selected"
    assert authority.source == "selected_support_contract"


def test_selected_authority_drifting_from_namesake_is_refused_not_inherited(selected):
    authority = authority_for("synthetic")
    assert not authority.declared and authority.cycles_from is None
    assert "drifts" in authority.lookup_error and "measurement" in authority.lookup_error


def test_absent_selected_measurement_is_undeclared_not_native(selected, monkeypatch, tmp_path):
    selected.write_text("name: synthetic\n")
    authority = authority_for("synthetic")
    assert not authority.declared and authority.cycles_from is None
    assert "drifts" in authority.lookup_error  # the namesake's timer is refused, not borrowed
    _without_namesake(monkeypatch, tmp_path)
    authority = authority_for("synthetic")
    assert not authority.declared
    assert authority.lookup_error is None


@pytest.mark.parametrize("invalid", ["absent", "yaml", "block"])
def test_bad_selected_contract_reports_lookup_failure_without_fallback(selected, invalid):
    if invalid == "absent":
        selected.unlink()
    else:
        selected.write_text("broken: [" if invalid == "yaml" else "name: synthetic\nmeasurement: [bad]\n")
    authority = authority_for("synthetic")
    assert not authority.declared and authority.lookup_error
    assert authority.cycles_from is None


def test_descriptor_remains_authoritative_even_with_selected_support(selected):
    authority = authority_for("synthetic", {"measurement": {"cycles_from": "descriptor"}})
    assert authority.cycles_from == "descriptor" and authority.source == "descriptor"


def test_legacy_tracked_declaration_still_available(selected, monkeypatch):
    monkeypatch.setenv("MERLIN_TARGET_PATH", "")
    authority = authority_for("synthetic")
    assert authority.cycles_from == "native" and authority.source == "tracked_contract"


def test_selected_custom_contract_location(selected, monkeypatch, tmp_path):
    _without_namesake(monkeypatch, tmp_path)
    custom = selected.with_name("custom.yaml")
    selected.rename(custom)
    metadata = selected.parent.parent / "provider.yaml"
    doc = yaml.safe_load(metadata.read_text())
    doc["contract"] = "contracts/custom.yaml"
    metadata.write_text(yaml.safe_dump(doc))
    assert authority_for("synthetic").cycles_from == "selected"


def test_external_only_authority_does_not_need_native_tree(selected, monkeypatch, tmp_path):
    _without_namesake(monkeypatch, tmp_path)
    assert authority_for("synthetic").cycles_from == "selected"
