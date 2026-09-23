"""Coefficients and vocabulary belong to the selected support, never its namesake."""

import json

import pytest

from merlin.perf.linear_cost import CostModelUnavailable, LinearCostModel, cost_model_artifact


@pytest.fixture
def selected(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path / "repo"))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    native = tmp_path / "repo/merlin/targets/synthetic/cost_model"
    native.mkdir(parents=True)
    (native / "coefficients.json").write_text(json.dumps({"const": 99, "coeff": {"native": 99}}))
    (native / "vocabulary.json").write_text(json.dumps({"events": ["native"]}))
    provider = tmp_path / "provider"
    (provider / "contracts").mkdir(parents=True)
    (provider / "contracts/target_contract.yaml").write_text("name: synthetic\n")
    model = provider / "cost_model"
    model.mkdir()
    (model / "coefficients.json").write_text(json.dumps({"const": 2, "coeff": {"selected": 3, "other": 5}}))
    (model / "vocabulary.json").write_text(json.dumps({"events": ["selected"]}))
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(provider))
    return model


def test_selected_coefficients_and_vocabulary_price_together(selected):
    assert cost_model_artifact("synthetic") == selected / "coefficients.json"
    model = LinearCostModel.for_target("synthetic")
    assert model.priced_events() == ("selected",)
    assert model.predict({"selected": 2, "native": 100}) == 8


def test_selected_absent_coefficient_is_unavailable_not_native(selected):
    (selected / "coefficients.json").unlink()
    assert cost_model_artifact("synthetic") is None
    with pytest.raises(CostModelUnavailable):
        LinearCostModel.for_target("synthetic")


def test_selected_absent_vocabulary_uses_own_coefficient_keys(selected):
    (selected / "vocabulary.json").unlink()
    assert LinearCostModel.for_target("synthetic").priced_events() == ("other", "selected")


@pytest.mark.parametrize("member", ["coefficients.json", "vocabulary.json"])
def test_selected_malformed_data_never_uses_native(selected, member):
    (selected / member).write_text("{")
    with pytest.raises(ValueError):
        LinearCostModel.for_target("synthetic")


@pytest.mark.parametrize("member", ["coefficients.json", "vocabulary.json"])
def test_selected_resources_must_not_escape_provider(selected, tmp_path, member):
    outside = tmp_path / member
    outside.write_bytes((selected / member).read_bytes())
    (selected / member).unlink()
    (selected / member).symlink_to(outside)
    with pytest.raises(ValueError):
        LinearCostModel.for_target("synthetic")


def test_legacy_native_calibration_remains_available(selected, monkeypatch):
    monkeypatch.delenv("MERLIN_TARGET_PATH")
    assert LinearCostModel.for_target("synthetic").predict({"native": 1}) == 198


def test_external_only_model_does_not_need_native_tree(selected, monkeypatch, tmp_path):
    monkeypatch.setenv("MERLIN_TARGETS_DIR", str(tmp_path / "absent-native"))
    assert LinearCostModel.for_target("synthetic").predict({"selected": 2}) == 8
