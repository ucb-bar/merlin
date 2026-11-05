"""Coarse API transport and expiry boundaries; these fake ELFs never execute."""

import json
from pathlib import Path

import pytest
from test_baremetal_saved_model import _console, _fake_build, _fixture
from test_source_dense_lowering import selection

from merlin.common.digest import sha256_file
from merlin.compile import baremetal_model as BM


def selected(tmp_path):
    directory = tmp_path / "capabilities"
    directory.mkdir()
    config, _, _ = selection(directory)
    return config


def fake_preparation(bundle, work, **kwargs):
    built = _fake_build(bundle, work, **kwargs)
    if kwargs.get("shared_source_lowerings"):
        directory = work / "prepared_model_transform"
        directory.mkdir()
        record = dict(schema="merlin.prepared_model_transform.v1")
        for role in ("source", "input_snapshot", "selected"):
            path = directory / (role + ".mlir")
            path.write_bytes((bundle / "model.mlir").read_bytes())
            record[role + "_path"] = str(path)
            record[role + "_sha256"] = sha256_file(path)
        (directory / "prepared_model_transform.json").write_text(json.dumps(record))
    return built


def test_shared_selections_are_forwarded_by_identity_only_when_selected(tmp_path, monkeypatch):
    inputs, out = _fixture(tmp_path, monkeypatch)
    config = selected(tmp_path)
    choices = (config,)
    calls = []

    def build(bundle, work, **kwargs):
        calls.append(kwargs)
        return fake_preparation(bundle, work, **kwargs)

    monkeypatch.setattr(BM.spike_model, "build", build)
    plain = BM.compile_saved_model(**inputs, output=out / "plain", run="none")
    tuned = BM.compile_saved_model(**inputs, output=out / "selected", run="none", shared_source_lowerings=choices)
    assert "shared_source_lowerings" not in calls[0]
    assert calls[1]["shared_source_lowerings"] is choices
    assert calls[1] == {**calls[0], "shared_source_lowerings": choices}
    assert "shared_source_lowerings" not in plain["inputs"]
    assert tuned["inputs"]["shared_source_lowerings"] == ["SourceDenseLowering"]
    assert "shared_source_preparation" not in plain["output"]
    record = tuned["output"]["shared_source_preparation"]
    assert record["sha256"] == sha256_file(Path(record["path"]))
    assert "host_provider" not in tuned["output"]


@pytest.mark.parametrize("invalid", [None, [], [object()], (object(),), "dense"])
def test_untyped_shared_selections_refuse_before_output_or_builder(tmp_path, monkeypatch, invalid):
    inputs, out = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(BM.spike_model, "build", lambda *a, **kw: pytest.fail("invalid selection reached builder"))
    with pytest.raises(BM.BaremetalModelError, match="typed selections|explicit dense"):
        BM.compile_saved_model(**inputs, output=out / "invalid", run="none", shared_source_lowerings=invalid)
    assert not (out / "invalid").exists()


def test_duplicate_shared_families_refuse_before_output(tmp_path, monkeypatch):
    inputs, out = _fixture(tmp_path, monkeypatch)
    config = selected(tmp_path)
    with pytest.raises(BM.BaremetalModelError, match="duplicate"):
        BM.compile_saved_model(**inputs, output=out / "duplicate", run="none", shared_source_lowerings=(config, config))
    assert not (out / "duplicate").exists()


@pytest.mark.parametrize("mutation", ["facts", "software", "receipt", "source", "input_snapshot", "selected"])
def test_execution_cannot_publish_success_with_changed_source_selection(tmp_path, monkeypatch, mutation):
    inputs, out = _fixture(tmp_path, monkeypatch)
    config = selected(tmp_path)
    monkeypatch.setattr(BM.spike_model, "build", fake_preparation)
    monkeypatch.setattr(BM.spike_backend, "available", lambda: True)
    monkeypatch.setattr(BM, "_spike_extension_backend", lambda target: None)

    def run(*args, **kwargs):
        if mutation in ("facts", "software"):
            contract = config.selected_contracts
            path = contract.facts_path if mutation == "facts" else contract.software_path
        else:
            name = "prepared_model_transform.json" if mutation == "receipt" else mutation + ".mlir"
            path = out / "execution" / "build" / "prepared_model_transform" / name
        path.write_bytes(path.read_bytes() + b"\nchanged after compilation\n")
        return dict(console=_console())

    monkeypatch.setattr(BM.spike_model, "run", run)
    with pytest.raises(ValueError, match="changed"):
        BM.compile_saved_model(
            **inputs, output=out / "execution", run="spike", reference_file="golden.npy",
            shared_source_lowerings=(config,),
        )
    receipt = json.loads((out / "execution" / "baremetal_model.json").read_text())
    assert receipt["status"] == "failed"


def test_selected_build_must_produce_preparation_custody(tmp_path, monkeypatch):
    inputs, out = _fixture(tmp_path, monkeypatch)
    config = selected(tmp_path)
    monkeypatch.setattr(BM.spike_model, "build", _fake_build)
    with pytest.raises(BM.BaremetalModelError, match="preparation receipt"):
        BM.compile_saved_model(**inputs, output=out / "missing", run="none", shared_source_lowerings=(config,))
