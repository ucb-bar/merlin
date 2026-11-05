"""Execution consumers retain target facts while selecting the verified runner configuration."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_bound_chipyard_rocc_backend import _bound
from test_bound_simulation_correspondence import _corresponding

from merlin.compile import model_execution_inputs as MI
from merlin.runtime.backends import base as backends

pytest_plugins = ("test_bound_chipyard_rocc_backend",)


@pytest.fixture
def execution_facts(selection, tmp_path, monkeypatch):
    _engine, receipt, _path = _corresponding(selection, tmp_path)
    contract, facts = selection
    contract["runner"]["chipyard_rocc"]["engines"]["gsim"]["argv"].append("+max-cycles=10000")
    target = tmp_path / "target.fir"
    target.write_text("original target FIRRTL")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    facts["inputs"].update(
        fir_path=str(target), fir_sha256=digest, firrtl_inputs=[{"path": str(target), "sha256": digest}]
    )
    facts["facts"]["source"]["fir_sha256"] = digest
    record = facts["facts"]["simulation_correspondence"]
    record["configs"]["target"]["firrtl"] = {"path": str(target), "sha256": digest}
    record["configs"]["simulation"]["firrtl"]["path"] = receipt["artifacts"]["firrtl"]["path"]
    path = tmp_path / "facts.json"
    path.write_text(json.dumps(facts))
    bound = _bound(selection)
    monkeypatch.setattr(backends, "get_backend", lambda target: bound)
    return path, facts, bound


def test_execution_selection_retains_target_identity_and_reopens_simulation_bytes(execution_facts):
    path, facts, bound = execution_facts
    selected = MI.selected_execution_firrtl(path, target="synthetic")
    assert selected["config"] == bound._block()["config"]
    assert selected["facts_config"] == "target_config"
    assert selected["sha256"] == MI.file_sha256(path)
    assert selected["firrtl_sha256"] != facts["inputs"]["fir_sha256"]
    Path(selected["firrtl"]).write_text("changed simulation")
    with pytest.raises(MI.ModelExecutionInputError, match="byte-verified"):
        MI.selected_execution_firrtl(path, target="synthetic")


@pytest.mark.parametrize("defect", ["missing", "different", "config", "firrtl", "hierarchy", "core_hw", "selection"])
def test_consumers_refuse_invalid_correspondence(execution_facts, defect):
    path, facts, _bound_backend = execution_facts
    correspondence = facts["facts"]["simulation_correspondence"]
    if defect == "missing":
        del facts["facts"]["simulation_correspondence"]
    elif defect == "different":
        correspondence["status"] = "different"
    elif defect == "config":
        correspondence["configs"]["simulation"]["config"] = "foreign_config"
    else:
        correspondence["configs"]["target"][defect]["sha256"] = "0" * 64
    path.write_text(json.dumps(facts))
    with pytest.raises((MI.ModelExecutionInputError, ValueError)):
        MI.selected_execution_firrtl(path, target="synthetic")


def test_model_check_selects_the_runner_board_and_receipt(execution_facts, monkeypatch):
    from merlin.model_check import engines

    path, _facts, bound = execution_facts
    configs = []

    def board(target, config, **kw):
        configs.append(config)
        return SimpleNamespace(record=lambda: {"config": config})

    monkeypatch.setattr(engines, "board_for", board)
    engine = engines.GsimEngine("synthetic", path)
    assert configs == [bound._block()["config"]]
    assert engine.identity()["facts_config"] == "target_config"
    assert engine.identity()["firrtl_sha256"] == engine.citation["receipt"]["firrtl_sha256"]


def test_owned_host_selection_uses_corresponding_execution_facts(execution_facts):
    from merlin.targetgen.owned_host_model_execution import _selected_facts

    path, _facts, bound = execution_facts
    selected = _selected_facts(path, target="synthetic")
    assert selected["config"] == bound._block()["config"]
    assert selected["facts_config"] == "target_config"
