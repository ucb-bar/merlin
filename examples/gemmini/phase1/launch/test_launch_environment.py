"""Operator preflight must inspect the runtime selected for native execution."""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.launch_config import Prepend


@pytest.fixture
def launcher():
    source = Path(__file__).with_name("launch_phase1.py")
    spec = importlib.util.spec_from_file_location("operator_environment_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selection():
    return SimpleNamespace(
        environment={
            "MERLIN_SPIKE": "/selected/fire-aligned/spike",
            "PATH": Prepend("/selected/bin"),
            "LD_LIBRARY_PATH": Prepend("/selected/lib"),
            "OVERRIDDEN": "runtime",
        },
        launch_environment={"OVERRIDDEN": "launcher", "MERLIN_ENGINE": "spike"},
    )


def test_prepare_selects_runtime_and_restores_ambient_environment(launcher, monkeypatch, tmp_path):
    monkeypatch.setenv("MERLIN_SPIKE", "/unselected/spike")
    monkeypatch.setenv("MERLIN_UNDECLARED", "must-not-leak")
    monkeypatch.setenv("PATH", "/ambient/bin")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    before = dict(os.environ)
    launch = selection()

    def inspect(actual, prep):
        assert actual is launch and prep == tmp_path
        assert os.environ["MERLIN_SPIKE"] == "/selected/fire-aligned/spike"
        assert os.environ["MERLIN_ENGINE"] == "spike"
        assert "MERLIN_UNDECLARED" not in os.environ
        assert os.environ["PATH"] == "/selected/bin:/ambient/bin"
        assert os.environ["LD_LIBRARY_PATH"] == "/selected/lib"
        assert os.environ["OVERRIDDEN"] == "launcher"
        assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
        return {"status": "ready"}

    monkeypatch.setattr(launcher, "_prepare", inspect)
    assert launcher.prepare(launch, tmp_path) == {"status": "ready"}
    assert dict(os.environ) == before


def test_failed_prepare_restores_environment(launcher, monkeypatch, tmp_path):
    before = dict(os.environ)

    def refuse(*_):
        assert os.environ["MERLIN_SPIKE"] == "/selected/fire-aligned/spike"
        os.environ["MUTATED_DURING_CHECK"] = "discard"
        raise ValueError("readiness refused")

    monkeypatch.setattr(launcher, "_prepare", refuse)
    with pytest.raises(ValueError, match="readiness refused"):
        launcher.prepare(selection(), tmp_path)
    assert dict(os.environ) == before


def test_receipt_builder_can_reuse_the_same_scoped_runtime(launcher, monkeypatch):
    monkeypatch.setenv("PATH", "/ambient/bin")
    before = dict(os.environ)
    with launcher.configured_environment(selection()):
        assert os.environ["PATH"] == "/selected/bin:/ambient/bin"
        assert os.environ["MERLIN_ENGINE"] == "spike"
    assert dict(os.environ) == before
