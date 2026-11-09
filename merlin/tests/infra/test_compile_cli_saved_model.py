"""The installed front door selects a saved bare-metal model without capsule fallback."""

import json

import pytest

from merlin import model as baremetal_model
from merlin.compile_cli import main
from merlin.targetgen import target_registry


def test_explicit_saved_model_route_and_input_guards(monkeypatch, capsys):
    monkeypatch.setattr(target_registry, "all_targets", lambda: ["test_device"])
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return {"status": "compiled", "execution_route": "host_baseline"}

    monkeypatch.setattr(baremetal_model, "compile_saved_model", build)
    args = [
        "--model-build",
        "--target",
        "test_device",
        "--capture-bundle",
        "/unused/capture",
        "--package",
        "/unused/host",
        "--board-catalog",
        "/unused/boards.yaml",
        "--board",
        "selected",
        "--host-dts",
        "/unused/host.dts",
        "--output",
        "/unused/out",
        "--arena-mb",
        "32",
        "--json",
    ]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["execution_route"] == "host_baseline"
    assert calls[0]["capture"] == "/unused/capture"
    assert calls[0]["run"] == "none"
    assert calls[0]["reference_file"] is None

    for extra in (["--run", "gsim"], ["--model-preflight"], ["--harts", "2"]):
        with pytest.raises(SystemExit) as error:
            main(args + extra)
        assert error.value.code == 2
    assert len(calls) == 1
