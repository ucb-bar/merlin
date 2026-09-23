"""Backend tool configuration cannot replace admitted Python import selection."""

import hashlib
import socket
import subprocess
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import checkpoint_admission as AD

from merlin.runtime.backends import base

KEYS = ("PYTHONPATH", "PYTHONSAFEPATH", "PYTHONHOME", "PYTHONNOUSERSITE", "PYTHONUSERBASE")


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("environment construction must not launch or listen")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket.socket, "bind", forbidden)
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    pins = {}
    for engine in ("gsim", "verilator"):
        path = tmp_path / engine
        path.write_bytes(b"synthetic binary identity, never executed")
        pins[engine + "_binary"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    config = SimpleNamespace(
        descriptor=tmp_path / "descriptor.yaml", gsim_max_cycles=123, telemetry_price_table=tmp_path / "prices.json"
    )
    certificate = SimpleNamespace(target="fixture", pins=pins)
    return config, certificate


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("mutation", ["add", "remove", "replace"])
def test_backend_cannot_change_import_selection(inputs, monkeypatch, key, mutation):
    if mutation != "add":
        monkeypatch.setenv(key, "admitted")

    def configure(*, environment, **kwargs):
        # Exercise in-place callback changes, not just returned replacement maps.
        if mutation == "remove":
            environment.pop(key)
        else:
            environment[key] = "different"
        return environment

    monkeypatch.setattr(base, "get_backend", lambda target: SimpleNamespace(runtime_environment=configure))
    with pytest.raises(AD.ExperimentError, match="Python source selection"):
        AD.child_environment(*inputs)


def test_same_import_selection_retains_runtime_overrides_and_pipeline_pins(inputs, monkeypatch):
    selected = {key: "selected-" + key for key in KEYS}
    for key, value in selected.items():
        monkeypatch.setenv(key, value)

    def configure(*, binaries, gsim_max_cycles, environment):
        assert set(binaries) == {"gsim", "verilator"}
        assert gsim_max_cycles == 123
        return {**environment, "PATH": "selected-tools", "RUNTIME_FIXTURE": "configured"}

    monkeypatch.setattr(base, "get_backend", lambda target: SimpleNamespace(runtime_environment=configure))
    result = AD.child_environment(*inputs)
    assert {key: result[key] for key in KEYS} == selected
    assert result["PATH"] == "selected-tools"
    assert result["RUNTIME_FIXTURE"] == "configured"
    assert result["MERLIN_REQUIRED_RTL_ENGINE"] == "gsim"
    assert result["MERLIN_CACHE_STATE"] == AD.MEASUREMENT_CACHE_CONDITION
    assert result["AET_PRICE_TABLE"] == str(inputs[0].telemetry_price_table)


def test_legacy_absent_selection_stays_absent(inputs, monkeypatch):
    monkeypatch.setattr(
        base, "get_backend", lambda target: SimpleNamespace(runtime_environment=lambda **kwargs: kwargs["environment"])
    )
    result = AD.child_environment(*inputs)
    assert all(key not in result for key in KEYS)
