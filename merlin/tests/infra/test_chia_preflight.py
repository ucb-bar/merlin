"""Readiness uses the selected interpreter's API, not merely its existence."""

import importlib.util
from types import SimpleNamespace

import pytest

from merlin.benchharness import chia_bridge
from merlin.common.paths import repo_root


@pytest.fixture
def preflight():
    spec = importlib.util.spec_from_file_location(
        "chia_repro_preflight", repo_root() / "build_tools/scripts/check_repro_env.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("returncode", [0, 1])
def test_chia_preflight_imports_selected_environment(preflight, monkeypatch, returncode):
    monkeypatch.setattr(chia_bridge, "chia_python", lambda: "/selected/venv/bin/python")
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=returncode, stderr="public API missing")

    monkeypatch.setattr(preflight.subprocess, "run", run)
    status, detail = preflight._probe("chia")
    assert status == ("available" if returncode == 0 else "unavailable")
    assert "/selected/venv/bin/python" in detail
    argv, kwargs = calls[0]
    assert argv[:3] == ["/selected/venv/bin/python", "-I", "-c"]
    assert "require_chia()" in argv[3]
    assert "ray.init" not in argv[3]
    assert kwargs["timeout"] == 20


def test_invalid_interpreter_does_not_launch_probe(preflight, monkeypatch):
    def invalid():
        raise RuntimeError("invalid explicit interpreter")

    def forbidden(*args, **kwargs):
        pytest.fail("invalid interpreter cannot launch a subprocess")

    monkeypatch.setattr(chia_bridge, "chia_python", invalid)
    monkeypatch.setattr(preflight.subprocess, "run", forbidden)
    status, detail = preflight._probe("chia")
    assert status == "error"
    assert "invalid explicit interpreter" in detail


def test_interpreter_report_reuses_bridge_resolvers(preflight, monkeypatch):
    monkeypatch.setattr(chia_bridge, "chia_python", lambda: "/chia/bin/python")
    monkeypatch.setattr(chia_bridge, "driver_python", lambda: "/driver/bin/python")
    result = preflight._interpreters()
    assert result["driver (MERLIN_EXPERIMENT_PYTHON)"] == "/driver/bin/python"
    assert result["chia (MERLIN_CHIA_PYTHON)"] == "/chia/bin/python"
