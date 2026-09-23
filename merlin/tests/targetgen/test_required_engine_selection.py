"""Required engine selection is a constraint, not a cost preference."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from merlin.common.paths import python_import_roots
from merlin.compile import mesh_backend
from merlin.runtime.backends import base
from merlin.targetgen import oracle_policy as policy
from merlin.targetgen import rtl_engine_policy as engines


@pytest.mark.parametrize("required", ["gsim", "  gsim\t", "verilator", "vcs"])
def test_required_engine_only_probes_requested_engine(monkeypatch, required):
    calls = []
    selected = required.strip()
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", required)
    monkeypatch.setattr(
        base, "get_backend", lambda target: SimpleNamespace(available=lambda engine: calls.append(engine) or True)
    )
    result = policy.chipyard_l3_selection("synthetic")
    assert calls == [selected]
    assert result["engine"] == result["required_engine"] == selected
    assert result["selection_constraint"] == "MERLIN_REQUIRED_RTL_ENGINE"


@pytest.mark.parametrize("required", [None, "", " \t"])
def test_unpinned_keeps_cost_priority(monkeypatch, required):
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    if required is not None:
        monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", required)
    calls = []
    monkeypatch.setattr(
        base, "get_backend", lambda target: SimpleNamespace(available=lambda engine: calls.append(engine) or True)
    )
    result = policy.chipyard_l3_selection("synthetic")
    assert result["engine"] == "vcs"
    assert "required_engine" not in result
    assert calls == ["vcs"]


@pytest.mark.parametrize("required", ["unknown", "GSIM"])
def test_unknown_pin_refuses_without_probing(monkeypatch, required):
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", required)
    monkeypatch.setattr(
        base,
        "get_backend",
        lambda target: SimpleNamespace(available=lambda engine: pytest.fail("unknown pin probed a substitute")),
    )
    with pytest.raises(RuntimeError, match="not registered"):
        policy.chipyard_l3_selection("synthetic")


@pytest.mark.parametrize("status", ["unavailable", "raises", "unexplained"])
def test_pinned_detailed_probe_refuses_without_fallback(monkeypatch, status):
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    calls = []

    def probe():
        calls.append("gsim")
        if status == "raises":
            raise RuntimeError("receipt rejected")
        return (True, "") if status == "unexplained" else (False, "receipt rejected")

    monkeypatch.setattr(
        base,
        "get_backend",
        lambda target: SimpleNamespace(gsim_status=probe, available=lambda engine: pytest.fail("substitute probed")),
    )
    expected = engines.UnrecordedSelection if status == "unexplained" else engines.NoEngineAvailable
    with pytest.raises(expected):
        policy.chipyard_l3_selection("synthetic")
    assert calls == ["gsim"]


def test_mesh_and_evaluator_bind_same_pin_and_keep_captured_choice(monkeypatch, tmp_path):
    from merlin.targetgen import capsule_runner as runner

    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    monkeypatch.delenv("MERLIN_MESH_SIM", raising=False)
    monkeypatch.setattr(
        base,
        "get_backend",
        lambda target: SimpleNamespace(
            available=lambda engine: True, gsim_status=lambda: (True, "validated fixture receipt")
        ),
    )
    assert mesh_backend._resolve_oot_mesh_simulator("synthetic") == "gsim"
    adapter = runner._sim_engine_adapters("chipyard", "synthetic")["L3"]
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "vcs")
    calls = []
    monkeypatch.setattr(
        runner.oot_compile, "run_on_oracle", lambda *args, **kwargs: calls.append(kwargs) or {"oracle": {}}
    )
    result = adapter({}, "ir", tmp_path, 1)
    assert calls[0]["simulator"] == "gsim"
    assert result["oracle"]["selection"]["required_engine"] == "gsim"
    assert result["oracle"]["selection"]["reason"] == "validated fixture receipt"


def test_unavailable_pin_has_no_evaluator_l3_or_mesh_substitute(monkeypatch):
    from merlin.targetgen import capsule_runner as runner

    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    monkeypatch.delenv("MERLIN_MESH_SIM", raising=False)
    monkeypatch.setattr(base, "get_backend", lambda target: SimpleNamespace(available=lambda engine: engine != "gsim"))
    assert set(runner._sim_engine_adapters("chipyard", "synthetic")) == {"L2"}
    with pytest.raises(engines.NoEngineAvailable):
        mesh_backend._resolve_oot_mesh_simulator("synthetic")


def test_required_selection_runs_without_optional_evaluator():
    code = """
import importlib.abc, sys
from types import SimpleNamespace
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = fullname in {'aet', 'merlin.targetgen.capsule_runner'}
        if blocked or fullname.startswith(('aet.', 'merlin_experiments')):
            raise AssertionError('optional evaluator imported: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from merlin.runtime.backends import base
from merlin.targetgen import oracle_policy
base.get_backend = lambda target: SimpleNamespace(available=lambda engine: True)
assert oracle_policy.chipyard_l3_selection('synthetic')['required_engine'] == 'gsim'
"""
    env = dict(
        os.environ,
        MERLIN_REQUIRED_RTL_ENGINE="gsim",
        PYTHONPATH=os.pathsep.join(str(path) for path in python_import_roots()),
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True)
