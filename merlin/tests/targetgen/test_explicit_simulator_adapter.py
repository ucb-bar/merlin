"""Explicit simulator construction must not select another target or engine."""

from types import SimpleNamespace

import pytest

from merlin.runtime.backends import base
from merlin.targetgen import capsule_runner as CR


def test_explicit_constructor_has_no_target_default_or_legacy_alias():
    assert not hasattr(CR, "default_adapters")
    assert not hasattr(CR, "_spike_verilator_adapter")
    with pytest.raises(TypeError):
        CR.simulator_adapter("spike")


@pytest.mark.parametrize("target", ["synthetic_alpha", "synthetic_beta"])
@pytest.mark.parametrize("sim", ["spike", "verilator"])
def test_explicit_target_engine_and_selection_are_preserved(monkeypatch, tmp_path, target, sim):
    calls = []

    def get_backend(requested):
        assert requested == target
        return SimpleNamespace(available=lambda selected: selected == sim)

    monkeypatch.setattr(base, "get_backend", get_backend)

    def run(cb, llvm, **kwargs):
        calls.append((cb, llvm, kwargs))
        return {"oracle": {"kind": "fixture"}, "outputs": {"result": 7}}

    monkeypatch.setattr(CR.oot_compile, "run_on_oracle", run)
    selection = {"engine": sim, "reason": "explicit experiment policy"}
    result = CR.simulator_adapter(sim, target, selection=selection)({"fixture": True}, "ir", tmp_path, 13)
    assert calls == [
        ({"fixture": True}, "ir", {"simulator": sim, "target": target, "workdir": tmp_path, "timeout": 13})
    ]
    assert result["outputs"] == {"result": 7}
    assert result["oracle"]["selection"] == selection
    assert result["oracle"]["selection"] is not selection


def test_explicit_unavailable_engine_never_substitutes(monkeypatch, tmp_path):
    monkeypatch.setattr(base, "get_backend", lambda target: SimpleNamespace(available=lambda sim: False))
    monkeypatch.setattr(CR.oot_compile, "run_on_oracle", lambda *a, **kw: pytest.fail("unavailable engine executed"))
    with pytest.raises(CR.OracleUnavailable, match="verilator not available"):
        CR.simulator_adapter("verilator", "synthetic")({}, "ir", tmp_path, 1)
