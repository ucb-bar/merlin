"""Data-bound selection precedes executable discovery and stays context-local."""

from types import ModuleType

import pytest

from merlin.runtime.backends import base
from merlin.targetgen.rtl import facts as rtl_facts
from merlin.targetgen.target_registry import observed_contract


def test_context_selected_contract_and_facts_precede_plugin_discovery(monkeypatch):
    import sys

    owner = ModuleType("merlin.runtime.backends.chipyard_rocc")
    selected = []

    def bind(**kwargs):
        selected.append(kwargs)
        return kwargs

    owner.bind = bind
    monkeypatch.setitem(sys.modules, owner.__name__, owner)
    monkeypatch.setattr(base, "_ensure_discovered", lambda: pytest.fail("data selection must not discover plugins"))
    monkeypatch.setattr(rtl_facts, "ensure_facts", lambda *args, **kwargs: pytest.fail("must not build RTL"))
    name = "synthetic_data_device"
    for token in ("first", "second"):
        contract = {"name": name, "runner": {"backend": "chipyard_rocc"}, "token": token}
        facts = {"facts": {"token": token}}
        with observed_contract(name, contract), rtl_facts.observed_facts(name, facts):
            assert base.get_backend(name)["contract"] == contract
    assert [item["facts"]["facts"]["token"] for item in selected] == ["first", "second"]


@pytest.mark.parametrize("runner", [{"backend": "unknown_family"}, {"backend": None}, []])
def test_explicit_invalid_generic_selection_never_falls_back(monkeypatch, runner):
    monkeypatch.setattr(
        base, "_ensure_discovered", lambda: pytest.fail("invalid selection must refuse before discovery")
    )
    name = "synthetic_data_device"
    with observed_contract(name, {"name": name, "runner": runner}):
        with pytest.raises(ValueError):
            base.get_backend(name)


def test_read_only_missing_facts_refuse_without_regeneration(monkeypatch):
    monkeypatch.setattr(rtl_facts, "find_facts", lambda *args, **kwargs: None)
    monkeypatch.setattr(rtl_facts, "ensure_facts", lambda *args, **kwargs: pytest.fail("must not build RTL"))
    with pytest.raises(FileNotFoundError, match="read-only"):
        rtl_facts.load_facts("synthetic_absent_facts", regenerate=False)


def test_generic_route_masks_all_executable_provider_hooks(monkeypatch):
    from merlin.targetgen import plugins, target_registry

    name = "synthetic_data_device"
    info = target_registry.resolve(name)
    monkeypatch.setattr(target_registry, "explicit_targets", lambda: [name])
    monkeypatch.setattr(plugins, "resolve_support", lambda target: info)
    monkeypatch.setattr(
        target_registry.TargetInfo,
        "_load_provider_contract",
        lambda info: pytest.fail("generic data cannot read provider implementations"),
    )
    contract = {"name": name, "runner": {"backend": "chipyard_rocc"}}
    with observed_contract(name, contract):
        assert info.plugin() == {}
        for key in ("backend", "dialect", "sim_oracle", "sim_oracle_metadata"):
            assert base._oot_plugin_modules(key) == []
    contract["plugin"] = {"sim_oracle": "compiler_bearing.py"}
    with observed_contract(name, contract):
        with pytest.raises(ValueError, match="executable provider hooks"):
            info.plugin()
        with pytest.raises(ValueError, match="executable provider hooks"):
            base.get_backend(name)
    with observed_contract(name, {"name": name, "runner": []}):
        with pytest.raises(ValueError, match="must be a mapping"):
            info.plugin()
        assert base._oot_plugin_modules("sim_oracle") == []
