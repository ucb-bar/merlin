"""The SIMT RTL introspect is registered from contract DATA, and mlc_bridge names no SIMT target."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from merlin.runtime.backends import base
from merlin.targetgen import plugins, target_registry
from merlin.targetgen.rtl import mlc_bridge as mb


def test_simt_introspect_is_a_recognised_consumed_plugin_key():
    spec = plugins.PLUGIN_KEYS["simt_introspect"]
    assert spec.consumed is True and spec.expects == "attr"


def test_an_empty_registry_is_filled_from_the_declaring_contract(monkeypatch):
    """The selected provider, not an in-tree reference target, supplies the introspect."""
    target = "synthetic_threads"
    introspect = SimpleNamespace(TARGET=target, build_facts=lambda: {"lanes": 8})
    backend = SimpleNamespace(introspect=introspect)
    contract = SimpleNamespace(plugin=lambda: {"backend": "backend.py", "simt_introspect": "backend.py:introspect"})
    monkeypatch.setattr(target_registry, "all_targets", lambda: [target])
    monkeypatch.setattr(target_registry, "resolve", lambda name: contract if name == target else None)
    monkeypatch.setattr(base, "get_backend", lambda name: backend if name == target else None)
    monkeypatch.setattr(mb, "_SIMT_INTROSPECTS", {})
    mb._register_declared_simt_introspects()
    assert mb._SIMT_INTROSPECTS == {target: introspect}


def test_mlc_bridge_carries_no_backend_name_literal():
    """The fallback used to call get_backend("<a SIMT target>"); the name must come from data now."""
    tree = ast.parse(Path(mb.__file__).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "get_backend"]
    assert calls and all(not isinstance(c.args[0], ast.Constant) for c in calls)
