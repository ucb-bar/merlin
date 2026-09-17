"""The SIMT RTL introspect is registered from contract DATA, and mlc_bridge names no SIMT target."""
from __future__ import annotations

import ast
from pathlib import Path

from merlin.targetgen import plugins
from merlin.targetgen.rtl import mlc_bridge as mb


def test_simt_introspect_is_a_recognised_consumed_plugin_key():
    spec = plugins.PLUGIN_KEYS["simt_introspect"]
    assert spec.consumed is True and spec.expects == "attr"


def test_an_empty_registry_is_filled_from_the_declaring_contract(monkeypatch):
    monkeypatch.setattr(mb, "_SIMT_INTROSPECTS", {})
    mb._register_declared_simt_introspects()
    assert mb._SIMT_INTROSPECTS, "no contract-declared SIMT introspect was registered"
    for module in mb._SIMT_INTROSPECTS.values():
        assert hasattr(module, "TARGET") and callable(getattr(module, "build_facts", None))


def test_mlc_bridge_carries_no_backend_name_literal():
    """The fallback used to call get_backend("<a SIMT target>"); the name must come from data now."""
    tree = ast.parse(Path(mb.__file__).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "get_backend"]
    assert calls and all(not isinstance(c.args[0], ast.Constant) for c in calls)
