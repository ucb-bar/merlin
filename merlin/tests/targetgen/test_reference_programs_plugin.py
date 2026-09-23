"""A target contributes its reference-program tool as data, and the shared tree holds no copy of one.

The bareMetalC corroboration tool used to sit in the shared targetgen package even though every fact in
it belonged to one target: its C programs include that target's test utilities, its anchors are that
target's capsules, and its build line is that target's Makefile. It now lives in the target's own
package, the contract declares it as ``plugin.reference_programs``, and callers reach it with
``plugins.load_declared(target, key)``. These tests pin both halves. A new target needs only a contract
entry and a file. A target that declares nothing fails closed with the reason instead of borrowing
another target's tool.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from merlin.targetgen import plugins, target_registry

KEY = "reference_programs"
#: What the callers use: the capsule-bench preflight table, the recipe-select evaluator and the IREE
#: perf bench.
CALLER_API = ("_anchors", "_build_cmd", "_matmul_golden", "build", "run", "matmul_source", "det_seed")


@pytest.fixture(autouse=True)
def isolated_selection(monkeypatch, tmp_path):
    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    monkeypatch.setenv("MERLIN_TARGETS_DIR", str(tmp_path / "references"))
    monkeypatch.setattr(target_registry, "generated_target_home", lambda: tmp_path / "generated")
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.startswith("merlin._oot_targets"):
            sys.modules.pop(name, None)


def _write_package(root: Path, name: str, *, declare: bool) -> str:
    """A minimal out-of-tree package. Returns the entry MERLIN_TARGET_PATH should name."""
    pkg = root / name
    (pkg / "contracts").mkdir(parents=True)
    plugin = f"plugin:\n  {KEY}: tools/refprog.py\n" if declare else ""
    (pkg / "contracts" / "target_contract.yaml").write_text(f"name: {name}\nversion: '0.1'\n{plugin}", encoding="utf-8")
    (pkg / "tools").mkdir()
    (pkg / "tools" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "tools" / "refprog.py").write_text(
        f"from pathlib import Path\nPath({str(pkg / 'executed')!r}).touch()\n"
        "def _anchors():\n    return [{'name': 'a0', 'capsule': 'c0', 'feature': 'f', 'golden': [[1]]}]\n"
        + "\n".join(f"def {name}(): return 'synthetic'" for name in CALLER_API if name != "_anchors"),
        encoding="utf-8",
    )
    return str(root)


def test_reference_programs_is_a_consumed_path_key():
    spec = plugins.PLUGIN_KEYS[KEY]
    assert spec.consumed is True and spec.expects == "path"


def test_a_new_target_contributes_the_tool_as_data(tmp_path, monkeypatch):
    """A contract entry plus a file is the whole change; no shared module learns the name."""
    monkeypatch.setenv("MERLIN_TARGET_PATH", _write_package(tmp_path, "synth_refprog_npu", declare=True))
    module = plugins.load_declared("synth_refprog_npu", KEY)
    assert module._anchors()[0]["capsule"] == "c0"
    assert tmp_path in Path(module.__file__).resolve().parents


def test_a_target_that_declares_none_fails_closed_with_the_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_TARGET_PATH", _write_package(tmp_path, "synth_refprog_bare", declare=False))
    with pytest.raises(plugins.PluginError, match=f"declares no plugin.{KEY}"):
        plugins.load_declared("synth_refprog_bare", KEY)


@pytest.mark.parametrize("key", ["path", "dialect_module", "no_such_key"])
def test_a_key_that_names_no_loadable_tool_is_refused(key):
    with pytest.raises(plugins.PluginError, match="not a loadable plugin key"):
        plugins.load_declared("any_target", key)


def test_selected_synthetic_tool_serves_callers_from_its_own_root(tmp_path, monkeypatch):
    name = "synthetic_api"
    monkeypatch.setenv("MERLIN_TARGET_PATH", _write_package(tmp_path, name, declare=True))
    module = plugins.load_declared(name, KEY)
    assert Path(module.__file__).resolve().is_relative_to(tmp_path / name)
    assert all(callable(getattr(module, attribute)) for attribute in CALLER_API)


@pytest.mark.parametrize("location", ["references", "generated"])
def test_unselected_inspectable_metadata_never_executes(tmp_path, location):
    name = "synthetic_unselected"
    _write_package(tmp_path / location, name, declare=True)
    assert target_registry.resolve(name).base == tmp_path / location / name
    with pytest.raises(plugins.PluginError):
        plugins.load_declared(name, KEY)
    assert not (tmp_path / location / name / "executed").exists()


@pytest.mark.parametrize("role", ["candidate_compiler", "host_schedule"])
def test_other_roles_cannot_borrow_same_name_reference(tmp_path, monkeypatch, role):
    name = "synthetic_roles"
    _write_package(tmp_path / "references", name, declare=True)
    monkeypatch.setenv("MERLIN_TARGET_PATH", _write_package(tmp_path / "selected", name, declare=True))
    package = tmp_path / "selected" / name
    (package / "provider.yaml").write_text(f"schema: merlin.provider.v1\nid: fixture\ntarget: {name}\nrole: {role}\n")
    with pytest.raises(plugins.PluginError):
        plugins.load_declared(name, KEY)
    assert not (package / "executed").exists()
    assert not (tmp_path / "references" / name / "executed").exists()


def test_removed_selection_cannot_reuse_cached_reference_module(tmp_path, monkeypatch):
    name = "synthetic_removed"
    _write_package(tmp_path / "references", name, declare=True)
    package = tmp_path / "references" / name
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(package))
    module = plugins.load_declared(name, KEY)
    assert module._anchors()
    monkeypatch.delenv("MERLIN_TARGET_PATH")
    assert target_registry.resolve(name).base == package
    with pytest.raises(plugins.PluginError):
        plugins.load_declared(name, KEY)
