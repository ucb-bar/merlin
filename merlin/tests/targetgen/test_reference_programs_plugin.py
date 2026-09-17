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

from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import plugins, target_registry

KEY = "reference_programs"
#: What the callers use: the capsule-bench preflight table, the recipe-select evaluator and the IREE
#: perf bench.
CALLER_API = ("_anchors", "_build_cmd", "_matmul_golden", "build", "run", "matmul_source", "det_seed")


def _write_package(root: Path, name: str, *, declare: bool) -> str:
    """A minimal out-of-tree package. Returns the entry MERLIN_TARGET_PATH should name."""
    pkg = root / name
    (pkg / "contracts").mkdir(parents=True)
    plugin = f"plugin:\n  {KEY}: tools/refprog.py\n" if declare else ""
    (pkg / "contracts" / "target_contract.yaml").write_text(f"name: {name}\nversion: '0.1'\n{plugin}", encoding="utf-8")
    (pkg / "tools").mkdir()
    (pkg / "tools" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "tools" / "refprog.py").write_text(
        "def _anchors():\n    return [{'name': 'a0', 'capsule': 'c0', 'feature': 'f', 'golden': [[1]]}]\n",
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


def _declaring_targets() -> dict[str, Path]:
    found = {}
    for name in target_registry.list_targets():
        info = target_registry.resolve(name)
        reference = info.plugin().get(KEY)
        if reference:
            found[name] = info.base / reference
    return found


def test_every_declared_tool_lives_in_its_own_package_and_serves_the_callers():
    declared = _declaring_targets()
    assert declared, "no reference target declares plugin.reference_programs; the callers have nothing"
    for name, path in declared.items():
        base = target_registry.resolve(name).base.resolve()
        assert base in path.resolve().parents, f"{name}: {path} is outside its own package"
        module = plugins.load_declared(name, KEY)
        missing = [attr for attr in CALLER_API if not hasattr(module, attr)]
        assert not missing, f"{name}: the declared tool lacks what its callers use: {missing}"


def test_the_shared_tree_holds_no_copy_of_a_declared_tool():
    """The relocated module is the only place its target's name and facts live."""
    shared = merlin_dir() / "python" / "merlin"
    for name, path in _declaring_targets().items():
        copies = sorted(str(p) for p in shared.rglob(path.name))
        assert not copies, f"{name}'s {path.name} is also in the shared tree: {copies}"
