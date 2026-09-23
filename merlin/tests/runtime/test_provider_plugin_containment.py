"""Discovery and direct plugin loading enforce the selected provider's path authority."""

import sys
from types import SimpleNamespace

import pytest

from merlin.runtime.backends import base
from merlin.targetgen import plugins, target_registry


@pytest.fixture
def selected(tmp_path, monkeypatch):
    root = tmp_path / "provider"
    root.mkdir()
    outside = tmp_path / "outside.py"
    marker = tmp_path / "executed"
    outside.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    block = {"backend": "backend.py", "path": str(root)}
    info = SimpleNamespace(base=root, plugin=lambda: dict(block))
    monkeypatch.setattr(target_registry, "resolve", lambda _: info)
    monkeypatch.setattr(target_registry, "explicit_targets", lambda: {"containment_fixture": root})
    monkeypatch.setattr(target_registry, "list_targets", lambda: [])
    monkeypatch.setattr(base, "_LOAD_FAILURES", {})
    monkeypatch.setattr(base, "_LOADED_PLUGIN_OWNERS", {})
    before = set(sys.modules)
    yield root, outside, marker, block
    for name in set(sys.modules) - before:
        if name.startswith("merlin._oot_backends.containment_fixture"):
            sys.modules.pop(name, None)


@pytest.mark.parametrize("kind", ["absolute", "traversal", "symlink", "init", "root_init", "root_override"])
def test_discovery_refuses_escaping_declared_reference(selected, kind):
    root, outside, marker, block = selected
    if kind == "absolute":
        block["backend"] = str(outside)
    elif kind == "traversal":
        block["backend"] = "../outside.py"
    elif kind == "symlink":
        (root / "backend.py").symlink_to(outside)
    elif kind == "init":
        (root / "backend").mkdir()
        (root / "backend/__init__.py").symlink_to(outside)
        block["backend"] = "backend"
    elif kind == "root_init":
        (root / "__init__.py").symlink_to(outside)
        block["backend"] = "."
    else:
        block.update(path=str(outside.parent), backend=outside.name)
    assert base._oot_plugin_modules() == []
    assert "PluginError" in base._LOAD_FAILURES["containment_fixture"]
    assert not marker.exists()


def test_direct_loader_cannot_bypass_discovery_containment(selected):
    root, outside, marker, _ = selected
    for path in (outside, root / "alias.py"):
        if path != outside:
            path.symlink_to(outside)
        with pytest.raises(plugins.PluginError, match="provider root"):
            base._load_oot_backend("containment_fixture", path)
        assert not marker.exists()
        assert "merlin._oot_backends.containment_fixture" not in sys.modules


def test_in_root_package_alias_preserves_imports_and_ownership(selected):
    root, _, _, block = selected
    backend = root / "implementation"
    backend.mkdir()
    (backend / "__init__.py").write_text("from .sibling import VALUE\n")
    (backend / "sibling.py").write_text("VALUE = 31\n")
    (root / "backend").symlink_to(backend, target_is_directory=True)
    block["backend"] = "backend"
    [(name, path)] = base._oot_plugin_modules()
    base._load_oot_backend(name, path)
    assert sys.modules["merlin._oot_backends.containment_fixture"].VALUE == 31
    base._assert_oot_plugin_ownership()


def test_loaded_owner_refuses_later_root_override(selected):
    root, outside, marker, block = selected
    (root / "backend.py").write_text("VALUE = 1\n")
    base._load_oot_backend("containment_fixture", root / "backend.py")
    block.update(path=str(outside.parent), backend=outside.name)
    with pytest.raises(base.PluginOwnershipError, match="selected provider root"):
        base._assert_oot_plugin_ownership()
    assert not marker.exists()


def test_invalid_selected_backend_never_borrows_same_name_builtin(selected, monkeypatch):
    root, outside, marker, block = selected
    block.update(path=str(outside.parent), backend=outside.name)
    monkeypatch.setattr(base, "_REGISTRY", {"containment_fixture": SimpleNamespace(module="native.fallback")})
    monkeypatch.setattr(base, "_ensure_discovered", lambda: base._oot_plugin_modules())
    monkeypatch.setattr(base.importlib, "import_module", lambda name: pytest.fail(f"borrowed {name}"))
    with pytest.raises(KeyError, match="backend module failed to load"):
        base.get_backend("containment_fixture")
    assert not marker.exists()


def test_unrelated_builtin_failure_record_does_not_invent_plugin_requirement(selected, monkeypatch):
    _, _, _, block = selected
    block.pop("backend")
    monkeypatch.setattr(base, "_REGISTRY", {"containment_fixture": SimpleNamespace(module="native.fixture")})
    monkeypatch.setattr(base, "_ensure_discovered", lambda: None)
    base._LOAD_FAILURES["containment_fixture"] = "unrelated optional feature"
    expected = object()
    monkeypatch.setattr(base.importlib, "import_module", lambda _: expected)
    assert base.get_backend("containment_fixture") is expected
