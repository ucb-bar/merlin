"""Declared references cannot execute code outside the selected support provider."""

import pytest

from merlin.targetgen import plugins


@pytest.fixture
def provider(tmp_path, monkeypatch):
    import sys

    root = tmp_path / "provider"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text(f"from pathlib import Path\nPath({str(tmp_path / 'executed')!r}).touch()\nVALUE = 99\n")
    # Imports in this file never leave synthetic module-cache state in other tests.
    before = set(sys.modules)
    yield root, outside
    for name in set(sys.modules) - before:
        if name.startswith("merlin._oot_targets.containment"):
            sys.modules.pop(name, None)


@pytest.mark.parametrize("kind", ["absolute", "traversal", "file_symlink", "package_init", "parent_init"])
def test_external_references_refuse_before_import(provider, kind):
    root, outside = provider
    if kind == "absolute":
        reference = str(outside)
    elif kind == "traversal":
        reference = "../outside.py"
    elif kind == "file_symlink":
        (root / "module.py").symlink_to(outside)
        reference = "module.py"
    else:
        package = root / "package"
        package.mkdir()
        (package / "__init__.py").symlink_to(outside)
        (package / "child.py").write_text("VALUE = 1\n")
        reference = "package" if kind == "package_init" else "package.child"
    assert plugins.validate({"backend": reference}, root=root)
    with pytest.raises(plugins.PluginError, match="under|escape|contain"):
        plugins.load_module(root, reference, package_name="containment_" + kind)
    assert not (outside.parent / "executed").exists()


def test_in_root_aliases_and_relative_imports_remain_supported(provider):
    root, _ = provider
    package = root / "package"
    package.mkdir()
    (package / "__init__.py").write_text("from .child import VALUE\n")
    (package / "child.py").write_text("VALUE = 17\n")
    (root / "alias.py").symlink_to(package / "child.py")
    for reference in ("package", "package.child", "package/child.py", "alias.py", "./alias.py"):
        assert plugins.validate({"backend": reference}, root=root) == []
        assert plugins.load_module(root, reference, package_name="containment_positive").VALUE == 17
    assert plugins.load_module(root, "package.child:VALUE", package_name="containment_positive").VALUE == 17
    assert plugins.load_object(root, "package.child:VALUE", package_name="containment_positive") == 17


def test_provider_root_alias_is_canonical(provider, tmp_path):
    root, _ = provider
    (root / "module.py").write_text("VALUE = 23\n")
    alias = tmp_path / "root-alias"
    alias.symlink_to(root, target_is_directory=True)
    first = plugins.load_module(alias, "module", package_name="containment_root")
    assert plugins.load_module(root, "module", package_name="containment_root") is first


def test_declared_root_override_is_rejected(provider, monkeypatch):
    from types import SimpleNamespace

    from merlin.targetgen import target_registry

    root, outside = provider
    info = SimpleNamespace(
        base=root,
        contract_path=root / "contract.yaml",
        plugin=lambda: {"backend": outside.name, "path": str(outside.parent)},
    )
    monkeypatch.setattr(target_registry, "resolve", lambda _: info)
    monkeypatch.setattr(target_registry, "explicit_targets", lambda: {"containment_override": root})
    with pytest.raises(plugins.PluginError, match="root|provider"):
        plugins.load_declared("containment_override", "backend")
    assert not (outside.parent / "executed").exists()


def test_import_parent_resolution_cannot_execute_external_shadow(provider):
    root, outside = provider
    (root / "package").mkdir()
    (root / "package/child.py").write_text("VALUE = 1\n")
    # A regular module takes precedence over a namespace-package directory.
    (root / "package.py").symlink_to(outside)
    with pytest.raises(plugins.PluginError):
        plugins.load_module(root, "package.child", package_name="containment_shadow")
    assert not (outside.parent / "executed").exists()
