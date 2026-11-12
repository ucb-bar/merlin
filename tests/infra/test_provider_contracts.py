"""Explicit support ownership, composition and contained plugin imports."""

import sys

import pytest

from merlin.common.yaml import write_yaml
from merlin.compiler.backend.composition import (
    BackendCompositionError,
    ComposeError,
    composition,
    materialize,
    validate_backend_composition,
)
from merlin.development.targetgen import plugins
from merlin.development.targetgen import target_registry as registry

REVISION = "a" * 40


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("MERLIN_TARGET_PATH", "")
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    monkeypatch.setattr(registry, "targets_dir", lambda: tmp_path / "curated")


def base_provider(root, target="base", revision=REVISION):
    write_yaml(
        root / "provider.yaml", {"schema": "merlin.provider.v1", "id": target, "target": target, "role": "support"}
    )
    write_yaml(
        root / "contracts/target_contract.yaml", {"name": target, "features": ["scalar"], "plugin": {"host": "base.py"}}
    )
    write_yaml(root.parent / "SOURCE.yaml", {"path": root.name, "source": {"commit": revision}})
    return root


def layer_provider(root, target="layer", base="base", revision=REVISION):
    write_yaml(
        root / "provider.yaml",
        {
            "schema": "merlin.provider.v2",
            "id": target,
            "target": target,
            "role": "support",
            "extends": {"target": base, "commit": revision},
        },
    )
    write_yaml(root / "contracts/overlay.yaml", {"features": ["matrix"], "plugin": {"matrix": "layer.py"}})
    return root


@pytest.fixture
def provider(tmp_path, monkeypatch):
    root = tmp_path / "provider"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text(f"from pathlib import Path\nPath({str(tmp_path / 'executed')!r}).touch()\nVALUE = 99\n")
    before = set(sys.modules)
    yield (root, outside)
    for name in set(sys.modules) - before:
        if name.startswith("merlin._oot_targets.containment"):
            sys.modules.pop(name, None)


def test_missing_ambiguous_and_wrong_revision_bases_refuse(tmp_path):
    base = base_provider(tmp_path / "first" / "support")
    other = base_provider(tmp_path / "second" / "support")
    layer = layer_provider(tmp_path / "layer" / "support")
    with pytest.raises(ComposeError, match="not explicitly selected"):
        composition(layer, bases=[])
    with pytest.raises(ComposeError, match="ambiguous"):
        composition(layer, bases=[base, other])
    base_provider(base, revision="b" * 40)
    with pytest.raises(ComposeError, match="pinned"):
        composition(layer, bases=[base])


def test_backend_replacements_require_explicit_preserved_contract(tmp_path):
    base = base_provider(tmp_path / "base" / "support")
    layer = layer_provider(tmp_path / "layer" / "support")
    materialize(layer, bases=[base])
    declaration = """schema = "merlin.backend_composition.v1"
target = "layer"
base_target = "base"
[plugin_owners]
matrix = "layer"
host = "base"
[[operations]]
name = "matrix"
action = "replace"
owner = "layer"
pipeline = "layer.matrix"
replaces = "base.matrix"
"""
    (layer / "backend.toml").write_text(declaration)
    with pytest.raises(BackendCompositionError, match="preserves"):
        validate_backend_composition(layer, bases=[base])
    (layer / "backend.toml").write_text(declaration + 'preserves = "matrix-boundary-v1"\n')
    result = validate_backend_composition(layer, bases=[base])
    assert result.base_commit == REVISION
    assert result.operations[0]["replaces"] == "base.matrix"


def test_external_plugin_alias_refuses_before_import(provider):
    root, outside = provider
    (root / "module.py").symlink_to(outside)
    assert plugins.validate({"backend": "module.py"}, root=root)
    with pytest.raises(plugins.PluginError, match="under|escape|contain"):
        plugins.load_module(root, "module.py", package_name="containment_negative")
    assert not (outside.parent / "executed").exists()
