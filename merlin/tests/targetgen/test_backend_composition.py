from __future__ import annotations

import pytest
import yaml

from merlin.common.yaml import write_yaml
from merlin.compiler.backend.composition import (
    BackendCompositionError,
    composition,
    materialize,
    validate_backend_composition,
)
from merlin.compiler.backend.load import read_provider

REVISION = "a" * 40


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


def test_explicit_composition_is_read_only_until_materialized(tmp_path):
    base = base_provider(tmp_path / "base-source" / "support")
    layer = layer_provider(tmp_path / "layer-source" / "support")
    result = composition(layer, bases=[base])
    assert not result.provider.contract_path.exists()
    effective = yaml.safe_load(result.text)
    assert effective["features"] == ["scalar", "matrix"]
    assert effective["plugin"] == {"matrix": "layer.py"}
    assert result.lock["base"]["revision"] == REVISION
    fresh, _ = materialize(layer, bases=[base])
    assert not fresh
    assert read_provider(layer).extends.target == "base"
    assert materialize(layer, bases=[base], check=True)[0]
    write_yaml(layer / "contracts/overlay.yaml", {"features": ["changed"]})
    assert not materialize(layer, bases=[base], check=True)[0]


def test_backend_replacements_require_explicit_preserved_contract(tmp_path):
    base = base_provider(tmp_path / "base" / "support")
    layer = layer_provider(tmp_path / "layer" / "support")
    materialize(layer, bases=[base])
    declaration = 'schema = "merlin.backend_composition.v1"\ntarget = "layer"\nbase_target = "base"\n[plugin_owners]\nmatrix = "layer"\nhost = "base"\n[[operations]]\nname = "matrix"\naction = "replace"\nowner = "layer"\npipeline = "layer.matrix"\nreplaces = "base.matrix"\n'
    (layer / "backend.toml").write_text(declaration)
    with pytest.raises(BackendCompositionError, match="preserves"):
        validate_backend_composition(layer, bases=[base])
    (layer / "backend.toml").write_text(declaration + 'preserves = "matrix-boundary-v1"\n')
    result = validate_backend_composition(layer, bases=[base])
    assert result.base_commit == REVISION
    assert result.operations[0]["replaces"] == "base.matrix"
