"""A chip backend must declare ownership before its plugins are imported."""

from __future__ import annotations

import pytest

from merlin.common.yaml import write_yaml
from merlin.targetgen import component_composition as cc
from merlin.targetgen import provider_compose

BASE_REVISION = "0123456789abcdef0123456789abcdef01234567"


def _layer(tmp_path, monkeypatch):
    base_parent = tmp_path / "base"
    base = base_parent / "support"
    base.mkdir(parents=True)
    write_yaml(
        base / "provider.yaml",
        {"schema": "merlin.provider.v1", "id": "base-support", "target": "base_isa", "role": "support"},
    )
    write_yaml(base / "contracts/target_contract.yaml", {"name": "base_isa", "plugin": {"rvv_codegen": "backend.py"}})
    write_yaml(base_parent / "SOURCE.yaml", {"path": "support", "source": {"commit": BASE_REVISION}})
    layer = tmp_path / "chip"
    write_yaml(
        layer / "provider.yaml",
        {
            "schema": "merlin.provider.v2",
            "id": "chip-support",
            "target": "chip",
            "role": "support",
            "extends": {"target": "base_isa", "commit": BASE_REVISION},
        },
    )
    write_yaml(layer / "contracts/overlay.yaml", {"plugin": {"matrix_lowering": "matrix_lowering.py"}})
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(base))
    provider_compose.materialize(layer)
    return layer


def _manifest(layer, operations=""):
    (layer / "backend.toml").write_text(
        'schema = "merlin.backend_composition.v1"\n'
        'target = "chip"\n'
        'base_target = "base_isa"\n'
        '[plugin_owners]\n'
        'rvv_codegen = "base_isa"\n'
        'matrix_lowering = "chip"\n'
        f"{operations}",
        encoding="utf-8",
    )


def test_composition_checks_pinned_base_and_explicit_ownership(tmp_path, monkeypatch):
    layer = _layer(tmp_path, monkeypatch)
    _manifest(layer, '[[operations]]\nname = "chip.matrix"\naction = "add"\nowner = "chip"\nunit = "matrix"\n')
    manifest = cc.validate_backend_composition(layer)
    assert manifest.base_commit == BASE_REVISION
    assert manifest.plugin_owners == {"rvv_codegen": "base_isa", "matrix_lowering": "chip"}
    assert manifest.operations[0]["name"] == "chip.matrix"

    write_yaml(
        layer / "provider.yaml",
        {
            "schema": "merlin.provider.v2",
            "id": "chip-support",
            "target": "chip",
            "role": "support",
            "extends": {"target": "base_isa", "commit": "f" * 40},
        },
    )
    with pytest.raises(ValueError, match="pinned at"):
        cc.validate_backend_composition(layer)


def test_ambiguous_operation_and_implicit_replacement_are_rejected(tmp_path, monkeypatch):
    layer = _layer(tmp_path, monkeypatch)
    _manifest(
        layer,
        '[[operations]]\nname = "chip.matrix"\naction = "add"\nowner = "chip"\nunit = "matrix"\n'
        '[[operations]]\nname = "chip.matrix"\naction = "add"\nowner = "chip"\nunit = "other"\n',
    )
    with pytest.raises(cc.BackendCompositionError, match="conflicting ownership"):
        cc.validate_backend_composition(layer)

    _manifest(layer, '[[operations]]\nname = "chip.matrix"\naction = "replace"\nowner = "chip"\nunit = "matrix"\n')
    with pytest.raises(cc.BackendCompositionError, match="replaces"):
        cc.validate_backend_composition(layer)


def test_plugin_claim_must_match_effective_contract(tmp_path, monkeypatch):
    layer = _layer(tmp_path, monkeypatch)
    _manifest(layer)
    (layer / "backend.toml").write_text(
        (layer / "backend.toml").read_text().replace('matrix_lowering = "chip"', 'matrix_lowering = "base_isa"'),
        encoding="utf-8",
    )
    with pytest.raises(cc.BackendCompositionError, match="claims base ownership"):
        cc.validate_backend_composition(layer)
