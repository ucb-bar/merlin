"""Tests for out-of-tree target discovery (MERLIN_TARGET_PATH) in the target registry."""
from __future__ import annotations

import pytest

from merlin.common.yaml import write_yaml
from merlin.targetgen import target_registry as tr


def _make_oot_target(root, name):
    """Write a minimal out-of-tree target package (contract + compute_units + plugin block)."""
    contracts = root / "contracts"
    write_yaml(
        contracts / "target_contract.yaml",
        {
            "name": name,
            "version": "0.1",
            "capabilities": {"ops": ["matmul"]},
            "memory_model": {"resident": True},
            "compiler_obligations": [],
            "hardware_promises": [],
            "runtime_promises": [],
            "legality": [],
            "runtime": {"default_backend": "simulator"},
            "compute_units": [
                {"name": "mx_pe", "kind": "systolic", "dtypes": ["mxfp4", "mxfp6", "mxfp8"],
                 "ops": ["matmul"], "accumulate": [{"in": "mxfp8", "weight": "mxfp8", "acc": "f32"}],
                 "scaling": "block_e8m0", "requant": {"ref": "radiance_mlir.lowering:requant_mx"}},
            ],
            "plugin": {
                "dialect_module": f"{name}_mlir.dialect",
                "lowering_entrypoint": f"{name}_mlir.lowering:lower",
            },
        },
    )
    return root


def test_no_env_means_no_external_targets(monkeypatch):
    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    assert tr.external_targets() == {}
    # reference targets still resolve normally.
    assert tr.resolve("gemmini").kind == "reference"


def test_discover_and_resolve_external_target(tmp_path, monkeypatch):
    root = _make_oot_target(tmp_path / "radiance", "radiance")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    assert "radiance" in tr.external_targets()
    info = tr.resolve("radiance")
    assert info.kind == "external"
    assert info.external_root == root
    assert info.backend == "simulator"
    # compute_units parse + plugin block reads (with path injected), lowering ref is opaque.
    from merlin.targetgen import compute_units as cu

    units = cu.compute_units(info.load_contract())
    assert {"mxfp4", "mxfp6", "mxfp8"} <= set(units[0].dtypes)
    plugin = info.plugin()
    assert plugin["dialect_module"] == "radiance_mlir.dialect"
    assert plugin["path"] == str(root)


def test_search_dir_of_targets(tmp_path, monkeypatch):
    _make_oot_target(tmp_path / "targets" / "radiance", "radiance")
    _make_oot_target(tmp_path / "targets" / "mx_gemmini", "mx_gemmini")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(tmp_path / "targets"))
    ext = tr.external_targets()
    assert {"radiance", "mx_gemmini"} <= set(ext)
    assert {"radiance", "mx_gemmini"} <= set(tr.all_targets())


def test_external_overrides_are_first(tmp_path, monkeypatch):
    # An external target named like nothing in-tree resolves external; reference names still resolve
    # reference when not shadowed.
    _make_oot_target(tmp_path / "mx_gemmini", "mx_gemmini")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(tmp_path / "mx_gemmini"))
    assert tr.resolve("mx_gemmini").kind == "external"
    assert tr.resolve("gemmini").kind == "reference"
