"""Tests for out-of-tree target discovery (MERLIN_TARGET_PATH) in the target registry."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common.yaml import write_yaml
from merlin.targetgen import target_registry as tr

pytestmark = pytest.mark.target("radiance", "mx_gemmini", "gemmini")


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
                {
                    "name": "mx_pe",
                    "kind": "systolic",
                    "dtypes": ["mxfp4", "mxfp6", "mxfp8"],
                    "ops": ["matmul"],
                    "accumulate": [{"in": "mxfp8", "weight": "mxfp8", "acc": "f32"}],
                    "scaling": "block_e8m0",
                    "requant": {"ref": "radiance_mlir.lowering:requant_mx"},
                },
            ],
            "plugin": {
                "dialect_module": f"{name}_mlir.dialect",
                "lowering_entrypoint": f"{name}_mlir.lowering:lower",
            },
        },
    )
    return root


def test_no_env_and_empty_generated_home_means_no_external_targets(tmp_path, monkeypatch):
    # external_targets() discovers env roots UNION the generated home (out/build/generated). With no env
    # AND an empty generated home (isolated via MERLIN_OUT_ROOT), there is nothing to discover.
    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path))  # empty generated home
    assert tr.external_targets() == {}
    # reference targets still resolve normally.
    assert tr.resolve("gemmini").kind == "reference"


def test_generated_home_is_auto_discovered_without_env(tmp_path, monkeypatch):
    # A package dropped into the generated home (out/build/generated/<pkg>) is picked up with ZERO env —
    # the seamless default for a just-generated target. (resolve() reports kind='external'.)
    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path))
    home = tr.generated_target_home()
    _make_oot_target(home / "radiance", "radiance")
    assert "radiance" in tr.external_targets()
    assert tr.resolve("radiance").kind == "external"


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


@pytest.fixture
def references(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    legacy = checkout / "merlin/targets"
    example = _make_oot_target(checkout / "examples/synthetic/target", "synthetic")
    monkeypatch.delenv("MERLIN_TARGETS_DIR", raising=False)
    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setattr(tr, "checkout_root", lambda: checkout)
    monkeypatch.setattr(tr, "targets_dir", lambda: Path(os.environ.get("MERLIN_TARGETS_DIR", legacy)))
    return checkout, legacy, example


def test_authored_example_is_reference_metadata_without_provider_execution(references, monkeypatch):
    _, _, example = references
    plugin = example / "synthetic_mlir"
    plugin.mkdir()
    (plugin / "__init__.py").write_text("raise AssertionError('discovery executed provider')\n")
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("discovery launched a process"))
    assert tr.reference_targets() == {"synthetic": example}
    assert tr.list_targets() == ["synthetic"]
    info = tr.resolve("synthetic")
    assert info.kind == "reference" and info.base == example
    assert info.plugin()["dialect_module"] == "synthetic_mlir.dialect"
    assert tr.explicit_targets() == {}
    assert "synthetic_mlir" not in sys.modules


def test_authored_example_wins_legacy_compatibility_alias(references):
    _, legacy, example = references
    _make_oot_target(legacy / "synthetic", "synthetic")
    assert tr.reference_targets() == {"synthetic": example}
    assert tr.resolve("synthetic").base == example


def test_example_identity_comes_from_contract_not_directory(references):
    checkout, _, example = references
    renamed = checkout / "examples/different-folder"
    example.parent.rename(renamed)
    assert tr.reference_targets() == {"synthetic": renamed / "target"}
    assert tr.resolve("synthetic").base == renamed / "target"
    assert "different-folder" not in tr.list_targets()


def test_duplicate_example_identities_refuse_ambiguous_selection(references):
    checkout, _, _ = references
    _make_oot_target(checkout / "examples/another-folder/target", "synthetic")
    with pytest.raises(tr.TargetCollisionError, match="synthetic"):
        tr.reference_targets()


def test_empty_legacy_directory_does_not_shadow_example_or_generated(references):
    _, legacy, example = references
    (legacy / "synthetic").mkdir(parents=True)
    (legacy / "generated_fixture").mkdir()
    generated = _make_oot_target(tr.generated_target_home() / "generated_fixture", "generated_fixture")
    assert tr.resolve("synthetic").base == example
    assert tr.resolve("generated_fixture").external_root == generated


def test_explicit_reference_root_suppresses_checkout_examples(references, tmp_path, monkeypatch):
    selected = _make_oot_target(tmp_path / "selected/other", "other")
    monkeypatch.setenv("MERLIN_TARGETS_DIR", str(selected.parent))
    assert tr.reference_targets() == {"other": selected}
    assert "synthetic" not in tr.list_targets()
    assert tr.resolve("synthetic").kind != "reference"


def test_explicit_oot_support_overrides_both_reference_sources(references, tmp_path, monkeypatch):
    _, legacy, _ = references
    _make_oot_target(legacy / "synthetic", "synthetic")
    selected = _make_oot_target(tmp_path / "explicit", "synthetic")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(selected))
    assert tr.resolve("synthetic").external_root == selected
    assert tr.explicit_targets() == {"synthetic": selected}


def test_installed_discovery_does_not_scan_host_examples(references, monkeypatch):
    checkout, legacy, _ = references
    bundled = _make_oot_target(legacy / "bundled", "bundled")
    monkeypatch.setattr(tr, "checkout_root", lambda: None)
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(checkout))
    assert tr.reference_targets() == {"bundled": bundled}
    assert "synthetic" not in tr.list_targets()


def test_residual_discovery_uses_selected_reference_over_incidental_generated(references):
    from merlin.targetgen import capability_manifests

    _, _, example = references
    write_yaml(example / "contracts/residual.yaml", {"name": "synthetic"})
    _make_oot_target(tr.generated_target_home() / "synthetic", "synthetic")
    assert tr.resolve("synthetic").base == example
    assert "synthetic" in capability_manifests.discovered_targets()


def test_residual_discovery_does_not_advertise_shadowed_reference(references, tmp_path, monkeypatch):
    from merlin.targetgen import capability_manifests

    _, _, example = references
    write_yaml(example / "contracts/residual.yaml", {"name": "synthetic"})
    selected = _make_oot_target(tmp_path / "explicit", "synthetic")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(selected))
    assert tr.resolve("synthetic").external_root == selected
    assert "synthetic" not in capability_manifests.discovered_targets()
