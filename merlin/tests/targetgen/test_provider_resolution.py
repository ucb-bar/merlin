"""Provider roles and discovery preserve package boundaries without implicit generation."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from merlin import targetgen
from merlin.common.yaml import write_yaml
from merlin.targetgen import target_registry as registry
from merlin.targetgen.providers import ProviderError, ProviderRole, read_provider


def support(root, target="synthetic", *, explicit=True, contract="contracts/target_contract.yaml"):
    write_yaml(root / contract, {"name": target, "runtime": {"default_backend": root.name}})
    if explicit:
        write_yaml(
            root / "provider.yaml",
            {
                "schema": "merlin.provider.v1",
                "id": root.name,
                "target": target,
                "role": "support",
                "contract": contract,
            },
        )
    return root


def fake_module(monkeypatch, name, **attributes):
    module = SimpleNamespace(**attributes)
    monkeypatch.setitem(sys.modules, f"merlin.targetgen.{name}", module)
    monkeypatch.setattr(targetgen, name, module, raising=False)


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    monkeypatch.setattr(registry, "targets_dir", lambda: tmp_path / "curated")


def test_order_is_intentional_and_shadows_are_observable(tmp_path, monkeypatch):
    first = support(tmp_path / "first")
    second = support(tmp_path / "second", contract="support/spec.yaml")
    monkeypatch.setenv("MERLIN_TARGET_PATH", os.pathsep.join(map(str, (first, second))))
    result = registry.discover([first, second])
    assert result.targets == {"synthetic": first}
    assert result.shadows == (registry.TargetShadow("synthetic", first, second),)
    assert registry.resolve("synthetic").base == first
    assert registry.backend_for("synthetic") == "first"
    monkeypatch.setenv("MERLIN_TARGET_PATH", os.pathsep.join(map(str, (second, first))))
    selected = registry.resolve("synthetic")
    assert selected.base == second
    assert selected.contract_path == second / "support/spec.yaml"
    assert selected.provider.id == "second"
    assert selected.provider.role == ProviderRole.SUPPORT
    assert registry.backend_for("synthetic") == "second"


def test_same_shelf_duplicate_is_not_alphabetical_selection(tmp_path):
    support(tmp_path / "shelf" / "a")
    support(tmp_path / "shelf" / "z")
    with pytest.raises(registry.TargetCollisionError, match="ambiguous target 'synthetic'"):
        registry.discover([tmp_path / "shelf"])


def test_same_physical_root_is_not_a_collision(tmp_path):
    actual = support(tmp_path / "shelf" / "actual")
    link = tmp_path / "shelf" / "link"
    link.symlink_to(actual, target_is_directory=True)
    result = registry.discover([link, tmp_path / "shelf", actual])
    assert result.targets == {"synthetic": actual}
    assert not result.shadows


def test_env_precedes_generated_and_legacy_contracts_still_work(tmp_path, monkeypatch):
    selected = support(tmp_path / "selected", explicit=False)
    support(registry.generated_target_home() / "generated")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(selected))
    assert registry.external_targets()["synthetic"] == selected
    assert registry.resolve("synthetic").base == selected
    assert registry.resolve("synthetic").provider.declared is False


def test_resolution_is_read_only_even_with_legacy_autofetch(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("resolution attempted to materialize or fetch")

    monkeypatch.setenv("MERLIN_TARGET_AUTOFETCH", "1")
    monkeypatch.setattr(registry, "materialize", forbidden)
    monkeypatch.setattr(registry, "declared_target_for", lambda _: None)
    fake_module(monkeypatch, "capability_manifests", write_oot_target=forbidden)
    fake_module(monkeypatch, "oot_fetch", fetch=forbidden)
    before = sorted(tmp_path.rglob("*"))
    resolved = registry.resolve("absent_synthetic")
    with pytest.raises(registry.TargetContractMissing):
        resolved.load_contract()
    assert sorted(tmp_path.rglob("*")) == before


def test_materialization_is_explicit_and_errors_are_visible(tmp_path, monkeypatch):
    calls = []

    def generate(name, destination):
        calls.append((name, destination))
        support(destination, name)

    fake_module(monkeypatch, "capability_manifests", write_oot_target=generate)
    destination = tmp_path / "new"
    result = registry.materialize("synthetic", destination=destination)
    assert calls == [("synthetic", destination)]
    assert result.contract_path.is_file()
    with pytest.raises(FileExistsError):
        registry.materialize("synthetic", destination=destination)
    assert len(calls) == 1

    def failure(*args):
        raise RuntimeError("derivation unavailable")

    fake_module(monkeypatch, "capability_manifests", write_oot_target=failure)
    with pytest.raises(RuntimeError, match="derivation unavailable"):
        registry.materialize("synthetic", destination=tmp_path / "failed")
    with pytest.raises(ValueError, match="invalid target name"):
        registry.materialize("../escape")


def test_roles_do_not_qualify_compilers_or_load_plugins(tmp_path, monkeypatch):
    provider = support(tmp_path / "sdk")
    write_yaml(provider / "contracts/target_contract.yaml", {"name": "synthetic", "plugin": {"backend": "missing.py"}})
    candidate = tmp_path / "candidate"
    write_yaml(candidate / "manifest.yaml", {"target": "synthetic", "artifact_type": "mlir_oot_target_backend"})
    schedule = tmp_path / "schedule"
    write_yaml(schedule / "manifest.yaml", {"target": "synthetic", "artifact_type": "mlir_oot_target_backend"})
    write_yaml(schedule / "payload/knobs.yaml", {"schedule_file": "schedule.mlir"})
    assert read_provider(provider).role == ProviderRole.SUPPORT
    assert read_provider(candidate).role == ProviderRole.CANDIDATE_COMPILER
    assert read_provider(schedule).role == ProviderRole.HOST_SCHEDULE
    assert read_provider(tmp_path / "unknown") is None
    assert registry.discover([tmp_path]).targets == {"synthetic": provider}
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(provider))
    assert registry.resolve("synthetic").plugin()["backend"] == "missing.py"


@pytest.mark.parametrize("role", ["candidate_compiler", "host_schedule"])
def test_explicit_non_support_role_is_never_target_support(tmp_path, role):
    root = support(tmp_path / role)
    write_yaml(
        root / "provider.yaml", {"schema": "merlin.provider.v1", "id": "claim", "target": "synthetic", "role": role}
    )
    assert read_provider(root).role.value == role
    assert registry.discover([root]).targets == {}


@pytest.mark.parametrize("resource", ["../elsewhere.yaml", "/tmp/elsewhere.yaml", "escape.yaml"])
def test_support_contract_escape_is_rejected(tmp_path, resource):
    root = support(tmp_path / "sdk")
    write_yaml(tmp_path / "elsewhere.yaml", {"name": "synthetic"})
    (root / "escape.yaml").symlink_to(tmp_path / "elsewhere.yaml")
    doc = {"schema": "merlin.provider.v1", "id": "bad", "target": "synthetic", "role": "support", "contract": resource}
    write_yaml(root / "provider.yaml", doc)
    with pytest.raises(ProviderError, match="relative path|escapes provider root"):
        registry.discover([root])


def test_contained_traversal_and_symlinked_user_root_are_supported(tmp_path):
    root = support(tmp_path / "sdk")
    (root / "link.yaml").symlink_to(root / "contracts/target_contract.yaml")
    write_yaml(
        root / "provider.yaml",
        {
            "schema": "merlin.provider.v1",
            "id": "sdk",
            "target": "synthetic",
            "role": "support",
            "contract": "contracts/../link.yaml",
        },
    )
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    assert read_provider(alias).contract_path == root / "contracts/target_contract.yaml"


@pytest.mark.parametrize(
    "change", [{"schema": "unknown"}, {"role": "trusted"}, {"target": "other"}, {"unexpected": True}]
)
def test_invalid_declaration_is_not_silently_treated_as_legacy(tmp_path, change):
    root = support(tmp_path / "sdk")
    doc = {"schema": "merlin.provider.v1", "id": "sdk", "target": "synthetic", "role": "support"}
    write_yaml(root / "provider.yaml", {**doc, **change})
    with pytest.raises(ProviderError):
        registry.discover([root])


def test_alias_is_one_hop_and_can_refer_to_external_provider(tmp_path, monkeypatch):
    root = support(tmp_path / "sdk")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    monkeypatch.setattr(registry, "declared_target_for", lambda name: "synthetic" if name == "alias" else "alias")
    assert registry.resolve("alias").base == root
    assert registry.resolve("missing").name == "missing"
