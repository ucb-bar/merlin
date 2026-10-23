"""Explicit private support dependencies; discovery never executes their code."""

import socket
import subprocess

import pytest

from merlin.common.yaml import write_yaml
from merlin.targetgen.providers import ProviderError, ProviderRole, read_provider, support_source_dependencies


def declaration(root, dependencies=None, *, role="support"):
    document = {"schema": "merlin.provider.v1", "id": "fixture", "target": "dependency_fixture", "role": role}
    if role == "support":
        write_yaml(root / "contracts/target_contract.yaml", {"name": "dependency_fixture"})
    if dependencies is not None:
        document["source_dependencies"] = dependencies
    write_yaml(root / "provider.yaml", document)
    return root


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("source dependency discovery must not execute code or open listeners")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(socket.socket, "bind", refuse)


def test_declared_sibling_and_legacy_behavior(tmp_path):
    helper = tmp_path / "helpers"
    helper.mkdir()
    (helper / "__init__.py").write_text("raise AssertionError('private helpers must not import')\n")
    root = declaration(tmp_path / "support", {"compiler": "../helpers"})
    provider = read_provider(root)
    assert provider.role == ProviderRole.SUPPORT
    assert provider.source_dependencies == (("compiler", helper),)
    assert support_source_dependencies(root) == {"compiler": helper}
    assert support_source_dependencies(declaration(tmp_path / "legacy")) == {}
    legacy = tmp_path / "contract_only"
    write_yaml(legacy / "contracts/target_contract.yaml", {"name": "legacy"})
    assert support_source_dependencies(legacy) == {}


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        "../helpers",
        {True: "../helpers"},
        {"not-safe": "../helpers"},
        {"": "../helpers"},
        {"compiler": 1},
        {"compiler": "/tmp/helpers"},
        {"compiler": "helpers"},
        {"compiler": "./helpers"},
        {"compiler": "../helpers/"},
        {"compiler": "../../helpers"},
        {"compiler": "../helpers/nested"},
        {"compiler": "../."},
        {"compiler": "../.."},
        {"compiler": "../helpers//"},
        {"compiler": "../helpers\\nested"},
        {"compiler": "../hélpers"},
    ],
)
def test_malformed_dependency_declarations_refuse(tmp_path, value):
    (tmp_path / "helpers").mkdir()
    root = declaration(tmp_path / "support")
    document = {
        "schema": "merlin.provider.v1",
        "id": "fixture",
        "target": "dependency_fixture",
        "role": "support",
        "source_dependencies": value,
    }
    write_yaml(root / "provider.yaml", document)
    with pytest.raises(ProviderError):
        read_provider(root)


@pytest.mark.parametrize("role", ["candidate_compiler", "host_schedule"])
def test_only_support_can_declare_dependencies(tmp_path, role):
    root = declaration(tmp_path / "candidate", {}, role=role)
    with pytest.raises(ProviderError, match="only support"):
        read_provider(root)
    declaration(root, role=role)
    with pytest.raises(ProviderError, match="support provider"):
        support_source_dependencies(root)


@pytest.mark.parametrize("kind", ["absent", "file", "symlink", "self", "duplicate", "metadata_link", "root_link"])
def test_indirect_or_overlapping_ownership_refuses(tmp_path, kind):
    helper = tmp_path / "helpers"
    if kind == "file":
        helper.write_text("not a directory")
    elif kind != "absent":
        helper.mkdir()
    dependencies = {"compiler": "../helpers"}
    if kind == "self":
        dependencies = {"compiler": "../support"}
    elif kind == "duplicate":
        dependencies["second"] = "../helpers"
    elif kind == "symlink":
        link = tmp_path / "helper_link"
        link.symlink_to(helper, target_is_directory=True)
        dependencies = {"compiler": "../helper_link"}
    root = declaration(tmp_path / "support", dependencies)
    if kind == "metadata_link":
        original = root / "original.yaml"
        (root / "provider.yaml").rename(original)
        (root / "provider.yaml").symlink_to(original)
    elif kind == "root_link":
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)
        root = alias
    with pytest.raises(ProviderError):
        support_source_dependencies(root)


def test_dependency_masks_are_private_and_metadata_only(tmp_path, monkeypatch):
    from merlin.targetgen import target_registry
    from merlin.targetgen.sandbox.answer_surfaces import _support_package_dirs

    helper = tmp_path / "helpers"
    helper.mkdir()
    (helper / "__init__.py").write_text("raise AssertionError('must remain private')\n")
    root = declaration(tmp_path / "support", {"compiler": "../helpers"})
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(root))
    monkeypatch.setattr(target_registry, "reference_targets", lambda: {})
    monkeypatch.setattr(target_registry, "in_repo_support", lambda: {})
    monkeypatch.setattr(target_registry, "vendored_support_dirs", lambda: ())
    masks = _support_package_dirs()
    assert root in masks and helper in masks
    (root / "provider.yaml").write_text((root / "provider.yaml").read_text().replace("../helpers", "../../helpers"))
    with pytest.raises(ProviderError):
        _support_package_dirs()


def test_shadowed_reference_dependencies_remain_masked(tmp_path, monkeypatch):
    from merlin.targetgen import target_registry
    from merlin.targetgen.sandbox.answer_surfaces import _support_package_dirs

    helpers = tmp_path / "old_helpers"
    helpers.mkdir()
    reference = declaration(tmp_path / "reference", {"compiler": "../old_helpers"})
    selected = declaration(tmp_path / "selected")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(selected))
    monkeypatch.setattr(target_registry, "reference_targets", lambda: {"dependency_fixture": reference})
    monkeypatch.setattr(target_registry, "in_repo_support", lambda: {})
    monkeypatch.setattr(target_registry, "vendored_support_dirs", lambda: ())
    assert helpers in _support_package_dirs()
