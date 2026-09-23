"""Descriptor-relative resource paths are lexical ownership references, not access grants."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from merlin.targetgen import target_experiment as experiments


@pytest.fixture
def descriptor(tmp_path):
    path = tmp_path / "example/descriptor.yaml"
    path.parent.mkdir()
    path.write_text("target: synthetic_device\n")
    return experiments.load_target_experiment(path)


@pytest.mark.parametrize("relative", ["task/TASK.md", Path("task/TASK.md")])
def test_nested_resource_under_absolute_descriptor(descriptor, relative):
    assert descriptor.resource_path(relative) == descriptor.path.parent / "task/TASK.md"


def test_relative_descriptor_preserves_lexical_relative_path(descriptor, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    relative = replace(descriptor, path=Path("example/descriptor.yaml"))
    assert relative.resource_path("task/TASK.md") == Path("example/task/TASK.md")


def test_dot_names_resource_root(descriptor):
    assert descriptor.resource_path(".") == descriptor.path.parent


@pytest.mark.parametrize("relative", ["/absolute", Path("/absolute"), "..", "../task", "task/../other"])
def test_absolute_and_parent_traversal_refused(descriptor, relative):
    with pytest.raises(ValueError):
        descriptor.resource_path(relative)
    with pytest.raises(ValueError):
        descriptor.experiment_resource(relative)


def test_resolution_does_not_probe_present_or_missing_files(descriptor, monkeypatch):
    existing = descriptor.path.parent / "existing"
    existing.write_text("public input\n")
    missing = descriptor.path.parent / "not-created/task.md"

    def forbidden(*args, **kwargs):
        raise AssertionError("resource resolution probed the filesystem")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "stat", forbidden)
        scoped.setattr(Path, "lstat", forbidden)
        scoped.setattr(Path, "resolve", forbidden)
        assert descriptor.resource_path("existing") == existing
        assert descriptor.resource_path("not-created/task.md") == missing


def test_symlink_is_preserved_without_claiming_physical_containment(descriptor, tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (external / "task.md").write_text("external content\n")
    link = descriptor.path.parent / "linked"
    link.symlink_to(external, target_is_directory=True)
    resource = descriptor.resource_path("linked/task.md")
    assert resource == link / "task.md"
    # Lexical declaration and physical access containment are separate contracts.
    assert resource.resolve() == external / "task.md"
    assert not resource.resolve().is_relative_to(descriptor.path.parent)


def test_legacy_grant_spelling_and_external_absolute(descriptor, tmp_path, monkeypatch):
    root = tmp_path / "repo"
    monkeypatch.setattr(experiments, "repo_root", lambda: root)
    legacy = replace(descriptor, path=root / "merlin/experiments/example/descriptor.yaml")
    assert legacy.experiment_resource("task") == "experiments/example/task"
    example = replace(descriptor, path=root / "examples/device/target/descriptor.yaml")
    assert example.experiment_resource("task") == "examples/device/target/task"
    assert descriptor.experiment_resource("task") == str(descriptor.path.parent / "task")


def test_grant_spelling_uses_shared_resource_resolution(descriptor, tmp_path, monkeypatch):
    calls = []
    selected = tmp_path / "selected/task"

    def resource_path(self, relative):
        calls.append((self, relative))
        return selected

    monkeypatch.setattr(experiments.TargetExperiment, "resource_path", resource_path)
    monkeypatch.setattr(experiments, "repo_root", lambda: tmp_path / "repo")
    assert descriptor.experiment_resource("task") == str(selected)
    assert calls == [(descriptor, "task")]


@pytest.mark.parametrize("relative", ["/absolute", "../harness"])
def test_curated_harness_consumer_rejects_escaping_declaration(descriptor, relative):
    from merlin.targetgen.sandbox.toolchain import curated_harness_dir

    invalid = replace(descriptor, curated_harness=relative)
    with pytest.raises(ValueError):
        curated_harness_dir(invalid)


@pytest.mark.parametrize("absolute", [False, True])
def test_declared_resources_root_is_independent_of_descriptor(descriptor, tmp_path, monkeypatch, absolute):
    root = tmp_path / "repository"
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(root))
    selected = tmp_path / "external" if absolute else root / "authored/phase1"
    declaration = str(selected) if absolute else "authored/phase1"
    descriptor.path.write_text(yaml.safe_dump({"target": "synthetic_device", "resources_root": declaration}))
    loaded = experiments.load_target_experiment(descriptor.path)
    assert loaded.resource_path(".") == selected
    assert loaded.resource_path("task/TASK.md") == selected / "task/TASK.md"
    assert loaded.resource_path("input_bundles") == selected / "input_bundles"
    expected_grant = str(selected / "task") if absolute else "authored/phase1/task"
    assert loaded.experiment_resource("task") == expected_grant


@pytest.mark.parametrize("value", [None, True, False, 1, 1.5, "", "  ", "..", "a/../b", "a\x00b"])
@pytest.mark.parametrize("field", ["resources_root", "task_root", "contracts_root"])
def test_invalid_resources_root_refused(descriptor, value, field):
    descriptor.path.write_text(yaml.safe_dump({"target": "synthetic_device", field: value}))
    with pytest.raises(ValueError, match=field):
        experiments.load_target_experiment(descriptor.path)


@pytest.mark.parametrize("absolute", [False, True])
def test_task_root_overrides_only_task_subtree(descriptor, tmp_path, monkeypatch, absolute):
    root = tmp_path / "repository"
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(root))
    task = tmp_path / "external-tasks" if absolute else root / "examples/device/phase1/task"
    value = str(task) if absolute else "examples/device/phase1/task"
    document = {"target": "synthetic_device", "resources_root": "shared", "task_root": value}
    descriptor.path.write_text(yaml.safe_dump(document))
    loaded = experiments.load_target_experiment(descriptor.path)
    assert loaded.task_root == task
    assert experiments.declared_task_root(document, root=root) == task
    assert loaded.resource_path("task") == task
    assert loaded.resource_path("task/nested/TASK.md") == task / "nested/TASK.md"
    assert loaded.resource_path("input_bundles") == root / "shared/input_bundles"
    assert loaded.resource_path("task_other") == root / "shared/task_other"
    assert loaded.resource_path(".") == root / "shared"
    assert loaded.experiment_resource("task") == (str(task) if absolute else value)


@pytest.mark.parametrize("absolute", [False, True])
def test_contracts_root_overrides_only_contracts_subtree(descriptor, tmp_path, monkeypatch, absolute):
    from merlin.targetgen.sandbox.toolchain import curated_harness_dir

    root = tmp_path / "repository"
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(root))
    selected = tmp_path / "oot-contracts" if absolute else root / "examples/device/phase1/contracts"
    value = str(selected) if absolute else "examples/device/phase1/contracts"
    document = {
        "target": "synthetic_device",
        "resources_root": "retained",
        "contracts_root": value,
        "hardware_spec": {"curated_harness": "contracts/harness"},
    }
    descriptor.path.write_text(yaml.safe_dump(document))
    loaded = experiments.load_target_experiment(descriptor.path)
    assert loaded.contracts_root == selected
    assert experiments.declared_contracts_root(document, root=root) == selected
    assert loaded.resource_path("contracts") == selected
    assert loaded.resource_path("contracts/harness") == selected / "harness"
    assert loaded.resource_path("contracts_other") == root / "retained/contracts_other"
    assert loaded.resource_path("task") == root / "retained/task"
    assert loaded.resource_path("input_bundles") == root / "retained/input_bundles"
    assert curated_harness_dir(loaded) == ""
    (selected / "harness").mkdir(parents=True)
    assert curated_harness_dir(loaded) == str(selected / "harness")
    assert loaded.experiment_resource("contracts") == (str(selected) if absolute else value)


def test_task_root_resolution_is_lexical_and_absence_preserves_siblings(descriptor, tmp_path, monkeypatch):
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "missing", target_is_directory=True)
    selected = replace(descriptor, task_root=alias)

    def forbidden(*args, **kwargs):
        raise AssertionError("task ownership resolution probed the filesystem")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "stat", forbidden)
        scoped.setattr(Path, "lstat", forbidden)
        scoped.setattr(Path, "resolve", forbidden)
        assert experiments.declared_task_root({}) is None
        assert experiments.declared_task_root({"task_root": str(alias)}, root=tmp_path) == alias
        assert selected.resource_path("task/TASK.md") == alias / "TASK.md"
        assert descriptor.resource_path("task/TASK.md") == descriptor.path.parent / "task/TASK.md"


@pytest.fixture
def discovery(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    monkeypatch.delenv("MERLIN_TARGET_EXPERIMENT", raising=False)

    def write(relative, target):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"target": target}))
        return path

    return write


def test_authored_descriptor_identity_not_example_folder_selects_target(discovery):
    authored = discovery("examples/friendly-name/target/descriptor.yaml", "synthetic_device")
    discovery("examples/synthetic_device/target/descriptor.yaml", "other_device")
    discovery("merlin/experiments/capsule_bench/targets/synthetic_device/target_experiment.yaml", "synthetic_device")
    assert experiments.descriptor_for("synthetic_device") == authored


def test_explicit_matching_descriptor_precedes_authored_example(discovery, monkeypatch):
    discovery("examples/friendly-name/target/descriptor.yaml", "synthetic_device")
    explicit = discovery("operator/selected.yaml", "synthetic_device")
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(explicit))
    assert experiments.descriptor_for("synthetic_device") == explicit


def test_wrong_target_override_does_not_shadow_authored_descriptor(discovery, monkeypatch):
    authored = discovery("examples/friendly-name/target/descriptor.yaml", "synthetic_device")
    explicit = discovery("operator/selected.yaml", "other_device")
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(explicit))
    assert experiments.descriptor_for("synthetic_device") == authored


def test_wrong_target_example_does_not_shadow_legacy_identity(discovery):
    discovery("examples/synthetic_device/target/descriptor.yaml", "other_device")
    legacy = discovery("merlin/experiments/capsule_bench/targets/alias/target_experiment.yaml", "synthetic_device")
    assert experiments.descriptor_for("synthetic_device") == legacy
    assert experiments.descriptor_for("undeclared_device") is None


def test_isa_contract_lookup_uses_selected_resources(descriptor, tmp_path, monkeypatch):
    from merlin.targetgen import corpora, isa_rtl_crosscheck

    selected = tmp_path / "resources"
    (selected / "contracts").mkdir(parents=True)
    (descriptor.path.parent / "contracts").mkdir()
    descriptor.path.write_text(yaml.safe_dump({"target": "synthetic_device", "resources_root": str(selected)}))
    monkeypatch.setattr(corpora, "descriptor_path", lambda target: descriptor.path)
    assert isa_rtl_crosscheck.contracts_dir("synthetic_device") == selected / "contracts"


@pytest.mark.parametrize("present", [True, False])
def test_declared_bringup_set_never_falls_back_to_resource_contracts(descriptor, tmp_path, monkeypatch, present):
    from merlin.targetgen import corpora, isa_rtl_crosscheck

    selected = tmp_path / "selected-bringup"
    if present:
        selected.mkdir()
    (descriptor.path.parent / "contracts").mkdir()
    descriptor.path.write_text(
        yaml.safe_dump({"target": "synthetic_device", "hardware_spec": {"hwbringup_set": str(selected)}})
    )
    monkeypatch.setattr(corpora, "descriptor_path", lambda target: descriptor.path)
    assert isa_rtl_crosscheck.contracts_dir("synthetic_device") == (selected if present else None)


@pytest.mark.parametrize("present", [True, False])
def test_crosscheck_contracts_root_has_no_sibling_fallback(descriptor, tmp_path, monkeypatch, present):
    from merlin.targetgen import corpora, isa_rtl_crosscheck

    selected = tmp_path / "authored-contracts"
    if present:
        selected.mkdir()
    (descriptor.path.parent / "contracts").mkdir()
    descriptor.path.write_text(yaml.safe_dump({"target": "synthetic_device", "contracts_root": str(selected)}))
    monkeypatch.setattr(corpora, "descriptor_path", lambda target: descriptor.path)
    assert isa_rtl_crosscheck.contracts_dir("synthetic_device") == (selected if present else None)
