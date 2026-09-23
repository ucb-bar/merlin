"""Authored target recipes have one example home; generated/private files do not."""

import yaml
from merlin_experiments.phase0.declarations import all_declarations

from merlin.common.paths import repo_root
from merlin.targetgen.target_experiment import load_target_experiment


def test_catalog_recipes_are_example_owned_and_shared_with_numeric_policy():
    declarations = all_declarations()
    assert declarations, "the public experiment catalog must not become empty"
    for declaration in declarations:
        expected = declaration.definition.parent / "phase0/recipe.yaml"
        assert declaration.recipe == expected
        assert expected.is_file()
        assert declaration.performance_template == repo_root() / "experiments/templates/phase0/performance.yaml"
        assert declaration.performance_template.is_file()
        expected_descriptor = declaration.definition.parent / "target/descriptor.yaml"
        assert declaration.descriptor == expected_descriptor
        assert expected_descriptor.is_file() and not expected_descriptor.is_symlink()
        descriptor = load_target_experiment(declaration.descriptor)
        legacy_root = repo_root() / "merlin/experiments/capsule_bench/targets" / declaration.definition.parent.name
        assert descriptor.resource_path(".") == legacy_root
        legacy_descriptor = legacy_root / "target_experiment.yaml"
        assert legacy_descriptor.is_symlink()
        assert legacy_descriptor.resolve() == expected_descriptor
        assert repo_root() / descriptor.numeric_profile == expected
        for sidecar in (declaration.synth_profile, declaration.smt_profile, declaration.hidden_profile):
            assert sidecar is None or not sidecar.is_relative_to(declaration.definition.parent)
        assert declaration.synth_profile.is_relative_to(repo_root() / "experiments/reference-data/phase0")
        assert declaration.synth_profile.is_file()


def test_phase0_examples_do_not_embed_generated_or_private_payloads():
    for declaration in all_declarations():
        directory = declaration.definition.parent / "phase0"
        assert {path.name for path in directory.iterdir()} <= {"recipe.yaml", "README.md", "AGENT.md"}
        target_directory = declaration.definition.parent / "target"
        assert (target_directory / "descriptor.yaml").is_file()
        assert not list(target_directory.rglob("golden.yaml"))
        assert not list(target_directory.rglob("*.hidden.yaml"))
        assert not list(target_directory.rglob("*.safetensors"))


def test_catalog_functional_examples_require_a_new_reviewed_corpus():
    for declaration in all_declarations():
        document = yaml.safe_load(declaration.definition.read_text(encoding="utf-8"))
        assert document["phases"][1]["config"]["require_reviewed_corpus"] is True


def test_declared_authored_tasks_are_example_owned_with_navigation_only_aliases():
    selected = []
    for declaration in all_declarations():
        descriptor = load_target_experiment(declaration.descriptor)
        if descriptor.task_root is None:
            continue
        tasks = declaration.definition.parent / "phase1/task"
        assert descriptor.resource_path("task") == tasks
        assert tasks.is_dir() and not tasks.is_symlink()
        assert {path.name for path in tasks.iterdir()} == {
            "AGENT.md",
            "TASK.md",
            "TASK_full.md",
            "TASK_pilot.md",
            "TASK_realistic.md",
        }
        for path in tasks.glob("TASK*.md"):
            assert not path.is_symlink()
            old = descriptor.resource_path(".") / "task" / path.name
            assert old.is_symlink() and old.resolve() == path
        selected.append(tasks)
    assert selected, "the task-ownership migration must exercise a real example"


def test_secondary_descriptors_share_the_authored_recipe_instead_of_a_copy():
    recipes = {item.recipe for item in all_declarations()}
    root = repo_root()
    selected = []
    for path in (root / "merlin/experiments/capsule_bench/targets").glob("*/target_experiment.yaml"):
        experiment = load_target_experiment(path)
        if experiment.numeric_profile is not None:
            recipe = root / experiment.numeric_profile
            assert recipe in recipes and recipe.is_file(), path
            selected.append(path)
    assert len(selected) >= len(recipes)
