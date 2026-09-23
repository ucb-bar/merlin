"""Repository gates consume explicit Phase 0 declarations, not directory conventions."""

import importlib.util

import yaml
from merlin_experiments.phase0 import declarations as D

from merlin.common.paths import repo_root


def definition(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "id": "fixture",
                "target": "runtime-device",
                "phases": {
                    0: {
                        "adapter": "capsule_derivation",
                        "config": {
                            "descriptor": "inputs/device.yaml",
                            "recipe": "inputs/arbitrary.recipe.yaml",
                            "performance_template": "templates/performance.yaml",
                            "hidden_profile": "private/withheld.yaml",
                        },
                    }
                },
            }
        )
    )
    return path


def tool(name):
    spec = importlib.util.spec_from_file_location(name, repo_root() / "build_tools/scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_holdout_gate_uses_declared_recipe_and_reports_private_absence(tmp_path, monkeypatch):
    declaration = D.from_definition(definition(tmp_path))
    declaration.recipe.parent.mkdir()
    declaration.recipe.write_text("datapath: {}\n")
    checker = tool("check_holdout_disjointness")
    monkeypatch.setattr(checker, "for_target", lambda target: declaration)
    assert checker.audit("runtime-device")["status"] == "no_holdout_sidecar"
    declaration.hidden_profile.parent.mkdir()
    declaration.hidden_profile.write_text("capsules: []\n")
    assert checker.audit("runtime-device")["status"] == "no_target_experiment"


def test_semantic_roster_uses_runtime_targets_not_sidecars(tmp_path, monkeypatch):
    declaration = D.from_definition(definition(tmp_path))
    monkeypatch.setattr(D, "all_declarations", lambda: (declaration,))
    checker = tool("check_semantic_coverage")
    assert checker._targets_with_profiles() == ["runtime-device"]
