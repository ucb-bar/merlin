"""Declared comparison preserves grouping without discovering private or sibling inputs."""

import json
from pathlib import Path

import pytest
import yaml
from merlin_experiments import cli
from merlin_experiments.phase0.comparison import build
from merlin_experiments.phase0.profiles import build_comparison_manifest
from merlin_experiments.spec import SpecError


@pytest.fixture
def definitions(tmp_path):
    paths = []
    for target, dtype in (("first", "i8"), ("second", "f32")):
        root = tmp_path / target
        root.mkdir()
        (root / "recipe.yaml").write_text(
            yaml.safe_dump(
                {
                    "datapath": {"operand_dtype": dtype},
                    "capsules": [{"name": "op", "op": "matmul"}, {"name": "model", "kind": "model"}],
                }
            )
        )
        (root / "performance.yaml").write_text("sweeps: []\n")
        path = root / "experiment.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "id": target,
                    "target": target,
                    "phases": {
                        0: {
                            "adapter": "capsule_derivation",
                            "config": {
                                "descriptor": "descriptor.yaml",
                                "recipe": "recipe.yaml",
                                "performance_template": "performance.yaml",
                                "hidden_profile": "private.yaml",
                            },
                        }
                    },
                }
            )
        )
        paths.append(path)
    return paths


def test_explicit_comparison_does_not_touch_private_files(definitions, monkeypatch):
    original = Path.stat

    def stat(path, *args, **kwargs):
        assert path.name != "private.yaml", "comparison touched private input"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    result = build(definitions)
    assert result["comparison_sets"] == {
        "matmul": [
            {"target": "first", "name": "op", "dtype": "i8", "label": "public"},
            {"target": "second", "name": "op", "dtype": "f32", "label": "public"},
        ]
    }
    assert result["scope"] == "current-declared-public-recipes"


def test_cli_comparison_uses_existing_experiment_entrypoint(definitions, capsys):
    assert cli.main(["corpus", "compare", *map(str, definitions)]) == 0
    assert json.loads(capsys.readouterr().out) == build(definitions)


def test_empty_or_duplicate_selection_refuses(definitions):
    for selected in ([], definitions[:1], [definitions[0], definitions[0]]):
        with pytest.raises(SpecError, match="distinct targets"):
            build(selected)
    with pytest.raises(ValueError, match="explicitly selected"):
        build_comparison_manifest([])
