"""Declaration routing uses definition paths, never recipe naming conventions."""

from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0 import declarations as D
from merlin_experiments.spec import SpecError


def definition(tmp_path, name="one", **overrides):
    config = dict(
        descriptor="inputs/device.yaml",
        recipe="inputs/arbitrary.recipe.yaml",
        performance_template="templates/performance.yaml",
        profile="corpus-alias",
        hidden_profile="private/withheld.yaml",
    )
    config.update(overrides)
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(
            dict(
                schema_version=1,
                id=name,
                target="runtime-device",
                phases={0: dict(adapter="capsule_derivation", config=config)},
            )
        )
    )
    return path


def catalog(tmp_path, *paths):
    path = tmp_path / "catalog.yaml"
    path.write_text(yaml.safe_dump(dict(schema_version=1, experiments={p.stem: p.name for p in paths})))
    return path


def test_paths_and_optional_absence_without_private_reads(tmp_path, monkeypatch):
    path = definition(tmp_path)
    original = Path.read_text

    def read(self, *args, **kwargs):
        assert "withheld" not in str(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.chdir(tmp_path.parent)
    declaration = D.from_definition(path)
    assert declaration.recipe == tmp_path / "inputs/arbitrary.recipe.yaml"
    assert declaration.descriptor == tmp_path / "inputs/device.yaml"
    assert declaration.hidden_profile == tmp_path / "private/withheld.yaml"
    assert declaration.synth_profile is None
    assert set(declaration.profile_inputs()) == {
        "recipe",
        "performance_template",
        "synth_profile",
        "smt_profile",
        "hidden_profile",
    }


def test_target_and_profile_selection_and_ambiguity(tmp_path):
    first = definition(tmp_path)
    source = catalog(tmp_path, first)
    assert D.for_target("runtime-device", catalog_path=source).id == "one"
    assert D.for_target("corpus-alias", catalog_path=source).id == "one"
    with pytest.raises(SpecError, match="found 0"):
        D.for_target("missing", catalog_path=source)
    second = definition(tmp_path, "two", profile="different")
    source = catalog(tmp_path, first, second)
    with pytest.raises(SpecError, match="found 2"):
        D.for_target("runtime-device", catalog_path=source)


def test_missing_explicit_recipe_refused(tmp_path):
    path = definition(tmp_path)
    document = yaml.safe_load(path.read_text())
    config = document["phases"][0]["config"]
    for name in ("recipe", "performance_template", "hidden_profile"):
        config.pop(name)
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(SpecError, match="recipe"):
        D.from_definition(path)


def test_templates_skipped(tmp_path):
    path = definition(tmp_path)
    document = yaml.safe_load(path.read_text())
    document["kind"] = "template"
    path.write_text(yaml.safe_dump(document))
    assert D.all_declarations(catalog_path=catalog(tmp_path, path)) == ()
