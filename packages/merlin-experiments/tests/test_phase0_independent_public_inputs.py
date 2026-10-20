"""Prospective functional derivation excludes private model inputs entirely."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import requirements as R

from merlin.common.paths import repo_root


def recipe(tmp_path, name):
    source = repo_root() / "examples/gemmini/phase0" / name
    path = tmp_path / "recipe.yaml"
    path.write_bytes(source.read_bytes())
    return path


def test_explicit_functional_policy_refuses_private_selection_before_opening_it(tmp_path, monkeypatch):
    path = recipe(tmp_path, "recipe-public-functional.yaml")
    monkeypatch.setattr(R, "from_definition", lambda _: SimpleNamespace(recipe=path))
    # No descriptor, private graph, output or compiler is needed to make this
    # selection refusal. In particular, never try to parse the private file.
    private = tmp_path / "private-model-shapes.json"
    private.write_bytes(b"not valid JSON and not a derivation input")
    monkeypatch.setattr(R, "load_target_experiment", lambda _: pytest.fail("descriptor read before refusal"))
    with pytest.raises(ValueError, match="private layer shapes are not inputs"):
        R.derive("definition", {}, rtl_facts="unused", output_root=tmp_path / "out", heldout_layers=private)


def test_functional_source_policy_does_not_require_private_selection(tmp_path, monkeypatch):
    path = recipe(tmp_path, "recipe-public-functional.yaml")
    monkeypatch.setattr(R, "from_definition", lambda _: SimpleNamespace(recipe=path, descriptor="selected"))

    def next_stage(_):
        raise RuntimeError("continued to actual descriptor selection")

    monkeypatch.setattr(R, "load_target_experiment", next_stage)
    with pytest.raises(RuntimeError, match="continued to actual descriptor"):
        R.derive("definition", {}, rtl_facts="unused", output_root=tmp_path / "out")


def test_legacy_policy_retains_original_private_guard(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import heldout_layers as H

    path = recipe(tmp_path, "recipe-public-connected.yaml")
    monkeypatch.setattr(R, "from_definition", lambda _: SimpleNamespace(recipe=path, descriptor="selected"))
    monkeypatch.setattr(R, "load_target_experiment", lambda _: SimpleNamespace(workload_spec={}))

    def original_guard(*args, **kwargs):
        raise RuntimeError("original legacy private guard")

    monkeypatch.setattr(H, "load", original_guard)
    with pytest.raises(RuntimeError, match="original legacy private guard"):
        R.derive("definition", {}, rtl_facts="unused", output_root=tmp_path / "out",
                 heldout_layers=tmp_path / "selected-private.json")


def test_malformed_prospective_policy_cannot_select_independence(tmp_path):
    path = recipe(tmp_path, "recipe-public-functional.yaml")
    body = yaml.safe_load(path.read_bytes())
    body["public_joint_boundaries"]["max_members"] = 0
    path.write_text(yaml.safe_dump(body))
    with pytest.raises(ValueError):
        R._independent_public_sources(path)


@pytest.mark.parametrize("payload", [b"[broken", b"[]", b"{}"])
def test_other_recipes_keep_validation_at_original_observation_point(tmp_path, payload):
    path = tmp_path / "recipe.yaml"
    path.write_bytes(payload)
    assert R._independent_public_sources(path) is False


def test_unreadable_legacy_recipe_keeps_original_error_order(tmp_path):
    assert R._independent_public_sources(Path(tmp_path / "absent.yaml")) is False
