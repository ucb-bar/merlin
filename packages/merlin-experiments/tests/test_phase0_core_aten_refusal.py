"""A retained Core ATen recipe is refused at every Phase 0 entry point; ordinary recipes derive.

Core ATen suites are Phase 0 spec-validation evidence, never capsules, so a recipe
with a top-level ``core_aten:`` block must fail closed instead of selecting a
parallel corpus. Ordinary recipes keep the megaphase derivation route. These tests
stop at the first step past the refusal: end-to-end Gemmini derivation still
raises in ``selected_facts`` until the contract's output_epilogue probe lands.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import core_aten_stage as stage
from merlin_experiments.spec import SpecError

REFUSAL = "Core ATen suites are Phase 0 spec-validation evidence, not capsules"
RETAINED_BLOCKS = [
    {"capsule_policy": "derived_only", "capsules": [], "core_aten": {"cohorts": {}}},
    # Malformed or empty blocks are refused the same way, not by stage validation.
    {"core_aten": None},
    {"capsule_policy": "authored", "capsules": [{"name": "x"}], "core_aten": []},
]


def _gemmini():
    from merlin.common.paths import repo_root

    root = repo_root() / "examples/gemmini"
    return root / "phase0/recipe.yaml", root / "target/descriptor.yaml"


def _write(tmp_path: Path, document: dict) -> Path:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.safe_dump(document))
    return recipe


@pytest.mark.parametrize("document", RETAINED_BLOCKS)
def test_helper_refuses_any_top_level_core_aten_block(tmp_path, document):
    with pytest.raises(ValueError, match=REFUSAL) as raised:
        stage.refuse_retained_stage(_write(tmp_path, document))
    assert "spec-validation case selection" in str(raised.value)
    assert "make_spec_validation_selection.py" in str(raised.value)


def test_helper_accepts_ordinary_recipes(tmp_path):
    recipe, _ = _gemmini()
    stage.refuse_retained_stage(recipe)
    stage.refuse_retained_stage(None)
    stage.refuse_retained_stage(tmp_path / "absent.yaml")
    stage.refuse_retained_stage(_write(tmp_path, {"capsule_policy": "derived_only", "capsules": []}))


@pytest.mark.parametrize("document", RETAINED_BLOCKS)
def test_requirements_derive_refuses_before_any_output(tmp_path, monkeypatch, document):
    from merlin_experiments.phase0 import requirements

    recipe = _write(tmp_path, document)
    monkeypatch.setattr(
        requirements,
        "from_definition",
        lambda _: SimpleNamespace(recipe=recipe, target="synthetic", descriptor=tmp_path / "descriptor.yaml"),
    )
    with pytest.raises(ValueError, match=REFUSAL):
        requirements.derive(
            tmp_path / "experiment.yaml", {}, rtl_facts=tmp_path / "facts.json", output_root=tmp_path / "derive"
        )
    assert not (tmp_path / "derive").exists()


def test_requirements_derive_routes_ordinary_recipe_to_model_derivation(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import requirements

    recipe, descriptor = _gemmini()
    monkeypatch.setattr(
        requirements,
        "from_definition",
        lambda _: SimpleNamespace(recipe=recipe, target="gemmini", descriptor=descriptor),
    )
    # The megaphase path's first gate: model derivation needs explicit RTL facts.
    with pytest.raises(ValueError, match="requires explicit --rtl-facts"):
        requirements.derive(tmp_path / "experiment.yaml", {}, rtl_facts=None, output_root=tmp_path / "derive")


@pytest.mark.parametrize("document", RETAINED_BLOCKS)
def test_generate_target_refuses_before_any_output(tmp_path, document):
    from merlin_experiments.phase0.generation import generate_target

    _, descriptor = _gemmini()
    with pytest.raises(ValueError, match=REFUSAL):
        generate_target(
            "gemmini",
            descriptor=descriptor,
            output_root=tmp_path / "out",
            recipe=_write(tmp_path, document),
            performance_template=tmp_path / "performance.yaml",
            evidence_mode="diagnostic",
        )
    assert not (tmp_path / "out").exists()


def test_generate_target_routes_ordinary_recipe_to_profile_derivation(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import generation

    class Reached(Exception):
        pass

    def load_profile(*args, **kwargs):
        raise Reached

    # Stop at the first megaphase step; the rest needs merlin-80's selected_facts fix.
    monkeypatch.setattr(generation, "load_profile", load_profile)
    recipe, descriptor = _gemmini()
    with pytest.raises(Reached):
        generation.generate_target(
            "gemmini",
            descriptor=descriptor,
            output_root=tmp_path / "out",
            recipe=recipe,
            performance_template=tmp_path / "performance.yaml",
            evidence_mode="diagnostic",
        )


@pytest.mark.parametrize("document", RETAINED_BLOCKS)
def test_selected_synthesis_refuses_even_without_a_profile(tmp_path, document):
    from merlin_experiments.phase0.profiles import verify_selected_synthesis

    with pytest.raises(ValueError, match=REFUSAL):
        verify_selected_synthesis(None, recipe=_write(tmp_path, document))


def test_runner_refuses_retained_recipe_as_operator_input(tmp_path):
    from merlin_experiments.runner import _phase0_operator_inputs

    recipe = _write(tmp_path, RETAINED_BLOCKS[0])
    template = tmp_path / "performance.yaml"
    template.write_text("{}\n")
    command = {
        "inputs": {"recipe": str(recipe), "performance_template": str(template)},
        "argv": ["--recipe", str(recipe), "--performance-template", str(template)],
    }
    with pytest.raises(SpecError, match=REFUSAL):
        _phase0_operator_inputs(command)
