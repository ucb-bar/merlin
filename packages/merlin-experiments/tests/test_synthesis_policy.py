"""Authored synthesis policy is explicit, narrow, and attributable."""

from copy import deepcopy
from hashlib import sha256

import pytest
import yaml
from merlin_experiments.phase0.synthesis_policy import apply_model_gates


def _recipe(tmp_path, policy):
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump({"synthesis_model_gates": policy}))
    return path


def _result():
    return {
        "capsules": [
            {"name": "composition", "kind": "model", "gate": {"other": "preserved"}},
            {"name": "another", "kind": "model"},
        ],
        "provenance": {"existing": True},
    }


def test_authored_gate_preserves_other_fields_and_records_exact_recipe(tmp_path):
    policy = {"composition": {"after_op_pass_fraction": 0.0, "reason": "Reviewed cheap composition"}}
    recipe = _recipe(tmp_path, policy)
    source = _result()
    before = deepcopy(source)
    result = apply_model_gates(source, recipe)
    assert source == before
    assert result["capsules"][0]["gate"] == {"other": "preserved", "after_op_pass_fraction": 0.0}
    assert result["capsules"][1] == before["capsules"][1]
    assert result["provenance"]["existing"] is True
    record = result["provenance"]["authored_model_gates"]
    assert record["recipe_sha256"] == sha256(recipe.read_bytes()).hexdigest()
    assert record["entries"][0]["previous_gate"] == {"other": "preserved"}
    assert record["entries"][0]["reason"] == policy["composition"]["reason"]


@pytest.mark.parametrize("fraction", [True, False, -0.1, 1.1, float("nan"), float("inf"), "0.0", None])
def test_invalid_threshold_refuses(tmp_path, fraction):
    recipe = _recipe(tmp_path, {"composition": {"after_op_pass_fraction": fraction, "reason": "review"}})
    with pytest.raises(ValueError, match="finite number"):
        apply_model_gates(_result(), recipe)


@pytest.mark.parametrize(
    "setting",
    [
        {},
        {"after_op_pass_fraction": 0},
        {"after_op_pass_fraction": 0, "reason": " "},
        {"after_op_pass_fraction": 0, "reason": "review", "atol": 100},
    ],
)
def test_incomplete_or_unrelated_policy_refuses(tmp_path, setting):
    with pytest.raises(ValueError):
        apply_model_gates(_result(), _recipe(tmp_path, {"composition": setting}))


@pytest.mark.parametrize(
    "entries", [[], [{"name": "composition", "kind": "op"}], [{"name": "composition", "kind": "model"}] * 2]
)
def test_stale_ambiguous_or_non_model_selector_refuses(tmp_path, entries):
    recipe = _recipe(tmp_path, {"composition": {"after_op_pass_fraction": 0, "reason": "review"}})
    with pytest.raises(ValueError, match="exactly one synthesized model"):
        apply_model_gates({"capsules": entries}, recipe)


def test_recipe_without_policy_preserves_derivation(tmp_path):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("capsules: []\n")
    result = _result()
    assert apply_model_gates(result, recipe) is result
