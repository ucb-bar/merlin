"""Existing source/law checks feed one emitter; no machine qualification claimed."""

import pytest
from author_skill_fixtures import load_script


@pytest.mark.parametrize("shape", [(7, 12, 19), (8, 12, 20)])
def test_actual_readout_transforms_reach_same_encoder_with_original_completion(shape, tmp_path):
    recipe = load_script("heavy-pass-recipes", "counted_emission_recipe")
    result = recipe.recipe(shape=shape, output=tmp_path)
    variants = result["variants"]
    assert len({r["same_effect_encoder"] for r in variants.values()}) == 1
    assert len({r["emitted_sha256"] for r in variants.values()}) == 4
    assert variants["batched"]["dynamic_epoch_drains"] < variants["original"]["dynamic_epoch_drains"]
    assert variants["ready"]["dynamic_epoch_drains"] < variants["batched"]["dynamic_epoch_drains"]
    for name, row in variants.items():
        assert row["last_effect"] == "original_full_completion"
        assert row["completion_sites"] == row["receipt"]["completion_mapping"]["emitted_static_drains"] > 0
        assert row["retained_loop_blocks"] > 1
        assert row["emitted_operations"] < 5000
        assert "original_full_completion" in (tmp_path / (name + ".llvm.mlir")).read_text()
    assert len({r["source_check"]["compute"] for r in variants.values()}) == 1
    assert result["execution_qualified"] is False and result["performance_claim"] is False
