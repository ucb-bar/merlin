"""Reference admission rules are data in the models file and are judged per split when a bundle is built."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.model_check.metrics import admission
from merlin.model_check.models import ModelsFileError, load


def test_absolute_bounds_take_the_aggregate_over_inputs():
    rows = [{"top1_agree": 1.0, "cosine": 0.995}, {"top1_agree": 0.0, "cosine": 0.991}]
    rules = [
        {"metric": "top1_agree", "aggregate": "mean", "min": 0.5},
        {"metric": "cosine", "aggregate": "mean", "min": 0.994},
    ]
    verdict = admission(rules, rows)
    assert [c["status"] for c in verdict["checks"]] == ["pass", "fail"]
    assert verdict["status"] == "fail"


def test_a_baseline_relative_bound_is_a_multiple_of_the_baseline_aggregate():
    rows = [{"rel_rmse_worst": 0.06}, {"rel_rmse_worst": 0.10}]
    baseline = [{"rel_rmse_worst": 0.03}, {"rel_rmse_worst": 0.05}]
    rules = [
        {"metric": "rel_rmse_worst", "aggregate": "mean", "max_ratio_to_baseline": 2.0},
        {"metric": "rel_rmse_worst", "aggregate": "max", "max_ratio_to_baseline": 2.0},
    ]
    verdict = admission(rules, rows, baseline)
    assert verdict["status"] == "pass"
    assert [round(c["bound"], 6) for c in verdict["checks"]] == [0.08, 0.1]


def test_a_rule_without_data_is_unknown_never_a_pass():
    rules = [{"metric": "rel_rmse_worst", "aggregate": "mean", "max_ratio_to_baseline": 2.0}]
    assert admission(rules, [{"rel_rmse_worst": 0.01}], [])["status"] == "unknown"
    assert admission([{"metric": "kl_mean", "max": 0.05}], [{"top1_agree": 1.0}])["status"] == "unknown"
    assert admission([], [{"kl_mean": 0.0}])["status"] == "unknown"


def test_the_target_models_file_declares_admission_for_every_model():
    models = load(repo_root() / "examples" / "gemmini" / "numerics" / "models.yaml")
    for model in models.models.values():
        assert model.judged.get("admission"), model.name


def _models_with(tmp_path: Path, judged_extra: dict) -> Path:
    data = yaml.safe_load((repo_root() / "examples" / "gemmini" / "numerics" / "models.yaml").read_text())
    data["models"] = {"resnet50": data["models"]["resnet50"]}
    data["models"]["resnet50"]["judged"] = {**data["models"]["resnet50"]["judged"], **judged_extra}
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_malformed_admission_rules_are_refused(tmp_path):
    with pytest.raises(ModelsFileError, match="one of"):
        load(_models_with(tmp_path, {"admission": [{"metric": "cosine", "min": 0.9, "max": 1.0}]}))
    with pytest.raises(ModelsFileError, match="reference.baseline"):
        load(_models_with(tmp_path, {"admission": [{"metric": "cosine", "max_ratio_to_baseline": 2.0}]}))
