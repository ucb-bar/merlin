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


def test_a_percentile_rule_and_a_report_only_rule():
    rows = [{"e": v} for v in (0.01, 0.02, 0.03, 0.04, 0.50)]
    base = [{"e": 0.05}] * 5
    rules = [
        {"metric": "e", "aggregate": "p50", "max_ratio_to_baseline": 1.0},
        {"metric": "e", "aggregate": "max", "max_ratio_to_baseline": 1.0, "report_only": True},
    ]
    verdict = admission(rules, rows, base)
    assert verdict["status"] == "pass"  # the one outlier only shows in the report-only max
    assert [c["status"] for c in verdict["checks"]] == ["pass", "info"]
    assert verdict["checks"][1]["within"] is False


def test_seeded_draws_are_reproducible_and_keep_the_stored_draw(tmp_path):
    import numpy as np

    from merlin.model_check import references as REF
    from merlin.model_check.bundle import Bundle
    from merlin.model_check.models import Model

    derived = tmp_path / "private" / "derived" / "m" / "visible" / "v00"
    derived.mkdir(parents=True)
    noise = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    np.savez(derived / "inputs.npz", noise=noise, other=np.ones(2, np.float32))
    judged = {"kind": "per_channel", "reference": {"seeds": {"input": "noise", "count": 3}}}
    model = Model(name="m", workload="w", chain={"kind": "single"}, programs=(), judged=judged)
    first = REF._cases(Bundle(tmp_path / "b1"), model, tmp_path / "private", [("visible", "v00")])  # noqa: SLF001
    again = REF._cases(Bundle(tmp_path / "b2"), model, tmp_path / "private", [("visible", "v00")])  # noqa: SLF001
    assert [c["draw"] for c in first] == [0, 1, 2] and first[0]["inputs"] == derived / "inputs.npz"
    draws = [np.load(c["inputs"])["noise"] for c in first[1:]]
    assert all(d.shape == noise.shape and d.dtype == noise.dtype for d in draws)
    assert not np.array_equal(draws[0], draws[1]) and not np.array_equal(draws[0], noise)
    assert np.array_equal(np.load(again[1]["inputs"])["noise"], draws[0])
    assert np.array_equal(np.load(first[1]["inputs"])["other"], np.ones(2, np.float32))


def test_a_changed_bar_rejudges_a_built_bundle_from_its_stored_metrics(tmp_path):
    import json

    from merlin.model_check import bundle as B
    from merlin.model_check import references as REF

    model = load(repo_root() / "examples" / "gemmini" / "numerics" / "models.yaml").model("tiny_llama")
    root = tmp_path / "bundle"
    (root).mkdir()
    (root / "MANIFEST.json").write_text(json.dumps({"schema": B.MANIFEST_SCHEMA, "models": {}}))
    # The launch recipe's measured KL: 0.043 visible, 0.066 sealed (and one extra draw, to cover seeds/<k>).
    cases = {"visible/v00": 0.043, "sealed/s00": 0.066, "sealed/s00#s1": 0.064}
    for name, kl in cases.items():
        where, _, draw = name.partition("#s")
        here = root / "tiny_llama" / where / ("seeds/" + draw if draw else "")
        here.mkdir(parents=True)
        (here / "metrics_int8.json").write_text(json.dumps({"kl_mean": kl, "top1_agree": 0.93}))
    stale = {"status": "fail", "checks": []}
    quality = {"model": "tiny_llama", "inputs": dict.fromkeys(cases, {}), "admission": {"sealed": stale}}
    (root / "tiny_llama" / "reference_quality.json").write_text(json.dumps(quality))

    summary = REF.rejudge(B.Bundle(root), model)
    assert {split: a["status"] for split, a in summary["admission"].items()} == {"sealed": "pass", "visible": "pass"}
    assert summary["admission"]["sealed"]["inputs"] == 2
    written = json.loads((root / "tiny_llama" / "reference_quality.json").read_text())
    assert written["admission"] == summary["admission"] and written["model"] == "tiny_llama"
