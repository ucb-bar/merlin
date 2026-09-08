from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen.evaluation_cohort import (
    materialize_evaluation_cohort,
    validate_evaluation_cohort,
)
from merlin.targetgen.target_experiment import load_target_experiment


RADIANCE = repo_root() / "merlin/experiments/capsule_bench/targets/radiance/target_experiment.yaml"


def test_radiance_derived_gsim_materialization_makes_l3_mandatory(tmp_path, monkeypatch):
    te = load_target_experiment(RADIANCE)
    monkeypatch.setattr(
        "merlin.targetgen.evaluation_cohort.engine_preflight",
        lambda _te, stage: {
            "stage": stage,
            "required_tier": "L3",
            "requested_engine": "gsim",
            "selected": {"available": True, "engine": "gsim", "reason": "receipt:test"},
            "ok": True,
            "problems": [],
        },
    )
    out = tmp_path / "derived-gsim"
    record = materialize_evaluation_cohort(out, te, "derived_gsim")

    assert record["n_capsules"] == 14
    assert record["after"] == "search_converged"
    assert record["engine_preflight"]["ok"] is True
    for name in te.evaluation_cohort("derived_gsim")["include_capsules"]:
        doc = yaml.safe_load((out / name / "capsule.yaml").read_text())
        assert "L3" in doc["required_oracle_tiers"]
        assert "max_oracle_tier" not in doc
        assert doc["evaluation_stage"] == {
            "name": "derived_gsim",
            "policy": "radiance_model_derived_l3_frozen_candidate_v1",
            "after": "search_converged",
            "required_oracle_tier": "L3",
            "oracle_engine": "gsim",
        }
    assert validate_evaluation_cohort(out, te) == record


def test_evaluation_materialization_refuses_stale_files(tmp_path, monkeypatch):
    te = load_target_experiment(RADIANCE)
    out = tmp_path / "not-empty"
    out.mkdir()
    (out / "stale").write_text("old")
    with pytest.raises(ValueError, match="not empty"):
        materialize_evaluation_cohort(out, te, "derived_gsim")


def test_evaluation_validation_detects_capsule_mutation(tmp_path, monkeypatch):
    te = load_target_experiment(RADIANCE)
    monkeypatch.setattr(
        "merlin.targetgen.evaluation_cohort.engine_preflight",
        lambda _te, stage: {"stage": stage, "ok": True},
    )
    out = tmp_path / "derived-gsim"
    materialize_evaluation_cohort(out, te, "derived_gsim")
    record = json.loads((out / ".evaluation_cohort.json").read_text())
    name = record["capsules"][0]["name"]
    with (out / name / "README.md").open("a") as handle:
        handle.write("\nmutation\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_evaluation_cohort(out, te)


def test_unknown_evaluation_stage_fails_closed():
    te = load_target_experiment(RADIANCE)
    with pytest.raises(KeyError, match="declared stages"):
        te.evaluation_cohort("not-a-stage")
