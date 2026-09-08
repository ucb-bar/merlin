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
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "schedule.mlir").write_text("module {}\n")
    out = tmp_path / "derived-gsim"
    record = materialize_evaluation_cohort(out, te, "derived_gsim", candidate)

    assert record["n_capsules"] == len(te.evaluation_cohort("derived_gsim")["include_capsules"])
    assert record["after"] == "search_converged"
    assert record["engine_preflight"]["ok"] is True
    for name in te.evaluation_cohort("derived_gsim")["include_capsules"]:
        doc = yaml.safe_load((out / name / "capsule.yaml").read_text())
        assert "L3" in doc["required_oracle_tiers"]
        assert "max_oracle_tier" not in doc
        assert "oracle_tier_ceiling" not in doc, (
            "the frozen evaluation copy must unlock L3 after the search view constrained execution")
        assert doc["evaluation_stage"] == {
            "name": "derived_gsim",
            "policy": "radiance_model_derived_l3_frozen_candidate_v1",
            "after": "search_converged",
            "required_oracle_tier": "L3",
            "oracle_engine": "gsim",
        }
    assert validate_evaluation_cohort(out, te, candidate) == record


def test_evaluation_materialization_refuses_stale_files(tmp_path, monkeypatch):
    te = load_target_experiment(RADIANCE)
    out = tmp_path / "not-empty"
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "schedule.mlir").write_text("module {}\n")
    out.mkdir()
    (out / "stale").write_text("old")
    with pytest.raises(ValueError, match="not empty"):
        materialize_evaluation_cohort(out, te, "derived_gsim", candidate)


def test_evaluation_materialization_refuses_empty_candidate(tmp_path):
    te = load_target_experiment(RADIANCE)
    candidate = tmp_path / "empty-candidate"
    candidate.mkdir()
    with pytest.raises(ValueError, match="has no files"):
        materialize_evaluation_cohort(tmp_path / "derived-gsim", te, "derived_gsim", candidate)


def test_evaluation_validation_detects_capsule_mutation(tmp_path, monkeypatch):
    te = load_target_experiment(RADIANCE)
    monkeypatch.setattr(
        "merlin.targetgen.evaluation_cohort.engine_preflight",
        lambda _te, stage: {"stage": stage, "ok": True},
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "schedule.mlir").write_text("module {}\n")
    out = tmp_path / "derived-gsim"
    materialize_evaluation_cohort(out, te, "derived_gsim", candidate)
    record = json.loads((out / ".evaluation_cohort.json").read_text())
    name = record["capsules"][0]["name"]
    with (out / name / "README.md").open("a") as handle:
        handle.write("\nmutation\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_evaluation_cohort(out, te, candidate)


def test_evaluation_validation_detects_candidate_mutation(tmp_path, monkeypatch):
    te = load_target_experiment(RADIANCE)
    monkeypatch.setattr(
        "merlin.targetgen.evaluation_cohort.engine_preflight",
        lambda _te, stage: {"stage": stage, "ok": True},
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    schedule = candidate / "schedule.mlir"
    schedule.write_text("module {}\n")
    out = tmp_path / "derived-gsim"
    materialize_evaluation_cohort(out, te, "derived_gsim", candidate)
    schedule.write_text("module { func.func private @changed() }\n")
    with pytest.raises(ValueError, match="candidate content digest mismatch"):
        validate_evaluation_cohort(out, te, candidate)


def test_unknown_evaluation_stage_fails_closed():
    te = load_target_experiment(RADIANCE)
    with pytest.raises(KeyError, match="declared stages"):
        te.evaluation_cohort("not-a-stage")


def _passing_predecessor_score(path: Path, candidate: Path, record: dict) -> None:
    names = sorted(row["name"] for row in record["capsules"])
    tier = record["required_oracle_tier"]
    path.write_text(json.dumps({
        "package": str(candidate.resolve()),
        "integrity_status": "clean",
        "gradeable": True,
        "n_capsules": len(names),
        "n_passed": len(names),
        "per_capsule": [
            {"capsule": name, "status": "pass", "tiers": {tier: "pass"}}
            for name in names
        ],
        "pass_evidence": {"rtl_backed": len(names)},
    }))


def test_kernel_comparison_requires_exact_derived_gsim_pass(tmp_path, monkeypatch):
    te = load_target_experiment(RADIANCE)
    monkeypatch.setattr(
        "merlin.targetgen.evaluation_cohort.engine_preflight",
        lambda _te, stage: {"stage": stage, "ok": True},
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "schedule.mlir").write_text("module {}\n")

    with pytest.raises(ValueError, match="requires --predecessor-cohort"):
        materialize_evaluation_cohort(
            tmp_path / "comparison-refused", te, "kernel_library_comparison", candidate)

    derived = tmp_path / "derived"
    prior = materialize_evaluation_cohort(derived, te, "derived_gsim", candidate)
    score = tmp_path / "derived-score.json"
    _passing_predecessor_score(score, candidate, prior)
    comparison = tmp_path / "comparison"
    record = materialize_evaluation_cohort(
        comparison, te, "kernel_library_comparison", candidate,
        predecessor_cohort=derived, predecessor_score=score)

    evidence = record["predecessor_pass_evidence"]
    assert evidence["stage"] == "derived_gsim"
    assert evidence["n_passed"] == evidence["n_capsules"] == prior["n_capsules"]
    assert evidence["candidate_tree_sha256"] == prior["candidate_tree_sha256"]
    assert validate_evaluation_cohort(comparison, te, candidate) == record
    score.write_text(score.read_text() + "\n")
    with pytest.raises(ValueError, match="predecessor score evidence digest mismatch"):
        validate_evaluation_cohort(comparison, te, candidate)


def test_kernel_comparison_rejects_partial_or_nonphysical_predecessor(tmp_path, monkeypatch):
    te = load_target_experiment(RADIANCE)
    monkeypatch.setattr(
        "merlin.targetgen.evaluation_cohort.engine_preflight",
        lambda _te, stage: {"stage": stage, "ok": True},
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "schedule.mlir").write_text("module {}\n")
    derived = tmp_path / "derived"
    prior = materialize_evaluation_cohort(derived, te, "derived_gsim", candidate)
    score = tmp_path / "derived-score.json"
    _passing_predecessor_score(score, candidate, prior)
    doc = json.loads(score.read_text())
    doc["n_passed"] -= 1
    doc["pass_evidence"]["rtl_backed"] -= 1
    score.write_text(json.dumps(doc))

    with pytest.raises(ValueError, match="predecessor pass evidence rejected"):
        materialize_evaluation_cohort(
            tmp_path / "comparison", te, "kernel_library_comparison", candidate,
            predecessor_cohort=derived, predecessor_score=score)
