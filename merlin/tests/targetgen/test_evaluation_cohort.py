from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen.evaluation_cohort import (
    create_search_pass_seal,
    materialize_evaluation_cohort,
    validate_evaluation_cohort,
    validate_search_pass_seal,
)
from merlin.targetgen.target_experiment import load_target_experiment


RADIANCE = repo_root() / "merlin/experiments/capsule_bench/targets/radiance/target_experiment.yaml"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _working_engine(tmp_path: Path, monkeypatch) -> dict:
    binary = tmp_path / "emulator"
    receipt_path = tmp_path / "build_receipt.json"
    binary.write_bytes(b"emulator")
    receipt_path.write_text("{}\n")
    receipt_identity = {
        "binary_sha256": _sha(binary),
        "receipt_sha256": _sha(receipt_path),
        "receipt_path": str(receipt_path.resolve()),
    }
    binding = {
        "engine": "gsim",
        "binary": str(binary.resolve()),
        "binary_sha256": _sha(binary),
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": _sha(receipt_path),
        "receipt_status": "bound",
        "receipt_identity": receipt_identity,
    }
    binding["binding_sha256"] = _canonical_sha(binding)
    preflight = {
        "stage": "derived_gsim",
        "required_tier": "L3",
        "requested_engine": "gsim",
        "selected": {"available": True, "engine": "gsim", "fidelity": "elaborated_rtl",
                     "reason": "receipt:test"},
        "engine_binding": binding,
        "ok": True,
        "problems": [],
    }
    monkeypatch.setattr(
        "merlin.targetgen.evaluation_cohort.engine_preflight",
        lambda _te, stage: {**preflight, "stage": stage},
    )
    monkeypatch.setattr("merlin.targetgen.gsim_emulator.gsim_home", lambda _target: tmp_path)
    return preflight


def _candidate(tmp_path: Path) -> Path:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "schedule.mlir").write_text("module {}\n")
    return candidate


def _passing_search_score(path: Path, te) -> None:
    names = sorted(te.graded_include)
    path.write_text(json.dumps({
        "n_capsules": len(names),
        "n_passed": len(names),
        "n_certified": len(names),
        "n_unchecked": 0,
        "all_pass": True,
        "per_capsule": [
            {"capsule": name, "pass": True, "barrier_tier": "L2",
             "barrier_status": "pass", "execution_digest": hashlib.sha256(name.encode()).hexdigest()}
            for name in names
        ],
    }))


def _search_seal(tmp_path: Path, te, candidate: Path) -> Path:
    score = tmp_path / "search-score.json"
    seal = tmp_path / "search-pass.json"
    _passing_search_score(score, te)
    create_search_pass_seal(seal, te, candidate, score)
    return seal


def test_radiance_derived_gsim_materialization_makes_l3_mandatory(tmp_path, monkeypatch):
    te = load_target_experiment(RADIANCE)
    _working_engine(tmp_path, monkeypatch)
    candidate = _candidate(tmp_path)
    seal = _search_seal(tmp_path, te, candidate)
    out = tmp_path / "derived-gsim"
    record = materialize_evaluation_cohort(
        out, te, "derived_gsim", candidate, search_pass_seal=seal)

    assert record["n_capsules"] == len(te.evaluation_cohort("derived_gsim")["include_capsules"])
    assert record["after"] == "search_l2_pass"
    assert record["search_pass_evidence"]["n_passed"] == 15
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
            "after": "search_l2_pass",
            "required_oracle_tier": "L3",
            "oracle_engine": "gsim",
        }
    assert validate_evaluation_cohort(out, te, candidate) == record


def test_derived_gsim_requires_exact_l2_pass_seal_and_leaves_no_tree(tmp_path, monkeypatch):
    te = load_target_experiment(RADIANCE)
    _working_engine(tmp_path, monkeypatch)
    candidate = _candidate(tmp_path)
    out = tmp_path / "derived-gsim"
    with pytest.raises(ValueError, match="requires --search-pass-seal"):
        materialize_evaluation_cohort(out, te, "derived_gsim", candidate)
    assert not out.exists()

    score = tmp_path / "search-score.json"
    _passing_search_score(score, te)
    doc = json.loads(score.read_text())
    doc["per_capsule"] = doc["per_capsule"][:-1]
    score.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="do not exactly cover"):
        create_search_pass_seal(tmp_path / "search-pass.json", te, candidate, score)


def test_search_pass_seal_freezes_candidate_and_score(tmp_path):
    te = load_target_experiment(RADIANCE)
    candidate = _candidate(tmp_path)
    score = tmp_path / "search-score.json"
    seal = tmp_path / "search-pass.json"
    _passing_search_score(score, te)
    record = create_search_pass_seal(seal, te, candidate, score)
    assert record["n_capsules"] == record["n_passed"] == 15
    assert {row["name"] for row in record["capsules"]} == set(te.graded_include)
    assert "SY_app_elementwise_add_f32_rank3_1x113x1_l2" in te.graded_include
    assert validate_search_pass_seal(seal, te, candidate)["seal_sha256"] == _sha(seal)

    cache = candidate / "mlir_oot/__pycache__"
    cache.mkdir(parents=True)
    (cache / "schedule.cpython-312.pyc").write_bytes(b"interpreter cache")
    assert validate_search_pass_seal(seal, te, candidate)["seal_sha256"] == _sha(seal)

    (candidate / "schedule.mlir").write_text("module { func.func private @changed() }\n")
    with pytest.raises(ValueError, match="candidate content digest mismatch"):
        validate_search_pass_seal(seal, te, candidate)


def test_engine_preflight_failure_does_not_materialize(tmp_path, monkeypatch):
    te = load_target_experiment(RADIANCE)
    candidate = _candidate(tmp_path)
    seal = _search_seal(tmp_path, te, candidate)
    monkeypatch.setattr(
        "merlin.targetgen.evaluation_cohort.engine_preflight",
        lambda _te, stage: {"stage": stage, "ok": False, "problems": ["receipt mismatch"]},
    )
    out = tmp_path / "derived-gsim"
    with pytest.raises(ValueError, match="receipt mismatch"):
        materialize_evaluation_cohort(
            out, te, "derived_gsim", candidate, search_pass_seal=seal)
    assert not out.exists()


def test_evaluation_validation_detects_engine_binary_mutation(tmp_path, monkeypatch):
    te = load_target_experiment(RADIANCE)
    preflight = _working_engine(tmp_path, monkeypatch)
    candidate = _candidate(tmp_path)
    seal = _search_seal(tmp_path, te, candidate)
    out = tmp_path / "derived-gsim"
    materialize_evaluation_cohort(
        out, te, "derived_gsim", candidate, search_pass_seal=seal)
    Path(preflight["engine_binding"]["binary"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="engine binary content digest mismatch"):
        validate_evaluation_cohort(out, te, candidate)


def test_evaluation_materialization_refuses_stale_files(tmp_path, monkeypatch):
    te = load_target_experiment(RADIANCE)
    _working_engine(tmp_path, monkeypatch)
    out = tmp_path / "not-empty"
    candidate = _candidate(tmp_path)
    seal = _search_seal(tmp_path, te, candidate)
    out.mkdir()
    (out / "stale").write_text("old")
    with pytest.raises(ValueError, match="not empty"):
        materialize_evaluation_cohort(
            out, te, "derived_gsim", candidate, search_pass_seal=seal)


def test_evaluation_materialization_refuses_empty_candidate(tmp_path):
    te = load_target_experiment(RADIANCE)
    candidate = tmp_path / "empty-candidate"
    candidate.mkdir()
    with pytest.raises(ValueError, match="has no files"):
        materialize_evaluation_cohort(tmp_path / "derived-gsim", te, "derived_gsim", candidate)


def test_evaluation_validation_detects_capsule_mutation(tmp_path, monkeypatch):
    te = load_target_experiment(RADIANCE)
    _working_engine(tmp_path, monkeypatch)
    candidate = _candidate(tmp_path)
    seal = _search_seal(tmp_path, te, candidate)
    out = tmp_path / "derived-gsim"
    materialize_evaluation_cohort(
        out, te, "derived_gsim", candidate, search_pass_seal=seal)
    record = json.loads((out / ".evaluation_cohort.json").read_text())
    name = record["capsules"][0]["name"]
    with (out / name / "README.md").open("a") as handle:
        handle.write("\nmutation\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_evaluation_cohort(out, te, candidate)


def test_evaluation_validation_detects_candidate_mutation(tmp_path, monkeypatch):
    te = load_target_experiment(RADIANCE)
    _working_engine(tmp_path, monkeypatch)
    candidate = _candidate(tmp_path)
    schedule = candidate / "schedule.mlir"
    seal = _search_seal(tmp_path, te, candidate)
    out = tmp_path / "derived-gsim"
    materialize_evaluation_cohort(
        out, te, "derived_gsim", candidate, search_pass_seal=seal)
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
            {"capsule": name, "status": "pass", "tiers": {tier: "pass"},
             "numeric": {"status": "pass"}}
            for name in names
        ],
        "pass_evidence": {"rtl_backed": len(names)},
    }))


def test_kernel_comparison_requires_exact_derived_gsim_pass(tmp_path, monkeypatch):
    te = load_target_experiment(RADIANCE)
    _working_engine(tmp_path, monkeypatch)
    candidate = _candidate(tmp_path)

    with pytest.raises(ValueError, match="requires --predecessor-cohort"):
        materialize_evaluation_cohort(
            tmp_path / "comparison-refused", te, "kernel_library_comparison", candidate)

    derived = tmp_path / "derived"
    seal = _search_seal(tmp_path, te, candidate)
    prior = materialize_evaluation_cohort(
        derived, te, "derived_gsim", candidate, search_pass_seal=seal)
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
    assert evidence["search_pass_seal_sha256"] == prior["search_pass_evidence"]["seal_sha256"]
    assert evidence["engine_binding_sha256"] == (
        prior["engine_preflight"]["engine_binding"]["binding_sha256"])
    assert validate_evaluation_cohort(comparison, te, candidate) == record
    score.write_text(score.read_text() + "\n")
    with pytest.raises(ValueError, match="predecessor score evidence digest mismatch"):
        validate_evaluation_cohort(comparison, te, candidate)


def test_kernel_comparison_rejects_partial_or_nonphysical_predecessor(tmp_path, monkeypatch):
    te = load_target_experiment(RADIANCE)
    _working_engine(tmp_path, monkeypatch)
    candidate = _candidate(tmp_path)
    derived = tmp_path / "derived"
    seal = _search_seal(tmp_path, te, candidate)
    prior = materialize_evaluation_cohort(
        derived, te, "derived_gsim", candidate, search_pass_seal=seal)
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
