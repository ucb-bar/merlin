"""Fail-closed completion gates for the Arm-4 functional experiment."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from merlin.common.paths import merlin_dir

HARNESS = merlin_dir() / "experiments/capsule_bench/harness"


def _mod(name: str):
    if str(HARNESS) not in sys.path:
        sys.path.insert(0, str(HARNESS))
    spec = importlib.util.spec_from_file_location(f"functional_gate_{name}", HARNESS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 — harness dependencies may be absent in small installs
        pytest.skip(f"{name} not importable here: {type(exc).__name__}: {exc}")
    return mod


def _score(*, n_capsules: int = 2, n_passed: int = 2, l3: int = 2,
           rtl_backed: int = 2, functional_pass: int = 1, gradeable: bool = True) -> dict:
    return {
        "n_capsules": n_capsules,
        "n_passed": n_passed,
        "functional_pass": functional_pass,
        "gradeable": gradeable,
        "numeric_all_exact": functional_pass == 1,
        "trace_all_pass": functional_pass == 1,
        "tier_reached": {"L3": l3},
        "highest_tier": "L3" if l3 == n_capsules and n_capsules else None,
        "pass_evidence": {"rtl_backed": rtl_backed},
        "public_passed": f"{n_passed}/{n_capsules}",
        "integrity_status": "clean",
        "n_not_graded_ineligible": 0,
        "n_gated_deferred": 0,
        "n_screened_only": 0,
        "n_budget_exhausted": 0,
        "n_incomplete": 0,
        "n_not_gradeable_no_oracle": 0,
    }


def _manifest(score: dict, *, complete: bool = True) -> dict:
    phase = {
        **score,
        "unmeasured_counts": {
            field: score[field] for field in (
                "n_not_graded_ineligible", "n_gated_deferred", "n_screened_only",
                "n_budget_exhausted", "n_incomplete", "n_not_gradeable_no_oracle",
            )
        },
        "formal_complete": complete,
        "completion_failures": [] if complete else ["failed"],
    }
    return {
        "public_dev": dict(phase),
        "hidden": dict(phase),
        "completion": {"formal_grade_complete": complete, "required_tier": "L3", "failures": []},
    }


def test_a_vacuous_hidden_zero_of_zero_is_never_complete():
    grader = _mod("grade_agent_run")
    complete, failures = grader.phase_completion(
        _score(n_capsules=0, n_passed=0, l3=0, rtl_backed=0, functional_pass=0, gradeable=False))
    assert complete is False
    assert "capsule_set_empty_or_malformed" in failures
    assert "numeric_grade_not_gradeable" in failures


@pytest.mark.parametrize("field", [
    "n_not_graded_ineligible", "n_gated_deferred", "n_screened_only",
    "n_budget_exhausted", "n_incomplete", "n_not_gradeable_no_oracle",
])
def test_no_unmeasured_capsule_can_disappear_from_formal_completion(field):
    grader = _mod("grade_agent_run")
    score = _score()
    score[field] = 1
    complete, failures = grader.phase_completion(score)
    assert complete is False
    assert f"{field}_nonzero" in failures


@pytest.mark.parametrize("field", ["integrity_status", "numeric_all_exact", "trace_all_pass"])
def test_formal_completion_requires_clean_exact_structural_evidence(field):
    grader = _mod("grade_agent_run")
    score = _score()
    score[field] = None
    complete, failures = grader.phase_completion(score)
    assert complete is False
    assert failures


def test_grader_main_returns_nonzero_and_records_failed_hidden_zero_of_zero(tmp_path, monkeypatch):
    grader = _mod("grade_agent_run")
    run_dir = tmp_path / "arm4"
    (run_dir / "submission").mkdir(parents=True)
    public = _score()
    hidden = _score(n_capsules=0, n_passed=0, l3=0, rtl_backed=0,
                    functional_pass=0, gradeable=False)
    scores = iter((public, hidden))
    monkeypatch.setattr(grader, "_score", lambda *_args, **_kwargs: next(scores))

    def freeze(rd: Path) -> dict:
        sha = grader.C.hash_tree(rd / "submission")["sha256"]
        record = {"submission_sha256": sha, "repo_sha": "test", "frozen_at": "now"}
        (rd / "freeze.json").write_text(json.dumps(record), encoding="utf-8")
        return record

    monkeypatch.setattr(grader.freeze_run, "freeze", freeze)
    rc = grader.main(["--run-dir", str(run_dir), "--arm", "merlin_assisted",
                      "--capsules", "public", "--hidden-capsules", "hidden"])
    manifest = yaml.safe_load((run_dir / "run_manifest.yaml").read_text(encoding="utf-8"))
    assert rc != 0
    assert manifest["hidden"]["passed"] == "0/0"
    assert manifest["hidden"]["formal_complete"] is False
    assert manifest["completion"]["formal_grade_complete"] is False


def test_launcher_rejects_nonzero_grader_even_if_manifest_claims_success(tmp_path):
    loop = _mod("run_baseline_qa_loop")
    (tmp_path / "run_manifest.yaml").write_text(yaml.safe_dump(_manifest(_score())), encoding="utf-8")
    result = loop._official_grade_result(9, tmp_path)
    assert result["complete"] is False
    assert "grader_exit_nonzero:9" in result["failures"]


def test_launcher_rejects_grader_failed_status_even_with_zero_exit(tmp_path):
    loop = _mod("run_baseline_qa_loop")
    (tmp_path / "run_manifest.yaml").write_text(
        yaml.safe_dump(_manifest(_score(), complete=False)), encoding="utf-8")
    result = loop._official_grade_result(0, tmp_path)
    assert result["complete"] is False
    assert "grader_reported_incomplete" in result["failures"]


def test_launcher_independently_rejects_a_hidden_unmeasured_capsule(tmp_path):
    loop = _mod("run_baseline_qa_loop")
    manifest = _manifest(_score())
    manifest["hidden"]["unmeasured_counts"]["n_screened_only"] = 1
    (tmp_path / "run_manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    result = loop._official_grade_result(0, tmp_path)
    assert result["complete"] is False
    assert "hidden:n_screened_only_not_zero" in result["failures"]


@pytest.mark.parametrize("override", [
    ["--bundle", "merlin_assisted_public_v0"],
    ["--bundle=raw_baseline_public_v0"],
    ["--bundle", "merlin_assisted_rtlchecks_public_v0", "--bundle", "other"],
])
def test_arm4_wrapper_rejects_every_noncanonical_bundle_override(monkeypatch, override):
    wrapper = _mod("run_rtlchecks_qa_loop")
    called = []
    monkeypatch.setattr(wrapper.L, "main", lambda argv: called.append(argv) or 0)
    assert wrapper.main(["--run-id", "test", *override]) == 4
    assert called == []


def test_arm4_wrapper_allows_an_explicit_identical_bundle_pin(monkeypatch):
    wrapper = _mod("run_rtlchecks_qa_loop")
    called = []
    monkeypatch.setattr(wrapper.L, "main", lambda argv: called.append(argv) or 0)
    bundle = "merlin_assisted_rtlchecks_public_v0"
    assert wrapper.main(["--run-id", "test", "--bundle", bundle]) == 0
    assert called and called[0][called[0].index("--bundle") + 1] == bundle
    assert called[0][called[0].index("--arm") + 1] == "merlin_assisted"
