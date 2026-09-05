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
        "model_execution_all_pass": None,
        "structural_evidence_all_pass": functional_pass == 1,
        "structural_evidence_scope": {
            "n_instruction_trace_capsules": n_capsules,
            "n_model_execution_capsules": 0,
        },
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
        "cohort_admission": {
            "policy": "all_discovered",
            "n_source_capsules": n_capsules,
            "n_admitted_capsules": n_capsules,
            "n_capability_excluded": 0,
            "n_resource_excluded": 0,
            "excluded_name_set_sha256": "0" * 64,
            "admitted_name_set_sha256": "1" * 64,
        },
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


@pytest.mark.parametrize("mutation", [
    lambda c: c.update({"n_source_capsules": 4}),
    lambda c: c.update({"n_admitted_capsules": 1}),
    lambda c: c.update({"n_capability_excluded": -1}),
    lambda c: c.update({"excluded_name_set_sha256": "not-a-digest"}),
    lambda c: c.update({"policy": "operator_cherry_pick"}),
])
def test_formal_completion_requires_a_coherent_sealed_cohort(mutation):
    grader = _mod("grade_agent_run")
    score = _score()
    mutation(score["cohort_admission"])
    complete, failures = grader.phase_completion(score)
    assert complete is False
    assert any(reason.startswith("cohort_admission_") for reason in failures)


def test_only_hidden_scoring_requests_capability_admission(monkeypatch):
    grader = _mod("grade_agent_run")
    calls = []
    monkeypatch.setattr(grader.CR, "oracle_adapters", lambda _target: {})
    monkeypatch.setattr(grader.CG, "grade", lambda *args, **kwargs: calls.append(kwargs) or {})
    grader._score("pkg", "public", "runs", {"public", "dev"}, False)
    grader._score("pkg", "hidden", "runs", {"hidden"}, False)
    assert calls[0]["capability_admission"] is False
    assert calls[1]["capability_admission"] is True


def test_chipyard_formal_model_sim_uses_dynamic_rtl_engine_policy(monkeypatch):
    """The formal model lane must use the same availability/cost policy as operator L3.

    A static ``tier_sim[L3]`` value is only the manifest's historical binding.  It must not pin a
    multi-hour model run to Verilator after an equally faithful GSIM adapter becomes available.
    """
    grader = _mod("grade_agent_run")
    monkeypatch.setattr(grader.CR, "_bespoke_sim_via", lambda _target: "chipyard")
    monkeypatch.setattr(
        grader.CR, "chipyard_l3_selection",
        lambda _target: {
            "engine": "gsim", "fidelity": "elaborated_rtl",
            "reason": "gsim available", "considered": [], "passed_over": ["vcs"],
        },
    )

    resolved = grader._formal_model_simulator("gemmini")

    assert resolved["engine"] == "gsim"
    assert resolved["fidelity"] == "elaborated_rtl"
    assert resolved["selection"] == "chipyard_l3_policy"


def test_chipyard_formal_model_sim_refuses_non_rtl_policy_result(monkeypatch):
    grader = _mod("grade_agent_run")
    monkeypatch.setattr(grader.CR, "_bespoke_sim_via", lambda _target: "chipyard")
    monkeypatch.setattr(
        grader.CR, "chipyard_l3_selection",
        lambda _target: {"engine": "spike", "fidelity": "functional_model"},
    )

    with pytest.raises(RuntimeError, match="elaborated_rtl"):
        grader._formal_model_simulator("gemmini")


def test_formal_model_install_overwrites_ambient_verilator_with_required_gsim(monkeypatch):
    grader = _mod("grade_agent_run")
    monkeypatch.setenv("MERLIN_MESH_SIM", "verilator")
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    monkeypatch.setattr(
        grader, "_formal_model_simulator",
        lambda _target: {"engine": "gsim", "fidelity": "elaborated_rtl"})

    selected, inherited = grader._install_formal_model_simulator("gemmini")

    assert selected["engine"] == "gsim"
    assert inherited == "verilator"
    assert grader.os.environ["MERLIN_MESH_SIM"] == "gsim"


def test_formal_model_install_refuses_engine_different_from_pin(monkeypatch):
    grader = _mod("grade_agent_run")
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "gsim")
    monkeypatch.setattr(
        grader, "_formal_model_simulator",
        lambda _target: {"engine": "verilator", "fidelity": "elaborated_rtl"})

    with pytest.raises(RuntimeError, match="differs from required"):
        grader._install_formal_model_simulator("gemmini")


@pytest.mark.parametrize("field", ["integrity_status", "numeric_all_exact", "trace_all_pass"])
def test_formal_completion_requires_clean_exact_structural_evidence(field):
    grader = _mod("grade_agent_run")
    score = _score()
    score[field] = None
    complete, failures = grader.phase_completion(score)
    assert complete is False
    assert failures


def test_formal_completion_accepts_distinct_operator_trace_and_model_execution_scopes():
    grader = _mod("grade_agent_run")
    score = _score(n_capsules=4, n_passed=4, l3=4, rtl_backed=4)
    score.update({
        "trace_all_pass": True,
        "model_execution_all_pass": True,
        "structural_evidence_all_pass": True,
        "structural_evidence_scope": {
            "n_instruction_trace_capsules": 2,
            "n_model_execution_capsules": 2,
        },
    })
    complete, failures = grader.phase_completion(score)
    assert complete is True, failures


@pytest.mark.parametrize("mutation,reason", [
    (lambda s: s.update({"model_execution_all_pass": False}),
     "model_execution_evidence_not_complete"),
    (lambda s: s["structural_evidence_scope"].update({"n_model_execution_capsules": 1}),
     "structural_evidence_scope_denominator_mismatch"),
    (lambda s: s.update({"structural_evidence_all_pass": False}),
     "structural_evidence_not_complete"),
])
def test_formal_completion_fails_closed_on_model_execution_evidence(mutation, reason):
    grader = _mod("grade_agent_run")
    score = _score(n_capsules=4, n_passed=4, l3=4, rtl_backed=4)
    score.update({
        "trace_all_pass": True,
        "model_execution_all_pass": True,
        "structural_evidence_all_pass": True,
        "structural_evidence_scope": {
            "n_instruction_trace_capsules": 2,
            "n_model_execution_capsules": 2,
        },
    })
    mutation(score)
    complete, failures = grader.phase_completion(score)
    assert complete is False
    assert reason in failures


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
