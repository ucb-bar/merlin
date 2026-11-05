"""Sealed fast grades require original kernels and every selected shape variant."""

import copy
import json
import threading

import pytest
import yaml
from merlin_experiments.corpus.preparation import stage_phase1_policy_descriptor
from merlin_experiments.phase1.feedback import model_check_evidence as E
from merlin_experiments.phase1.feedback import model_check_official as O
from merlin_experiments.phase1.feedback import private_full_model_execution as PFX
from merlin_experiments.phase1.feedback import private_full_models as PFM


def body():
    return {
        "gates": {k: {"status": "pass", "partial": False}
                  for k in ["build", "kernel_exact", "model_numerics"]},
        "model_metrics": {"worst": {}},
        "programs": [
            {"program": "main", "build": {"built": 1, "refused": 0},
             "census": {"device": 1, "denominator": 1},
             "kernels": [{"status": "pass", "source_region_id": "group", "cases": 3}]},
            {"program": "main", "source": "sealed_variants", "split": "sealed",
             "build": {"built": 2, "refused": 0},
             "kernels": [{"status": "pass", "cases": 2}]},
        ],
    }


def test_fast_grade_keeps_original_census_and_requires_variants():
    report = body()
    observed, problems = E.assess(report, ["main"], mode="fast", shape_counts={"main": 2})
    assert not problems
    assert observed["main"]["census"] == {"device": 1, "denominator": 1}
    assert E.assess(report, ["main"], mode="fast", shape_counts={})[1]
    report["programs"].pop()
    assert E.assess(report, ["main"], mode="fast", shape_counts={"main": 2})[1]


@pytest.mark.parametrize("mutation", ["partial", "wrong_count", "failed_variant", "duplicate", "no_metrics"])
def test_fast_grade_rejects_incomplete_evidence(mutation):
    report = body()
    if mutation == "partial":
        report["gates"]["kernel_exact"]["partial"] = True
    elif mutation == "wrong_count":
        report["programs"][1]["kernels"][0]["cases"] = 1
    elif mutation == "failed_variant":
        report["programs"][1]["kernels"][0]["status"] = "fail"
    elif mutation == "duplicate":
        report["programs"].append(copy.deepcopy(report["programs"][0]))
    else:
        report["model_metrics"] = None
    assert E.assess(report, ["main"], mode="fast", shape_counts={"main": 2})[1]


def test_fast_grade_runs_models_in_parallel_and_records_full_execution_deferred(tmp_path, monkeypatch):
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "manifest.yaml").write_text("commands: {}\n")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    names = ["a", "b", "c"]
    (bundle / "MANIFEST.json").write_text(json.dumps({
        "schema": "merlin.model_check.bundle.v1",
        "models": {name: {"sealed_variants": {"programs": {"main": {"variants": 2}}}} for name in names},
    }))
    rendezvous = threading.Barrier(3, timeout=5)
    modes = []

    def invoke(command, **kwargs):
        assert command == "check"
        modes.append(kwargs["mode"])
        rendezvous.wait()
        return {"status": "pass", "body": body()}

    monkeypatch.setattr(O, "_invoke", invoke)
    result = O.run_gate(submission, tmp_path / "run", {name: ("main",) for name in names},
                        bundle=bundle, mode="fast", model_workers=3)
    assert result["passed"] and modes == ["fast"] * 3
    assert result["whole_model_status"] == "deferred_post_experiment"
    assert result["rtl_required"] is False and E.rtl_requirement_satisfied(result)


@pytest.mark.parametrize("mutation", ["missing_flag", "false_pass", "unexpected_replay"])
def test_model_rtl_deferral_requires_explicit_consistent_status(mutation):
    value = {"rtl_required": False, "rtl_status": "not_requested", "rtl": [], "rtl_complete": False}
    assert E.rtl_requirement_satisfied(value)
    if mutation == "missing_flag":
        value.pop("rtl_required")
    elif mutation == "false_pass":
        value["rtl_complete"] = True
    else:
        value["rtl"] = [{"status": "pass"}]
    assert not E.rtl_requirement_satisfied(value)


def test_reviewed_policy_can_defer_execution_while_retaining_compile_link_roster(tmp_path):
    gates = {
        "private_full_models": {"required": True, "models": ["net"], "programs": {"net": ["main"]}},
        "whole_model": {"private_full_models": {"required": True, "programs": {
            "net": {"main": {"engine": "spike", "timeout_s": 60,
                             "host_execution": {"atol": 0.01, "rtol": 0.0},
                             "fp32_sanity": {"reference": "reference.json", "cosine_margin": 4, "top1": True}}},
        }}},
    }
    source = tmp_path / "source.yaml"
    source.write_text(yaml.safe_dump({"target": "t", "phase1_gates": gates}))
    roster = PFM.program_requirements_for(source)
    assert roster == {"net": ("main",)}
    assert PFX.gate_for(source, required_programs=roster) is not None
    selected = tmp_path / "selected.yaml"
    selected.write_text(yaml.safe_dump({"target": "t", "phase1_gates": {
        "private_full_models": gates["private_full_models"],
    }}))
    private = tmp_path / "private"
    private.mkdir()
    receipt, _ = stage_phase1_policy_descriptor(source, selected, private)
    retained = private / "phase1-policy-descriptor.yaml"
    assert receipt["sha256"] and PFM.program_requirements_for(retained) == roster
    assert PFX.gate_for(retained, required_programs=roster) is None
    assert PFX.gate_for(source, required_programs=roster) is not None
