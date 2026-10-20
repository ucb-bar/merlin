"""Synthetic inference contracts do not qualify actual native engines or hardware."""

import copy

import pytest
from data_helpers import h, sample
from merlin_timing import cli
from merlin_timing import contracts as C
from merlin_timing import feedback as B
from merlin_timing import fitting as F
from merlin_timing import validation as V
from test_event_validation import original_reports


def inference_fixture(*, profile="phase1-public", residuals=False):
    dataset = original_reports(sample(profile=profile))
    if residuals:
        calibration = {case["case_id"] for case in dataset["cases"] if case["split"] == "calibration"}
        for row in dataset["records"]:
            if row["case_id"] in calibration:
                row["label"]["cycles"] *= 0.9 if row["case_id"].startswith("calibration-4-") else 1.1
    receipt = {"dataset": dataset, "report": V.validate(dataset, prediction_kind="event_timing")}
    cases = {
        "schema": C.NATIVE_CASES,
        "profile": dataset["profile"],
        "domain": copy.deepcopy(dataset["domain"]),
        "feature_names": list(dataset["feature_names"]),
        "cases": [],
    }
    by_id = {case["case_id"]: case for case in dataset["cases"]}
    for row in dataset["records"]:
        if row["case_id"] not in {"test-6-0", "test-6-1"}:
            continue
        identity = {key: copy.deepcopy(value) for key, value in by_id[row["case_id"]].items() if key != "split"}
        identity["case_id"] = "candidate-" + row["case_id"]
        identity["elf_sha256"] = h(identity["case_id"])
        identity["source"]["phase"] = "phase2" if profile == "phase1-public" else "historical"
        native = copy.deepcopy(row["event_report"]["report"])
        native["elf_sha256"] = identity["elf_sha256"]
        cases["cases"].append(
            {
                "case": identity,
                "features": copy.deepcopy(row["features"]),
                "event_report": {"sha256": C.digest(native), "report": native},
                "numeric": {
                    "status": "pass",
                    "receipt_sha256": h([identity["case_id"], "numeric"]),
                    "execution_sha256": C.execution_identity(identity),
                    "measured_now": True,
                },
            }
        )
    return cases, receipt


def test_native_cli_uses_original_predictions_and_never_a_linear_screen(tmp_path, monkeypatch):
    cases, receipt = inference_fixture()
    assert receipt["report"]["event_engine_qualified"]  # Synthetic policy arithmetic only.

    def no_screen(*args, **kwargs):
        raise AssertionError("native inference must not consume linear predictions")

    monkeypatch.setattr(F, "predict", no_screen)
    source, authority, output = tmp_path / "cases.json", tmp_path / "receipt.json", tmp_path / "comparison.json"
    C.write(source, cases)
    C.write(authority, receipt)
    assert (
        cli.main(["compare-native", "--cases", str(source), "--qualification", str(authority), "--output", str(output)])
        == 0
    )
    result = C.read(output)
    assert result["prediction_kind"] == "event_timing" and result["event_engine_qualified"]
    assert result["public_feedback_eligible"]
    assert result["cases"][0]["estimated_cycles"] == cases["cases"][0]["event_report"]["report"]["estimated_cycles"]
    assert result["comparisons"][0]["predicted_faster"] == cases["cases"][0]["case"]["case_id"]
    assert result["model_sha256"] == cases["domain"]["target_model_sha256"]


def test_actual_count_v2_inference_uses_report_completeness_without_training_record_fields():
    """Synthetic original v2 products exercise the production report grammar."""
    cases, receipt = inference_fixture()

    def v2(row):
        raw = row["event_report"]["report"]
        count = raw["feature_counts"]["host_instructions"]
        raw.update(schema="merlin.annotated_spike_report.v2", instructions=count, execution_completed=True)
        raw["feature_observation"] = {
            "schema": "merlin.spike_feature_observation.v1",
            "complete": True,
            "scope_instructions": count,
            "counted_cpu_instructions": count,
            "scope_kind": "whole_program",
            "window_index": None,
            "primary_cpu_features": ["host_instructions"],
            "unresolved": [],
            "qualification": "not_established",
        }
        row["event_report"]["sha256"] = C.digest(raw)

    for row in receipt["dataset"]["records"]:
        v2(row)
    receipt["report"] = V.validate(receipt["dataset"], prediction_kind="event_timing")
    for row in cases["cases"]:
        v2(row)
        assert "feature_provenance" not in row
    result = B.compare_native(cases, qualification=receipt)
    assert all(row["status"] == "qualified_native" for row in result["cases"])
    # Training provenance remains mandatory and independently cross-checked.
    row = receipt["dataset"]["records"][0]
    row["feature_provenance"]["complete"] = False
    case = next(case for case in receipt["dataset"]["cases"] if case["case_id"] == row["case_id"])
    with pytest.raises(C.ContractError, match="contradicts"):
        V.native_prediction(row, case, receipt["dataset"])


def test_without_qualified_receipt_only_diagnostics_are_available():
    cases, receipt = inference_fixture()
    result = B.compare_native(cases)
    assert result["cases"][0]["estimated_cycles"] is not None
    assert result["comparisons"][0]["status"] == "unknown"
    assert not result["public_feedback_eligible"] and not result["event_engine_qualified"]
    for row in receipt["dataset"]["records"][-2:]:
        raw = row["event_report"]["report"]
        raw["estimated_cycles"] *= 2
        row["event_report"]["sha256"] = C.digest(raw)
    receipt["report"] = V.validate(receipt["dataset"], prediction_kind="event_timing")
    assert not receipt["report"]["qualified"]
    result = B.compare_native(cases, qualification=receipt)
    assert result["cases"][0]["diagnostic_interval"] is not None
    assert result["cases"][0]["interval"] is None and result["comparisons"][0]["status"] == "unknown"


def test_changed_receipt_or_screen_qualification_cannot_authorize_native_ranking():
    cases, receipt = inference_fixture()
    receipt["report"]["metrics"]["rank_agreement"] = 0
    with pytest.raises(C.ContractError, match="original held-out"):
        B.compare_native(cases, qualification=receipt)
    cases, receipt = inference_fixture()
    screen = F.fit(receipt["dataset"])
    receipt["report"] = V.validate(receipt["dataset"], screen)
    with pytest.raises(C.ContractError, match="original held-out"):
        B.compare_native(cases, qualification=receipt)


@pytest.mark.parametrize("field", sorted(C.DOMAIN_KEYS))
def test_native_inference_cannot_transfer_qualification_across_any_domain_identity(field):
    cases, receipt = inference_fixture()
    cases["domain"][field] = "verilator" if field == "label_engine" else h("different")
    for row in cases["cases"]:
        if field in {"scope_sha256", "device_sha256"}:
            row["case"][field] = cases["domain"][field]
            row["case"]["workload_sha256"] = C.workload(row["case"])
            row["numeric"]["execution_sha256"] = C.execution_identity(row["case"])
    with pytest.raises(C.ContractError, match="profile/domain/feature schema"):
        B.compare_native(cases, qualification=receipt)


@pytest.mark.parametrize("mutation", ["private", "heldout", "unknown", "profile", "features"])
def test_public_inference_rejects_hidden_sources_and_mismatched_receipt_profiles(mutation):
    cases, receipt = inference_fixture()
    if mutation == "private":
        cases["cases"][0]["case"]["source"]["visibility"] = "private"
    elif mutation in {"heldout", "unknown"}:
        cases["cases"][0]["case"]["source"]["phase"] = mutation
    elif mutation == "profile":
        _, receipt = inference_fixture(profile="historical-private")
    else:
        cases["feature_names"] = ["other_feature"]
        for row in cases["cases"]:
            row["features"] = {"other_feature": 1}
    with pytest.raises(C.ContractError):
        B.compare_native(cases, qualification=receipt)


@pytest.mark.parametrize("mutation", ["numeric_fail", "stale", "missing_numeric", "incomplete", "range", "family"])
def test_failed_numerics_incomplete_native_or_ood_candidates_cannot_rank(mutation):
    cases, receipt = inference_fixture()
    row = cases["cases"][0]
    raw = row["event_report"]["report"]
    if mutation == "numeric_fail":
        row["numeric"]["status"] = "fail"
    elif mutation == "stale":
        row["numeric"]["measured_now"] = False
    elif mutation == "missing_numeric":
        row["numeric"] = None
    elif mutation == "incomplete":
        raw["timing_complete"] = False
    elif mutation == "range":
        row["features"]["host_instructions"] = raw["feature_counts"]["host_instructions"] = 100
    else:
        for member in cases["cases"]:
            member["case"]["mechanism_family"] = "unseen"
    row["event_report"]["sha256"] = C.digest(raw)
    result = B.compare_native(cases, qualification=receipt)
    assert result["comparisons"][0]["status"] == "unknown"
    assert result["cases"][0]["status"] == "diagnostic" and result["cases"][0]["interval"] is None


def test_overlapping_intervals_and_distinct_workloads_do_not_decide_order():
    cases, receipt = inference_fixture(residuals=True)
    for index, row in enumerate(cases["cases"]):
        row["event_report"]["report"]["estimated_cycles"] = 100 + index
        row["event_report"]["sha256"] = C.digest(row["event_report"]["report"])
    result = B.compare_native(cases, qualification=receipt)
    assert all(row["status"] == "qualified_native" for row in result["cases"])
    assert result["comparisons"][0]["status"] == "unknown"
    row = cases["cases"][-1]
    row["case"]["input_sha256"] = h("another input")
    row["case"]["workload_sha256"] = C.workload(row["case"])
    row["numeric"]["execution_sha256"] = C.execution_identity(row["case"])
    row["event_report"]["report"]["input_sha256"] = row["case"]["input_sha256"]
    row["event_report"]["sha256"] = C.digest(row["event_report"]["report"])
    assert B.compare_native(cases, qualification=receipt)["comparisons"] == []


def test_private_native_ranking_never_becomes_public_feedback():
    cases, receipt = inference_fixture(profile="historical-private")
    result = B.compare_native(cases, qualification=receipt)
    assert result["comparisons"][0]["predicted_faster"] is not None
    assert result["public_feedback_eligible"] is False
    assert not any(row["public_feedback_eligible"] for row in result["cases"])


def test_public_phase2_inference_does_not_admit_phase2_calibration_sources():
    cases, receipt = inference_fixture()
    assert B.compare_native(cases, qualification=receipt)["public_feedback_eligible"]
    receipt["dataset"]["cases"][0]["source"]["phase"] = "phase2"
    with pytest.raises(C.ContractError, match="source visibility/phase"):
        V.validate(receipt["dataset"], prediction_kind="event_timing")


def test_native_case_identity_receipts_and_resource_limits_are_closed(monkeypatch):
    cases, receipt = inference_fixture()
    cases["cases"][0]["numeric"]["execution_sha256"] = h("wrong execution")
    with pytest.raises(C.ContractError, match="exact execution"):
        B.compare_native(cases, qualification=receipt)
    cases, receipt = inference_fixture()
    cases["cases"][0]["features"]["not_declared"] = 1
    with pytest.raises(C.ContractError, match="features requires exactly"):
        B.compare_native(cases, qualification=receipt)
    cases, receipt = inference_fixture()
    monkeypatch.setattr(C, "MAX_PAIR_COMPARISONS", 0)
    with pytest.raises(C.ContractError, match="pair comparisons"):
        B.compare_native(cases)
