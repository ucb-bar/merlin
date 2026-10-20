"""Synthetic numerical contracts exercise policy without qualifying actual hardware."""

import copy
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from data_helpers import h, sample
from merlin_timing import cli
from merlin_timing import contracts as C
from merlin_timing import feedback as B
from merlin_timing import fitting as F
from merlin_timing import validation as V


def test_interpretable_nnls_is_nonnegative_and_never_engine_qualification():
    data = sample()
    model = F.fit(data)
    assert model["coefficients"]["host_instructions"] == pytest.approx(3, abs=1e-6)
    assert model["fixed_cycles"] == pytest.approx(5, abs=1e-6)
    assert model["kind"] == "empirical_screen" and model["engine_qualification"] == "not_established"
    report = V.validate(data, model)
    assert report["qualified"]  # synthetic policy arithmetic only
    assert report["metrics"]["eligible_held_pairs"] == 36 and report["metrics"]["mechanism_families"] == 3
    assert report["event_engine_qualified"] is False and report["qualification_scope"] == "empirical_screen_only"


def test_all_fitted_features_and_explicit_fixed_term_have_nonnegative_prices():
    coefficients, _ = F.nnls(
        [[1, i / 10] for i in range(1, 9)], [1 - i / 20 for i in range(1, 9)], maximum_condition=1e6
    )
    assert coefficients[1] == 0 and all(c >= 0 for c in coefficients)
    data = sample()
    for row in data["records"]:
        row["label"]["cycles"] = 3 * row["features"]["host_instructions"]
    model = F.fit(data, include_fixed=False)
    assert model["fixed_cycles"] == 0 and model["solver"]["include_fixed"] is False
    assert V.validate(data, model)["qualified"]


def test_held_failures_do_not_disappear_behind_good_training_error():
    data = sample(held_failure=True)
    model = F.fit(data)
    assert model["coefficients"]["host_instructions"] == pytest.approx(3, abs=1e-6)
    report = V.validate(data, model)
    assert report["status"] == "diagnostic" and report["metrics"]["rank_agreement"] == 0
    assert report["checks"]["median_error"] is False and report["event_engine_qualified"] is False


@pytest.mark.parametrize("profile", ["phase1-public", "historical-private"])
def test_profiles_share_math_but_private_never_public_feedback(profile):
    data = sample(profile=profile)
    model = F.fit(data)
    report = V.validate(data, model)
    assert report["qualified"] and report["public_feedback_eligible"] is (profile == "phase1-public")
    assert model["profile"] == profile


@pytest.mark.parametrize(
    "mutation",
    [
        "hidden",
        "validation",
        "historical",
        "mixed_profile",
        "group_split",
        "source_split",
        "workload_forged",
        "repeat_case",
        "label_engine",
        "feature_engine",
        "feature_model",
        "label_input",
        "feature_input",
        "extra_field",
        "nan",
        "bool",
    ],
)
def test_invalid_or_leaking_source_records_refuse(mutation):
    data = sample()
    case, record = data["cases"][0], data["records"][0]
    if mutation in {"hidden", "validation"}:
        case["source"]["visibility"] = mutation
    elif mutation == "historical":
        case["source"]["phase"] = "historical"
    elif mutation == "mixed_profile":
        data["profile"] = "historical-private"
    elif mutation == "group_split":
        data["cases"][-1]["source_group_sha256"] = case["source_group_sha256"]
    elif mutation == "source_split":
        case["source_group_sha256"] = h("another-group")
    elif mutation == "workload_forged":
        case["workload_sha256"] = h("wrong-workload")
    elif mutation == "repeat_case":
        duplicate = copy.deepcopy(case)
        duplicate["case_id"] = "another-repeat"
        data["cases"].append(duplicate)
    elif mutation == "label_engine":
        record["label"]["engine_sha256"] = h("another-engine")
    elif mutation == "feature_engine":
        record["feature_provenance"]["engine_sha256"] = h("another-engine")
    elif mutation == "feature_model":
        record["feature_provenance"]["model_sha256"] = h("another-model")
    elif mutation == "label_input":
        record["label"]["execution_sha256"] = h("another-input")
    elif mutation == "feature_input":
        record["feature_provenance"]["execution_sha256"] = h("another-input")
    elif mutation == "extra_field":
        record["secret_cycles"] = 4
    elif mutation == "nan":
        record["features"]["host_instructions"] = math.nan
    else:
        record["features"]["host_instructions"] = True
    with pytest.raises(C.ContractError):
        F.fit(data)


def test_repeats_remain_one_case_and_unknown_labels_cannot_train():
    data = sample()
    repeat = copy.deepcopy(data["records"][0])
    repeat["record_id"] = "repeat"
    data["records"].append(repeat)
    model = F.fit(data)
    assert len(model["training_case_ids"]) == 16
    assert V.validate(data, model)["metrics"]["eligible_held_pairs"] == 36
    repeat["label"]["numeric"] = "fail"
    with pytest.raises(C.ContractError, match="every declared training"):
        F.fit(data)


def test_rank_deficiency_and_underidentified_parameters_refuse():
    with pytest.raises(C.ContractError, match="rank deficient"):
        F.nnls([[1, i, 2 * i] for i in range(1, 9)], [i for i in range(1, 9)], maximum_condition=1e6)
    with pytest.raises(C.ContractError, match="two distinct"):
        F.nnls([[1, 1], [1, 2]], [1, 2], maximum_condition=1e6)
    with pytest.raises(C.ContractError, match="maximum condition"):
        F.nnls([[1, i / 10] for i in range(1, 9)], [i for i in range(1, 9)], maximum_condition=1)


def test_declared_coverage_counts_unknown_cases_and_not_only_successes():
    data = sample()
    for case, row in zip(data["cases"], data["records"]):
        if case["split"] == "test" and case["source_group_sha256"] in {h(["group", 6]), h(["group", 7])}:
            row["features"]["host_instructions"] = None
    report = V.validate(data, F.fit(data))
    assert report["metrics"]["prediction_coverage"] == pytest.approx(2 / 3)
    assert report["checks"]["declared_case_coverage"] is False and report["qualified"] is False


def test_out_of_domain_private_and_unqualified_predictions_do_not_rank():
    data = sample()
    model = F.fit(data)
    cases = {
        "schema": C.CASES,
        "profile": data["profile"],
        "domain": data["domain"],
        "cases": [
            {"case_id": "small", "workload_sha256": h("same-workload"), "features": {"host_instructions": 10}},
            {"case_id": "large", "workload_sha256": h("same-workload"), "features": {"host_instructions": 40}},
        ],
    }
    diagnostic = B.compare(model, cases)
    assert diagnostic["comparisons"][0]["status"] == "unknown"
    assert diagnostic["cases"][0]["interval"] is None and diagnostic["cases"][0]["diagnostic_interval"]
    qualified = B.compare(model, cases, qualification={"dataset": data, "report": V.validate(data, model)})
    assert qualified["comparisons"][0]["predicted_faster"] == "small" and not qualified["event_engine_qualified"]
    cases["cases"][1]["features"]["host_instructions"] = 4000
    assert B.compare(model, cases)["cases"][1]["estimated_cycles"] is None
    cases["profile"] = "historical-private"
    with pytest.raises(C.ContractError, match="profile"):
        B.compare(model, cases)


def test_pair_budget_counts_only_actual_same_workload_comparisons():
    data = sample()
    model = F.fit(data)
    cases = {
        "schema": C.CASES,
        "profile": data["profile"],
        "domain": data["domain"],
        "cases": [
            {"case_id": f"{member}-{arm}", "workload_sha256": h(str(member)),
             "features": {"host_instructions": 10 + arm}}
            for member in range(257) for arm in range(2)
        ],
    }
    result = B.compare(model, cases)
    assert len(result["cases"]) == 514 and len(result["comparisons"]) == 257
    assert all(pair["status"] == "unknown" for pair in result["comparisons"])
    for row in cases["cases"]:
        row["workload_sha256"] = h("same-workload")
    with pytest.raises(C.ContractError, match="pair resource limit"):
        B.compare(model, cases)


def test_edited_screen_or_validation_receipt_cannot_acquire_qualification():
    data = sample()
    model = F.fit(data)
    model["fixed_cycles"] += 20
    model["parameters"]["fixed_cycles"] += 20
    with pytest.raises(C.ContractError, match="reproducible"):
        V.validate(data, model)


def test_probe_plan_exports_only_missing_original_evidence_without_launcher():
    data = sample()
    data["records"][-1]["label"] = None
    model = F.fit(data)
    plan = B.probe_plan(data, model)
    assert plan["launch_authority"] is False
    missing = [probe for probe in plan["probes"] if probe["case_id"] == data["cases"][-1]["case_id"]][0]
    assert "fresh_passing_cycle_label" in missing["required"]
    assert "original_native_event_report" in missing["required"]
    assert "argv" not in json.dumps(plan)


def test_cli_roundtrip_and_isolated_runtime_without_compiler_imports(tmp_path):
    data = sample()
    dataset, model, report = (tmp_path / name for name in ("dataset.json", "model.json", "report.json"))
    C.write(dataset, data)
    assert cli.main(["fit", "--dataset", str(dataset), "--output", str(model)]) == 0
    assert cli.main(["validate", "--dataset", str(dataset), "--model", str(model), "--output", str(report)]) == 0
    assert C.read(report)["qualification_scope"] == "empirical_screen_only"
    assert (
        cli.main(
            ["probe-plan", "--dataset", str(dataset), "--model", str(model), "--output", str(tmp_path / "probes.json")]
        )
        == 0
    )
    # A small source archive supplies only this package. Isolated Python has no
    # checkout, PYTHONPATH, user site or compiler framework dependency.
    import merlin_timing

    archive = tmp_path / "runtime.zip"
    package = Path(merlin_timing.__file__).parent
    with zipfile.ZipFile(archive, "w") as zipped:
        for source in package.glob("*.py"):
            zipped.write(source, f"merlin_timing/{source.name}")
    code = (
        "import sys; sys.path.insert(0,sys.argv[1]); "
        "from merlin_timing.cli import main; "
        "assert not any(k == 'merlin' or k.startswith('xdsl') for k in sys.modules); "
        "raise SystemExit(main(sys.argv[2:]))"
    )
    process = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            code,
            str(archive),
            "validate",
            "--dataset",
            str(dataset),
            "--model",
            str(model),
            "--output",
            str(tmp_path / "isolated.json"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.returncode == 0, process.stderr
    assert C.read(tmp_path / "isolated.json") == C.read(report)


def test_json_duplicate_keys_refuse(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":1,"schema":2}')
    with pytest.raises(C.ContractError, match="duplicate"):
        C.read(path)
