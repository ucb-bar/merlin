"""Synthetic native responses test diagnostics, never target qualification."""

import copy
import math

import pytest
from data_helpers import h, sample
from merlin_timing import cli
from merlin_timing import contracts as C
from merlin_timing import native_sensitivity as S


def fixture():
    dataset = sample()
    cases = [case for case in dataset["cases"] if case["split"] == "train"]
    rows = {row["case_id"]: row for row in dataset["records"]}
    evaluations = []
    for index, parameters in enumerate(
        (
            {"a": 1, "b": 1, "masked": 1},
            {"a": 2, "b": 1, "masked": 1},
            {"a": 1, "b": 2, "masked": 1},
            {"a": 1, "b": 1, "masked": 2},
        )
    ):
        predictions = {
            case["case_id"]: 5
            + (parameters["a"] + 2 * parameters["b"]) * rows[case["case_id"]]["features"]["host_instructions"]
            for case in cases
        }
        evaluations.append(
            {
                "model_sha256": h(["model", index]),
                "fixed_model_sha256": h("fixed"),
                "parameters": parameters,
                "predictions": predictions,
            }
        )
    return dataset, {"schema": S.SCHEMA, "dataset_sha256": C.digest(dataset), "evaluations": evaluations}


def test_coupled_prices_and_masked_cost_are_not_physical_identification():
    dataset, document = fixture()
    report = S.analyze(dataset, document)
    assert report["declared_train_cases"] == report["available_train_cases"] == 16
    assert report["available_train_source_groups"] == 4
    assert report["response_matrix_rank"] == 1
    assert report["response_matrix_columns"] == 3
    assert report["coupled_responses"][0]["parameters"] == ["a", "b"]
    assert report["coupled_responses"][0]["response_cosine"] == pytest.approx(1)
    masked = next(row for row in report["parameter_sensitivity"] if row["parameter"] == "masked")
    assert masked["status"] == "no_observed_response"
    assert report["physical_price_identification"] == "NOT_ESTABLISHED"
    assert report["qualification_issued"] is report["launch_authority"] is False


def test_held_and_calibration_labels_do_not_affect_sensitivity_or_training_scores():
    dataset, document = fixture()
    before = S.analyze(dataset, document)
    cases = {case["case_id"]: case for case in dataset["cases"]}
    for row in dataset["records"]:
        if cases[row["case_id"]]["split"] != "train":
            row["label"]["cycles"] *= 1000
    document["dataset_sha256"] = C.digest(dataset)
    after = S.analyze(dataset, document)
    assert {key: value for key, value in before.items() if key != "dataset_sha256"} == {
        key: value for key, value in after.items() if key != "dataset_sha256"
    }


def test_unavailable_original_cases_and_missing_predictions_do_not_improve_scores():
    dataset, document = fixture()
    original_missing = dataset["cases"][0]["case_id"]
    dataset["records"] = [row for row in dataset["records"] if row["case_id"] != original_missing]
    document["dataset_sha256"] = C.digest(dataset)
    current_missing = dataset["cases"][1]["case_id"]
    document["evaluations"][1]["predictions"][current_missing] = None
    report = S.analyze(dataset, document)
    assert report["declared_train_cases"] == 16 and report["available_train_cases"] == 15
    assert report["unavailable_train_cases"][0]["case_id"] == original_missing
    assert report["evaluations"][1]["metrics"] is None
    assert report["evaluations"][1]["missing_predictions"] == [current_missing]
    assert (
        next(row for row in report["parameter_sensitivity"] if row["parameter"] == "a")["status"]
        == "no_complete_one_parameter_contrast"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "private",
        "held_prediction",
        "drop_case",
        "structure",
        "duplicate",
        "nan",
        "bool",
        "negative",
        "roster",
        "conflicting_repeat",
        "dataset",
    ],
)
def test_invalid_or_leaking_native_probes_refuse(mutation):
    dataset, document = fixture()
    second = document["evaluations"][1]
    if mutation == "private":
        dataset = sample(profile="historical-private")
        document["dataset_sha256"] = C.digest(dataset)
    elif mutation == "held_prediction":
        held = next(case for case in dataset["cases"] if case["split"] == "test")
        second["predictions"][held["case_id"]] = 10
    elif mutation == "drop_case":
        second["predictions"].pop(next(iter(second["predictions"])))
    elif mutation == "structure":
        second["fixed_model_sha256"] = h("other fixed")
    elif mutation == "duplicate":
        second["model_sha256"] = document["evaluations"][0]["model_sha256"]
    elif mutation in {"nan", "bool", "negative"}:
        second["parameters"]["a"] = {"nan": math.nan, "bool": True, "negative": -1}[mutation]
    elif mutation == "roster":
        second["parameters"]["extra"] = 1
    elif mutation == "conflicting_repeat":
        second["parameters"] = copy.deepcopy(document["evaluations"][0]["parameters"])
    else:
        document["dataset_sha256"] = h("different dataset")
    with pytest.raises(C.ContractError):
        S.analyze(dataset, document)


def test_cli_exposes_the_same_analysis_without_launches(tmp_path):
    dataset, document = fixture()
    data, probes, output = (tmp_path / name for name in ("dataset.json", "probes.json", "report.json"))
    C.write(data, dataset)
    C.write(probes, document)
    assert (
        cli.main(["native-sensitivity", "--dataset", str(data), "--evaluations", str(probes), "--output", str(output)])
        == 0
    )
    assert C.read(output) == S.analyze(dataset, document)


def test_work_budget_refuses_before_allocating_all_pair_response_vectors(monkeypatch):
    dataset, document = fixture()
    monkeypatch.setattr(S, "MAX_SCALAR_WORK", 1)
    with pytest.raises(C.ContractError, match="scalar-work budget"):
        S.analyze(dataset, document)


def test_finite_large_values_do_not_silently_overflow_correlation_or_norm():
    assert S._cosine([1e200, 1e200], [1e200, -1e200]) == pytest.approx(0)
    with pytest.raises(C.ContractError, match="finite diagnostic range"):
        S._norm([1e308] * 10)
