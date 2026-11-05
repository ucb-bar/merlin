"""Explicit record availability never changes declared held coverage or gates."""

import copy

import pytest
from data_helpers import sample
from merlin_timing import contracts as C
from merlin_timing import feedback as B
from merlin_timing import fitting as F
from merlin_timing import validation as V


def test_eligible_training_is_explicit_and_preserves_the_original_dataset():
    data = sample()
    missing = data["records"].pop(0)
    original = copy.deepcopy(data)
    with pytest.raises(C.ContractError, match="every declared training"):
        F.fit(data)
    model = F.fit(data, training_intake="eligible_records_v1")
    assert model["schema"] == C.MODEL_V2
    assert model["training_intake"]["unavailable_training_cases"][0]["case_id"] == missing["case_id"]
    assert len(model["training_case_ids"]) == 15
    assert len(model["training_intake"]["declared_training_case_ids"]) == 16
    assert data == original
    assert V.validate(data, model)["metrics"]["eligible_held_pairs"] == 36
    assert model["engine_qualification"] == "not_established"


def test_missing_held_cases_remain_in_the_coverage_denominator():
    data = sample()
    data["records"] = [row for row in data["records"] if not row["case_id"].startswith("test-6")]
    model = F.fit(data, training_intake="eligible_records_v1")
    report = V.validate(data, model)
    assert report["metrics"]["prediction_coverage"] == pytest.approx(20 / 24)
    assert len(report["refusals"]) == 4


@pytest.mark.parametrize("mutation", ["reason", "drop", "overlap", "policy"])
def test_unavailable_training_inventory_cannot_be_forged(mutation):
    data = sample()
    data["records"].pop(0)
    model = F.fit(data, training_intake="eligible_records_v1")
    intake = model["training_intake"]
    if mutation == "reason":
        intake["unavailable_training_cases"][0]["reason"] = "invented"
    elif mutation == "drop":
        intake["declared_training_case_ids"].pop()
    elif mutation == "overlap":
        intake["unavailable_training_cases"][0]["case_id"] = model["training_case_ids"][0]
    else:
        intake["policy"] = "choose_fast_cases"
    with pytest.raises(C.ContractError):
        V.validate(data, model)


def test_eligible_records_preserve_functional_and_source_independence_checks():
    data = sample()
    data["records"][0]["feature_provenance"]["complete"] = False
    model = F.fit(data, training_intake="eligible_records_v1")
    assert len(model["training_intake"]["unavailable_training_cases"]) == 1
    data["cases"][-1]["source"]["visibility"] = "hidden"
    with pytest.raises(C.ContractError):
        F.fit(data, training_intake="eligible_records_v1")


def test_explicit_relative_loss_optimizes_training_relative_error_and_is_rebuilt():
    data = sample()
    data["records"][0]["label"]["cycles"] *= 10
    absolute = F.fit(data)
    relative = F.fit(data, loss="relative_squared")
    assert absolute["schema"] == C.MODEL and "fit_loss" not in absolute
    assert relative["schema"] == C.MODEL_V3 and relative["fit_loss"] == "relative_squared"
    train = [r for r in data["records"] if r["case_id"].startswith("train-")]

    def error(model):
        return sum(
            (F.predict(model, r["features"], domain_sha256=model["domain_sha256"])[0] / r["label"]["cycles"] - 1) ** 2
            for r in train
        )

    assert error(relative) < error(absolute)
    assert V.validate(data, relative)["event_engine_qualified"] is False
    relative["fit_loss"] = "absolute_squared"
    with pytest.raises(C.ContractError):
        V.validate(data, relative)


def test_feedback_exposes_observed_terms_without_claiming_hardware_attribution():
    data = sample()
    model = F.fit(data, loss="relative_squared")
    cases = {
        "schema": C.CASES,
        "profile": data["profile"],
        "domain": data["domain"],
        "cases": [
            {
                "case_id": "case",
                "workload_sha256": data["cases"][0]["workload_sha256"],
                "features": {"host_instructions": 10},
            }
        ],
    }
    result = B.compare(model, cases)["cases"][0]
    terms = result["screen_terms"]
    assert terms["fixed_cycles"] + sum(r["estimated_cycles"] for r in terms["features"]) == result["estimated_cycles"]
    assert terms["features"][0]["observed_count"] == 10
    assert "not physical" in terms["interpretation"]
    assert result["status"] == "diagnostic" and result["interval"] is None
    cases["cases"][0]["features"]["host_instructions"] = 1000
    assert B.compare(model, cases)["cases"][0]["screen_terms"] is None
