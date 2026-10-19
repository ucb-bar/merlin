"""Explicit record availability never changes declared held coverage or gates."""

import copy

import pytest
from data_helpers import sample
from merlin_timing import contracts as C
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
