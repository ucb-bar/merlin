"""Synthetic source/split/receipt controls grant no target timing authority."""

import copy

import pytest
from merlin_experiments.phase2 import calibration_domain_coverage as D
from merlin_timing import contracts as C
from merlin_timing import observations as O


@pytest.fixture
def inputs():
    domain = {name: C.digest(name) for name in C.DOMAIN_KEYS - {"label_engine"}}
    domain["label_engine"] = "gsim"
    data = {
        "schema": C.DATASET,
        "profile": "phase1-public",
        "domain": domain,
        "feature_names": ["instructions"],
        "cases": [],
        "records": [],
    }
    catalog = {
        "schema": "merlin.phase2.retained_archive_feedback.v1",
        "domain": domain,
        "archive_sha256": C.digest("original archive"),
        "sources": {},
        "cases": [],
    }
    for index, split in enumerate(("train", "calibration", "test")):
        source = C.digest(["source", index])
        profile = {"family": "matmul", "m": index + 1, "k": 16, "n": 32, "layout": "rank2", "readout": "raw_i32"}
        catalog["sources"][source] = {"semantic_profile": profile}
        case = {
            "case_id": split,
            "source_group_sha256": source,
            "workload_sha256": "",
            "mechanism_family": "matmul",
            "split": split,
            "source": {
                "visibility": "public",
                "phase": "phase1",
                "manifest_sha256": C.digest("manifest"),
                "source_sha256": source,
            },
            "elf_sha256": C.digest(["elf", index]),
            "input_sha256": C.digest(["input", index]),
            "scope_sha256": domain["scope_sha256"],
            "device_sha256": domain["device_sha256"],
        }
        case["workload_sha256"] = C.workload(case)
        record = {
            "record_id": split,
            "case_id": split,
            "features": {"instructions": 12},
            "feature_provenance": {
                "engine_sha256": domain["feature_engine_sha256"],
                "model_sha256": domain["target_model_sha256"],
                "trace_sha256": C.digest(["trace", index]),
                "execution_sha256": C.execution_identity(case),
                "complete": True,
            },
            "label": {
                "engine": "gsim",
                "engine_sha256": domain["label_engine_sha256"],
                "receipt_sha256": C.digest(["receipt", index]),
                "execution_sha256": C.execution_identity(case),
                "status": "pass",
                "numeric": "pass",
                "measured_now": True,
                "cycles": 100,
            },
            "event_report": None,
        }
        report = {
            "schema": O.REPORT_V1,
            "profile": data["profile"],
            "domain_sha256": C.digest(domain),
            "model_sha256": domain["target_model_sha256"],
            "feature_engine_sha256": domain["feature_engine_sha256"],
            **{name: case[name] for name in ("elf_sha256", "input_sha256", "scope_sha256", "device_sha256")},
            "functional_status": "PASS",
            "timing_complete": True,
            "estimated_cycles": 110,
            "feature_counts": record["features"],
            "unresolved": [],
            "cycle_breakdown": {},
        }
        record["event_report"] = {"report": report, "sha256": C.digest(report)}
        original = {
            "record_id": split,
            "originally_measured": True,
            "original_numerical_status": "pass",
            "cycles": 100,
            "native_observation_available": True,
            "scope_binding": {name: case[name] for name in ("elf_sha256", "input_sha256", "scope_sha256")},
            "label_receipt_sha256": record["label"]["receipt_sha256"],
            "engine": "gsim",
            "engine_sha256": domain["label_engine_sha256"],
        }
        data["cases"].append(case)
        data["records"].append(record)
        catalog["cases"].append({"case": case, "records": [original]})
    return catalog, data


def request(catalog, split="train"):
    source = catalog["cases"][("train", "calibration", "test").index(split)]["case"]["source"]["source_sha256"]
    return {"identity": "required", "profile": copy.deepcopy(catalog["sources"][source]["semantic_profile"])}


def test_exact_train_profile_is_only_source_support(inputs):
    catalog, data = inputs
    result = D.assess(catalog, data, [request(catalog)])
    assert result["counts"]["available_native_records"] == 3
    assert result["requirements"][0]["status"] == "exact_public_train_profile_present"
    assert result["requirements"][0]["source_groups_by_split"]["test"] == []
    assert result["current_prediction_issued"] is False
    assert result["promotion_authority"] == "NONE"


@pytest.mark.parametrize(
    "field,value", [("layout", "transposed"), ("readout", "saturated_i8"), ("m", 2), ("n", 31), ("k", 32)]
)
def test_joint_profile_not_axis_range_or_just_shape(inputs, field, value):
    catalog, data = inputs
    req = request(catalog)
    req["profile"][field] = value
    assert D.assess(catalog, data, [req])["requirements"][0]["status"] == "UNKNOWN"


@pytest.mark.parametrize("split", ["test", "calibration"])
def test_nontraining_source_cannot_supply_training_support(inputs, split):
    catalog, data = inputs
    row = D.assess(catalog, data, [request(catalog, split)])["requirements"][0]
    assert row["status"] == "UNKNOWN"
    assert len(row["source_groups_by_split"][split]) == 1


def test_missing_repeat_stays_in_denominator(inputs):
    catalog, data = inputs
    missing = copy.deepcopy(catalog["cases"][0]["records"][0])
    missing["record_id"] = "missing-repeat"
    missing["native_observation_available"] = False
    missing["scope_binding"] = None
    catalog["cases"][0]["records"].append(missing)
    result = D.assess(catalog, data, [request(catalog)])
    assert result["counts"]["declared_records"] == 4
    assert result["counts"]["available_labels"] == 4
    assert result["counts"]["available_native_records"] == 3
    assert result["requirements"][0]["status"] == "UNKNOWN"
    assert "missing-repeat" in result["requirements"][0]["record_ids"]


@pytest.mark.parametrize("field", ["cycles", "label_receipt_sha256", "engine_sha256"])
def test_foreign_original_label_refuses(inputs, field):
    catalog, data = inputs
    catalog["cases"][0]["records"][0][field] = 101 if field == "cycles" else C.digest("foreign")
    with pytest.raises(ValueError, match="original per-record cycle label"):
        D.assess(catalog, data, [])


def test_counter_scope_cannot_be_substituted(inputs):
    catalog, data = inputs
    catalog["cases"][0]["records"][0]["scope_binding"]["elf_sha256"] = C.digest("foreign")
    with pytest.raises(ValueError, match="counter scope"):
        D.assess(catalog, data, [])


def test_catalog_flag_cannot_replace_original_observation(inputs):
    catalog, data = inputs
    data["records"].pop(0)
    result = D.assess(catalog, data, [request(catalog)])
    assert result["requirements"][0]["status"] == "UNKNOWN"
    assert result["counts"]["available_native_records"] == 2


def test_unknown_profile_and_all_members_preserved(inputs):
    catalog, data = inputs
    result = D.assess(catalog, data, [{"identity": "unsupported", "profile": None}, request(catalog)])
    assert len(result["requirements"]) == 2
    assert result["requirements"][0]["status"] == "UNKNOWN"
    assert len(result["cases"]) == 3


def test_source_group_leakage_rejected_by_original_contract(inputs):
    catalog, data = inputs
    data["cases"][1]["source_group_sha256"] = data["cases"][0]["source_group_sha256"]
    with pytest.raises(ValueError):
        D.assess(catalog, data, [])


def test_native_report_missing_cannot_be_replaced_by_complete_features(inputs):
    catalog, data = inputs
    data["records"][0]["event_report"] = None
    result = D.assess(catalog, data, [request(catalog)])
    assert result["requirements"][0]["status"] == "UNKNOWN"
    assert result["counts"]["available_labels"] == 3
    assert result["counts"]["available_native_records"] == 2


def test_unresolved_native_effects_are_not_complete_timing(inputs):
    catalog, data = inputs
    event = data["records"][0]["event_report"]
    event["report"]["unresolved"] = ["unpriced completion"]
    event["sha256"] = C.digest(event["report"])
    result = D.assess(catalog, data, [request(catalog)])
    assert result["requirements"][0]["status"] == "UNKNOWN"


def test_incomplete_counts_do_not_establish_source_training_support(inputs):
    catalog, data = inputs
    data["records"][0]["feature_provenance"]["complete"] = False
    assert D.assess(catalog, data, [request(catalog)])["requirements"][0]["status"] == "UNKNOWN"


def test_exact_shape_does_not_create_missing_schedule_pair(inputs):
    catalog, data = inputs
    plan = {
        "plan_sha256": C.digest("original plan"),
        "acceptance": {"ordering_agreement_min": 0.9, "separation_min_fraction": 0.05},
        "points": [
            {
                "category": "bank_conflicts",
                "label": "bank_pair",
                "schedules": ["same_bank", "interleaved"],
                "reconcile_with_board": False,
            }
        ],
    }
    result = D.assess(catalog, data, [request(catalog)], calibration_plan=plan)
    verdict = result["capacity_validation"]["categories"]["bank_conflicts"]
    assert verdict["status"] == "unknown"
    assert verdict["missing"] == ["bank_pair"]


def test_catalog_case_omission_cannot_improve_coverage(inputs):
    catalog, data = inputs
    catalog["cases"].pop()
    with pytest.raises(ValueError, match="case roster"):
        D.assess(catalog, data, [])
