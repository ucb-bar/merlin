"""Native count coverage is independent of prices, never a timing qualification."""

import copy

import pytest
from merlin_timing import archive as A
from merlin_timing import contracts as C
from merlin_timing import observations as O
from merlin_timing import validation as V
from test_archive import prepared, reports_for


def count_report(report):
    report = copy.deepcopy(report)
    binding = report["scope_binding"]
    report.update(schema=O.REPORT_V2, instructions=12, timing_complete=False, estimated_cycles=None)
    report["unresolved"] = ["event prices are unspecified"]
    report["counter_windows"] = [{"begin_pc": binding["begin_pc"], "end_pc": binding["end_pc"], "instructions": 10}]
    report["feature_observation"] = {
        "schema": O.COUNT_SCHEMA,
        "complete": True,
        "scope_instructions": 10,
        "counted_cpu_instructions": 10,
        "scope_kind": "counter_pc_pair",
        "window_index": 0,
        "primary_cpu_features": ["host_instructions"],
        "unresolved": [],
        "qualification": "not_established",
    }
    return report


def test_original_count_observations_attach_without_qualifying_event_timing(tmp_path):
    plan, *_ = prepared(tmp_path)
    reports = reports_for(plan)
    for envelope in reports["reports"].values():
        report = count_report(envelope["report"])
        envelope.update(report=report, sha256=C.digest(report))
    data = A.attach(plan, reports)
    assert len(data["records"]) == 6
    assert all(row["feature_provenance"]["complete"] for row in data["records"])
    cases = {case["case_id"]: case for case in data["cases"]}
    for row in data["records"]:
        prediction, reason = V.native_prediction(row, cases[row["case_id"]], data)
        assert prediction is None and "incomplete" in reason


def test_a_new_field_cannot_upgrade_saved_v1_reports(tmp_path):
    plan, *_ = prepared(tmp_path)
    report = count_report(next(iter(reports_for(plan)["reports"].values()))["report"])
    report["schema"] = O.REPORT_V1
    assert not O.feature_complete(report)
    report["timing_complete"] = True
    assert O.feature_complete(report)


@pytest.mark.parametrize(
    "damage", ["count", "scope_count", "duplicate", "missing", "pc", "window", "bool", "reasons", "exit", "qualified"]
)
def test_count_contract_contradictions_refuse(tmp_path, damage):
    plan, *_ = prepared(tmp_path)
    report = count_report(next(iter(reports_for(plan)["reports"].values()))["report"])
    observation = report["feature_observation"]
    if damage == "count":
        observation["counted_cpu_instructions"] += 1
    elif damage == "scope_count":
        observation["scope_instructions"] += 1
    elif damage == "duplicate":
        observation["primary_cpu_features"] *= 2
    elif damage == "missing":
        observation["primary_cpu_features"] = ["not_observed"]
    elif damage == "pc":
        report["counter_windows"][0]["begin_pc"] += 2
    elif damage == "window":
        observation["window_index"] = True
    elif damage == "bool":
        observation["counted_cpu_instructions"] = True
    elif damage == "reasons":
        observation["unresolved"] = ["unsupported count semantics"]
    elif damage == "exit":
        report["raw_functional_exit_code"] = 1
    elif damage == "qualified":
        observation["qualification"] = "qualified"
    with pytest.raises(C.ContractError):
        O.feature_complete(report)


def test_semantic_refusal_remains_incomplete_even_with_event_cycles(tmp_path):
    plan, *_ = prepared(tmp_path)
    reports = reports_for(plan)
    for envelope in reports["reports"].values():
        report = count_report(envelope["report"])
        report["feature_observation"].update(complete=False, unresolved=["counter-derived address"])
        report.update(timing_complete=True, estimated_cycles=100)
        envelope.update(report=report, sha256=C.digest(report))
    data = A.attach(plan, reports)
    assert not any(row["feature_provenance"]["complete"] for row in data["records"])
    assert all(
        C.case_point(case, C.grouped_records(data).get(case["case_id"], []), data["feature_names"])[0] is None
        for case in data["cases"]
    )
