"""Original native timing predictions use a separate qualification path."""

import copy

import pytest
from data_helpers import h, sample
from merlin_timing import cli
from merlin_timing import contracts as C
from merlin_timing import validation as V


def original_reports(data):
    by_case = {case["case_id"]: case for case in data["cases"]}
    for row in data["records"]:
        case = by_case[row["case_id"]]
        report = {
            "schema": "merlin.annotated_spike_report.v1",
            "profile": data["profile"],
            "domain_sha256": C.digest(data["domain"]),
            "model_sha256": data["domain"]["target_model_sha256"],
            "feature_engine_sha256": data["domain"]["feature_engine_sha256"],
            **{key: case[key] for key in ("device_sha256", "scope_sha256", "elf_sha256", "input_sha256")},
            "domain": data["domain"],
            "functional_status": "PASS",
            "raw_functional_exit_code": 0,
            "functional_status_scope": "ELF exit status; independent numerical gate remains required",
            "timing_complete": True,
            "estimated_cycles": row["label"]["cycles"],
            "feature_counts": dict(row["features"]),
            "unresolved": [],
            "cycle_breakdown": {"synthetic": row["label"]["cycles"]},
        }
        row["event_report"] = {"sha256": C.digest(report), "report": report}
    return data


def test_only_actual_original_event_predictions_can_qualify_the_event_path():
    data = original_reports(sample())
    report = V.validate(data, prediction_kind="event_timing")
    # Arithmetic fixture qualification never claims actual device validation.
    assert report["qualified"] and report["event_engine_qualified"]
    assert report["qualification_scope"] == "native_event_timing"
    assert report["model_sha256"] == data["domain"]["target_model_sha256"]
    missing = V.validate(sample(), prediction_kind="event_timing")
    assert missing["event_engine_qualified"] is False and missing["metrics"]["prediction_coverage"] == 0


def test_bad_held_native_predictions_fail_independently_of_the_linear_screen():
    data = original_reports(sample())
    test_ids = {case["case_id"] for case in data["cases"] if case["split"] == "test"}
    for row in data["records"]:
        if row["case_id"] in test_ids:
            raw = row["event_report"]["report"]
            raw["estimated_cycles"] = 150 - 3 * row["features"]["host_instructions"]
            row["event_report"]["sha256"] = C.digest(raw)
    report = V.validate(data, prediction_kind="event_timing")
    assert report["metrics"]["rank_agreement"] == 0 and not report["event_engine_qualified"]


@pytest.mark.parametrize(
    "field",
    [
        "profile",
        "domain_sha256",
        "model_sha256",
        "feature_engine_sha256",
        "device_sha256",
        "elf_sha256",
        "input_sha256",
        "scope_sha256",
        "feature_counts",
    ],
)
def test_original_native_report_cannot_be_reattributed(field):
    data = original_reports(sample())
    row = data["records"][-1]
    raw = row["event_report"]["report"]
    raw[field] = {"host_instructions": 999} if field == "feature_counts" else h("different")
    row["event_report"]["sha256"] = C.digest(raw)
    with pytest.raises(C.ContractError):
        V.validate(data, prediction_kind="event_timing")


def test_changed_report_and_incomplete_native_execution_do_not_supply_cycles():
    data = original_reports(sample())
    data["records"][-1]["event_report"]["report"]["estimated_cycles"] = 1
    with pytest.raises(C.ContractError, match="digest"):
        V.validate(data, prediction_kind="event_timing")
    for row in data["records"]:
        raw = row["event_report"]["report"]
        raw["timing_complete"] = False
        row["event_report"]["sha256"] = C.digest(raw)
    report = V.validate(data, prediction_kind="event_timing")
    assert report["metrics"]["predicted_held_cases"] == 0 and report["qualified"] is False


def test_repeat_native_reports_cannot_inflate_pair_counts_or_choose_a_faster_result():
    data = original_reports(sample())
    repeated = copy.deepcopy(data["records"][-1])
    repeated["record_id"] = "native-repeat"
    data["records"].append(repeated)
    assert V.validate(data, prediction_kind="event_timing")["metrics"]["eligible_held_pairs"] == 36
    repeated["event_report"]["report"]["estimated_cycles"] += 1
    repeated["event_report"]["sha256"] = C.digest(repeated["event_report"]["report"])
    report = V.validate(data, prediction_kind="event_timing")
    assert report["metrics"]["predicted_held_cases"] == 23
    assert any("repeat native" in row["reason"] for row in report["refusals"])


def test_native_cli_has_no_screen_or_linear_qualification_shortcut(tmp_path):
    data = original_reports(sample())
    source, result = tmp_path / "source.json", tmp_path / "validation.json"
    C.write(source, data)
    assert (
        cli.main(["validate", "--dataset", str(source), "--prediction-kind", "event_timing", "--output", str(result)])
        == 0
    )
    assert C.read(result)["qualification_scope"] == "native_event_timing"
    assert (
        cli.main(
            [
                "validate",
                "--dataset",
                str(source),
                "--prediction-kind",
                "event_timing",
                "--model",
                str(source),
                "--output",
                str(result),
            ]
        )
        == 2
    )


def test_gate_requires_twenty_real_held_pairs_and_three_distinct_families():
    data = sample()
    # Three variants of each held workload yield only 18 independent pairs.
    remove = {case["case_id"] for case in data["cases"] if case["split"] == "test" and case["case_id"].endswith("-3")}
    data["cases"] = [case for case in data["cases"] if case["case_id"] not in remove]
    data["records"] = [row for row in data["records"] if row["case_id"] not in remove]
    report = V.validate(original_reports(data), prediction_kind="event_timing")
    assert report["metrics"]["eligible_held_pairs"] == 18 and report["checks"]["eligible_pairs"] is False
    assert report["qualified"] is False
    data = sample()
    for case in data["cases"]:
        case["mechanism_family"] = "one-mechanism"
    report = V.validate(original_reports(data), prediction_kind="event_timing")
    assert report["checks"]["mechanism_families"] is False and report["qualified"] is False


def test_five_percent_measured_separation_and_prediction_ties_cannot_earn_rank_credit():
    data = sample()
    held = {case["case_id"] for case in data["cases"] if case["split"] == "test"}
    for row in data["records"]:
        if row["case_id"] in held:
            row["label"]["cycles"] = 95 if row["features"]["host_instructions"] < 30 else 100
    report = V.validate(original_reports(data), prediction_kind="event_timing")
    assert report["metrics"]["eligible_held_pairs"] == 0  # exactly 5% of the slower measurement
    data = original_reports(sample())
    for row in data["records"]:
        if row["case_id"] in held:
            row["event_report"]["report"]["estimated_cycles"] = 90
            row["event_report"]["sha256"] = C.digest(row["event_report"]["report"])
    assert V.validate(data, prediction_kind="event_timing")["metrics"]["rank_agreement"] == 0


def test_good_median_does_not_hide_tail_errors_or_missing_calibration_intervals():
    data = original_reports(sample())
    for row in data["records"][-2:]:
        raw = row["event_report"]["report"]
        raw["estimated_cycles"] *= 2
        row["event_report"]["sha256"] = C.digest(raw)
    report = V.validate(data, prediction_kind="event_timing")
    assert report["checks"]["median_error"] and report["checks"]["p95_error"] is False
    assert report["qualified"] is False
    data = original_reports(sample())
    remove = {case["case_id"] for case in data["cases"] if case["source_group_sha256"] == h(["group", 4])}
    for row in data["records"]:
        if row["case_id"] in remove:
            row["label"] = None
    report = V.validate(data, prediction_kind="event_timing")
    assert report["metrics"]["rank_agreement"] == 1
    assert report["residual_interval"]["status"] == "unknown" and report["checks"]["calibration_intervals"] is False
    assert report["qualified"] is False


def test_descriptive_native_extensions_do_not_override_selected_timing_or_identity():
    data = original_reports(sample())
    raw = data["records"][-1]["event_report"]["report"]
    raw["instructions"] = 400
    raw["estimated_cycles_override"] = 1
    raw["arbitrary_diagnostic"] = {"cache": "unknown"}
    data["records"][-1]["event_report"]["sha256"] = C.digest(raw)
    report = V.validate(data, prediction_kind="event_timing")
    last = data["cases"][-1]["case_id"]
    assert report["predictions"][last]["predicted_cycles"] == raw["estimated_cycles"]


@pytest.mark.parametrize("mutation", ["exit", "domain", "status"])
def test_native_writer_status_and_optional_identity_fields_are_not_descriptive_overrides(mutation):
    data = original_reports(sample())
    raw = data["records"][-1]["event_report"]["report"]
    if mutation == "exit":
        raw["raw_functional_exit_code"] = 1
    elif mutation == "domain":
        raw["domain"] = {**raw["domain"], "memory_regime_sha256": h("different")}
    else:
        raw["functional_status"] = "pass"  # not the native writer's explicit PASS enum
    data["records"][-1]["event_report"]["sha256"] = C.digest(raw)
    if mutation in {"exit", "domain"}:
        with pytest.raises(C.ContractError):
            V.validate(data, prediction_kind="event_timing")
    else:
        report = V.validate(data, prediction_kind="event_timing")
        assert data["cases"][-1]["case_id"] not in report["predictions"]


def test_native_feature_projection_keeps_selected_roster_and_ignores_digest_bound_diagnostics():
    data = original_reports(sample())
    raw = data["records"][-1]["event_report"]["report"]
    raw["feature_counts"].update({f"diagnostic.counter.{index}": index for index in range(100)})
    data["records"][-1]["event_report"]["sha256"] = C.digest(raw)
    report = V.validate(data, prediction_kind="event_timing")
    assert report["metrics"]["predicted_held_cases"] == 24
    raw["feature_counts"].pop("host_instructions")
    data["records"][-1]["event_report"]["sha256"] = C.digest(raw)
    with pytest.raises(C.ContractError, match="omits selected"):
        V.validate(data, prediction_kind="event_timing")


@pytest.mark.parametrize("completion", [False, 1, None, "true"])
def test_native_execution_completion_cannot_be_a_descriptive_status_override(completion):
    data = original_reports(sample())
    row = data["records"][-1]
    raw = row["event_report"]["report"]
    raw["execution_completed"] = completion
    row["event_report"]["sha256"] = C.digest(raw)
    with pytest.raises(C.ContractError, match="completion|incomplete execution"):
        V.validate(data, prediction_kind="event_timing")
    raw["execution_completed"] = True
    row["event_report"]["sha256"] = C.digest(raw)
    assert V.validate(data, prediction_kind="event_timing")["metrics"]["predicted_held_cases"] == 24


def test_strict_profile_rejects_one_bad_case_hidden_by_good_percentiles():
    data = original_reports(sample())
    raw = data["records"][-1]["event_report"]["report"]
    raw["estimated_cycles"] *= 1.05
    data["records"][-1]["event_report"]["sha256"] = C.digest(raw)
    legacy = V.validate(data, prediction_kind="event_timing", qualification_profile="legacy_v1")
    assert legacy["qualified"]
    strict = V.validate(data, prediction_kind="event_timing")
    assert strict["checks"]["median_error"] and strict["checks"]["p95_error"]
    assert not strict["checks"]["every_case_error"] and not strict["qualified"]
    assert strict["metrics"]["maximum_absolute_relative_error"] == pytest.approx(0.05)


def test_strict_profile_requires_every_declared_case_and_keeps_refusal():
    data = original_reports(sample())
    data["records"][-1]["event_report"] = None
    legacy = V.validate(data, prediction_kind="event_timing", qualification_profile="legacy_v1")
    assert legacy["qualified"]
    strict = V.validate(data, prediction_kind="event_timing")
    assert not strict["checks"]["declared_case_coverage"] and not strict["qualified"]
    assert strict["metrics"]["declared_held_cases"] == 24
    assert strict["refusals"][-1]["case_id"] == data["cases"][-1]["case_id"]


def test_four_percent_boundary_passes_but_larger_error_does_not():
    data = original_reports(sample())
    raw = data["records"][-1]["event_report"]["report"]
    assert raw["estimated_cycles"] == 125
    raw["estimated_cycles"] = 130
    data["records"][-1]["event_report"]["sha256"] = C.digest(raw)
    assert V.validate(data, prediction_kind="event_timing")["qualified"]
    raw["estimated_cycles"] = 130.0001
    data["records"][-1]["event_report"]["sha256"] = C.digest(raw)
    assert not V.validate(data, prediction_kind="event_timing")["qualified"]
    with pytest.raises(C.ContractError, match="profile"):
        V.validate(data, prediction_kind="event_timing", qualification_profile="forgiving")
