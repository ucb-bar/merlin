"""Synthetic integration checks; these fixtures qualify no actual timing model."""

import copy
import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import archive_native_screen as N
from merlin_experiments.phase2 import archive_screen_selection as SELECT
from merlin_experiments.phase2 import ordinary_fast_feedback as FAST
from merlin_experiments.phase2 import ordinary_search_policy as OS
from merlin_experiments.phase2.contracts import StageGateError, sha256_file
from merlin_timing import contracts as C
from test_event_validation import original_reports
from test_native_comparison import inference_fixture
from test_ordinary_archive_screen import _SyntheticCorrespondence
from test_ordinary_archive_screen import source as source
from test_ordinary_fast_feedback import cheap_inputs as cheap_inputs
from test_ordinary_fast_feedback import revisions
from test_ordinary_search_policy import ordinary_inputs as ordinary_inputs

from merlin.llvmlower.reviewed_pass_capabilities import ReviewedPassCapability


def native_source(source):
    original_reports(source["dataset"])
    for row in source["dataset"]["records"]:
        raw = row["event_report"]["report"]
        raw["feature_counts"].update({"unfitted.fp": 0, "constant.issue": 2})
        row["event_report"]["sha256"] = C.digest(raw)
        source["observations"]["reports"][row["record_id"]]["report"] = raw
    source["observations_path"].write_text(json.dumps(source["observations"]))
    arguments = {**source["arguments"], "screen_path": None}
    arguments["native_semantics"] = ReviewedPassCapability(
        "native_timing_semantics", arguments["correspondence"].binding, _SyntheticCorrespondence()
    )
    return arguments


def test_native_selector_reconstructs_original_event_gates_separately_from_screen(source):
    selection = SELECT.NativeArchiveScreenSelection(**native_source(source))
    assert selection.preflight()["report"]["event_engine_qualified"]
    owner = SELECT.install(selection, source["policy"])
    assert owner.summary()["event_engine_qualified"]
    assert owner.summary()["prediction_kind"] == "event_timing"
    assert json.loads(owner.screen_json) is None
    assert json.loads(owner.report_json)["qualification_scope"] == "native_event_timing"


@pytest.mark.parametrize(
    "damage", ["absent", "absent_with_linear", "copied", "wrong_mechanism", "linear", "bad_native", "stale"]
)
def test_native_selection_requires_live_semantics_and_original_event_gates(source, damage):
    arguments = native_source(source)
    if damage in {"absent", "absent_with_linear"}:
        arguments["native_semantics"] = None
        if damage == "absent_with_linear":
            arguments["screen_path"] = source["screen_path"]
    elif damage == "copied":
        arguments["native_semantics"] = arguments["native_semantics"].record()
    elif damage == "wrong_mechanism":
        arguments["native_semantics"] = arguments["correspondence"]
    elif damage == "linear":
        arguments["screen_path"] = source["screen_path"]
    elif damage == "bad_native":
        for row in source["dataset"]["records"]:
            raw = row["event_report"]["report"]
            raw["estimated_cycles"] = 150 - 3 * row["features"]["host_instructions"]
            row["event_report"]["sha256"] = C.digest(raw)
    else:
        arguments["native_semantics"].owner.identity = "changed synthetic review"
    with pytest.raises((ValueError, StageGateError)):
        SELECT.install(SELECT.NativeArchiveScreenSelection(**arguments), source["policy"])


def test_training_presence_alone_cannot_qualify_an_uncovered_attention_family(source):
    arguments = native_source(source)
    for case in source["dataset"]["cases"]:
        if case["source_group_sha256"] == C.digest(["group", 0]):
            case["mechanism_family"] = "attention"
    owner = SELECT.install(SELECT.NativeArchiveScreenSelection(**arguments), source["policy"])
    assert "attention" not in owner.families
    assert set(owner.families) == {"matmul", "conv", "resident_chain"}


def test_native_queue_preserves_unknown_members_and_original_measured_final_gates(source, cheap_inputs, tmp_path):
    arguments = native_source(source)
    policy = OS.prepare_ordinary_search_policy(**source["ordinary_inputs"].arguments)
    SELECT.install(SELECT.NativeArchiveScreenSelection(**arguments), policy)
    candidate, baseline = revisions(policy, tmp_path)
    result = policy.evaluate_retained(
        candidate, baseline, kind=FAST.KIND, work_root=tmp_path / "native-queue", call_index=0, timeout_s=10
    )
    assert result["screening"]["event_engine_qualified"]
    assert result["screening"]["promotion_authority"] == "NONE"
    assert len(result["screening"]["evidence"]) == len(policy.search_corpus.capsules)
    assert all(row["conservative_saved_cycles"] is None for row in result["screening"]["evidence"])
    for cell in result["cells"]:
        for arm in ("baseline", "candidate"):
            assert cell[arm]["timing_cycles"] is None and cell[arm]["pending_tiers"] == ["L3"]


def test_identical_native_arms_are_one_inference_and_both_remain_visible():
    cases, receipt = inference_fixture()
    baseline = cases["cases"][0]
    candidate = copy.deepcopy(baseline)
    candidate["case"]["case_id"] = "candidate-identical"
    result = N.compare([baseline, candidate], dataset=receipt["dataset"], qualification=receipt)
    assert len(result["cases"]) == 2 and len(result["comparisons"]) == 0
    assert {row["case_id"] for row in result["cases"]} == {baseline["case"]["case_id"], "candidate-identical"}
    assert result["cases"][0]["estimated_cycles"] == result["cases"][1]["estimated_cycles"]
    candidate["event_report"]["report"]["estimated_cycles"] += 1
    candidate["event_report"]["sha256"] = C.digest(candidate["event_report"]["report"])
    with pytest.raises(StageGateError, match="disagree"):
        N.compare([baseline, candidate], dataset=receipt["dataset"], qualification=receipt)


def test_identical_predictions_preserve_distinct_original_counter_reviews_and_numerical_receipts():
    cases, receipt = inference_fixture()
    baseline = cases["cases"][0]
    candidate = copy.deepcopy(baseline)
    candidate["case"]["case_id"] = "another-current-execution"
    for index, row in enumerate((baseline, candidate)):
        raw = row["event_report"]["report"]
        raw["scope_binding"] = {"begin_pc": 16, "end_pc": 32, "occurrence": 0, "evidence_sha256": C.digest(index)}
        raw["scope_binding_sha256"] = C.digest(raw["scope_binding"])
        raw["cycle_breakdown"] = {"scope": "whole_trace", "modeled_horizon": 100 + index}
        row["event_report"]["sha256"] = C.digest(raw)
        row["numeric"]["receipt_sha256"] = C.digest([index, "fresh numerical receipt"])
    result = N.compare([baseline, candidate], dataset=receipt["dataset"], qualification=receipt)
    assert len(result["cases"]) == 2 and not result["comparisons"]
    for actual, original in zip(result["cases"], (baseline, candidate), strict=True):
        assert actual["original_execution_evidence"] == {
            "event_report_sha256": original["event_report"]["sha256"],
            "numeric_receipt_sha256": original["numeric"]["receipt_sha256"],
        }
    candidate["event_report"]["report"]["scope_binding"]["begin_pc"] = 17
    candidate["event_report"]["sha256"] = C.digest(candidate["event_report"]["report"])
    with pytest.raises(StageGateError, match="disagree"):
        N.compare([baseline, candidate], dataset=receipt["dataset"], qualification=receipt)


@pytest.mark.parametrize("damage", ["event_digest", "numeric_status", "numeric_freshness"])
def test_every_identical_arm_is_checked_before_collapsing_inference(damage):
    cases, receipt = inference_fixture()
    baseline = cases["cases"][0]
    candidate = copy.deepcopy(baseline)
    candidate["case"]["case_id"] = "another-current-execution"
    if damage == "event_digest":
        candidate["event_report"]["sha256"] = C.digest("copied verdict")
    else:
        candidate["numeric"]["status" if damage == "numeric_status" else "measured_now"] = (
            "fail" if damage == "numeric_status" else False
        )
    with pytest.raises((StageGateError, C.ContractError)):
        N.compare([baseline, candidate], dataset=receipt["dataset"], qualification=receipt)


@pytest.mark.parametrize("damage", [None, "copied_value", "changed_file", "failed_tier", "other_elf"])
def test_native_inference_reopens_actual_original_complete_numerics(tmp_path, damage):
    cases, receipt = inference_fixture()
    row = cases["cases"][0]
    report = row["event_report"]["report"]
    original = {
        "functional_status": "PASS",
        "compiled": {"elf_sha256": report["elf_sha256"]},
        "original_grade": {"tiers": {tier: {"status": "pass"} for tier in ("L0", "L1", "L2")}},
    }
    if damage == "failed_tier":
        original["original_grade"]["tiers"]["L2"]["status"] = "fail"
    if damage == "other_elf":
        original["compiled"]["elf_sha256"] = C.digest("another ELF")
    path = tmp_path / "ordinary_observation.json"
    path.write_text(json.dumps(original))
    value = {k: v for k, v in original.items() if k != "original_grade"} | {"evidence_sha256": sha256_file(path)}
    if damage == "copied_value":
        value["functional_status"] = "UNKNOWN"
    elif damage == "changed_file":
        path.write_text("changed bytes")
    arguments = dict(
        key="own",
        member=SimpleNamespace(source_sha256=row["case"]["source"]["source_sha256"]),
        family=row["case"]["mechanism_family"],
        dataset=receipt["dataset"],
        manifest_sha256=C.digest("manifest"),
        report=report,
        directory=tmp_path,
        value=value,
    )
    if damage:
        with pytest.raises(StageGateError):
            N.inference_case(**arguments)
    else:
        actual = N.inference_case(**arguments)
        assert actual["numeric"]["receipt_sha256"] == sha256_file(path)
        assert actual["numeric"]["execution_sha256"] == C.execution_identity(actual["case"])
