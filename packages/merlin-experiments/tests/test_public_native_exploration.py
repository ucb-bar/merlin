"""Synthetic transport controls; no fixture qualifies actual target timing."""

import copy
import json
from dataclasses import replace

import pytest
from merlin_experiments.phase2 import _public_native_feedback as FEEDBACK
from merlin_experiments.phase2 import _public_native_observation as OBS
from merlin_experiments.phase2 import archive_screen_selection as SELECTION
from merlin_experiments.phase2 import public_native_exploration as N
from merlin_experiments.phase2 import public_source_diagnostic as D
from merlin_experiments.phase2.contracts import StageGateError, document_sha256, sha256_file
from test_native_comparison import inference_fixture
from test_ordinary_native_archive_screen import native_source
from test_ordinary_native_archive_screen import ordinary_inputs as ordinary_inputs
from test_ordinary_native_archive_screen import source as source
from test_public_source_diagnostic import inputs as inputs

from merlin.llvmlower.reviewed_pass_capabilities import ReviewedPassCapability


def selection(source):
    return SELECTION.NativeArchiveScreenSelection(**native_source(source))


def test_original_public_native_gates_and_owner_mutation_refusal(source):
    owner = N.prepare_public_native_exploration(selection(source))
    record = owner.record()
    assert record["prediction_kind"] == "event_timing" and record["measured_cycles"] is None
    assert record["promotion_authority"] == "NONE"
    with pytest.raises(StageGateError, match="issued owners"):
        copy.copy(owner).verify()
    object.__setattr__(owner, "catalog_json", b"{}")
    with pytest.raises(StageGateError, match="issued owners"):
        owner.verify()


@pytest.mark.parametrize("fault", ["saved", "linear", "absent_semantics", "wrong_domain"])
def test_saved_or_incompatible_selection_never_issues_native_owner(source, fault):
    selected = selection(source)
    if fault == "saved":
        selected = {"status": "qualified_native"}
    elif fault == "linear":
        selected = replace(selected, screen_path=source["screen_path"])
    elif fault == "absent_semantics":
        selected = replace(selected, native_semantics=None)
    else:
        selected = replace(selected, expected_rtl_facts_sha256="f" * 64)
    with pytest.raises((ValueError, StageGateError)):
        N.prepare_public_native_exploration(selected)


def test_changed_original_observation_or_function_refuses(source, monkeypatch):
    owner = N.prepare_public_native_exploration(selection(source))
    original = N.DOMAIN.current
    monkeypatch.setattr(N.DOMAIN, "current", lambda *_a: {"status": "source_domain_supported"})
    with pytest.raises(StageGateError, match="issued owners"):
        owner.verify()
    monkeypatch.setattr(N.DOMAIN, "current", original)
    source["observations_path"].write_text("changed evidence")
    with pytest.raises(StageGateError, match="evidence changed"):
        owner.verify()


def test_explicit_native_policy_keeps_no_certificate_and_no_ordinary_screen(source, tmp_path, monkeypatch):
    public_root = tmp_path / "public-native-policy"
    public_root.mkdir()
    args, *_ = inputs.__wrapped__(public_root, monkeypatch)
    selected = selection(source)
    binding = replace(
        selected.correspondence.binding,
        hardware=sha256_file(args["service"].rtl_facts_path)
        if hasattr(args["service"], "rtl_facts_path")
        else args["compiler_pass_selection"].binding.hardware,
        software=args["compiler_pass_selection"].binding.software,
    )
    selected = replace(
        selected,
        expected_rtl_facts_sha256=binding.hardware,
        expected_observed_contract_sha256=binding.software,
        correspondence=ReviewedPassCapability("timing_model_correspondence", binding, selected.correspondence.owner),
        native_semantics=ReviewedPassCapability("native_timing_semantics", binding, selected.native_semantics.owner),
    )
    owner = N.prepare_public_native_exploration(selected)
    policy = D.prepare_public_source_diagnostic(**args, native_exploration=owner)
    assert policy._ordinary_screen is None
    assert policy.binding()["schema"] == "merlin.public_source_diagnostic_policy.v2"
    assert policy.binding()["native_exploration"] == owner.record()
    policy._native_exploration = None
    with pytest.raises(StageGateError, match="original execution owner"):
        policy.validate_search_inputs()


def test_uncovered_joint_source_abstains_before_any_count_job(source, tmp_path, monkeypatch):
    monkeypatch.setattr(N.DOMAIN, "current", lambda *_a: {"status": "UNKNOWN", "reason": "new capacity domain"})
    monkeypatch.setattr(N.COUNTS, "observe", lambda *_a, **_kw: pytest.fail("uncovered source must not execute"))
    owner = N.prepare_public_native_exploration(selection(source))
    result = OBS.observe(
        owner, None, None, tmp_path, {"functional_status": "PASS", "compiled": {}}, sandbox=None, remaining_s=10
    )
    assert result["status"] == "UNKNOWN" and result["prediction"] is None
    assert result["reason"] == "new capacity domain"


def projection(owner, interval):
    return {
        "schema": OBS.SCHEMA,
        "status": "predicted",
        "reason": None,
        "selection_sha256": document_sha256(owner.record()),
        "receipt_sha256": "a" * 64,
        "prediction": {
            "status": "qualified_native",
            "public_feedback_eligible": True,
            "prediction_kind": "event_timing",
            "interval": interval,
        },
        "measured_cycles": None,
        "promotion_authority": "NONE",
    }


@pytest.mark.parametrize("interval,expected", [([50, 60], "candidate"), ([95, 125], None), ([130, 160], "baseline")])
def test_intervals_only_prioritize_selected_members_never_claim_objective(source, interval, expected):
    owner = N.prepare_public_native_exploration(selection(source))
    doc = {
        "selected_members": [["family", "selected"]],
        "cells": [
            {
                "family": "family",
                "capsule": "selected",
                "baseline": {"native_exploration": projection(owner, [100, 120])},
                "candidate": {"native_exploration": projection(owner, interval)},
            },
            {"family": "family", "capsule": "unselected", "baseline": None, "candidate": None},
        ],
    }
    result = FEEDBACK.join(doc, owner=owner)
    assert result["members"][0]["recommendation"] == expected
    assert result["members"][1]["status"] == "UNKNOWN"
    assert result["measured_cycles"] is None and result["attainment"] is None
    assert result["complete_performance_objective"] is False and result["promotion_authority"] == "NONE"
    doc["native_exploration"] = result
    FEEDBACK.validate(doc, owner=owner)
    doc["native_exploration"]["members"][0]["recommendation"] = "forged"
    with pytest.raises(StageGateError):
        FEEDBACK.validate(doc, owner=owner)


@pytest.mark.parametrize("fault", ["cycles", "authority", "negative_interval", "different_owner"])
def test_projection_cannot_claim_measurement_or_other_selection(source, fault):
    owner = N.prepare_public_native_exploration(selection(source))
    value = projection(owner, [90, 110])
    if fault == "cycles":
        value["measured_cycles"] = 100
    elif fault == "authority":
        value["promotion_authority"] = "measured"
    elif fault == "negative_interval":
        value["prediction"]["interval"] = [-1, 2]
    else:
        value["selection_sha256"] = "0" * 64
    with pytest.raises(StageGateError):
        OBS.validate_projection(value, owner=owner)


@pytest.mark.parametrize("fault", [None, "saved_selection", "different_record", "incomplete_source"])
def test_native_inference_reopens_checked_source_without_editing_original_receipt(inputs, tmp_path, fault):
    from types import SimpleNamespace

    from merlin_experiments.phase2.ordinary_checked_source import record

    args, *_ = inputs
    checked = args["checked_source_diagnostic"], args["compiler_pass_selection"]
    cases, receipt = inference_fixture()
    row = cases["cases"][0]
    report = row["event_report"]["report"]
    original = {
        "functional_status": "PASS",
        "compiled": {"elf_sha256": report["elf_sha256"]},
        "original_grade": {
            "tiers": {tier: {"status": "pass"} for tier in ("L0", "L1", "L2")},
            "checked_source_status": "complete_source_L0_L1_L2_diagnostic_pass",
        },
        "checked_source_diagnostic": record(*checked),
    }
    if fault == "different_record":
        original["checked_source_diagnostic"] = {"saved": "foreign"}
    elif fault == "incomplete_source":
        original["original_grade"]["checked_source_status"] = "incomplete"
    path = tmp_path / "ordinary_observation.json"
    path.write_text(json.dumps(original))
    before = path.read_bytes()
    value = {
        key: value for key, value in original.items() if key not in {"original_grade", "checked_source_diagnostic"}
    }
    value["evidence_sha256"] = sha256_file(path)
    arguments = dict(
        key="own",
        member=SimpleNamespace(source_sha256=row["case"]["source"]["source_sha256"]),
        family=row["case"]["mechanism_family"],
        dataset=receipt["dataset"],
        manifest_sha256=document_sha256("public"),
        report=report,
        directory=tmp_path,
        value=value,
        checked_source=checked if fault != "saved_selection" else original["checked_source_diagnostic"],
    )
    if fault:
        with pytest.raises((StageGateError, TypeError)):
            N.NATIVE.inference_case(**arguments)
    else:
        actual = N.NATIVE.inference_case(**arguments)
        assert actual["numeric"]["receipt_sha256"] == sha256_file(path)
    assert path.read_bytes() == before
