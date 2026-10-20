"""Neutral original count/process custody, not actual target qualification."""

import json
import time
from dataclasses import replace

import pytest
from merlin_experiments.phase2 import _public_native_feedback as FEEDBACK
from merlin_experiments.phase2 import _public_native_observation as OBS
from merlin_experiments.phase2 import archive_screen_review as REVIEW
from merlin_experiments.phase2 import exploratory_development_priority as PRIORITY
from merlin_experiments.phase2 import public_native_exploration as N
from merlin_experiments.phase2 import public_source_diagnostic as D
from test_public_native_count_projection import inputs as inputs
from test_public_native_count_projection import ordinary_inputs as ordinary_inputs
from test_public_native_count_projection import products
from test_public_native_count_projection import source as source
from test_public_native_exploration import selection

from merlin.llvmlower.reviewed_pass_capabilities import ReviewedPassCapability
from merlin.runtime import _native_files as F


@pytest.mark.parametrize("case", ["raw_in_range", "timing_absent", "timing_false", "unresolved", "cycles_absent"])
def test_original_parent_keeps_exploratory_numbers_independent_of_source_and_event_support(
    source, inputs, tmp_path, monkeypatch, case
):
    args, *_ = inputs
    receipts = {}

    def unsupported_geometry(*_):
        return {"status": "UNKNOWN", "reason": "unseen source geometry"}

    def neutral_count(elf, **kwargs):
        assert kwargs["numerical_pass"] is True
        return receipts[elf]

    # Both callbacks are explicitly neutral fixtures, installed before original
    # live owner issuance. Complete process/input/output pins are still real.
    monkeypatch.setattr(OBS.DOMAIN, "current", unsupported_geometry)
    monkeypatch.setattr(OBS.COUNTS, "observe", neutral_count)
    selected = selection(source)
    binding = replace(selected.correspondence.binding,
                      hardware=args["compiler_pass_selection"].binding.hardware,
                      software=args["compiler_pass_selection"].binding.software)
    selected = replace(
        selected, expected_rtl_facts_sha256=binding.hardware,
        expected_observed_contract_sha256=binding.software,
        correspondence=ReviewedPassCapability("timing_model_correspondence", binding, selected.correspondence.owner),
        native_semantics=ReviewedPassCapability("native_timing_semantics", binding, selected.native_semantics.owner),
    )
    owner = N.prepare_public_native_exploration(
        selected, count_protocol=N.UNQUALIFIED_COUNT_PROTOCOL,
        count_feedback_protocol=N.OBSERVED_COUNT_FEEDBACK_PROTOCOL,
        exploratory_estimate_protocol=N.EXPLORATORY_ESTIMATE_PROTOCOL,
    )
    policy = D.prepare_public_source_diagnostic(**args, native_exploration=owner)
    root = policy.feedback_evaluator.work_root / "exploratory-parent"
    root.mkdir(parents=True)
    members = sorted(policy.search_corpus.capsules, key=lambda row: (row.family, row.capsule))
    member = members[0]
    rows = [dict(family=m.family, capsule=m.capsule, source_sha256=m.source_sha256,
                 baseline=None, candidate=None) for m in members]
    original_bytes = {}
    for arm, cycles in (("baseline", 9999), ("candidate", 9000)):
        workspace = root / f"m000_{arm}"
        workspace.mkdir()

        def transform(report):
            report["estimated_cycles"] = cycles
            report["counter_windows"][0]["estimated_cycles"] = cycles
            if case == "raw_in_range":
                # Use the original neutral archive's declared count vocabulary
                # and a TRAIN observation, without changing its ranges or owner.
                train = next(c for c in source["dataset"]["cases"] if c["split"] == "train")
                row = next(r for r in source["dataset"]["records"] if r["case_id"] == train["case_id"])
                report["feature_counts"] = dict(row["features"], **{"unfitted.fp": 0, "constant.issue": 2})
                primary = list(row["features"])
                instructions = sum(row["features"].values())
                report["feature_observation"].update(primary_cpu_features=primary,
                    scope_instructions=instructions, counted_cpu_instructions=instructions)
                report["counter_windows"][0]["instructions"] = instructions
            elif case == "timing_absent":
                report.pop("timing_complete")
            elif case == "timing_false":
                report["timing_complete"] = False
            elif case == "unresolved":
                report["unresolved"] = ["neutral event has no timing price"]
            else:
                report["timing_complete"] = False
                report.pop("estimated_cycles")
                report["counter_windows"][0].pop("estimated_cycles")

        _, _, _, numerical, receipt, report_path, _ = products(
            source, inputs, workspace, monkeypatch, selected_owner=owner, selected_policy=policy,
            selected_member=member, exploratory=True, report_transform=transform, elf_tag=arm.encode(),
        )
        receipts[workspace / "original.elf"] = receipt
        original_bytes[report_path] = report_path.read_bytes()
        numeric_path = workspace / "ordinary_observation.json"
        original_bytes[numeric_path] = numeric_path.read_bytes()
        value = OBS.observe(owner, policy, member, workspace, numerical, sandbox=None,
                            remaining_s=10, include_unqualified_counts=True)
        assert value["status"] == "UNKNOWN" and value["prediction"] is None
        assert value["reason"] == "Fresh counts observed; timing UNKNOWN: unseen source geometry"
        assert value["measured_cycles"] is None and value["promotion_authority"] == "NONE"
        estimate = value["exploratory_estimate"]
        if case == "cycles_absent":
            assert estimate["status"] == "unavailable" and estimate["estimated_cycles"] is None
        else:
            assert estimate["status"] == "estimated" and estimate["estimated_cycles"] == cycles
        assert estimate["event_accounting_complete"] is (case == "raw_in_range")
        if case == "raw_in_range":
            raw = json.loads(report_path.read_bytes())
            assert REVIEW.domain_reason(raw, json.loads(owner.reviewed_json)["raw_activity_ranges"]) is None
            assert estimate["domain_deviations"] == []
        assert value["observed_counts"]["provenance"]["numerical_receipt_sha256"] == F.digest(numeric_path)
        rows[0][arm] = dict(numerical, native_exploration=value)

    document = dict(selected_members=[[member.family, member.capsule]], cells=rows)
    policy.consume_feedback_child_result(document, work_root=root,
        selected_members=((member.family, member.capsule),), deadline=time.monotonic() + 30)
    document["native_exploration"] = FEEDBACK.join(document, owner=owner)
    FEEDBACK.validate(document, owner=owner)
    assert document["native_exploration"]["members"][0]["status"] == "UNKNOWN"
    advice = PRIORITY.decide(document, required_members=((member.family, member.capsule),))
    if case == "raw_in_range":
        assert advice["next_exploration_parent"] == "candidate"  # Explicitly exploratory only.
    else:
        assert advice["next_exploration_parent"] is None
        assert advice["latency_recommendation_coverage_complete"] is False
        if case != "cycles_absent":
            assert advice["members"][0]["candidate_to_baseline_ratio"] == 9000 / 9999
            assert "event accounting" in advice["members"][0]["reason"]
    assert advice["qualified_for_promotion"] is False and advice["plateau_evidence"] is False
    assert all(path.read_bytes() == before for path, before in original_bytes.items())
