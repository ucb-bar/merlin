"""Neutral public count transport; these fixtures qualify no target or timing domain."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import _public_native_count_projection as COUNTS
from merlin_experiments.phase2 import _public_native_exploratory_estimate as ESTIMATE
from merlin_experiments.phase2 import _public_native_feedback as FEEDBACK
from merlin_experiments.phase2 import _public_native_observation as OBS
from merlin_experiments.phase2 import ordinary_checked_source as CHECKED
from merlin_experiments.phase2 import public_native_exploration as N
from merlin_experiments.phase2 import public_source_diagnostic as D
from merlin_experiments.phase2.contracts import StageGateError, document_sha256, write_json
from test_ordinary_native_archive_screen import ordinary_inputs as ordinary_inputs
from test_ordinary_native_archive_screen import source as source
from test_public_native_exploration import selection
from test_public_source_diagnostic import inputs as diagnostic_inputs

from merlin.llvmlower.reviewed_pass_capabilities import ReviewedPassCapability
from merlin.runtime import _native_files as F


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root = tmp_path / "checked-source-inputs"
    root.mkdir()
    return diagnostic_inputs.__wrapped__(root, monkeypatch)

def products(source, inputs, tmp_path, monkeypatch, *, selected_member=None, selected_owner=None,
             selected_policy=None, exploratory=False, report_transform=None, elf_tag=b""):
    args, *_ = inputs
    checked = args["checked_source_diagnostic"], args["compiler_pass_selection"]
    policy = selected_policy or SimpleNamespace(
        _checked_source_diagnostic=checked[0], _compiler_pass_selection=checked[1]
    )
    member = selected_member or SimpleNamespace(source_sha256="a" * 64)
    elf = tmp_path / "original.elf"
    elf.write_bytes(b"neutral executable bytes; never executed" + elf_tag)
    runtime = source["runtime"]
    binding = {
        "schema": "merlin.spike_scope_binding.v1", "elf_sha256": F.digest(elf),
        "input_sha256": document_sha256({"kind": "original_frozen_public_capsule", "sha256": member.source_sha256}),
        "scope_sha256": json.loads(runtime.model_json)["domain"]["scope_sha256"],
        "begin_pc": 100, "end_pc": 200, "occurrence": 0, "evidence_sha256": "b" * 64,
    }
    report = {
        "schema": "merlin.annotated_spike_report.v2", "instructions": 1000,
        "execution_completed": True, "functional_status": "PASS", "raw_functional_exit_code": 0,
        "elf_sha256": binding["elf_sha256"], "input_sha256": binding["input_sha256"],
        "model_sha256": F.digest(runtime.model_path),
        "device_sha256": json.loads(runtime.model_json)["domain"]["device_sha256"],
        "feature_engine_sha256": json.loads(runtime.model_json)["domain"]["feature_engine_sha256"],
        "scope_sha256": binding["scope_sha256"],
        "loaded_library_sha256": json.loads(runtime.model_json)["libraries"],
        "scope_binding": binding,
        "counter_windows": [{"begin_pc": 100, "end_pc": 200, "instructions": 8, "estimated_cycles": 9999}],
        "estimated_cycles": 9999,
        "feature_observation": {
            "schema": "merlin.spike_feature_observation.v1", "complete": True,
            "scope_instructions": 8, "counted_cpu_instructions": 8,
            "scope_kind": "counter_pc_pair", "window_index": 0,
            "primary_cpu_features": ["cpu.integer", "cpu.custom", "cpu.fence"],
            "unresolved": [], "qualification": "not_established",
        },
        "feature_counts": {
            "cpu.integer": 4, "cpu.custom": 3, "cpu.fence": 1,
            "target.commands": 3, "target.class.load": 1, "target.class.config": 1,
            "target.class.store": 1, "target.fences": 1,
            "device.requested_read_bytes": 128, "device.requested_write_bytes": 64,
            "target.local_read_bytes": 128, "modeled.dram_request_bytes": 999,
        },
    }
    if exploratory:
        report.update(domain=dict(json.loads(runtime.model_json)["domain"],
                                  target_model_sha256=F.digest(runtime.model_path)),
                      timing_complete=True, unresolved=[])
    if report_transform is not None:
        report_transform(report)
    count_root = tmp_path / "public_native_counts"
    count_root.mkdir()
    report_path = count_root / "report.json"
    write_json(report_path, report)
    receipt = {
        "schema": "merlin.phase2.archive_count_observation.v1", "status": "observed",
        "elf_sha256": binding["elf_sha256"], "input_sha256": binding["input_sha256"],
        "scope_binding": binding, "runtime_sha256": runtime.sha256,
        "functional_identical": True, "complete_counts": True, "native_event_timing_qualified": False,
        "report": {"path": str(report_path), "sha256": F.digest(report_path)},
        "feature_counts": report["feature_counts"],
        "processes": [],
    }
    scope_path = count_root / "scope.json"
    write_json(scope_path, binding)
    for arm in ("functional", "annotated"):
        # Actual neutral host subprocess receipts exercise custody, not Spike or numerical qualification.
        argv = [sys.executable, "-c", "print('neutral complete output')"]
        with COUNTS.I.observe(
            count_root / arm, stage="archive_screen_count", argv=argv, inputs=(elf, scope_path),
            outputs=(report_path,) if arm == "annotated" else (),
            dependencies=tuple(Path(p) for p, _ in runtime.files),
        ) as invocation:
            actual = subprocess.run(argv, capture_output=True, check=True)
            invocation.complete(actual)
            document = COUNTS.I.verify(invocation.path)
            receipt["processes"].append({
                "path": str(invocation.path), "sha256": F.digest(invocation.path),
                "stdout_sha256": document["stdout"]["sha256"],
            })
    original = {
        "functional_status": "PASS", "compiled": {"elf_sha256": binding["elf_sha256"]},
        "original_grade": {
            "tiers": {tier: {"status": "pass"} for tier in ("L0", "L1", "L2")},
            "checked_source_status": "complete_source_L0_L1_L2_diagnostic_pass",
        },
        "checked_source_diagnostic": CHECKED.record(*checked),
    }
    if elf_tag:
        original["compiled"]["status"] = "observed"
    write_json(tmp_path / "ordinary_observation.json", original)
    numerical = {key: value for key, value in original.items()
                 if key not in {"original_grade", "checked_source_diagnostic"}}
    numerical["evidence_sha256"] = F.digest(tmp_path / "ordinary_observation.json")
    def observe(_elf, **kwargs):
        assert _elf == elf and kwargs["numerical_pass"] is True
        return receipt

    if selected_owner is None:
        monkeypatch.setattr(OBS.DOMAIN, "current", lambda *_: {"status": "UNKNOWN", "reason": "new source geometry"})
        monkeypatch.setattr(OBS.COUNTS, "observe", observe)
    owner = selected_owner or N.prepare_public_native_exploration(
        selection(source), count_protocol=N.UNQUALIFIED_COUNT_PROTOCOL,
        count_feedback_protocol=N.OBSERVED_COUNT_FEEDBACK_PROTOCOL,
        exploratory_estimate_protocol=ESTIMATE.PROTOCOL if exploratory else None,
    )
    return owner, policy, member, numerical, receipt, report_path, original


def observe(owner, policy, member, directory, numerical):
    return OBS.observe(owner, policy, member, directory, numerical, sandbox=None,
                       remaining_s=10, include_unqualified_counts=True)


def test_unknown_timing_exposes_original_observed_counts_and_requested_bytes(source, inputs, tmp_path, monkeypatch):
    owner, policy, member, numerical, receipt, report, _ = products(source, inputs, tmp_path, monkeypatch)
    before = (tmp_path / "ordinary_observation.json").read_bytes(), report.read_bytes()
    result = observe(owner, policy, member, tmp_path, numerical)
    assert result["schema"] == OBS.COUNT_SCHEMA and result["status"] == "UNKNOWN"
    assert result["prediction"] is None and result["measured_cycles"] is None
    assert owner.record()["schema"] == "merlin.public_native_exploration.v6"
    assert owner.record()["count_feedback_protocol"] == COUNTS.PROTOCOL
    counts = result["observed_counts"]
    assert counts["metrics"]["scope_instructions"] == {"value": 8, "unit": "instructions"}
    assert counts["metrics"]["target.commands"] == {"value": 3, "unit": "commands"}
    assert counts["metrics"]["device.requested_read_bytes"] == {"value": 128, "unit": "encoded requested bytes"}
    assert all(not key.startswith("modeled.") for key in counts["metrics"])
    assert "estimated_cycles" not in counts["metrics"]
    assert set(counts["unknown"].values()) == {None} and result["promotion_authority"] == "NONE"
    assert counts["provenance"]["receipt_sha256"] == F.digest(tmp_path / "public_native_counts_receipt.json")
    assert counts["provenance"]["report_sha256"] == receipt["report"]["sha256"]
    assert before == ((tmp_path / "ordinary_observation.json").read_bytes(), report.read_bytes())
    assert str(tmp_path) not in json.dumps(counts)  # no private paths, consoles, operands or outputs
    OBS.validate_projection(result, owner=owner, numerical=numerical, capsule_sha256=member.source_sha256)
    document = {
        "selected_members": [["family", "member"]],
        "cells": [{"family": "family", "capsule": "member", "source_sha256": member.source_sha256,
                   "baseline": {**numerical, "native_exploration": result},
                   "candidate": {**numerical, "native_exploration": result}}],
    }
    joined = FEEDBACK.join(document, owner=owner)
    assert joined["members"][0]["recommendation"] is None
    assert joined["members"][0]["status"] == "UNKNOWN" and joined["attainment"] is None


@pytest.mark.parametrize("fault", ["report", "receipt_elf", "input", "runtime", "window", "feature_counts",
                                   "missing_processes", "numeric", "source_selection", "tiers"])
def test_projection_reopens_exact_original_source_numerics_count_report_and_joins(
    source, inputs, tmp_path, monkeypatch, fault
):
    owner, policy, member, numerical, receipt, report, original = products(source, inputs, tmp_path, monkeypatch)
    if fault == "report":
        report.write_text("changed raw report")
    elif fault == "receipt_elf":
        receipt["elf_sha256"] = "f" * 64
    elif fault == "input":
        receipt["input_sha256"] = "f" * 64
    elif fault == "runtime":
        receipt["runtime_sha256"] = "f" * 64
    elif fault == "window":
        receipt["scope_binding"] = {**receipt["scope_binding"], "occurrence": 1}
    elif fault == "feature_counts":
        receipt["feature_counts"] = {**receipt["feature_counts"], "target.commands": 0}
    elif fault == "missing_processes":
        receipt.pop("processes")
    else:
        if fault == "numeric":
            original["functional_status"] = "UNKNOWN_OR_FAILURE"
        elif fault == "source_selection":
            original["checked_source_diagnostic"] = {"saved": "not an original live selector"}
        else:
            original["original_grade"]["tiers"]["L1"]["status"] = "unknown"
        write_json(tmp_path / "ordinary_observation.json", original)
    with pytest.raises((StageGateError, ValueError)):
        observe(owner, policy, member, tmp_path, numerical)


@pytest.mark.parametrize("fault", ["modeled", "unit", "negative", "boolean", "cycles", "elf", "numeric", "scope"])
def test_public_transport_cannot_forge_units_physical_timing_or_original_arm(
    source, inputs, tmp_path, monkeypatch, fault
):
    owner, policy, member, numerical, *_ = products(source, inputs, tmp_path, monkeypatch)
    result = observe(owner, policy, member, tmp_path, numerical)
    counts = result["observed_counts"]
    if fault == "modeled":
        counts["metrics"]["modeled.dram_request_bytes"] = {"value": 999, "unit": "encoded requested bytes"}
    elif fault == "unit":
        counts["metrics"]["target.commands"]["unit"] = "cycles"
    elif fault in {"negative", "boolean"}:
        counts["metrics"]["target.commands"]["value"] = -1 if fault == "negative" else True
    elif fault == "cycles":
        counts["unknown"]["cycles"] = 1
    elif fault == "scope":
        counts["provenance"]["scope_binding"]["begin_pc"] = 300
    else:
        counts["provenance"]["elf_sha256" if fault == "elf" else "numerical_receipt_sha256"] = "f" * 64
    with pytest.raises(StageGateError):
        OBS.validate_projection(result, owner=owner, numerical=numerical, capsule_sha256=member.source_sha256)


def test_failed_functional_checks_never_count_or_expose_metrics(source, tmp_path, monkeypatch):
    monkeypatch.setattr(OBS.COUNTS, "observe", lambda *_a, **_kw: pytest.fail("failed numerics must not count"))
    owner = N.prepare_public_native_exploration(
        selection(source), count_protocol=N.UNQUALIFIED_COUNT_PROTOCOL,
        count_feedback_protocol=N.OBSERVED_COUNT_FEEDBACK_PROTOCOL,
    )
    result = OBS.observe(owner, None, None, tmp_path, {"functional_status": "UNKNOWN_OR_FAILURE"},
                         sandbox=None, remaining_s=10, include_unqualified_counts=True)
    assert result["schema"] == OBS.COUNT_SCHEMA and result["observed_counts"] is None
    assert result["receipt_sha256"] is None


def test_default_wire_projection_remains_unchanged(source):
    owner = N.prepare_public_native_exploration(selection(source))
    result = OBS.unknown(owner, "unchanged original reason")
    assert result == {
        "schema": OBS.SCHEMA, "status": "UNKNOWN", "reason": "unchanged original reason",
        "selection_sha256": document_sha256(owner.record()), "prediction": None, "receipt_sha256": None,
        "measured_cycles": None, "promotion_authority": "NONE",
    }
    assert "count_feedback_protocol" not in owner.record()


@pytest.mark.parametrize("protocol", [True, {}, "", "unreviewed"])
def test_saved_or_unknown_count_feedback_protocol_refuses(source, protocol):
    with pytest.raises(StageGateError, match="explicit original"):
        N.prepare_public_native_exploration(selection(source), count_protocol=N.UNQUALIFIED_COUNT_PROTOCOL,
                                           count_feedback_protocol=protocol)


def test_count_feedback_requires_explicit_count_owner_and_detects_mutation(source):
    with pytest.raises(StageGateError, match="explicit original"):
        N.prepare_public_native_exploration(selection(source), count_feedback_protocol=COUNTS.PROTOCOL)
    owner = N.prepare_public_native_exploration(
        selection(source), count_protocol=N.UNQUALIFIED_COUNT_PROTOCOL, count_feedback_protocol=COUNTS.PROTOCOL
    )
    object.__setattr__(owner, "count_feedback_protocol", None)
    with pytest.raises(StageGateError, match="issued owners"):
        owner.verify()


@pytest.mark.parametrize("exploratory", [False, True])
@pytest.mark.parametrize("fault", [None, "positive_count", "missing_receipt", "report", "escape", "alias", "deadline",
                                   "drop_projection"])
def test_original_parent_policy_reopens_both_trusted_arm_workspaces_before_completion(
    source, inputs, tmp_path, monkeypatch, fault, exploratory
):
    import os
    import time

    args, *_ = inputs
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
        selected, count_protocol=N.UNQUALIFIED_COUNT_PROTOCOL, count_feedback_protocol=COUNTS.PROTOCOL,
        exploratory_estimate_protocol=ESTIMATE.PROTOCOL if exploratory else None,
    )
    policy = D.prepare_public_source_diagnostic(**args, native_exploration=owner)
    root = policy.feedback_evaluator.work_root / "job"
    root.mkdir(parents=True)
    members = sorted(policy.search_corpus.capsules, key=lambda row: (row.family, row.capsule))
    member = members[0]
    rows = [{"family": m.family, "capsule": m.capsule, "source_sha256": m.source_sha256,
             "baseline": None, "candidate": None} for m in members]
    workspaces = []
    for arm in ("baseline", "candidate"):
        workspace = root / f"m000_{arm}"
        workspace.mkdir()
        workspaces.append(workspace)
        _, _, _, value, receipt, *_ = products(
            source, inputs, workspace, monkeypatch, selected_owner=owner, selected_policy=policy,
            selected_member=member, exploratory=exploratory,
        )
        path = workspace / "public_native_counts_receipt.json"
        write_json(path, receipt)
        projection = OBS.unknown(owner, "unchanged unsupported source")
        projection.update(receipt_sha256=F.digest(path),
                          observed_counts=COUNTS.project(owner, policy, member, workspace, value, path))
        if exploratory:
            projection["exploratory_estimate"] = ESTIMATE.project(
                owner, projection["observed_counts"], workspace / "public_native_counts/report.json")
        rows[0][arm] = {**value, "native_exploration": projection}
    document = {"selected_members": [[member.family, member.capsule]], "cells": rows}
    if fault == "positive_count":
        rows[0]["candidate"]["native_exploration"]["observed_counts"]["metrics"]["target.commands"]["value"] = 2
    elif fault == "estimate":
        rows[0]["candidate"]["native_exploration"]["exploratory_estimate"]["estimated_cycles"] = 1
    elif fault == "missing_receipt":
        (workspaces[1] / "public_native_counts_receipt.json").unlink()
    elif fault == "report":
        path = workspaces[1] / "public_native_counts/report.json"
        raw = json.loads(path.read_bytes())
        raw["feature_counts"]["target.commands"] = 2
        write_json(path, raw)
    elif fault == "escape":
        (workspaces[1] / "public_native_counts/report.json").rename(tmp_path / "outside-report.json")
        (workspaces[1] / "public_native_counts/report.json").symlink_to(tmp_path / "outside-report.json")
    elif fault == "alias":
        os.link(workspaces[1] / "public_native_counts_receipt.json", tmp_path / "alias-receipt.json")
    elif fault == "drop_projection":
        rows[0]["candidate"]["native_exploration"].update(observed_counts=None, receipt_sha256=None)
    deadline = time.monotonic() + (30 if fault != "deadline" else -1)
    if fault is None:
        policy.consume_feedback_child_result(document, work_root=root,
                                             selected_members=((member.family, member.capsule),), deadline=deadline)
    else:
        with pytest.raises((StageGateError, ValueError, OSError)):
            policy.consume_feedback_child_result(document, work_root=root,
                                                 selected_members=((member.family, member.capsule),), deadline=deadline)


def test_exploratory_cycles_survive_domain_abstention_without_qualification(source, inputs, tmp_path, monkeypatch):
    owner, policy, member, numerical, *_ = products(source, inputs, tmp_path, monkeypatch, exploratory=True)
    value = observe(owner, policy, member, tmp_path, numerical)
    assert owner.record()["schema"] == "merlin.public_native_exploration.v7"
    assert value["schema"] == OBS.ESTIMATE_SCHEMA
    assert value["status"] == "UNKNOWN" and value["prediction"] is None
    estimate = value["exploratory_estimate"]
    assert estimate["estimated_cycles"] == 9999 and estimate["event_accounting_complete"] is True
    assert estimate["qualified_for_promotion"] is False and estimate["uncertainty_interval_cycles"] is None
    assert estimate["model_sha256"] == F.digest(owner.selection.runtime.model_path)
    assert estimate["domain_deviations"] and estimate["calibration_residuals"]["held_domain_metrics"]
    OBS.validate_projection(value, owner=owner, numerical=numerical, capsule_sha256=member.source_sha256)
    joined = FEEDBACK.join({"selected_members": [["f", "m"]], "cells": [
        {"family": "f", "capsule": "m", "source_sha256": member.source_sha256,
         "baseline": {**numerical, "native_exploration": value},
         "candidate": {**numerical, "native_exploration": value}}]}, owner=owner)
    assert joined["schema"] == "merlin.public_native_exploration_feedback.v2"
    assert joined["members"][0]["status"] == "UNKNOWN"
    assert joined["members"][0]["exploratory_estimates"]["candidate_to_baseline_ratio"] == 1
    estimate["estimated_cycles"] = 0
    with pytest.raises(StageGateError):
        OBS.validate_projection(value, owner=owner)


@pytest.mark.parametrize("protocol", [True, {}, "", "unreviewed", ESTIMATE.PROTOCOL])
def test_estimate_requires_explicit_count_projection_owner(source, protocol):
    with pytest.raises(StageGateError, match="explicit verified"):
        N.prepare_public_native_exploration(selection(source), exploratory_estimate_protocol=protocol)


def test_estimate_owner_mutation_refuses(source):
    owner = N.prepare_public_native_exploration(
        selection(source), count_protocol=N.UNQUALIFIED_COUNT_PROTOCOL,
        count_feedback_protocol=COUNTS.PROTOCOL, exploratory_estimate_protocol=ESTIMATE.PROTOCOL)
    object.__setattr__(owner, "exploratory_estimate_protocol", None)
    with pytest.raises(StageGateError, match="issued owners"):
        owner.verify()


def test_parent_rejects_plausible_positive_cycle_mutation(source, inputs, tmp_path, monkeypatch):
    test_original_parent_policy_reopens_both_trusted_arm_workspaces_before_completion(
        source, inputs, tmp_path, monkeypatch, "estimate", True)
