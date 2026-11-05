"""Campaign bookkeeping must not turn parity, skips or saved labels into proof credit."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from build_tools.scripts import qualify_pass_cutoff as campaign


def bound_file(directory, name, text):
    path = directory / name
    path.write_text(text)
    return campaign.file_ref(path)


def arms(tmp_path, *, after="llvm"):
    return (
        {"status": "lowered", "output": bound_file(tmp_path, "baseline.ll", "llvm")},
        {"status": "lowered", "output": bound_file(tmp_path, "checked.ll", after)},
    )


def accepted_job(tmp_path, *, after="after"):
    bindings = [
        bound_file(tmp_path, name, text)
        for name, text in (
            ("request.json", "{}"),
            ("verdict.json", "{}"),
            ("before.mlir", "before"),
            ("after.mlir", after),
        )
    ]
    return {"verdict": "accepted", "changed": True, "bindings": bindings}


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    source = tmp_path / "compiler.py"
    source.write_text("original source")
    monkeypatch.setattr(campaign, "source_bindings", lambda: [campaign.file_ref(source)])
    monkeypatch.setattr(campaign, "runtime_bindings", lambda: {"test_runtime": True})
    input_path = tmp_path / "original.mlir"
    input_path.write_text("module {}")
    external = tmp_path / "inputs.json"
    campaign.write(
        external,
        {
            "schema": "merlin.pass-cutoff.inputs.v1",
            "inputs": [{"id": "declared", "path": input_path.name, "sha256": campaign.digest(input_path)}],
        },
    )
    destination = tmp_path / "inventory"
    campaign.inventory(destination, external=external)
    return destination / "inventory.json"


def retained_report(tmp_path, inventory):
    manifest = campaign.validate_manifest(inventory)
    result = campaign.summarize_case(*arms(tmp_path), [])
    result.update(id=manifest["cases"][0]["id"], input=manifest["cases"][0]["input"], selectors=manifest["selectors"])
    record_path = tmp_path / "case.json"
    campaign.write(record_path, result)
    report = {
        "schema": "merlin.pass-cutoff.campaign.v1",
        "inventory": campaign.file_ref(inventory),
        "records": [campaign.file_ref(record_path)],
        "results": [result],
        "summary": campaign.summary(manifest, [result]),
        "runtime": campaign.runtime_bindings(),
    }
    report_path = tmp_path / "campaign.json"
    campaign.write(report_path, report)
    return report_path, report


def test_equal_compiler_failures_are_not_success():
    result = campaign.summarize_case({"status": "lowering_failed"}, {"status": "lowering_failed"}, [])
    result["id"] = "declared"
    totals = campaign.summary({"cases": [{"id": "declared"}]}, [result])
    assert result["status"] == "baseline_lowering_failed"
    assert totals["complete"] and not totals["all_lowered"]


@pytest.mark.parametrize("status", ["gate_unsupported", "gate_not_established", "unavailable"])
def test_runtime_refusal_or_missing_tool_stays_explicit(tmp_path, status):
    before, _ = arms(tmp_path)
    assert campaign.summarize_case(before, {"status": status}, [])["status"] == status


def test_identity_has_zero_positive_proof_credit(tmp_path):
    result = campaign.summarize_case(*arms(tmp_path), [])
    assert result["status"] == "lowered_no_changed_selected_boundary"
    assert result["positive_checked_boundaries"] == 0
    assert result["completed_jobs"] == result["accepted_jobs"] == result["changed_jobs"] == 0


def test_changed_accepted_boundary_has_separate_job_counts(tmp_path):
    result = campaign.summarize_case(*arms(tmp_path), [accepted_job(tmp_path)])
    assert result["status"] == "lowered_checked"
    assert (
        result["positive_checked_boundaries"]
        == result["completed_jobs"]
        == result["accepted_jobs"]
        == result["changed_jobs"]
        == 1
    )


@pytest.mark.parametrize("artifact", ["checked.ll", "verdict.json", "before.mlir"])
def test_missing_artifact_cannot_be_accepted(tmp_path, artifact):
    before, after = arms(tmp_path)
    job = accepted_job(tmp_path)
    (tmp_path / artifact).unlink()
    assert campaign.summarize_case(before, after, [job])["status"] == "unavailable"


def test_saved_accepted_label_without_artifacts_is_not_evidence(tmp_path):
    result = campaign.summarize_case(*arms(tmp_path), [{"verdict": "accepted", "changed": True, "bindings": []}])
    assert result["status"] == "unavailable"


def test_modified_nested_receipt_invalidates_retained_evidence(tmp_path):
    before, after = arms(tmp_path)
    job = accepted_job(tmp_path)
    job["bindings"].append(bound_file(tmp_path, "receipt.json", '{"checked": true}'))
    (tmp_path / "receipt.json").write_text('{"checked": false}')
    assert campaign.summarize_case(before, after, [job])["status"] == "unavailable"


def test_unchanged_job_cannot_claim_positive_credit(tmp_path):
    assert campaign.summarize_case(*arms(tmp_path), [accepted_job(tmp_path, after="before")])["status"] == "unavailable"


def test_different_actual_compiler_output_is_not_success(tmp_path):
    result = campaign.summarize_case(*arms(tmp_path, after="corrupted llvm"), [])
    assert result["status"] == "llvm_mismatch"
    assert result["positive_checked_boundaries"] == 0


def test_partial_or_empty_denominator_cannot_claim_success():
    assert not campaign.summary({"cases": []}, [])["all_lowered"]
    assert not campaign.summary({"cases": [{"id": "missing"}]}, [])["complete"]


def test_source_cut_change_requires_new_inventory(inventory, tmp_path):
    (tmp_path / "compiler.py").write_text("modified source")
    with pytest.raises(ValueError, match="source cut changed"):
        campaign.validate_manifest(inventory)


def test_inventory_mutation_is_detected_by_retained_binding(tmp_path, inventory):
    path, _ = retained_report(tmp_path, inventory)
    with inventory.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="bound file changed"):
        campaign.audit(path)


def test_audit_checks_integrity_and_grants_no_fresh_proof(tmp_path, inventory):
    path, _ = retained_report(tmp_path, inventory)
    result = campaign.audit(path)
    assert result["all_lowered"] and result["retained_bytes_consistent"]
    assert not result["fresh_proof"] and not result["saved_record_authority"]


def test_audit_detects_output_tampering(tmp_path, inventory):
    path, _ = retained_report(tmp_path, inventory)
    (tmp_path / "checked.ll").write_text("poisoned after the run")
    with pytest.raises(ValueError, match="actual artifact bindings"):
        campaign.audit(path)


def test_audit_recomputes_case_status_instead_of_trusting_saved_success(tmp_path, inventory):
    path, report = retained_report(tmp_path, inventory)
    record = report["results"][0]
    record["status"] = "lowered_checked"
    record["positive_checked_boundaries"] = 1
    record_path = Path(report["records"][0]["path"])
    campaign.write(record_path, record)
    report["records"][0] = campaign.file_ref(record_path)
    report["summary"] = campaign.summary(campaign.validate_manifest(inventory), [record])
    campaign.write(path, report)
    with pytest.raises(ValueError, match="actual artifact bindings"):
        campaign.audit(path)


def test_audit_rejects_missing_case_record(tmp_path, inventory):
    path, report = retained_report(tmp_path, inventory)
    Path(report["records"][0]["path"]).unlink()
    with pytest.raises(FileNotFoundError):
        campaign.audit(path)


def test_audit_rejects_success_label_without_compiler_records(tmp_path, inventory):
    path, report = retained_report(tmp_path, inventory)
    record = report["results"][0]
    del record["baseline"]
    record_path = Path(report["records"][0]["path"])
    campaign.write(record_path, record)
    report["records"][0] = campaign.file_ref(record_path)
    campaign.write(path, report)
    with pytest.raises(ValueError, match="without compiler records"):
        campaign.audit(path)


def test_missing_external_input_is_preserved_as_roster_failure(tmp_path, inventory):
    external = tmp_path / "inputs.json"
    declaration = json.loads(external.read_text())
    declaration["inputs"].append({"id": "missing", "path": "absent.mlir", "sha256": "0" * 64})
    campaign.write(external, declaration)
    result = campaign.inventory(tmp_path / "incomplete", external=external)
    assert result["status"] == "incomplete"
    assert len(result["cases"]) == len(result["errors"]) == 1
    assert result["errors"][0]["declared_input"]["id"] == "missing"
    with pytest.raises(ValueError, match="complete frozen inventory"):
        campaign.validate_manifest(tmp_path / "incomplete/inventory.json")


def test_compile_reads_actual_gate_job_ids_and_retains_boundaries(tmp_path, monkeypatch):
    from merlin.verify.pass_validation import DEFAULT_CHECKS, CheckerTools

    # Only this infrastructure test substitutes lowering and checker construction.
    # Production campaign execution always calls the ordinary compiler and workers.
    tools = object.__new__(CheckerTools)
    object.__setattr__(tools, "sdk_bin", tmp_path)
    object.__setattr__(tools, "compiler_python", Path("unused-test-tool"))

    def lower(source, destination, *, textual, validation):
        destination.mkdir()
        output = destination / "kernel.ll"
        output.write_text("llvm")
        if validation is not None:
            folder = validation.gate.directory / "0001-test"
            folder.mkdir(parents=True)
            accepted_job(folder)
            (folder / "proof").mkdir()
            (folder / "proof/receipt.json").write_text("test receipt")
            validation.gate.verdicts.append(
                {"job": folder.name, "check": "tensor_refinement", "label": "cse", "verdict": "accepted"}
            )
        return SimpleNamespace(ll_path=output, stats={})

    monkeypatch.setattr(campaign, "_lower", lower)
    case = {"input": bound_file(tmp_path, "input.mlir", "module {}"), "route": "xdsl"}
    result = campaign.compile_case(case, tmp_path / "case", tools, sorted(DEFAULT_CHECKS), job_timeout=1)
    assert result["status"] == "lowered_checked"
    assert result["positive_checked_boundaries"] == 1
    assert any(ref["path"].endswith("proof/receipt.json") for ref in result["jobs"][0]["bindings"])
