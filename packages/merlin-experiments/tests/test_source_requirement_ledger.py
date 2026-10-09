"""Actual source references never turn pending compiler criteria into passes."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0 import component_numerics
from merlin_experiments.phase0 import source_requirement_ledger as L
from merlin_experiments.phase0.component_generation import digest
from merlin_experiments.phase1.source_inputs import fingerprint

from merlin.targetgen import golden_store


def _fixtures():
    path = Path(__file__).with_name("test_component_source_binding.py")
    spec = importlib.util.spec_from_file_location("source_ledger_native_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixtures = _fixtures()
automatic = fixtures.automatic
independent = fixtures.independent
selected = fixtures.selected


@pytest.fixture
def inputs(automatic, tmp_path):
    options = fixtures._source_options(automatic, tmp_path, version=fixtures.A.LOGICAL_POLICY_SCHEMA)
    coverage = fixtures.fixtures.run(options)
    return {
        "root": options["output_root"],
        "coverage": coverage,
        "hardware": options["hardware_intake"],
        "software": options["software_intake"],
        "purpose": "source_preparation",
    }


@pytest.mark.parametrize("automatic", [{"logical": True}], indirect=True)
def test_original_denominator_full_references_and_logical_cases_remain_distinct_from_verdicts(inputs):
    ledger = L.prepare_requirement_ledger(**inputs)
    actual = ledger.record()
    coverage = inputs["coverage"]
    assert actual["original_required_ids"] == [row["id"] for row in coverage["obligations"]]
    assert actual["original_mandatory_ids"] == [row["id"] for row in coverage["obligations"] if row["mandatory"]]
    assert actual["checked_source_witnesses"]
    for row in actual["requirements"]:
        assert row["source_producer_phase"] == 0
        assert row["candidate_verdict_phase"] == 1 and row["candidate_verdict"] == "not_evaluated"
        if row["kind"] == "physical_interaction":
            assert row["logical_testcase_members"]
            assert row["source_input_state"] == "unavailable"
            assert row["missing_source_producers"]
    assert actual["release_authority"] == "not_issued"
    assert actual["status"] == "diagnostic_incomplete"
    assert L.verify_requirement_ledger(ledger, **inputs) == actual


@pytest.mark.parametrize("automatic", [{"logical": True}], indirect=True)
def test_changed_last_output_refuses_even_after_resigning_member_and_report(inputs):
    coverage = inputs["coverage"]
    row = next(row for row in coverage["obligations"] if row["members"] and row["state"] == "source_generated")
    member = row["members"][-1]
    directory = inputs["root"] / member["member"]
    golden = golden_store.load_golden(directory)
    name = sorted(golden["outputs"])[-1]
    golden["outputs"][name][-1][-1] += 1
    golden_store.write_golden(directory, golden)
    member["sha256"] = fingerprint(directory)
    coverage["sha256"] = digest({key: value for key, value in coverage.items() if key != "sha256"})
    with pytest.raises(ValueError, match="complete original independent reference"):
        L.prepare_requirement_ledger(**inputs)


@pytest.mark.parametrize("automatic", [{"logical": True}], indirect=True)
def test_dropped_original_missing_id_refuses_resigned_coverage(inputs):
    coverage = inputs["coverage"]
    missing = next(row for row in coverage["obligations"] if row["state"] == "unavailable")
    coverage["obligations"].remove(missing)
    coverage["sha256"] = digest({key: value for key, value in coverage.items() if key != "sha256"})
    with pytest.raises(ValueError, match="original required obligation"):
        L.prepare_requirement_ledger(**inputs)


@pytest.mark.parametrize("automatic", [{"logical": True}], indirect=True)
def test_resigned_deferred_stamp_cannot_change_actual_reference_or_pending_verdict(inputs):
    ledger = L.prepare_requirement_ledger(**inputs)
    data = ledger.record()
    physical = next(row for row in data["requirements"] if row["kind"] == "physical_interaction")
    physical.update(source_input_state="checked", missing_source_producers=[], candidate_verdict="accepted")
    data["sha256"] = digest({key: value for key, value in data.items() if key != "sha256"})
    altered = L.SourceRequirementLedger(json.dumps(data))
    with pytest.raises(ValueError, match="full denominator changed"):
        L.verify_requirement_ledger(altered, **inputs)


@pytest.mark.parametrize("automatic", [{"logical": True}], indirect=True)
def test_performance_purpose_keeps_empty_selection_and_absent_contracts_blocking(inputs):
    inputs["purpose"] = "performance_campaign"
    actual = L.prepare_requirement_ledger(**inputs).record()
    performance = actual["performance_preparation"]
    assert performance["required_for_purpose"] is True and performance["objective_count"] == 0
    assert performance["checked_source_members_by_cohort"]["development"] == 0
    assert "explicit nonempty independent performance objectives" in performance["missing_producers"]
    assert any("guard producer" in item for item in performance["missing_producers"])
    assert any("measurement/cold-warm/timer" in item for item in performance["missing_producers"])
    assert actual["status"] == "diagnostic_incomplete" and actual["release_authority"] == "not_issued"


@pytest.mark.parametrize("automatic", [{"logical": True}], indirect=True)
def test_denied_original_budget_cannot_reach_reference_allocation(automatic, tmp_path, monkeypatch):
    options = fixtures._source_options(automatic, tmp_path, version=fixtures.A.LOGICAL_POLICY_SCHEMA)
    policy = yaml.safe_load(options["component_coverage"].read_bytes())
    policy["execution_budget"]["max_reference_work"] = 2
    fixtures.fixtures.write(options["component_coverage"], policy)
    coverage = fixtures.fixtures.run(options)
    assert any(row["state"] != "admitted" for row in coverage["execution_admission"]["decisions"])
    monkeypatch.setattr(component_numerics, "evaluate", lambda *_: pytest.fail("denied roster allocated references"))
    actual = L.prepare_requirement_ledger(
        root=options["output_root"],
        coverage=coverage,
        hardware=options["hardware_intake"],
        software=options["software_intake"],
        purpose="source_preparation",
    ).record()
    assert actual["checked_source_witnesses"] == [] and actual["mandatory_source_blockers"]


@pytest.mark.parametrize("automatic", [{"logical": True}], indirect=True)
def test_selected_recipe_drift_cannot_be_a_new_performance_declaration(inputs):
    path = Path(inputs["coverage"]["generation_identity"]["recipe"]["path"])
    path.write_bytes(path.read_bytes() + b"\n# new unchecked purpose\n")
    with pytest.raises(ValueError, match="source changed|source bytes changed"):
        L.prepare_requirement_ledger(**inputs)


@pytest.mark.parametrize("automatic", [{"logical": True}], indirect=True)
def test_reference_alias_is_rejected_before_opening_excluded_content(inputs, tmp_path):
    row = next(row for row in inputs["coverage"]["obligations"] if row["members"])
    directory = inputs["root"] / row["members"][0]["member"]
    excluded = tmp_path / "excluded-reference"
    excluded.write_bytes(b"not an output document")
    document = directory / golden_store.DOCUMENT
    document.unlink()
    document.symlink_to(excluded)
    with pytest.raises(ValueError, match="indirect product|ordinary source/product"):
        L.prepare_requirement_ledger(**inputs)
    assert excluded.read_bytes() == b"not an output document"


def test_plain_saved_ledger_is_not_actual_reference_evidence():
    with pytest.raises(ValueError, match="exact diagnostic data type"):
        L.verify_requirement_ledger(
            {}, root=None, coverage=None, hardware=None, software=None, purpose="source_preparation"
        )
