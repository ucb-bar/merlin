"""Private logical features consume the complete original source contract."""

import copy
import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0 import component_source_demand as D

from merlin.perf.component_source_demand import SourceDemandLimits


def _fixtures():
    spec = importlib.util.spec_from_file_location(
        "source_demand_fixtures", Path(__file__).with_name("test_component_source_performance.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fixtures = _fixtures()
automatic = fixtures.automatic
independent = fixtures.independent
selected = fixtures.selected


def selection_for(record, inputs, path):
    contract = inputs["root"] / "_evidence/coverage/source-performance-contracts.json"
    members = []
    for row in record["requested_members"]:
        if row["state"] != "source_checked":
            continue
        capsule = yaml.safe_load((inputs["root"] / row["member"] / "capsule.yaml").read_bytes())
        members.append(
            {
                "member": row["member"],
                "member_sha256": row["original"]["member_sha256"],
                "source_sha256": row["original"]["source"]["sha256"],
                "schedule": [node["name"] for node in capsule["component_program"]["nodes"]],
            }
        )
    value = {
        "schema": D.SELECTION_SCHEMA,
        "generation_identity_sha256": record["generation_identity_sha256"],
        "source_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
        "limits": asdict(SourceDemandLimits(16, 64, 64, 64, 64, 256)),
        "members": members,
    }
    path.write_text(json.dumps(value))
    return dict(
        **inputs,
        source_contract=contract,
        selection=path,
        max_document_bytes=1 << 24,
        max_members=512,
        max_total_nodes=8192,
    )


def original_case(automatic, tmp_path):
    record, inputs = fixtures.run(fixtures.options_for(automatic, tmp_path))
    return record, selection_for(record, inputs, tmp_path / "schedules.json")


def test_complete_live_source_contract_roster_is_observed_without_phase_credit(automatic, tmp_path):
    original, inputs = original_case(automatic, tmp_path)
    record = D.prepare(**inputs)
    assert {row["member"] for row in record["requested_members"]} == {
        row["member"] for row in original["requested_members"]
    }
    assert {row["cohort"] for row in record["requested_members"]} == {
        "development",
        "functional_guard",
        "withheld_transfer",
    }
    assert all(
        row["features"]["status"] == "observed" for row in record["requested_members"] if row["state"] == "observed"
    )
    assert record["original_required_ids"] == original["original_required_ids"]
    assert record["mandatory_missing_ids"] == original["mandatory_missing_ids"]
    assert record["authority"] == "none" and record["hardware_guard_link"] == "not_established"
    assert record["candidate_verdict"] == "not_evaluated" and record["release_authority"] == "not_issued"
    assert D.verify(record, **inputs)
    for substitute in (False, 0.0):
        changed = copy.deepcopy(record)
        zero_macs = next(
            row
            for row in changed["requested_members"]
            if row["state"] == "observed" and row["features"]["demand"]["macs"] == 0
        )
        zero_macs["features"]["demand"]["macs"] = substitute
        with pytest.raises(ValueError, match="complete original source derivation"):
            D.verify(changed, **inputs)
    changed = copy.deepcopy(record)
    next(row for row in changed["requested_members"] if row["state"] == "observed")["features"]["demand"]["macs"] += 1
    changed["sha256"] = fixtures.P.digest({key: value for key, value in changed.items() if key != "sha256"})
    with pytest.raises(ValueError, match="complete original source derivation"):
        D.verify(changed, **inputs)
    stored = json.loads(inputs["source_contract"].read_bytes())
    stored["source_checked_counts"]["development"] = float(stored["source_checked_counts"]["development"])
    inputs["source_contract"].write_text(json.dumps(stored))
    selected_record = json.loads(inputs["selection"].read_bytes())
    selected_record["source_contract_sha256"] = hashlib.sha256(inputs["source_contract"].read_bytes()).hexdigest()
    inputs["selection"].write_text(json.dumps(selected_record))
    with pytest.raises(ValueError, match="fixed complete source/reference generation replay"):
        D.prepare(**inputs)


@pytest.mark.parametrize(
    "defect",
    ["omit_guard", "duplicate", "foreign_member", "source_pin", "dependency", "aggregate", "contract", "generation"],
)
def test_missing_foreign_stale_or_resigned_membership_refuses(automatic, tmp_path, defect):
    _original, inputs = original_case(automatic, tmp_path)
    selected = json.loads(inputs["selection"].read_bytes())
    if defect == "omit_guard":
        selected["members"].pop()
    elif defect == "duplicate":
        selected["members"].append(copy.deepcopy(selected["members"][0]))
    elif defect == "foreign_member":
        selected["members"][0]["member"] = "foreign/source"
    elif defect == "source_pin":
        selected["members"][0]["source_sha256"] = "0" * 64
    elif defect == "dependency":
        row = next(row for row in selected["members"] if len(row["schedule"]) > 1)
        row["schedule"] = list(reversed(row["schedule"]))
    elif defect == "aggregate":
        inputs["max_total_nodes"] = 1
    elif defect == "generation":
        selected["generation_identity_sha256"] = "0" * 64
    else:
        contract = json.loads(inputs["source_contract"].read_bytes())
        contract["mandatory_missing_ids"] = []
        contract["sha256"] = fixtures.P.digest({key: value for key, value in contract.items() if key != "sha256"})
        inputs["source_contract"].write_text(json.dumps(contract))
        selected["source_contract_sha256"] = hashlib.sha256(inputs["source_contract"].read_bytes()).hexdigest()
    inputs["selection"].write_text(json.dumps(selected))
    with pytest.raises(ValueError):
        D.prepare(**inputs)


def test_altered_complete_last_reference_is_not_a_feature_source(automatic, tmp_path):
    _original, inputs = original_case(automatic, tmp_path)
    selected = json.loads(inputs["selection"].read_bytes())
    member = inputs["root"] / selected["members"][-1]["member"]
    stored = fixtures.golden_store.load_golden(member)
    value = next(iter(stored["outputs"].values()))
    value[-1][-1] += 1
    (member / "golden.json").write_text(json.dumps(stored))
    with pytest.raises(ValueError):
        D.prepare(**inputs)


def test_over_budget_profiles_remain_unknown_without_dropping_original_requests(automatic, tmp_path):
    original, inputs = original_case(automatic, tmp_path)
    selected = json.loads(inputs["selection"].read_bytes())
    selected["limits"]["max_payload_bits"] = 1
    inputs["selection"].write_text(json.dumps(selected))
    record = D.prepare(**inputs)
    assert len(record["requested_members"]) == len(original["requested_members"])
    assert all(
        row["features"]["status"] == "UNKNOWN" for row in record["requested_members"] if row["state"] == "UNKNOWN"
    )
    assert record["mandatory_missing_ids"] == original["mandatory_missing_ids"]
