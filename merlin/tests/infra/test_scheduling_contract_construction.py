"""Drafts expose missing semantics without manufacturing selection or qualification."""
import copy
import hashlib
import json

import pytest
import yaml

from merlin.compile.scheduling.dense_contracts import DENSE_FIELDS, DENSE_POINTER_FIELDS, dense_contract_draft
from merlin.compile.scheduling.selected_contracts import (
    inspect_scheduling_draft, read_scheduling_contracts, scheduling_spec_draft,
)
from test_selected_dense_contracts import inputs


def encoded(value):
    return yaml.safe_dump(value).encode()


def test_missing_sections_have_editable_unknowns_and_no_assumed_semantics():
    software = {"schema": "merlin.software_spec.v1", "target": "independent", "status": "reviewed",
                "existing_public_field": {"retain": True}}
    before = copy.deepcopy(software)
    draft = scheduling_spec_draft(software, target="independent", dense_schema="merlin.dense_schedule.v2")
    assert software == before and draft["status"] == "draft"
    assert draft["existing_public_field"] == software["existing_public_field"]
    dense = draft["scheduling"]["dense"]
    assert set(dense) == set(DENSE_FIELDS) | {"logical_pointer_abi", "external_pointer_arguments"}
    assert set(dense["logical_pointer_abi"]) == set(DENSE_POINTER_FIELDS)
    assert all(value is None for value in dense["semantics"].values())
    assert all(value is None for value in dense["instructions"].values())
    assert draft["scheduling"]["configuration"] is None  # Empty would assert no config effects.
    result = inspect_scheduling_draft(b'{"facts":{"target":"independent"}}', encoded(draft),
                                     target="independent", require_dense=True)
    assert result["status"] == "incomplete" and result["qualification"] == "not_established"
    assert ["scheduling", "dense", "semantics", "completion"] in result["unknown_fields"]
    assert not result["dense"]["family_requirements_are_target_facts"]


def test_completed_draft_uses_same_resolver_without_creating_review_marker(tmp_path):
    selected, *_ = inputs(tmp_path)
    original = yaml.safe_load(selected.software_bytes)
    draft = scheduling_spec_draft(original, target="neutral")
    payload = encoded(draft)
    result = inspect_scheduling_draft(selected.facts_bytes, payload, target="neutral", require_dense=True)
    assert result["status"] == "declaration_resolved"
    assert result["isa"]["identity"] == selected.isa.identity()
    assert result["qualification"] == "not_established" and result["software_status"] == "draft"
    assert result["dense"]["resolved"] and not result["dense"]["family_semantic_mismatches"]
    path = tmp_path / "authored-draft.yaml"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="reviewed"):
        read_scheduling_contracts(selected.facts_path, path, target="neutral")
    # Explicit author assertion only; this marker is not a signed review or native grade.
    draft["status"] = "reviewed"
    path.write_bytes(encoded(draft))
    resolved = read_scheduling_contracts(selected.facts_path, path, target="neutral")
    assert resolved.isa.identity() == selected.isa.identity() and resolved.dense == selected.dense


def test_raw_encoding_with_unknown_effects_cannot_unlock_dense(tmp_path):
    selected, *_ = inputs(tmp_path)
    document = yaml.safe_load(selected.software_bytes)
    for instruction in document["isa"]["instructions"]:
        instruction.pop("effects")
    document["isa"]["resources"] = []
    result = inspect_scheduling_draft(selected.facts_bytes, encoded(document), target="neutral", require_dense=True)
    assert result["isa"]["status"] == "resolved_declaration"
    assert len(result["isa"]["unknown_effects"]) == 5
    assert result["status"] == "incomplete" and not result["dense"]["resolved"]


def test_queued_acceptance_does_not_become_synchronous_completion(tmp_path):
    selected, *_ = inputs(tmp_path)
    document = yaml.safe_load(selected.software_bytes)
    document["scheduling"]["dense"]["semantics"]["completion"] = "queued_acceptance"
    result = inspect_scheduling_draft(selected.facts_bytes, encoded(document), target="neutral", require_dense=True)
    assert result["status"] == "incomplete"
    assert result["dense"]["family_semantic_mismatches"] == [
        {"field": "completion", "declared": "queued_acceptance", "family_requires": "synchronous"}]


def test_changed_facts_pin_is_not_repaired_and_different_isa_is_not_overwritten(tmp_path):
    selected, *_ = inputs(tmp_path)
    document = yaml.safe_load(selected.software_bytes)
    changed = selected.facts_bytes + b'\n'
    result = inspect_scheduling_draft(changed, encoded(document), target="neutral", require_dense=True)
    assert result["status"] == "incomplete" and "pin" in result["isa"]["reason"]
    with pytest.raises(ValueError, match="replace"):
        scheduling_spec_draft(document, target="neutral", isa={"different": True})
    assert result["facts_sha256"] == hashlib.sha256(changed).hexdigest()


def test_draft_version_refuses_unknown_and_has_no_shared_mutable_cells():
    with pytest.raises(ValueError, match="version"):
        dense_contract_draft(schema="unknown")
    left = dense_contract_draft(schema="merlin.dense_schedule.v1")
    right = dense_contract_draft(schema="merlin.dense_schedule.v1")
    left["storage"]["lhs"]["row_bytes"]["fact"] = ["a"]
    assert right["storage"]["lhs"]["row_bytes"]["fact"] is None
