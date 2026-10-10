"""Fixed live factory replay joins; substituted coverage is diagnostic only."""

import copy
import json

import pytest
import test_original_factory_prerequisites as fixtures
from merlin_experiments.phase0 import original_call_sources as C
from merlin_experiments.phase0 import original_factory_prerequisites as F
from merlin_experiments.phase0 import source_requirement_ledger as L

from merlin.common.jsonio import canonical_json

original = fixtures.original


def old_document(original):
    unknowns = C.required_unknowns(original["source_record"], basis=original["basis"], unknown=F.A._unknown)
    unknowns.append(
        F.A._unknown("effect_domain", {"input": "independently-defined"}, "original effect remains missing")
    )
    rows = [
        {
            "original_id": row["id"],
            "kind": row["kind"],
            "original_selector": row["selector"],
            "mandatory": True,
            "source_input_state": "unavailable",
            "missing_source_producers": [row["reason"]],
            "candidate_verdict_phase": 1,
            "candidate_verdict": "not_evaluated",
        }
        for row in unknowns
    ]
    return {
        "schema": L.SCHEMA,
        "original_required_ids": [row["original_id"] for row in rows],
        "original_mandatory_ids": [row["original_id"] for row in rows],
        "requirements": rows,
        "mandatory_source_blockers": [row["original_id"] for row in rows],
        "candidate_verdicts": "not_evaluated",
        "release_authority": "not_issued",
        "status": "diagnostic_incomplete",
        "sha256": "original-projection",
    }


@pytest.fixture
def selected(original, monkeypatch, tmp_path):
    document = old_document(original)
    calls = []

    def old_prepare(**kwargs):
        calls.append(kwargs)
        return L.SourceRequirementLedger(json.dumps(document))

    monkeypatch.setattr(L, "prepare_requirement_ledger", old_prepare)
    return {
        "inputs": {
            "root": tmp_path,
            "coverage": {
                "automatic_derivation": {"original_call_sources": original["source_record"]},
                "obligations": [{"id": row["original_id"]} for row in document["requirements"]],
            },
            "hardware": original["hardware"],
            "software": original["software"],
            "purpose": "source_preparation",
            "schema_intake": original["schema_intake"],
            "semantic_basis": original["basis"],
        },
        "document": document,
        "calls": calls,
        "original": original,
    }


@pytest.mark.parametrize("historical_schema", [L.SCHEMA, "merlin.phase0.source_requirement_ledger.v2"])
def test_new_union_retains_original_coverage_and_every_fulfilled_prerequisite(selected, historical_schema):
    before = selected["document"]
    before["schema"] = historical_schema
    ledger = L.prepare_prerequisite_ledger(**selected["inputs"])
    after = ledger.record()
    assert after["schema"] == L.PREREQUISITE_SCHEMA and after["coverage_projection_schema"] == historical_schema
    assert after["original_required_ids"] == before["original_required_ids"]
    assert set(before["original_required_ids"]) < set(after["original_prerequisite_ids"])
    for key in (
        "requirements",
        "mandatory_source_blockers",
        "original_mandatory_ids",
        "candidate_verdicts",
        "release_authority",
        "status",
    ):
        assert canonical_json(after[key]) == canonical_json(before[key])
    assert len(after["original_factory_prerequisites"]["factory_prerequisites"]) == 9
    assert after["original_factory_prerequisites"]["admission"] == "not_issued"
    assert L.verify_prerequisite_ledger(ledger, **selected["inputs"]) == after
    assert all(call.get("standard_ir") is None for call in selected["calls"])


@pytest.mark.parametrize(
    "change", ["duplicate_id", "omitted_id", "omitted_row_and_id", "kind", "selector", "mandatory_int", "extra_factory"]
)
def test_original_projection_ids_and_selector_collisions_refuse(selected, change):
    document = selected["document"]
    factory = next(row for row in document["requirements"] if row["kind"] == "original_operator_factory")
    if change == "duplicate_id":
        document["original_required_ids"].append(document["original_required_ids"][0])
        document["requirements"].append(copy.deepcopy(document["requirements"][0]))
    elif change == "omitted_id":
        document["requirements"].pop()
    elif change == "omitted_row_and_id":
        document["requirements"].pop()
        document["original_required_ids"].pop()
    elif change == "kind":
        factory["kind"] = "effect_domain"
    elif change == "selector":
        factory["original_selector"]["extent"] = True
    elif change == "mandatory_int":
        factory["mandatory"] = 1
    else:
        extra = copy.deepcopy(factory)
        extra["original_id"] = "extra-unbound-original"
        document["requirements"].append(extra)
        document["original_required_ids"].append(extra["original_id"])
        selected["inputs"]["coverage"]["obligations"].append({"id": extra["original_id"]})
    with pytest.raises(ValueError):
        L.prepare_prerequisite_ledger(**selected["inputs"])


@pytest.mark.parametrize(
    "change",
    ["union_drop", "union_extra", "union_reorder", "factory_drop", "factory_state", "candidate_pass", "blocker_drop"],
)
def test_resigned_or_saved_ledger_cannot_replace_full_live_replay(selected, change):
    ledger = L.prepare_prerequisite_ledger(**selected["inputs"])
    document = ledger.record()
    if change == "union_drop":
        document["original_prerequisite_ids"].pop()
    elif change == "union_extra":
        document["original_prerequisite_ids"].append("extra")
    elif change == "union_reorder":
        document["original_prerequisite_ids"].reverse()
    elif change == "factory_drop":
        document["original_factory_prerequisites"]["factory_prerequisites"].pop()
    elif change == "factory_state":
        document["original_factory_prerequisites"]["factory_prerequisites"][-1]["factory_state"] = "source_constructed"
    elif change == "candidate_pass":
        document["requirements"][0]["candidate_verdict"] = "accepted"
    else:
        document["mandatory_source_blockers"] = []
    document["sha256"] = L.digest({key: value for key, value in document.items() if key != "sha256"})
    with pytest.raises(ValueError, match="stable original prerequisite"):
        L.verify_prerequisite_ledger(L.SourceRequirementLedger(json.dumps(document)), **selected["inputs"])


def test_unchanged_standard_owner_is_forwarded_and_never_replaced_by_factory_state(selected):
    standard = object()
    L.prepare_prerequisite_ledger(**selected["inputs"], standard_ir=standard)
    assert selected["calls"][-1]["standard_ir"] is standard


def test_complete_union_is_stable_when_a_real_factory_budget_refuses_previous_construction(selected):
    before = L.prepare_prerequisite_ledger(**selected["inputs"]).record()
    original = selected["original"]
    source = copy.deepcopy(original["source_record"])
    source["budget"]["max_sources"] = 8
    total = dict.fromkeys(("tensor_elements", "scalar_products", "source_bytes"), 0)
    for graph in source["members"]:
        graph["source_members"] = [
            row
            for row, _ in C._sources(
                graph["calls"], graph["forms"], budget=source["budget"], total=total, requested=9, version=7
            )
        ]
    original["source_record"] = source
    selected["inputs"]["coverage"]["automatic_derivation"]["original_call_sources"] = source
    selected["document"].clear()
    selected["document"].update(old_document(original))
    selected["inputs"]["coverage"]["obligations"] = [
        {"id": row["original_id"]} for row in selected["document"]["requirements"]
    ]
    after = L.prepare_prerequisite_ledger(**selected["inputs"]).record()
    assert before["original_prerequisite_ids"] == after["original_prerequisite_ids"]
    assert len(after["original_required_ids"]) > len(before["original_required_ids"])
    assert len(after["original_factory_prerequisites"]["unavailable_factory_prerequisite_ids"]) == 9
    assert after["release_authority"] == "not_issued" and after["candidate_verdicts"] == "not_evaluated"


@pytest.mark.parametrize("change", ["schema", "software", "hardware"])
def test_different_live_selection_refuses_before_coverage_producer(selected, change, monkeypatch):
    selected["inputs"][{"schema": "schema_intake", "software": "software", "hardware": "hardware"}[change]] = object()
    monkeypatch.setattr(
        L, "prepare_requirement_ledger", lambda **kwargs: pytest.fail("different original owner reached ledger")
    )
    with pytest.raises(ValueError, match="identical live"):
        L.prepare_prerequisite_ledger(**selected["inputs"])


def test_plain_saved_ledger_is_not_fixed_prerequisite_replay(selected):
    with pytest.raises(ValueError, match="exact diagnostic"):
        L.verify_prerequisite_ledger({}, **selected["inputs"])
