"""Actual finite comparisons cannot replace the full original requirements.

Diagnostic metadata below isolates the join; only the ordinary coverage owner
can prepare the enclosing ledger. The reference/stdIR objects are actual live
public-schema preparations, never synthetic authority or saved status.
"""

import copy
import importlib.util
from pathlib import Path

import pytest
from merlin_experiments.phase0 import original_reference_requirements as J

path = Path(__file__).with_name("test_original_reference_standard_ir.py")
spec = importlib.util.spec_from_file_location("original_reference_join_native_fixtures", path)
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
live_originals = fixtures.live_originals
references = fixtures.references
selection = fixtures.selection
observed = fixtures.observed


def _inputs(observed):
    intake = observed.references.schema_intake
    reference = observed.references.record()
    members, unknowns, requirements = [], [], []
    for defaults in reference["defaults"]:
        calls, seen = [], set()
        for row in reference["members"]:
            if row["graph_path"] != defaults["graph_path"] or row["node"] in seen:
                continue
            seen.add(row["node"])
            calls.append(row["call"])
            identity = "original-" + row["original_member_id"] + "-" + row["node"]
            unknowns.append(
                {
                    "id": identity,
                    "kind": "original_operator_admission",
                    "selector": {
                        key: row[value]
                        for key, value in (("member", "original_member_id"), ("node", "node"), ("target", "target"))
                    },
                }
            )
            requirements.append(
                {
                    "original_id": identity,
                    "kind": "original_operator_admission",
                    "source_input_state": "unavailable",
                    "missing_source_producers": ["original full source/reference numerical comparison"],
                    "candidate_verdict": "not_evaluated",
                }
            )
        members.append({"graph_path": defaults["graph_path"], "calls": calls})
    document = {
        "schema": "merlin.phase0.source_requirement_ledger.v1",
        "original_required_ids": [row["original_id"] for row in requirements],
        "requirements": requirements,
        "mandatory_source_blockers": [row["original_id"] for row in requirements],
        "release_authority": "not_issued",
    }
    coverage = {"automatic_derivation": {"original_call_sources": {"members": members}, "required_unknowns": unknowns}}
    return document, {
        "coverage": coverage,
        "hardware": intake.software.hardware,
        "software": intake.software,
        "standard_ir": observed,
    }


def test_live_complete_cohorts_join_exact_original_ids_without_admitting_whole_rows(observed):
    document, arguments = _inputs(observed)
    original = copy.deepcopy(document)
    actual = J.join(document, **arguments)
    assert actual["original_required_ids"] == original["original_required_ids"]
    assert actual["mandatory_source_blockers"] == original["mandatory_source_blockers"]
    assert actual["release_authority"] == "not_issued"
    assert actual["original_reference_observations"]["required_source_slots"] == 24
    assert actual["original_reference_observations"]["source_reference_ir_checked_slots"] == 18
    assert len(actual["original_source_reference_witnesses"]) == 24
    checked, missing = [], []
    for row in actual["requirements"]:
        facet = row["original_reference_facets"]
        assert row["source_input_state"] == "unavailable" and row["candidate_verdict"] == "not_evaluated"
        assert len(facet["required_source_slots"]) == 3
        assert "original_operation_correspondence" in facet["missing_original_premises"]
        (checked if facet["complete_comparison"] == "checked" else missing).append(row)
    assert len(checked) == 6 and len(missing) == 2
    assert "upstream_compiled_semantics" in checked[0]["original_reference_facets"]["missing_original_premises"]


@pytest.mark.parametrize("change", ["wrong_call", "missing_call", "changed_dtype", "wrong_node", "missing_id"])
def test_join_cannot_drop_or_substitute_original_call_abi_or_denominator(observed, change):
    document, arguments = _inputs(observed)
    coverage = arguments["coverage"]
    if change == "wrong_node":
        coverage["automatic_derivation"]["required_unknowns"][0]["selector"]["node"] = "other"
    elif change == "missing_id":
        document["requirements"].pop()
    else:
        calls = coverage["automatic_derivation"]["original_call_sources"]["members"][0]["calls"]
        if change == "missing_call":
            calls.pop()
        elif change == "changed_dtype":
            calls[0]["result_dtypes"] = ["float16"]
        else:
            calls[0]["target"] = "other.operation"
    with pytest.raises(ValueError):
        J.join(document, **arguments)


def test_saved_or_copied_owner_is_not_replayed_evidence(observed):
    document, arguments = _inputs(observed)
    for saved in (observed.record(), copy.copy(observed)):
        arguments["standard_ir"] = saved
        with pytest.raises(ValueError, match="live|actual"):
            J.join(document, **arguments)


def test_other_software_or_hardware_owner_refuses_before_attribution(observed):
    document, arguments = _inputs(observed)
    for key in ("hardware", "software"):
        altered = dict(arguments, **{key: object()})
        with pytest.raises(ValueError, match="same"):
            J.join(document, **altered)


def test_changed_original_native_product_cannot_retain_checked_facets(observed, tmp_path):
    document, arguments = _inputs(observed)
    record = observed.record()
    product = Path(
        next(row for row in record["members"] if row["state"] == "source_reference_ir_checked")["products"]["actual"][
            "path"
        ]
    )
    original = product.read_bytes()
    try:
        product.write_bytes(original + b"\n")
        with pytest.raises(ValueError):
            J.join(document, **arguments)
    finally:
        product.write_bytes(original)
