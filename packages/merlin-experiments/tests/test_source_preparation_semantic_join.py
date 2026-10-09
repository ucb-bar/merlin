"""Finite live source cases join exact original slots, never whole readiness.

The diagnostic requirement table isolates attribution. It cannot construct the
enclosing live preparation; its real coverage consumer has separate controls.
Native schema/reference/standard-source owners below are genuine fixed runs.
"""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
from merlin_experiments.phase0 import original_reference_requirements as J
from merlin_experiments.phase0 import source_preparation_release as P


def _load(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("preparation_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _load("test_original_semantic_review")
F = _load("test_original_reference_requirement_join")
live_originals = M.live_originals
references = M.references
observed = M.observed
checked = M.checked


@pytest.fixture(scope="module")
def joined(checked):
    document, arguments = F._inputs(checked.standard_ir)
    ledger = J.join(document, **arguments)
    result = P._semantic_facets(ledger, checked)
    return checked, ledger, result


def test_actual_original_semantic_case_join_retains_every_unsupported_and_pending_row(joined):
    checked, ledger, cases = joined
    assert cases["sha256"] == checked.sha256
    assert len(cases["members"]) == 24
    assert sum(row["state"] == "source_case_checked" for row in cases["members"]) == 18
    assert sum(row["state"] == "finite_source_stress_checked" for row in cases["original_calls"]) == 6
    assert len(ledger["requirements"]) == len(ledger["mandatory_source_blockers"]) == 8
    for requirement in ledger["requirements"]:
        assert requirement["source_input_state"] == "unavailable"
        assert requirement["candidate_verdict"] == "not_evaluated"
        assert len(requirement["original_semantic_facets"]["required_source_slots"]) == 3
        assert requirement["missing_source_producers"]
    assert (
        sum(row["original_semantic_facets"]["complete_cohort_stress"] == "checked" for row in ledger["requirements"])
        == 6
    )
    assert cases["remaining_by_phase"] == M.M._UNKNOWN


@pytest.mark.parametrize("defect", ["missing_private", "swapped", "wrong_source", "missing_result"])
def test_actual_case_attribution_cannot_drop_private_or_substitute_source_products(joined, defect):
    checked, ledger, _ = joined
    changed = copy.deepcopy(ledger)
    rows = changed["original_source_reference_witnesses"]
    if defect == "missing_private":
        rows.pop()
    elif defect == "swapped":
        rows[0], rows[1] = rows[1], rows[0]
    elif defect == "wrong_source":
        rows[0]["standard_ir_products"]["source"]["sha256"] = "f" * 64
    else:
        rows[0]["reference_products"].pop("comparison")
    with pytest.raises(ValueError, match="original|complete"):
        P._semantic_facets(changed, checked)


def test_plain_or_reconstructed_semantic_data_cannot_grant_source_facets(joined):
    checked, ledger, _ = joined
    for saved in (json.loads(checked.receipt_json), copy.copy(checked)):
        with pytest.raises(ValueError, match="live|actual"):
            P._semantic_facets(copy.deepcopy(ledger), saved)


def test_changed_last_actual_comparison_product_refuses_join(joined):
    checked, ledger, _ = joined
    rows = json.loads(checked.receipt_json)["members"]
    selected = [row for row in rows if row["state"] == "source_case_checked"][-1]
    path = Path(selected["reference_products"]["comparison"]["path"])
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"\n")
        with pytest.raises(ValueError):
            P._semantic_facets(copy.deepcopy(ledger), checked)
    finally:
        path.write_bytes(original)
