"""Actual complete source/review stress cases, without target or phase grants."""

import copy
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
from merlin_experiments.phase0 import original_reference_plan as P
from merlin_experiments.phase0 import original_semantic_review as M
from merlin_experiments.phase0 import original_semantic_review_plan as Q

from merlin.common.jsonio import canonical_json


def _fixtures():
    path = Path(__file__).with_name("test_original_reference_standard_ir.py")
    spec = importlib.util.spec_from_file_location("semantic_review_standard_ir_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = _fixtures()
live_originals = F.live_originals
references = F.references
observed = F.observed
BUDGET = Q.OriginalSemanticReviewBudget(300000, 300000, 3000000, 100)


def review_selection(observed):
    """Independent fixture review choices; no production review is inferred."""
    references = observed.references
    original = json.loads(references.receipt_json)
    selected = json.loads(references.selection.read_bytes())
    schemas = json.loads(references.schema_intake.receipt_json)
    canonical = json.loads(Path(schemas["selection_path"]).read_bytes())["canonical_source"]
    defaults = M._defaults(original)
    owners = {}
    for source in original["members"]:
        if source["state"] != "reference_checked":
            continue
        selector = Q.selector(
            source["form"],
            defaults[(source["graph_path"], source["target"], source["call"]["schema"])],
        )
        key = canonical_json(selector)
        context = {
            "aten.matmul.default": "LinearAlgebra.cpp",
            "aten.add.Tensor": "BinaryOps.cpp",
            "aten.conv2d.default": "Convolution.cpp",
        }[source["target"]]
        path = Path(canonical["checkout"]) / "aten/src/ATen/native" / context
        dtypes = dict.fromkeys(source["policy"]["operand_dtypes"])
        owners[key] = {
            "id": "owner_" + str(len(owners)) if key not in owners else owners[key]["id"],
            "selector": selector,
            "numerical_policy": source["policy"],
            "input_palettes": [{"dtype": dtype, "values": P.palette(selected, dtype)} for dtype in dtypes],
            "implementation_context": M.R._pin(path),
            "stress": {
                "per_member": ["signed_inputs"],
                "across_complete_cohorts": ["odd_logical_extent", "nonzero_output"],
            },
        }
    execution_budget = copy.deepcopy(selected["execution_budget"])
    # Additional exact Fraction stress observations have their own explicitly
    # selected logical budget. Original native/reference budgets stay intact.
    for metric in M.E._METRICS:
        execution_budget["max_" + metric] = 100000000
    return {
        "schema": Q.SCHEMA,
        "canonical_source": canonical,
        "cohorts": selected["cohorts"],
        "owners": list(owners.values()),
        "budget": dict(vars(BUDGET)),
        "execution_budget": execution_budget,
    }


@pytest.fixture(scope="module")
def checked(observed, tmp_path_factory):
    owner = tmp_path_factory.mktemp("actual-original-semantic-review")
    path = owner / "review.json"
    F.F.write(path, review_selection(observed))
    return M.prepare(standard_ir=observed, review=path, budget=BUDGET, forbidden_roots=(), destination=owner / "cases")


def test_actual_complete_original_cohorts_and_realized_stress(checked):
    record = checked.record()
    assert len(record["members"]) == 24
    complete = [row for row in record["members"] if row["state"] == "source_case_checked"]
    assert len(complete) == 18, [(row["original"], row["reason"]) for row in record["members"]]
    assert len([row for row in record["original_calls"] if row["state"] == "finite_source_stress_checked"]) == 6
    for row in complete:
        assert row["stress"]["counts"]["negative_inputs"] > 0
        assert row["stress"]["counts"]["positive_inputs"] > 0
        assert row["native_invocation"] and row["upstream_invocation"] and row["parse_invocation"]
        assert set(row["checked_facets"]) == {
            "exact_public_schema_defaults_form_and_storage",
            "protected_original_semantic_selection",
            "complete_independent_native_reference_comparison",
            "ordered_upstream_standard_source_abi",
        }
        assert row["remaining_by_phase"] == M._UNKNOWN
    assert record["remaining_by_phase"] == M._UNKNOWN
    assert "no target/global numeric or phase release" in record["scope"]
    assert record["stress_logical_totals"]["reference_work"] > 0
    assert len(record["implementation_contexts"]) == 6


def test_selector_preserves_semantics_without_selecting_graph_or_geometry(checked):
    source = next(
        row
        for row in json.loads(checked.standard_ir.references.receipt_json)["members"]
        if row["state"] == "reference_checked"
    )
    defaults = M._defaults(json.loads(checked.standard_ir.references.receipt_json))
    observed = defaults[(source["graph_path"], source["target"], source["call"]["schema"])]
    original = Q.selector(source["form"], observed)
    changed = copy.deepcopy(source["form"])
    changed["node"] = "unrelated_node"
    for index, argument in enumerate(changed["arguments"]):
        if argument["value"]["kind"] == "ssa":
            argument["value"]["value"].update(id="fresh_" + str(index), shape=[997, 991])
    assert Q.selector(changed, observed) == original
    changed["arguments"][0]["alias"] = "a!"
    assert Q.selector(changed, observed) != original


@pytest.mark.parametrize("defect", ["defaults", "argument", "alias", "tolerance", "palette", "context"])
def test_original_semantic_review_drift_refuses_actual_source_join(checked, tmp_path, defect):
    selected = review_selection(checked.standard_ir)
    owner = selected["owners"][0]
    if defect == "defaults":
        owner["selector"]["defaults"].reverse()
    elif defect == "argument":
        owner["selector"]["arguments"].reverse()
    elif defect == "alias":
        owner["selector"]["arguments"][0]["alias"] = "a!"
    elif defect == "tolerance":
        owner["numerical_policy"]["atol"] = 0.1
    elif defect == "palette":
        owner["input_palettes"][0]["values"] = [-2.0, 1.0, 4.0]
    else:
        owner["implementation_context"]["sha256"] = "0" * 64
    path = tmp_path / "changed.json"
    F.F.write(path, selected)
    with pytest.raises(ValueError):
        M.prepare(
            standard_ir=checked.standard_ir,
            review=path,
            budget=BUDGET,
            forbidden_roots=(),
            destination=tmp_path / "cases",
        )


def test_missing_original_owner_preserves_every_required_case(checked, tmp_path):
    selected = review_selection(checked.standard_ir)
    missing = selected["owners"].pop()["id"]
    path = tmp_path / "missing.json"
    F.F.write(path, selected)
    actual = M.prepare(
        standard_ir=checked.standard_ir, review=path, budget=BUDGET, forbidden_roots=(), destination=tmp_path / "cases"
    )
    record = json.loads(actual.receipt_json)
    assert len(record["members"]) == 24 and len(record["original_calls"]) == 8
    assert len([row for row in record["members"] if row["state"] == "source_case_checked"]) == 15
    assert all(row["owner"] != missing for row in record["members"])
    assert sum("no independently protected" in str(row["reason"]) for row in record["members"]) == 3


@pytest.mark.parametrize("predicate", ["rounded_product", "wrapped_output"])
def test_full_value_and_abi_pass_cannot_grant_unrealized_required_stress(checked, tmp_path, predicate):
    selected = review_selection(checked.standard_ir)
    for owner in selected["owners"]:
        owner["stress"]["across_complete_cohorts"].append(predicate)
    path = tmp_path / "stress.json"
    F.F.write(path, selected)
    actual = M.prepare(
        standard_ir=checked.standard_ir, review=path, budget=BUDGET, forbidden_roots=(), destination=tmp_path / "cases"
    )
    record = json.loads(actual.receipt_json)
    assert len(record["members"]) == 24
    assert sum(row["state"] == "source_case_checked" for row in record["members"]) == 18
    assert all(row["state"] == "unavailable" for row in record["original_calls"])
    assert sum(predicate in row["unrealized_complete_cohort_stress"] for row in record["original_calls"]) == 6


@pytest.mark.parametrize("limit", ["max_members", "max_total_tensor_payload_bytes"])
def test_complete_stress_preflight_precedes_new_shaped_evaluation(checked, tmp_path, monkeypatch, limit):
    selected = review_selection(checked.standard_ir)
    if limit == "max_members":
        selected["budget"][limit] = 1
    else:
        selected["execution_budget"][limit] = 1
    budget = Q.OriginalSemanticReviewBudget(**selected["budget"])
    path = tmp_path / "denied.json"
    F.F.write(path, selected)
    monkeypatch.setattr(M, "_stress", lambda *args: pytest.fail("stress allocated before the complete roster budget"))
    actual = M.prepare(
        standard_ir=checked.standard_ir, review=path, budget=budget, forbidden_roots=(), destination=tmp_path / "cases"
    )
    record = json.loads(actual.receipt_json)
    assert len(record["members"]) == 24
    assert all(row["state"] == "unavailable" for row in record["members"])
    assert all("stress" not in row for row in record["members"])


def test_saved_object_and_changed_private_product_cannot_recreate_checked_cases(checked):
    with pytest.raises(ValueError, match="actual live"):
        replace(checked).verify()
    original = checked.output.read_bytes()
    try:
        checked.output.write_bytes(original[:-2])
        with pytest.raises(ValueError, match="complete private record"):
            checked.verify()
    finally:
        checked.output.write_bytes(original)


@pytest.mark.parametrize("field", ["model", "accepted", "candidate", "shape", "callback"])
def test_protected_review_has_no_answer_or_candidate_selector(checked, field):
    selected = review_selection(checked.standard_ir)
    selected["owners"][0][field] = "injected"
    with pytest.raises(ValueError, match="saved status, callbacks"):
        Q.validate(selected)


def test_review_byte_budget_refuses_before_document_decode(checked, tmp_path):
    selected = review_selection(checked.standard_ir)
    selected["budget"]["max_review_bytes"] = 1
    path = tmp_path / "oversized.json"
    F.F.write(path, selected)
    with pytest.raises(ValueError, match="before decoding"):
        M.prepare(
            standard_ir=checked.standard_ir,
            review=path,
            budget=Q.OriginalSemanticReviewBudget(**selected["budget"]),
            forbidden_roots=(),
            destination=tmp_path / "cases",
        )
