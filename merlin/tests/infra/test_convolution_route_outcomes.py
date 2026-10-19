"""Retained full routes never claim to satisfy an unmet packing payload budget."""

from dataclasses import replace

import pytest
from test_source_convolution_routes import configured
from xdsl.dialects import builtin, func

from merlin.llvmlower.convolution_routes import normalize_convolution_source
from merlin.transforms.certificates import ir_digest, ir_text


def normalize(config, module):
    return normalize_convolution_source(module, config, config.selected_contracts.revalidate)


@pytest.mark.parametrize("budget,status", [(60, "exceeded"), (4096, "met")])
def test_explicit_full_is_a_legal_baseline_with_an_honest_payload_outcome(tmp_path, budget, status):
    config, _, module = configured(tmp_path)
    config = replace(config, convolution_route="full", max_convolution_packing_bytes=budget)
    before = ir_text(module)
    selected, report = normalize(config, module)
    assert selected is module and ir_text(selected) == before
    (outcome,) = report["outcomes"]
    assert outcome["selected_route"] == "full"
    assert outcome["selected_packing_payload_bytes"] == outcome["full_packing_payload_bytes"] == 252
    assert outcome["requested_packing_budget_bytes"] == budget
    assert outcome["budget_status"] == status and outcome["reason"] == "explicit_full_route"
    assert "final allocation unestablished" in outcome["evidence"]


@pytest.mark.parametrize("route", ["auto", "panel"])
def test_checked_panel_outcome_counts_both_backings_instead_of_largest_arm(tmp_path, route):
    config, _, module = configured(tmp_path, convolution_route=route)
    selected, report = normalize(config, module)
    assert selected is not module
    (outcome,), (realized,) = report["outcomes"], report["routes"]
    assert outcome["selected_route"] == "panel" and outcome["budget_status"] == "met"
    assert outcome["selected_packing_payload_bytes"] == 48
    assert outcome["full_packing_payload_bytes"] == 252
    assert outcome["source_operation_ordinal"] == realized["source_operation_ordinal"]
    assert outcome["selected_packing_payload_bytes"] == realized["realized_distinct_panel_payload_bytes"]
    assert outcome["reason"] == "checked_source_panel_route"
    positions = tuple(selected.walk())
    assert realized["selected_source_sha256"] == report["selected_sha256"] == ir_digest(selected)
    assert realized["entry_function"] == "forward"
    contractions = [positions[index] for index in realized["selected_contraction_ordinals"]]
    assert len(contractions) == 2 and len(set(contractions)) == 2
    for contraction in contractions:
        assert contraction.name == "linalg.generic"
        assert tuple(contraction.inputs[1].type.get_shape()) in ((12, 3), (12, 1))
        assert tuple(contraction.results[0].type.get_shape()) in ((3, 3), (3, 1))


def test_auto_retains_full_when_its_known_payload_fits(tmp_path):
    config, _, module = configured(tmp_path)
    config = replace(config, max_convolution_packing_bytes=252)
    selected, report = normalize(config, module)
    assert selected is module and not report["rewritten"]
    (outcome,) = report["outcomes"]
    assert outcome["selected_route"] == "full" and outcome["budget_status"] == "met"
    assert outcome["reason"] == "full_route_within_budget"


@pytest.mark.parametrize("route", ["auto", "panel"])
@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"max_convolution_packing_bytes": 1}, "no_legal_panel_within_packing_budget"),
        ({"max_convolution_candidates": 1}, "candidate_budget"),
        ({"max_convolution_added_operations": 1}, "source_growth_budget"),
    ],
)
def test_failed_panel_choices_preserve_full_source_and_report_exceeded(tmp_path, route, changes, reason):
    config, _, module = configured(tmp_path, convolution_route=route)
    config = replace(config, **changes)
    before = ir_text(module)
    selected, report = normalize(config, module)
    assert selected is module and ir_text(selected) == before and report["rewritten"] == 0
    (outcome,) = report["outcomes"]
    assert outcome["selected_route"] == "full" and outcome["selected_packing_payload_bytes"] == 252
    assert outcome["budget_status"] == "exceeded" and outcome["reason"] == reason


@pytest.mark.parametrize("premise", ["qualification", "ownership", "borrowed_effects"])
def test_missing_execution_premises_can_report_source_payload_without_granting_a_provider(tmp_path, premise):
    config, _, module = configured(tmp_path)
    if premise == "qualification":
        config = replace(config, adapter=replace(config.adapter, qualification_identity=None))
    elif premise == "ownership":
        config = replace(config, private_ownership=False)
    else:
        config = replace(config, borrowed_effects=None)
    selected, report = normalize(config, module)
    assert selected is module and not report["rewritten"]
    (outcome,) = report["outcomes"]
    assert outcome["full_packing_payload_bytes"] == 252
    assert outcome["selected_route"] == "full" and outcome["budget_status"] == "exceeded"
    assert outcome["reason"] == "missing_qualified_dense_ownership"


@pytest.mark.parametrize("route", ["full", "panel"])
def test_identical_islands_have_distinct_source_locations_and_individual_budgets(tmp_path, route):
    config, _, module = configured(tmp_path, convolution_route=route)
    original = next(op for op in module.body.block.ops if type(op) is func.FuncOp)
    second = original.clone()
    second.properties["sym_name"] = builtin.StringAttr("second")
    module.body.block.add_op(second)
    module.verify()
    selected, report = normalize(config, module)
    assert (selected is module) == (route == "full")
    first, second = report["outcomes"]
    assert first["source_identity"] == second["source_identity"]
    assert first["source_operation_ordinal"] != second["source_operation_ordinal"]
    assert all(row["budget_status"] == ("exceeded" if route == "full" else "met") for row in report["outcomes"])
    assert "per-island" in report["budget_scope"] and "total host RAM" in report["budget_scope"]
    if route == "panel":
        positions = tuple(selected.walk())
        ordinals = [ordinal for row in report["routes"] for ordinal in row["selected_contraction_ordinals"]]
        assert len(ordinals) == len(set(ordinals)) == 4
        assert {row["entry_function"] for row in report["routes"]} == {"forward", "second"}
        assert all(row["selected_source_sha256"] == ir_digest(selected) for row in report["routes"])
        assert all(positions[index].name == "linalg.generic" for index in ordinals)
        assert sum(row["selected_packing_payload_bytes"] for row in report["outcomes"]) == 96 > 60


def test_unknown_source_coverage_is_not_reported_as_a_met_budget(tmp_path):
    config, _, module = configured(tmp_path)
    config = replace(config, max_source_elements=1)
    selected, report = normalize(config, module)
    assert selected is module and report["outcomes"] == []
    assert "unsupported operations" in report["outcome_coverage"]


def test_retained_full_reporting_rejects_replaced_cached_dense_semantics(tmp_path):
    config, _, module = configured(tmp_path, convolution_route="full")
    dense = config.selected_contracts.dense
    object.__setattr__(config.selected_contracts, "dense", replace(dense, dtypes=("i16", "i16", "i32")))
    with pytest.raises(ValueError, match="original selected declarations"):
        normalize(config, module)
