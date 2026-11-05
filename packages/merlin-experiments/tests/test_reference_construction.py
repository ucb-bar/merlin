"""Reference advice reaches automatic construction through the existing catalogs and tools."""
from __future__ import annotations

import copy
import hashlib
import json

import pytest
import test_selfcheck_next_steps as step_fixtures
from merlin_experiments.phase1 import operation_optimizations as operations
from merlin_experiments.phase1.feedback import next_steps, reference_map
from merlin_experiments.phase1.feedback import reference_construction as construction
from merlin_experiments.phase1.feedback import reference_experiments as reference
from test_reference_experiment_feedback import _fixture
from test_selfcheck_next_steps import FIXTURE


@pytest.fixture
def workspace(tmp_path):
    return step_fixtures.workspace.__wrapped__(tmp_path)


@pytest.fixture
def guided(workspace, monkeypatch):
    owner = workspace / "selection-owner"
    owner.mkdir()
    selected = _fixture(owner)
    doc = dict(schema=reference_map.SCHEMA, reference_manifest_sha256="e" * 64, entries=[
        dict(id="dense-region", file="kernel.c", locator=dict(function="expert"),
             operation=dict(kind="matmul", dims=dict(M=16, K=32, N=16)),
             techniques=["retain a panel", "overlap a legal next panel"], uses_hardware_loops=False,
             confidence="inferred", evidence="pinned source function")])
    path = workspace / "selected-reference-map.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setenv(reference_map.ENV, json.dumps(dict(path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    catalog = dict(schema=operations.SELECTION_SCHEMA, entries=[dict(id="contraction", mechanisms=[
        dict(kind="catalog", id="construction-alternative")])])
    (workspace / operations.FILENAME).write_text(json.dumps(catalog))
    selection = copy.deepcopy(FIXTURE["suggestions"])
    selection["all_mechanisms"].append(dict(id="construction-alternative", api="merlin.compile.capacity",
        symbol="capacity_fit", family="storage", requirements=dict(**{"workload.dense_contraction": "unknown"}),
        legality="derive a legal block from the current geometry and assigned storage"))
    (workspace / "OPTIMIZATION_SUGGESTIONS.json").write_text(json.dumps(selection))
    return selected


def step(workspace):
    return next_steps._correctness(dict(family="dense", label="dense", left=["MM_case"], correct=0, cases=1,
        failing_count=1, left_large=0, left_more=0), next_steps.Context(workspace))


def test_off_has_same_map_and_library_but_no_added_construction_advice(workspace, guided):
    found = step(workspace)
    assert found["kind"] == "finish_correctness"
    assert "reference_construction" not in found


def test_on_gets_actionable_advice_before_correctness_and_missing_adapter_is_not_ready(workspace, guided):
    with reference.invocation(guided["path"], guided["sha"], tasks=False):
        found = step(workspace)
    construction = found["reference_construction"]
    assert construction["phase"] == "construction"
    assert construction["references"][0]["id"] == "dense-region"
    assert "--reference-entry dense-region" in construction["references"][0]["inspect"]
    chosen = construction["alternatives"][0]
    assert chosen["api"] == "merlin.compile.capacity:capacity_fit"
    assert chosen["opportunity_status"] == "missing_adapter" and "apply" not in chosen
    assert "library_pass_adapter.py" in chosen["next_action"]
    assert construction["ready_for_performance"] is False
    assert "general compiler rules" in construction["objective"]
    assert "Trust supplied expert measurements" in construction["reference_performance"]
    assert construction["learning"]["policy_api"].startswith("merlin.transforms.shape_policy:")
    assert construction["semantic_compatibility"] == construction["mechanism_status"] == "unknown"
    assert "compare" not in construction  # source-only reference is not silently compared to disassembly
    assert any(gap["kind"] == "reference_machine_view_missing" for gap in construction["gaps"])
    lines = next_steps._reference_lines(["header", "construction"], [found], {})
    assert any("reference experiment: construction-alternative via" in line for line in lines)
    assert any("schedule.plan" in line for line in lines)


def test_existing_adapter_path_reuses_native_ab_tool_and_source_calls_stay_unconfirmed(workspace, guided):
    adapter = workspace / "submission/tools/library_pass_adapter.py"
    adapter.parent.mkdir()
    adapter.write_text("# Presence is discovery; the existing apply tool still verifies actual hooks.\n")
    with reference.invocation(guided["path"], guided["sha"], tasks=False):
        chosen = step(workspace)["reference_construction"]["alternatives"][0]
        assert chosen["opportunity_status"] == "candidate"
        assert "apply_library_pass.py" in chosen["apply"]
        assert "--mechanism construction-alternative" in chosen["apply"]
        assert "variants/" in chosen["apply"]
        (workspace / "submission/backend.py").write_text(
            "from merlin.compile.capacity import capacity_fit\ncapacity_fit(problem, geometry)\n")
        updated = step(workspace)["reference_construction"]
    chosen = updated["alternatives"][0]
    assert chosen["opportunity_status"] == "already_observed" and chosen["source_call_sites"]
    assert "runtime application unconfirmed" in chosen["observation_basis"]
    assert "apply" not in chosen and updated["mechanism_status"] == "unknown"


def test_partial_geometry_is_not_an_exact_match(workspace, guided):
    declaration = dict(kind="matmul", dims=dict(M=16, K=32, N=16), shape=None)
    entry = dict(operation=dict(kind="matmul", dims=dict(M=16)))
    assert reference_map.grade(entry, declaration, edge=16) == "technique"


def test_unsupported_catalog_entry_is_not_advertised(workspace, guided):
    p = workspace / "OPTIMIZATION_SUGGESTIONS.json"
    data = json.loads(p.read_text())
    data["all_mechanisms"][-1].update(status="unsupported", requirements={})
    p.write_text(json.dumps(data))
    with reference.invocation(guided["path"], guided["sha"], tasks=False):
        found = step(workspace)["reference_construction"]
    assert found["alternatives"] == []
    assert any(row["kind"] == "library_binding_gap" for row in found["gaps"])


def test_declared_conv_pool_gets_existing_packing_and_composition_route(workspace, guided, monkeypatch):
    declaration = step_fixtures.conv("CHAIN_case")
    declaration["operation"]["attributes"].update(layout="nhwc", epilogue=["requantize", "maxpool"])
    from merlin_experiments.phase1.feedback import case_declarations

    monkeypatch.setattr(case_declarations, "load", lambda name, ws: declaration)
    entry = dict(id="spatial-expert", file="expert.dis", locator=dict(lines=[1, 9]),
        operation=dict(kind="conv2d"), techniques=[], uses_hardware_loops=False,
        confidence="inferred", evidence="fixture")
    monkeypatch.setattr(reference_map, "selected", lambda: dict(entries=[entry]))
    with reference.invocation(guided["path"], guided["sha"], tasks=False):
        found = construction.construction("CHAIN_case", next_steps.Context(workspace))
    pool = next(row for row in found["composition"]["routes"] if row["route"] == "producer_pool")
    assert "bounded_spatial_packing_and_rolling" in pool["catalog_rows"]
    assert "spatial_chain_recipe.py" in pool["next"]
    assert any("pack input patches" in text and "producer_pool" in text for text in found["lines"])
    assert found["semantic_compatibility"] == "unknown" and not found["ready_for_performance"]


def test_exact_lm_head_is_targeted_before_correctness_without_bundle_membership(workspace, guided, monkeypatch):
    from merlin_experiments.phase1.feedback import case_declarations

    declaration = dict(operation=dict(op="matmul", attributes=dict(lhs="A", weight="W")),
        inputs=[dict(name="A", role="input", shape=[1, 2048]),
                dict(name="W", role="weight", shape=[2048, 32000])])
    monkeypatch.setattr(case_declarations, "load", lambda name, ws: declaration)
    entry = dict(id="tinyllama.mm_lm", file="tinyllama_raw/exo_llm.c", locator=dict(function="mm_lm"),
        operation=dict(kind="matmul", dims=dict(M=1, K=2048, N=32000)), techniques=[],
        uses_hardware_loops=False, confidence="verified", evidence="fixture")
    monkeypatch.setattr(reference_map, "selected", lambda: dict(entries=[entry]))
    with reference.invocation(guided["path"], guided["sha"], tasks=False):
        found = construction.construction("anonymous-output-projection", next_steps.Context(workspace))
    assert found["references"][0]["id"] == "tinyllama.mm_lm"
    assert found["references"][0]["match"] == "exact"
    assert "--reference-entry tinyllama.mm_lm" in found["references"][0]["inspect"]


def test_actual_shared_rhs_source_gets_bound_ab_route_without_a_reference_name_match(workspace, guided, monkeypatch):
    from test_reference_views import inputs

    source, _, _ = inputs()
    (workspace / "layers/MM_case/capsule.interface.mlir").write_bytes(source)
    monkeypatch.setattr(reference_map, "selected", lambda: None)
    path = workspace / "OPTIMIZATION_SUGGESTIONS.json"
    selection = json.loads(path.read_text())
    selection["all_mechanisms"].append(dict(id="shared_rhs_epochs",
        api="merlin.compile.scheduling.dense_shared_rhs", symbol="schedule_dense_shared_rhs",
        family="scheduling", requirements={}))
    path.write_text(json.dumps(selection))
    with reference.invocation(guided["path"], guided["sha"], tasks=False):
        missing = construction.construction("MM_case", next_steps.Context(workspace))
        adapter = workspace / "submission/tools/library_pass_adapter.py"
        adapter.parent.mkdir()
        adapter.write_text("# existing apply tool validates the actual hooks\n")
        bound = construction.construction("MM_case", next_steps.Context(workspace))
    assert missing["references"] == [] and missing["grouped_source"]["products"] == 2
    assert missing["alternatives"][0]["opportunity_status"] == "missing_adapter"
    route = bound["alternatives"][0]
    assert route["mechanism"] == "shared_rhs_epochs"
    assert "apply_library_pass.py" in route["apply"] and "--mechanism shared_rhs_epochs" in route["apply"]
    assert bound["mechanism_status"] == "unknown" and not bound["ready_for_performance"]
    assert any("configuration/fence/alias" in line for line in bound["lines"])
