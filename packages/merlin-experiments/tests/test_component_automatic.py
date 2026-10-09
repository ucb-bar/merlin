"""Automatic source classes reach the actual bounded writer and full oracle.

The native fixture observes only a synthetic unit, not accelerator functionality.
Generated semantics are independently evaluated; hardware roles stay UNKNOWN.
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import test_component_generation as generation_fixtures
import test_component_minimal_spec as minimal_fixtures
import yaml
from merlin_experiments.phase0 import component_automatic as A
from merlin_experiments.phase0 import component_coverage as C
from merlin_experiments.phase0 import generation
from merlin_experiments.phase0.component_semantic_basis import PROVENANCE
from merlin_experiments.phase0.component_semantic_basis import SCHEMA as BASIS_SCHEMA
from merlin_experiments.phase0.evidence import select_evidence
from merlin_experiments.phase0.rtl_intake import issue_independent_hardware_intake
from merlin_experiments.phase0.software_intake import REVIEW_SCHEMA, issue_independent_software_intake
from test_component_execution_budget import policy as execution_policy
from test_independent_rtl_intake import selected  # noqa: F401 -- real deterministic native observation fixture

from merlin.targetgen import golden_store
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves
from merlin.targetgen.frontend_trace import _digest, original_operation_semantics
from merlin.targetgen.rtl import facts
from merlin.targetgen.rtl.source_selection import produce_selection

independent = generation_fixtures.independent
write = generation_fixtures.write


def example(*, unknown=False, extent=17, logical=False):
    nodes, edges = [], []

    def add(name, operation, target, inputs, shape):
        nodes.append(
            {
                "id": name,
                "ordinal": len(nodes),
                "op": operation,
                "target": target,
                "args": [{"node_id": source, "value_id": source + ":v"} for source in inputs],
                "kwargs": {},
                "results": [{"id": name + ":v", "kind": "tensor", "dtype": "int8", "shape": shape}],
            }
        )
        for index, source in enumerate(inputs):
            result = next(node for node in nodes if node["id"] == source)["results"][0]
            edges.append(
                {
                    "producer_node_id": source,
                    "producer_value_id": source + ":v",
                    "consumer_node_id": name,
                    "argument_path": "args/" + str(index),
                    "value_kind": "tensor",
                    "dtype": result["dtype"],
                    "shape": result["shape"],
                }
            )

    add("A", "placeholder", "a", [], [extent, 19])
    add("W0", "placeholder", "w0", [], [19, 23])
    add("W1", "placeholder", "w1", [], [19, 23])
    add("P0", "call_function", "aten.matmul.default", ["A", "W0"], [extent, 23])
    add("P1", "call_function", "aten.matmul.default", ["A", "W1"], [extent, 23])
    add("Copy", "call_function", "aten.clone.default", ["A"], [extent, 19])
    if logical:
        add("Fork0", "call_function", "aten.clone.default", ["Copy"], [extent, 19])
        add("Fork1", "call_function", "aten.clone.default", ["Copy"], [extent, 19])
    if unknown:
        add("Unknown", "call_function", "unreviewed.operation", ["P0"], [extent, 23])
    add(
        "Output",
        "output",
        "output",
        ["P0", "P1", "Copy"] + (["Fork0", "Fork1"] if logical else []) + (["Unknown"] if unknown else []),
        [1, 1],
    )
    graph = {
        "schema": "m2m.frontend_graph.v1",
        "stage": "original",
        "status": "complete",
        "nodes": nodes,
        "edges": edges,
        "call_count": sum(node["op"] == "call_function" for node in nodes),
    }
    graph["sha256"] = _digest(graph)
    return {"schema": "m2m.frontend_trace.v1", "graphs": {"original": graph}}


@pytest.fixture
def automatic(independent, selected, tmp_path, monkeypatch, request):  # noqa: F811 -- registered real producer fixture
    contract = yaml.safe_load(independent["capability_contract"].read_bytes())
    # This native synthetic unit has no observed fixed compute geometry. Do
    # not retain the old fixture's declared systolic mesh as factual authority.
    contract["compute_units"][0]["kind"] = "vector"
    contract["compute_units"][0]["accumulate"] = [{"in": "int8", "weight": "int8", "acc": "i32"}]
    contract["compute_units"][0]["semantic_capabilities"][1]["result_dtypes"] = ["int8", "i32"]
    write(independent["capability_contract"], contract)
    original = json.loads(selected["source_bundle"].read_bytes())
    bundle = produce_selection(
        target="fixture",
        firrtl=Path(original["sources"]["firrtl"]["path"]),
        generator="test_unit",
        config="TestConfiguration",
        core_root="Unit",
        firtool=Path(original["production"]["tool"]["path"]),
        output=tmp_path / "fixture-rtl",
    )
    hardware = issue_independent_hardware_intake(
        target="fixture",
        descriptor=independent["descriptor"],
        source_bundle=bundle,
        forbidden_roots=selected["forbidden_roots"],
        output=tmp_path / "issued-hardware",
    )
    independent.update(rtl_facts=tmp_path / "issued-hardware/facts.json", hardware_intake=hardware)
    monkeypatch.setattr(
        facts,
        "find_facts",
        lambda target, explicit=None: Path(explicit) if explicit is not None else independent["rtl_facts"],
    )
    original_spec = yaml.safe_load(independent["software_spec"].read_bytes())
    original_spec["component_performance"]["hardware"] = {
        "contract_sha256": hashlib.sha256((json.dumps(contract, sort_keys=True, indent=2) + "\n").encode()).hexdigest(),
        "raw_facts_sha256": hashlib.sha256(independent["rtl_facts"].read_bytes()).hexdigest(),
    }
    write(independent["software_spec"], original_spec)
    recipe = minimal_fixtures.recipe_objective(independent)
    spec = yaml.safe_load(independent["software_spec"].read_bytes())
    spec["numerical_semantics"]["overflow"] = "bounded_exact"
    spec["operations"] = {
        owner: {"families": [owner], "placement": "accelerator", "signature": {"operand_dtypes": ["int8"]}}
        for owner in ("movement", "contraction")
    }
    write(independent["software_spec"], spec)
    choices = getattr(request, "param", {})
    source = write(tmp_path / "independent-example.json", example(**choices))
    roster = {
        "schema": BASIS_SCHEMA,
        "status": "reviewed",
        "provenance": dict(PROVENANCE),
        "members": [
            {
                "id": "source-example",
                "kind": "model2mlir_frontend_trace",
                "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "schema": "m2m.frontend_trace.v1",
                "operation_semantics": list(original_operation_semantics(json.loads(source.read_bytes()))[1]),
                "effect_semantics": [],
            }
        ],
    }
    basis = write(tmp_path / "example-basis.json", roster)
    recipe["semantic_basis"] = {"path": str(basis), "sha256": hashlib.sha256(basis.read_bytes()).hexdigest()}
    write(independent["recipe"], recipe)
    review = write(
        tmp_path / "protected-review.json",
        {
            "schema": REVIEW_SCHEMA,
            "target": "fixture",
            "source": {
                "path": str(independent["software_spec"]),
                "sha256": hashlib.sha256(independent["software_spec"].read_bytes()).hexdigest(),
            },
            "semantic_basis": recipe["semantic_basis"],
            "numerical_choices": spec["numerical_semantics"],
            "operation_basis": [
                {
                    "owner": owner,
                    "member": "source-example",
                    "operations": [
                        operation
                        for operation in operations
                        if operation in roster["members"][0]["operation_semantics"]
                    ],
                }
                for owner, operations in (
                    ("movement", ["aten.clone.default", "aten.reshape.default"]),
                    ("contraction", ["aten.matmul.default"]),
                )
            ],
        },
    )
    software = issue_independent_software_intake(
        hardware=hardware,
        source=independent["software_spec"],
        review=review,
        forbidden_roots=selected["forbidden_roots"],
        output_root=tmp_path / "issued-software",
    )
    independent["software_intake"] = software
    evidence = select_evidence(
        "fixture",
        descriptor=independent["descriptor"],
        capability_contract_path=independent["capability_contract"],
        facts_path=independent["rtl_facts"],
        software_spec=independent["software_spec"],
        hardware_intake=hardware,
        software_intake=software,
    )
    recipe["component_performance"]["hardware"] = {
        key: evidence.derivation_identity[key] for key in ("contract_sha256", "raw_facts_sha256")
    }
    write(independent["recipe"], recipe)
    policy = {
        "schema": A.SCHEMA,
        "status": "reviewed",
        "hardware": {key: evidence.derivation_identity[key] for key in ("contract_sha256", "raw_facts_sha256")},
        "software_spec_sha256": software.source.sha256,
        "numerical_semantics_sha256": C.digest(evidence.software_spec["numerical_semantics"]),
        "semantic_basis_sha256": recipe["semantic_basis"]["sha256"],
        "budget": {"max_members": 32, "max_interaction_cells": 128},
        "execution_budget": execution_policy(),
    }
    independent["component_coverage"] = write(tmp_path / "automatic-policy.json", policy)
    return independent


def report(options):
    return json.loads((options["output_root"] / "_evidence/coverage/component-coverage.json").read_bytes())


def run(options):
    with pytest.raises(RuntimeError, match="component coverage"):
        generation.generate_target("fixture", **options)
    return report(options)


def test_real_normal_generation_derives_shared_source_and_all_independent_outputs(automatic):
    observed = run(automatic)
    A.verify(observed["automatic_derivation"], report=observed)
    assert observed["status"] == "incomplete"
    missing = observed["automatic_derivation"]["required_unknowns"]
    assert {(row["kind"], row["selector"]) for row in missing} == {
        ("resource_role", "rtl_boundary_axis_mapping"),
        ("effect_domain", "original_operator_effects"),
    }
    requested = [row for row in observed["obligations"] if row["id"].startswith("auto_shared_input_")]
    assert {row["cohort"] for row in requested} == {"functional_guard", "withheld_transfer"}
    for obligation in requested:
        assert obligation["state"] == "generated"
        for member in obligation["members"]:
            directory = automatic["output_root"] / member["member"]
            capsule = yaml.safe_load((directory / "capsule.yaml").read_bytes())
            assert all(
                17 not in row["shape"] and 19 not in row["shape"] and 23 not in row["shape"]
                for row in capsule["inputs"]
            )
            leaves = materialize_capsule_leaves(capsule)
            a = leaves["A"]
            m, k = a.shape
            expected = {}
            for ordinal in (0, 1):
                w = leaves["W" + str(ordinal)]
                n = w.shape[1]
                expected["Y" + str(ordinal)] = [
                    [sum(a.data[i * k + p] * w.data[p * n + j] for p in range(k)) for j in range(n)] for i in range(m)
                ]
            assert golden_store.load_golden(directory)["outputs"] == expected
            assert capsule["component_program"]["uses"]["A"] == 2
    public = (automatic["output_root"] / "MANIFEST.yaml").read_text()
    assert "independent-example.json" not in public
    assert "rtl_boundary_axis_mapping" not in public
    with pytest.raises(ValueError, match="mandatory component coverage"):
        C.verify_report(automatic["output_root"])


@pytest.mark.parametrize("automatic", [{"unknown": True}], indirect=True)
def test_unreviewed_operator_and_unsupported_interaction_remain_required(automatic):
    observed = run(automatic)
    A.verify(observed["automatic_derivation"], report=observed)
    missing = {(row["kind"], row["selector"]) for row in observed["automatic_derivation"]["required_unknowns"]}
    assert ("operation", "unreviewed.operation") in missing
    assert ("interaction", "publication_and_further_use") in missing
    unknown_ids = {row["id"] for row in observed["automatic_derivation"]["required_unknowns"]}
    assert all(
        row["mandatory"] and row["state"] == "unavailable" and not row["members"]
        for row in observed["obligations"]
        if row["id"] in unknown_ids
    )
    assert any(row["state"] == "generated" for row in observed["obligations"])


@pytest.mark.parametrize("defect", ["missing_unknown", "fake_unknown_pass", "changed_source_template"])
def test_resigned_report_does_not_replace_actual_derivation(automatic, defect):
    observed = copy.deepcopy(run(automatic))
    record = observed["automatic_derivation"]
    if defect == "missing_unknown":
        record["required_unknowns"] = []
        observed["obligations"].pop()
        reason = "original graph replay"
    elif defect == "fake_unknown_pass":
        observed["obligations"][-1]["state"] = "generated"
        reason = "fabricated witness"
    else:
        observed["declaration"]["obligations"][0]["base"]["program"]["inputs"][0]["shape"][0] = 17
        path = Path(record["derived_plan"]["path"])
        write(path, observed["declaration"])
        record["derived_plan"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        observed["plan"]["sha256"] = record["derived_plan"]["sha256"]
        reason = "source factory"
    record["sha256"] = C.digest({key: value for key, value in record.items() if key != "sha256"})
    observed["generation_identity"]["automatic_derivation_sha256"] = C.digest(record)
    observed["sha256"] = C.digest({key: value for key, value in observed.items() if key != "sha256"})
    with pytest.raises(ValueError, match=reason):
        C.verify_report(automatic["output_root"], observed)


def test_author_supplied_obligations_refuse_before_any_component_builder(automatic, monkeypatch):
    from merlin.targetgen import component_program

    declaration = json.loads(automatic["component_coverage"].read_bytes())
    declaration["obligations"] = []
    write(automatic["component_coverage"], declaration)
    monkeypatch.setattr(
        component_program, "build", lambda *a, **kw: pytest.fail("authored obligations reached a builder")
    )
    with pytest.raises(ValueError, match="closed reviewed preauthor policy"):
        generation.generate_target("fixture", **automatic)


def test_substituted_roster_refuses_before_reading_its_source(automatic, tmp_path, monkeypatch):
    substituted = tmp_path / "unselected-source.json"
    substituted.write_text("unselected sentinel")
    recipe = yaml.safe_load(automatic["recipe"].read_bytes())
    recipe["semantic_basis"]["path"] = str(substituted)
    write(automatic["recipe"], recipe)
    read = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: pytest.fail("unselected roster was opened") if path == substituted else read(path),
    )
    with pytest.raises(ValueError, match="roster differs from the protected source selection"):
        generation.generate_target("fixture", **automatic)


def test_source_derived_budget_stops_expensive_shared_contraction_before_build(automatic, monkeypatch):
    from merlin.targetgen import component_program

    declaration = json.loads(automatic["component_coverage"].read_bytes())
    declaration["execution_budget"]["max_reference_work"] = 60
    write(automatic["component_coverage"], declaration)
    original = component_program.build

    def bounded(entry, *args, **kwargs):
        if any(node["op"] == "matmul" for node in entry["program"]["nodes"]):
            assert all(max(row["shape"]) <= 2 for row in entry["program"]["inputs"])
        return original(entry, *args, **kwargs)

    monkeypatch.setattr(component_program, "build", bounded)
    observed = run(automatic)
    denied = [row for row in observed["execution_admission"]["decisions"] if row["state"] != "admitted"]
    assert len(denied) == 2
    assert all("reference_work" in row["reason"] for row in denied)
    assert any("shared_input" in row["name"] for row in denied)
    assert all(
        member["state"] == "unavailable"
        for row in observed["obligations"]
        for member in row["members"]
        if member["name"] in {denial["name"] for denial in denied}
    )


@pytest.mark.parametrize("automatic", [{"extent": 10**9}], indirect=True)
def test_large_original_metadata_cannot_inflate_generated_reference_work(automatic):
    observed = run(automatic)
    A.verify(observed["automatic_derivation"], report=observed)
    for row in observed["obligations"]:
        for member in row["members"]:
            directory = automatic["output_root"] / member["member"]
            capsule = yaml.safe_load((directory / "capsule.yaml").read_bytes())
            assert max(dim for value in capsule["inputs"] for dim in value["shape"]) <= 3
    decisions = observed["execution_admission"]["decisions"]
    assert max(row["cost"]["reference_work"] for row in decisions) == 162
    assert len({member["program_sha256"] for row in observed["obligations"] for member in row["members"]}) > 1
