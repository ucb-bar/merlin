"""Automatic sources reach the ordinary writer, reference, discovery and freeze.

The native fixture derives a synthetic memory volume. It has no array mapping,
execution or timing authority. Those unknowns remain in every resulting receipt.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_component_source_performance as fixtures
import yaml
from merlin_experiments.phase0 import component_source_performance as P
from merlin_experiments.phase0 import sweeps
from merlin_experiments.phase0.component_generation import digest
from merlin_experiments.phase0.software_intake import issue_independent_software_intake
from merlin_experiments.phase2 import corpus as C

from merlin.targetgen import golden_store
from merlin.targetgen.frontend_trace import _digest, original_operation_semantics
from merlin.targetgen.op_form_regimes import Bound, boundary_witnesses

automatic = fixtures.automatic
independent = fixtures.independent
selected = fixtures.selected


def policy(**changes):
    return {
        "schema": P.CURRICULUM_SCHEMA,
        "probe_extents": [2, 3],
        "max_members": 128,
        "max_extent_bits": 32,
        **changes,
    }


def options_for(options, tmp_path, selected):
    options = fixtures.fixtures._source_options(options, tmp_path, version=fixtures.fixtures.A.LOGICAL_POLICY_SCHEMA)
    spec = yaml.safe_load(options["software_spec"].read_bytes())
    spec["numerical_semantics"]["overflow"] = "modular_wrap"
    spec["operations"] = {
        "movement": {"ops": ["copy"], "placement": "host", "signature": {"broadcasting": "none"}},
        "contraction": {"ops": ["matmul"], "placement": "accelerator", "signature": {"broadcasting": "none"}},
        "cast": {"ops": ["cast"], "placement": "host", "signature": {"broadcasting": "none"}},
        "reduction": {"ops": ["reduce_sum"], "placement": "host", "signature": {"broadcasting": "none"}},
        "add": {"ops": ["add"], "placement": "host", "signature": {"broadcasting": "none"}},
    }
    fixtures.fixtures.fixtures.write(options["software_spec"], spec)
    recipe = yaml.safe_load(options["recipe"].read_bytes())
    basis_path = Path(recipe["semantic_basis"]["path"])
    basis = json.loads(basis_path.read_bytes())
    source_path = Path(basis["members"][0]["path"])
    trace = json.loads(source_path.read_bytes())
    graph = trace["graphs"]["original"]
    output = graph["nodes"].pop()
    assert output["op"] == "output"
    for name, target in (
        ("SourceCast", "aten.to.dtype"),
        ("SourceSum", "aten.sum.dim_IntList"),
        ("SourceAdd", "aten.add.Tensor"),
    ):
        original = deepcopy(next(row for row in graph["nodes"] if row["id"] == "Copy"))
        original.update(id=name, target=target, ordinal=len(graph["nodes"]))
        original["results"][0]["id"] = name + ":v"
        graph["nodes"].append(original)
        original_edge = deepcopy(next(row for row in graph["edges"] if row["consumer_node_id"] == "Copy"))
        original_edge["consumer_node_id"] = name
        graph["edges"].append(original_edge)
    graph["nodes"].append(output)
    for ordinal, row in enumerate(graph["nodes"]):
        row["ordinal"] = ordinal
    graph["call_count"] += 3
    graph["sha256"] = _digest({key: value for key, value in graph.items() if key != "sha256"})
    fixtures.fixtures.fixtures.write(source_path, trace)
    basis["members"][0].update(
        sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        operation_semantics=list(original_operation_semantics(trace)[1]),
    )
    fixtures.fixtures.fixtures.write(basis_path, basis)
    recipe["semantic_basis"]["sha256"] = hashlib.sha256(basis_path.read_bytes()).hexdigest()
    review_path = tmp_path / "protected-review.json"
    review = json.loads(review_path.read_bytes())
    review.update(
        source={
            "path": str(options["software_spec"]),
            "sha256": hashlib.sha256(options["software_spec"].read_bytes()).hexdigest(),
        },
        semantic_basis=recipe["semantic_basis"],
        numerical_choices=spec["numerical_semantics"],
        operation_basis=[
            {"owner": owner, "member": "source-example", "operations": [operation]}
            for owner, operation in (
                ("movement", "aten.clone.default"),
                ("contraction", "aten.matmul.default"),
                ("cast", "aten.to.dtype"),
                ("reduction", "aten.sum.dim_IntList"),
                ("add", "aten.add.Tensor"),
            )
        ],
    )
    fixtures.fixtures.fixtures.write(review_path, review)
    software = issue_independent_software_intake(
        hardware=options["hardware_intake"],
        source=options["software_spec"],
        review=review_path,
        forbidden_roots=selected["forbidden_roots"],
        output_root=tmp_path / "curriculum-software",
    )
    options["software_intake"] = software
    coverage = yaml.safe_load(options["component_coverage"].read_bytes())
    coverage.update(
        software_spec_sha256=software.source.sha256,
        numerical_semantics_sha256=digest(software.public_facts()["numerical_semantics"]),
        semantic_basis_sha256=recipe["semantic_basis"]["sha256"],
    )
    fixtures.fixtures.fixtures.write(options["component_coverage"], coverage)
    template = P.derive_curriculum(software=software, hardware=options["hardware_intake"], policy=policy())
    fixtures.fixtures.fixtures.write(options["performance_template"], template)
    recipe["component_performance"]["objectives"] = template["curriculum_derivation"]["objectives"]
    fixtures.fixtures.fixtures.write(options["recipe"], recipe)
    options["source_preparation"] = P.SCHEMA_V2
    return options, template


def test_automatic_curriculum_normal_generation_reference_discovery_and_family_closed_freeze(
    automatic, selected, tmp_path
):
    options, template = options_for(automatic, tmp_path, selected)
    record, inputs = fixtures.run(options)
    assert record["schema"] == P.SCHEMA_V2
    assert record["curriculum_derivation"] == template["curriculum_derivation"]
    assert record["source_checked_counts"]["development"] == 36  # six 2x2 families, volume and joint boundary points
    assert any(
        row["family"] == "joint_tail_capacity" and row["status"] == "UNKNOWN" for row in record["missing_families"]
    )
    assert record["mandatory_missing_ids"] and record["hardware_guard_link"] == "not_established"
    assert P.verify_source_contracts(record, **inputs) == record
    mixed = next(row for row in record["requested_members"] if row["member"].endswith("mixed_closed_boundary_2_3"))
    path = inputs["root"] / mixed["member"]
    cap = yaml.safe_load((path / "capsule.yaml").read_bytes())
    assert [row["source_operation"] for row in mixed["typed_node_owners"]] == [
        "copy",
        "matmul",
        "reduce_sum",
        "cast",
        "matmul",
    ]
    assert set(golden_store.load_golden(path)["outputs"]) == {"producer", "host_sum", "converted", "result"}
    assert (path / "capsule.interface.mlir").read_text().count("arith.muli") == 2
    assert cap["performance"]["curriculum"]["point"] == {"R": 2, "C": 3}
    joint = next(row for row in template["sweeps"] if row["id"] == "joint_source_depth_volume_0")
    assert joint["axes"] == {"R": [3, 4, 5], "C": [3, 4, 5]}
    assert joint["base"]["performance"]["curriculum"]["resource_boundary"]["hardware_mapping"] == "UNKNOWN"
    selected_corpus = SimpleNamespace(
        target="fixture", capsule_corpus=options["output_root"] / "isa", graded_roots=lambda: []
    )
    discovered = C.discover_performance_corpus(selected_corpus)
    assert discovered.test_justification["schema"] == "merlin.phase2.test_justification.v2"
    assert all(
        row["source_execution_basis"]["status"] == "source_prepared_execution_unknown"
        for row in discovered.test_justification["members"]
    )
    assert discovered.test_justification["families_not_generated"]["mandatory_unknowns"]
    frozen = C.freeze_performance_corpus(discovered, tmp_path / "frozen-source")
    C.verify_frozen_performance_corpus(frozen)
    represented = C.discover_performance_corpus(selected_corpus, capsules="representative")
    selected_families = {member.family for member in represented.capsules}
    assert represented.capsules and all(
        sum(member.family == family for member in represented.capsules)
        == sum(member.family == family for member in discovered.capsules)
        for family in selected_families
    )


def test_re_signed_template_and_forbidden_query_hints_cannot_replace_live_derivation(automatic, tmp_path):
    software, hardware = automatic["software_intake"], automatic["hardware_intake"]
    with pytest.raises(ValueError, match="teaching extents"):
        P.derive_curriculum(software=software, hardware=hardware, policy=policy(workload="private_selector"))
    template = P.derive_curriculum(software=software, hardware=hardware, policy=policy())
    assert any(
        row["family"] == "mixed_closed_boundary" and row["mandatory"]
        for row in template["curriculum_derivation"]["required_unknowns"]
    )
    template["sweeps"][0]["axes"]["R"] = [7, 9]
    derivation = template["curriculum_derivation"]
    derivation["sha256"] = digest({key: value for key, value in derivation.items() if key != "sha256"})
    path = fixtures.fixtures.fixtures.write(tmp_path / "altered-template.yaml", template)
    with pytest.raises(ValueError, match="live complete automatic"):
        P.verify_curriculum_template(path, software=software, hardware=hardware)


def test_host_gate_cannot_gain_execution_authority_from_saved_satisfied_labels():
    from merlin_experiments.phase0.profiles import HOST_SOURCE_BASIS

    basis = {
        "schema": HOST_SOURCE_BASIS,
        "kind": "host",
        "software_intake_sha256": "1" * 64,
        "operations": ["copy"],
        "numerical_reference": "integer_reference",
        "boundary": "complete_host_call",
    }
    ok, decision = sweeps.evaluate_gate(
        {"traits": [], "source_execution_basis": basis},
        {"source_execution_bases": {digest(basis): {"selection": basis, "satisfied": True}}},
    )
    assert not ok and decision["outcome"] == "unestablished"
    assert decision["source_execution_basis"]["satisfied"] is None


def test_adjacent_regime_witnesses_preserve_missing_positive_comparisons():
    assert boundary_witnesses(Bound("axis", 4, "observed.depth")) == {"<": 3, "=": 4, ">": 5}
    assert boundary_witnesses(Bound("axis", 1, "observed.depth")) == {"<": None, "=": 1, ">": 2}
    assert boundary_witnesses(Bound("axis", None, "", "not derived")) == {"<": None, "=": None, ">": None}
