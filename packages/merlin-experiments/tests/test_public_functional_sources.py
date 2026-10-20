"""Independent tiny source graphs exercise functional derivation, never a target."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from merlin_experiments.phase0 import freeze, profiles
from merlin_experiments.phase0 import public_boundaries as B
from merlin_experiments.phase0 import public_boundary_semantics as S
from merlin_experiments.phase0 import public_functional_sources as F
from merlin_experiments.phase0.evidence import _materialize_evidence
from merlin_experiments.phase0.generation import _select_generation_scope
from test_public_joint_boundaries import trace

from merlin.common.paths import repo_root
from merlin.targetgen import golden_store

POLICY = {
    "schema": B.SCHEMA_V3,
    "max_members": 16,
    "max_nodes": 16,
    "max_logical_elements": 4096,
    "max_extent": 64,
    "max_source_bytes": 16384,
}


def program():
    inputs = [
        {"name": n, "shape": s, "dtype": "float32"}
        for n, s in (("X", [2, 3]), ("Wq", [4, 3]), ("Wk", [4, 3]), ("R", [2, 4]))
    ]
    nodes = [
        ("q", "aten.linear.default", [{"value": "X"}, {"value": "Wq"}], [2, 4]),
        ("k", "aten.linear.default", [{"value": "X"}, {"value": "Wk"}], [2, 4]),
        ("residual", "aten.add.Tensor", [{"value": "q"}, {"value": "R"}], [2, 4]),
        ("active", "aten.relu.default", [{"value": "residual"}], [2, 4]),
        ("mean", "aten.mean.dim", [{"value": "active"}, [-1], True], [2, 1]),
    ]
    return {
        "inputs": inputs,
        "nodes": [dict(name=n, target=t, args=a, kwargs={}, shape=s, dtype="float32") for n, t, a, s in nodes],
        "outputs": ["k", "mean"],
    }


def inputs(source=None):
    raw = json.dumps(trace(source or program())).encode()
    receipt = json.dumps(
        {"artifacts": {"frontend-trace.json": {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}}}
    ).encode()
    origin = {
        "materialized_capture": {
            "path": "materialized-performance/independent/model.mlir",
            "workload_id": "independent",
            "workload_role": "performance_scale",
            "capture_sha256": "1" * 64,
            "receipt_sha256": hashlib.sha256(receipt).hexdigest(),
            "loader_sha256": "2" * 64,
            "full_inventory_sha256": "3" * 64,
        }
    }
    outputs = {
        "materialized-performance/independent/frontend-trace.json": raw,
        "materialized-performance/independent/capture_receipt.json": receipt,
    }
    facts = {"facts": {"arrays": [{"rows": 4, "cols": 4}]}}
    evidence = SimpleNamespace(
        refreshed_facts=facts,
        raw_facts_sha256=B.digest(facts),
        software_spec={"numerical_semantics": {"model": {"engine": "integer_reference"}}},
    )
    return [origin], outputs, evidence


def derive(source=None, policy=POLICY):
    origins, outputs, evidence = inputs(source)
    return B.derive(origins, outputs, evidence=evidence, policy=policy)


def test_sibling_projection_adjacency_is_explicit_and_legacy_unchanged():
    source = trace(program())
    previous, _ = S.cuts(source, typed=True, max_nodes=16)
    current, gaps = S.cuts(source, typed=True, max_nodes=16, shared_inputs=True)
    assert len(previous) == 2 and len(current) == 1 and not gaps
    assert len(current[0]["nodes"]) == 5 and len(current[0]["outputs"]) == 2
    assert S.cuts(source, typed=True, max_nodes=16)[0] == previous


def test_residual_facet_needs_distinct_paths_and_equal_shapes():
    entry = derive()[0][0]
    actual = entry["public_boundary"]["program"]
    assert F.facets(actual)["residual_join"]
    add = next(node for node in actual["nodes"] if node["target"] == "aten.add.Tensor")
    add["args"][1] = copy.deepcopy(add["args"][0])
    assert F.facets(actual)["residual_join"] == []


def test_facts_derived_functional_members_remain_in_functional_scope():
    entries, outputs, report = derive()
    assert len(entries) == 2 and report["requested"] == 2
    assert _select_generation_scope(entries, "phase1-functional") == (entries, [])
    for entry in entries:
        member = entry["public_boundary"]
        assert entry["cat"] == "_synthetic" and entry["label"] == "public" and "performance" not in entry
        assert member["logical_facets"]["residual_join"]
        assert member["logical_facets"]["mean_reduction"]
        assert len(member["logical_facets"]["shared_input_projections"][0]) == 2
        assert member["stages"] == {"generated": True, "evaluated": False, "admitted": False, "executed": False}
        assert outputs[entry["loader"]] == B.render(member["program"]).encode()
        assert member["source_execution_basis"]["numerical_semantics_sha256"] == report["numerical_semantics_sha256"]
    assert all(row["mandatory"] and row["status"] == "UNKNOWN" for row in report["missing"])
    assert any(row.get("domain") == "attention_state" for row in report["missing"])
    assert report["functional_completeness"] == report["physical_reuse"] == report["timing"] == "UNKNOWN"


def test_tiny_complete_host_reference_preserves_all_liveouts():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    entries, _, _ = derive()
    for entry in entries:
        p = entry["public_boundary"]["program"]
        scope = {}
        exec(compile(B.render(p), "independent-functional-source", "exec"), scope)
        model, operands = scope["get_model_and_inputs"]()
        actual = model(*operands)
        x, wq, wk, residual = (value.numpy() for value in operands)
        expected = (x @ wk.T, np.maximum(x @ wq.T + residual, 0).mean(-1, keepdims=True))
        assert len(actual) == len(expected) == len(p["outputs"])
        for a, e in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a.numpy(), e)


def test_excluded_inputs_and_result_values_do_not_select_sources():
    origins, outputs, evidence = inputs()
    baseline = B.derive(origins, outputs, evidence=evidence, policy=POLICY)
    # These are never opened: changing answers, validation artifacts or compiler
    # successes cannot select source geometry, operations, source bytes or IDs.
    for path in (
        "private-final/model.mlir",
        "supplied-implementation.zip",
        "golden.npy",
        "build/model.elf",
        "compiler-result.json",
    ):
        outputs[path] = b"invalid bytes that would fail any attempted parse"
    assert B.derive(origins, outputs, evidence=evidence, policy=POLICY) == baseline
    assert derive() == derive()


@pytest.mark.parametrize("defect", ["role", "receipt", "trace", "numeric"])
def test_changed_or_ineligible_selection_refuses(defect):
    origins, outputs, evidence = inputs()
    if defect == "role":
        origins[0]["materialized_capture"]["workload_role"] = "validation"
    elif defect in {"receipt", "trace"}:
        suffix = "capture_receipt.json" if defect == "receipt" else "frontend-trace.json"
        outputs["materialized-performance/independent/" + suffix] += b" "
    else:
        evidence.software_spec["numerical_semantics"] = {}
    with pytest.raises(ValueError):
        B.derive(origins, outputs, evidence=evidence, policy=POLICY)


@pytest.mark.parametrize(
    "budget,value",
    [("max_members", 1), ("max_nodes", 1), ("max_extent", 1), ("max_source_bytes", 1), ("max_logical_elements", 1)],
)
def test_every_denied_obligation_is_retained(budget, value):
    entries, _, report = derive(policy={**POLICY, budget: value})
    assert len(entries) <= 1 and report["missing"]
    assert any(
        row["reason"] != "complete attention mask/probability and carried KV state contracts are not established"
        for row in report["missing"]
    )


def test_replay_and_freeze_reopen_source_purpose_and_complete_graph(tmp_path):
    entries, outputs, _ = derive()
    _materialize_evidence(tmp_path, outputs)
    entry = entries[0]
    assert B.selected_loader(entry, tmp_path).read_bytes() == outputs[entry["loader"]]
    synthesis = tmp_path / "synthesis.yaml"
    synthesis.write_text(yaml.safe_dump({"capsules": entries}))
    owned = freeze._materialized_inputs({"inputs": {"synth_profile": str(synthesis)}})
    assert len(owned) == len(entries) + 1
    changed = copy.deepcopy(entry)
    changed["public_boundary"]["logical_facets"]["shared_input_projections"] = []
    with pytest.raises(ValueError, match="logical witness"):
        B.selected_loader(changed, tmp_path)
    changed = copy.deepcopy(entry)
    changed["public_boundary"]["stages"]["admitted"] = True
    with pytest.raises(ValueError, match="logical witness"):
        B.selected_loader(changed, tmp_path)
    changed = copy.deepcopy(entry)
    changed["performance"] = {"invented": "timing"}
    with pytest.raises(ValueError, match="logical witness"):
        B.selected_loader(changed, tmp_path)


def test_written_source_reference_records_evaluation_without_admission(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    entry = derive()[0][0]
    program = entry["public_boundary"]["program"]
    scope = {}
    exec(compile(B.render(program), "tiny-independent-source", "exec"), scope)
    model, operands = scope["get_model_and_inputs"]()
    raw = json.dumps(trace(program)).encode()
    (tmp_path / "capsule.pytorch.py").write_text(B.render(program))
    (tmp_path / "frontend-trace.json").write_bytes(raw)
    for name in ("capsule.interface.mlir", "frontend-source.mlir"):
        (tmp_path / name).write_bytes(b"synthetic source fixture; no compiler/target authority")
    golden_store.write_golden(
        tmp_path,
        {
            "golden_source": "host_torch_eager",
            "outputs": {str(i): value.tolist() for i, value in enumerate(model(*operands))},
        },
    )
    cap = {
        "public_boundary": entry["public_boundary"],
        "frontend_trace": {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "source_mlir_sha256": hashlib.sha256((tmp_path / "frontend-source.mlir").read_bytes()).hexdigest(),
        },
    }
    result = B.verify_written(cap, tmp_path)
    assert result["stages"] == {"generated": True, "evaluated": True, "admitted": False, "executed": False}
    cap["public_boundary"]["source_execution_basis"]["operations"] = []
    with pytest.raises(ValueError, match="full derivation"):
        B.verify_written(cap, tmp_path)


def test_selected_ordinary_recipe_factory_and_legacy_definitions():
    root = repo_root()
    recipe = yaml.safe_load((root / "examples/gemmini/phase0/recipe-public-functional.yaml").read_bytes())
    shared = yaml.safe_load((root / "experiments/templates/phase0/performance.yaml").read_bytes())
    assert B.template_for(recipe["public_joint_boundaries"]) == shared["public_boundary_factories"][B.SCHEMA_V3]
    assert shared["public_boundary_factory"] == B.TEMPLATE
    assert shared["public_boundary_factories"][B.SCHEMA_V2] == B.TEMPLATE_V2
    profile = {}
    profiles._merge_shared_perf(
        profile,
        source=root / "examples/gemmini/phase0/recipe-public-functional.yaml",
        performance_template=root / "experiments/templates/phase0/performance.yaml",
    )
    assert profile["_performance_template"]["public_boundary_factories"][B.SCHEMA_V3] == B.TEMPLATE_V3
