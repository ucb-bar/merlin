"""Public graph planning metadata cannot disclose private source values."""

import hashlib
import json

import pytest

from merlin_experiments.corpus.model_shape_roster import prepare, project


def trace():
    return {"graphs": {"prepared": {"sha256": "c" * 64, "nodes": [
        {"id": name, "target": target, "op": "call_function", "ordinal": i,
         "args": ["PRIVATE_WEIGHT_CONTENT"], "kwargs": {"secret": "PRIVATE_REFERENCE"},
         "results": [{"id": name + ":v0", "dtype": "int32", "shape": [4, 8],
                      "kind": "tensor", "values": ["PRIVATE_ARRAY"]}],
         "source": "/private/checkpoint"}
        for i, (name, target) in enumerate([("a", "aten.mm.default"), ("b", "aten.add.Tensor")])
    ], "edges": [
        {"producer_node_id": "input", "producer_value_id": "input:v0", "consumer_node_id": "a",
         "shape": [4, 8], "dtype": "int8", "value": "PRIVATE_INPUT"},
        {"producer_node_id": "a", "producer_value_id": "a:v0", "consumer_node_id": "b",
         "shape": [4, 8], "dtype": "int32"}
    ]}}}


def test_projection_keeps_source_join_and_connected_boundary_without_values():
    result = project(trace(), model="test_model", program="decode", source_sha256="d" * 64)
    assert "PRIVATE" not in json.dumps(result)
    assert "/private" not in json.dumps(result)
    graph = result["graphs"][0]
    assert graph["nodes"][0]["results"][0]["shape"] == [4, 8]
    region = graph["connected_region_proposals"][0]
    assert region["nodes"] == ["a", "b"]
    assert region["boundary_inputs"][0]["producer_node_id"] == "input"
    assert region["execution_status"] == "not_admitted"
    assert result["historical_holdout_claim"] is False


def test_prepare_rechecks_trace_pin_and_emits_no_private_path(tmp_path):
    source = tmp_path / "trace.json"
    source.write_text(json.dumps(trace()))
    row = {"model": "test_model", "program": "model", "path": str(source),
           "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    output = tmp_path / "public"
    record = prepare([row], output)
    assert str(source) not in (output / "roster.json").read_text()
    assert record["private_source_pins"][0]["path"] == str(source)
    source.write_text("{}")
    with pytest.raises(ValueError, match="bytes changed"):
        prepare([row], tmp_path / "changed")


def test_compact_projection_preserves_every_source_shape_occurrence():
    from merlin_experiments.corpus.model_shape_roster import compact_projection

    full = project(trace(), model="test_model", program="model", source_sha256="d" * 64)
    compact = compact_projection(full)
    graph = compact["graphs"][0]
    assert graph["node_count"] == 2
    assert sorted(name for row in graph["shape_signatures"] for name in row["source_node_ids"]) == ["a", "b"]
    assert graph["connected_region_proposals"] == full["graphs"][0]["connected_region_proposals"]
    assert "PRIVATE" not in json.dumps(compact)


def test_connected_cut_keeps_graph_liveout_and_pattern_boundary():
    from merlin_experiments.corpus.workload_patterns import analyze

    source = trace()
    graph = source["graphs"]["prepared"]
    graph["edges"].append({"producer_node_id": "b", "producer_value_id": "b:v0",
                           "consumer_node_id": "output", "shape": [4, 8], "dtype": "int32"})
    projected = project(source, model="test_model", program="model", source_sha256="d" * 64)
    region = projected["graphs"][0]["connected_region_proposals"][0]
    assert region["nodes"] == ["a", "b"]
    assert region["boundary_outputs"][0]["consumer_node_id"] == "output"
    analysis = analyze(projected)
    assert analysis["graphs"][0]["connected_pattern_families"] == 1
    assert analysis["graphs"][0]["contraction_boundary_patterns"] == [
        {"producer": "aten.mm.default", "consumer": "aten.add.Tensor", "occurrences": 1}]
    assert "PRIVATE" not in json.dumps(analysis)
