"""A complete source graph remains evidence even when no target region is executable."""

from __future__ import annotations

import hashlib
import json

import pytest

from merlin.capture.graph_regions import capture_graph_regions
from merlin.common.jsonio import canonical_sha256


def _write_capture(root):
    nodes = []
    for identity, op, args in (
        ("input", "placeholder", []),
        ("first", "call_function", [{"node_id": "input", "value_id": "input:v0"}, 0.125]),
        ("second", "call_method", [{"node_id": "first", "value_id": "first:v0"}]),
    ):
        nodes.append(
            dict(
                id=identity,
                graph_id="original",
                ordinal=len(nodes),
                op=op,
                target="custom." + identity,
                args=args,
                kwargs={},
                results=[
                    dict(
                        id=identity + ":v0",
                        kind="tensor",
                        dtype="float32",
                        shape=[2, 3],
                        layout="torch.strided",
                        stride=[3, 1],
                    )
                ],
            )
        )
    graph = dict(
        schema="m2m.frontend_graph.v1",
        stage="original",
        status="complete",
        counting_unit="static_captured_call_sites",
        graph_id="original",
        nodes=nodes,
        call_count=2,
        by_target={"custom.first": 1, "custom.second": 1},
    )
    graph["sha256"] = canonical_sha256(graph)
    quant = dict(
        schema="m2m.frontend_graph.v1",
        stage="quantized",
        status="complete",
        nodes=[
            dict(id="qscale", op="placeholder", input_target="scale", origin_node_ids=[]),
            dict(
                id="qfirst",
                op="call_function",
                origin_node_ids=["first"],
                args=[dict(node_id="qscale", value_id="qscale:v0")],
            ),
        ],
    )
    quant["sha256"] = canonical_sha256(quant)
    module = b"module {}\n"
    (root / "model.mlir").write_bytes(module)
    trace = dict(
        schema="m2m.frontend_trace.v1",
        status="complete",
        graphs={"original": graph, "quantized": quant},
        mlir=dict(sha256=hashlib.sha256(module).hexdigest(), operations=[dict(ordinal=1, origin_node_ids=["first"])]),
    )
    (root / "frontend-trace.json").write_text(json.dumps(trace))
    (root / "meta.json").write_text(
        json.dumps(
            dict(
                frontend_trace=dict(sha256=hashlib.sha256((root / "frontend-trace.json").read_bytes()).hexdigest()),
                recipe=dict(rounding="half_even"),
                recipe_sha256="selected-recipe",
            )
        )
    )
    (root / "weights.safetensors").write_bytes(b"selected captured bytes; this test does not interpret storage")
    (root / "weights.safetensors.manifest.json").write_text(
        json.dumps({"0": dict(weight="scale", kind="buffer", shape=[1], dtype="float32")})
    )
    return trace


def _replace_trace(root, trace):
    (root / "frontend-trace.json").write_text(json.dumps(trace))
    meta = json.loads((root / "meta.json").read_text())
    meta["frontend_trace"]["sha256"] = hashlib.sha256((root / "frontend-trace.json").read_bytes()).hexdigest()
    (root / "meta.json").write_text(json.dumps(meta))


def test_all_calls_frontiers_literals_and_buffer_lineage_retained(tmp_path):
    _write_capture(tmp_path)
    first = capture_graph_regions(tmp_path)
    assert first == capture_graph_regions(tmp_path)
    assert len(first["regions"]) == 2
    a, b = first["regions"]
    assert a["region_id"] == "first" and a["operation"]["args"][1] == 0.125
    assert a["inputs"][0]["value"]["stride"] == [3, 1]
    assert b["dependency_region_ids"] == ["first"]
    assert a["output_consumers"]["first:v0"] == [dict(node_id="second", argument_path="/args/0")]
    assert a["lineage"] == dict(graphs={"quantized": ["qfirst"]}, mlir_operation_ordinals=[1])
    assert a["buffer_bindings"][0]["manifest"]["weight"] == "scale"
    assert a["source_binding"]["standalone"] is False and a["source_region"] is None
    assert a["reference_ids"] == [] and {g["kind"] for g in a["gaps"]} == {
        "source_region_extraction_gap",
        "reference_alignment_gap",
    }
    assert b["lineage"]["mlir_operation_ordinals"] == []
    assert "numerical_equivalence" in first["not_established"]


@pytest.mark.parametrize(
    "change", ["original_digest", "future_value", "duplicate_value", "source_module", "lineage_digest", "trace_pin"]
)
def test_broken_source_custody_or_complete_graph_refused(tmp_path, change):
    trace = _write_capture(tmp_path)
    if change == "source_module":
        (tmp_path / "model.mlir").write_text("module { changed }")
    elif change == "trace_pin":
        (tmp_path / "frontend-trace.json").write_text("{}")
    else:
        graph = trace["graphs"]["original"]
        if change == "original_digest":
            graph["nodes"][1]["args"][1] = 2.0
        elif change == "lineage_digest":
            trace["graphs"]["quantized"]["nodes"][0]["input_target"] = "other"
        elif change == "future_value":
            graph["nodes"][1]["args"][0] = dict(node_id="second", value_id="second:v0")
            graph["sha256"] = canonical_sha256({k: v for k, v in graph.items() if k != "sha256"})
        else:
            graph["nodes"][1]["results"][0]["id"] = "input:v0"
            graph["sha256"] = canonical_sha256({k: v for k, v in graph.items() if k != "sha256"})
        _replace_trace(tmp_path, trace)
    with pytest.raises(ValueError):
        capture_graph_regions(tmp_path)


def test_bounds_symlink_and_duplicate_json_keys_refused(tmp_path):
    _write_capture(tmp_path)
    with pytest.raises(ValueError):
        capture_graph_regions(tmp_path, max_bytes=10)
    original = tmp_path / "original.mlir"
    (tmp_path / "model.mlir").rename(original)
    (tmp_path / "model.mlir").symlink_to(original)
    with pytest.raises(ValueError):
        capture_graph_regions(tmp_path)
    (tmp_path / "model.mlir").unlink()
    original.rename(tmp_path / "model.mlir")
    (tmp_path / "weights.safetensors.manifest.json").write_text('{"0":{},"0":{}}')
    with pytest.raises(ValueError):
        capture_graph_regions(tmp_path)
