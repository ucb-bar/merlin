"""Frontend trace capacity is separate from ordinary strict authority records."""

from __future__ import annotations

import hashlib
import inspect
import json

import pytest

from merlin.common import strict_json
from merlin.targetgen import frontend_trace as T


def test_trace_uses_dedicated_capacity_without_changing_global(monkeypatch):
    original = strict_json.loads
    observed = []

    def record(raw, *, max_bytes):
        observed.append(max_bytes)
        return original(raw, max_bytes=max_bytes)

    monkeypatch.setattr(strict_json, "loads", record)
    assert T.loads_frontend_trace(b'{"graphs": {}}') == {"graphs": {}}
    assert observed == [256 * 1024 * 1024]
    assert strict_json.MAX_RECORD_BYTES == 128 * 1024 * 1024
    assert inspect.signature(original).parameters["max_bytes"].default == 128 * 1024 * 1024


def test_trace_above_generic_bound_is_complete_and_bounded(monkeypatch):
    raw = b'{"complete":true}' + b" " * 64
    with pytest.raises(ValueError, match="bounded reader"):
        strict_json.loads(raw, max_bytes=64)
    monkeypatch.setattr(T, "MAX_FRONTEND_TRACE_BYTES", len(raw))
    assert T.loads_frontend_trace(raw) == {"complete": True}
    with pytest.raises(ValueError, match="bounded reader"):
        T.loads_frontend_trace(raw + b" ")
    with pytest.raises(ValueError, match="bounded reader"):
        T.loads_frontend_trace(raw.decode() + "é")


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"nested":{"a":1,"a":2}}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":-Infinity}',
        b'{"a":1e999}',
    ],
)
def test_trace_preserves_duplicate_and_finite_checks(raw):
    with pytest.raises(ValueError, match="duplicate|non-finite"):
        T.loads_frontend_trace(raw)


def _emitted_alias(tmp_path):
    from merlin.targetgen.application_graph import application_graph_inventory

    model = tmp_path / "model.mlir"
    model.write_text("""module {
      func.func @forward(%x: tensor<2xf32>) -> tensor<2xf32> {
        %alias = "tensor.cast"(%x) {
          prov.source_node_ids = ["g:prepared:root:n1"],
          prov.origin_node_ids = ["g:original:root:n1", "g:quantized:root:n1"],
          prov.trace_role = "lowering"
        } : (tensor<2xf32>) -> tensor<2xf32>
        func.return %alias : tensor<2xf32>
      }
    }""")
    graph = application_graph_inventory(model)
    graphs = {}
    for stage in ("original", "quantized", "prepared"):
        snapshot = {
            "schema": "m2m.frontend_graph.v1",
            "stage": stage,
            "status": "complete",
            "call_count": 1,
            "nodes": [
                {"id": f"g:{stage}:root:n0", "ordinal": 0, "op": "placeholder", "target": "x", "results": []},
                {
                    "id": f"g:{stage}:root:n1",
                    "ordinal": 1,
                    "op": "call_function",
                    "target": "aten.alias.default",
                    "results": [],
                },
            ],
            "edges": [],
            "runtime_versions": {"torch": "fixture"},
        }
        snapshot["sha256"] = hashlib.sha256(
            json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        graphs[stage] = snapshot
    operations = [
        dict(
            ordinal=op["ordinal"],
            operation=op["mlir_operation"],
            source_node_ids=op["source_node_ids"],
            origin_node_ids=op["origin_node_ids"],
            role=op["trace_role"] or "structural",
            operand_types=[v["type"] for v in op["operands"]],
            result_types=[v["type"] for v in op["results"]],
        )
        for op in graph["capture_graph"]["operations"]
    ]
    receipt = {
        "node_id": "g:prepared:root:n1",
        "alias_of_node_id": "g:prepared:root:n0",
        "result_index": 0,
        "status": "alias",
        "mlir_ordinals": [op["ordinal"] for op in operations if op["source_node_ids"]],
    }
    trace = {
        "schema": "m2m.frontend_trace.v1",
        "status": "complete",
        "blockers": [],
        "graphs": graphs,
        "transformations": [
            dict(
                from_stage=a,
                to_stage=b,
                status="complete",
                relations=[
                    {
                        "source_ids": [f"g:{a}:root:n1"],
                        "destination_ids": [f"g:{b}:root:n1"],
                        "kind": "identity",
                    }
                ],
            )
            for a, b in (("original", "quantized"), ("quantized", "prepared"))
        ],
        "mlir": {
            "sha256": graph["capture_sha256"],
            "bytes": graph["capture_bytes"],
            "operations": operations,
            "source_correspondence": [receipt],
        },
    }
    return trace, graph, receipt


def test_alias_disposition_can_own_exact_emitted_cast(tmp_path):
    trace, graph, _ = _emitted_alias(tmp_path)
    joined = T.join_frontend_trace(trace, graph, capture_sha256=graph["capture_sha256"])
    assert joined["status"] == "complete" and joined["errors"] == []
    assert joined["prepared_lowering_obligations"]["status"] == "verified"


@pytest.mark.parametrize(
    "change",
    [
        {"alias_of_node_id": "missing"},
        {"result_index": -1},
        {"result_index": True},
        {"result_index": None},
        {"mlir_ordinals": []},
        {"status": "eliminated", "reason": "dropped"},
    ],
)
def test_emitted_alias_still_requires_valid_target_result_and_exact_ordinals(tmp_path, change):
    trace, graph, receipt = _emitted_alias(tmp_path)
    receipt.update(change)
    joined = T.join_frontend_trace(trace, graph, capture_sha256=graph["capture_sha256"])
    assert joined["status"] == "partial"
    assert "prepared call-site lowering receipt disagrees with exact MLIR source identities" in joined["errors"]
