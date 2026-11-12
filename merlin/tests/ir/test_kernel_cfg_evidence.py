"""CFG source custody is checked independently of producer/evaluator code."""

import hashlib
import json

import pytest

from merlin.frontends.kernel_cfg_evidence import checked_control_flow_sources


def record(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def fixture():
    source = b"original multi-block source"
    graph = {
        "operations": [{"id": "source.branch", "successors": ["exit"]}],
        "blocks": [{"id": "entry"}, {"id": "exit"}],
        "attributes": "complete source fields",
    }
    data = {
        "cfg-input.mlir": b"typed function input",
        "cfg-lifted.mlir": b"structured native output",
        "cfg-output.ttir": b"typed TT output",
        "cfg-stderr.log": b"",
        "cfg-source-graph.json": json.dumps(graph).encode(),
    }
    stage = {
        "schema": "t2m.control-flow-stage.v1",
        "stage_completed": True,
        "grants_semantics": False,
        "flags": [
            "--mlir-disable-threading",
            "--verify-each",
            "--mlir-print-op-generic",
            "--mlir-print-debuginfo",
            "--mlir-print-local-scope",
            "--canonicalize",
            "--lift-cf-to-scf",
            "--canonicalize",
        ],
        "optimizer": {"sha256": "a" * 64, "bytes": 100},
        "artifacts": {
            "source.ttir": record(data["cfg-input.mlir"]),
            "lowered.mlir": record(data["cfg-lifted.mlir"]),
            "stderr.log": record(data["cfg-stderr.log"]),
        },
    }
    data["cfg-stage.json"] = json.dumps(stage).encode()
    provenance = {
        "spec": {"entry": "kernel"},
        "source_operations": graph["operations"],
        "original_blocks": graph["blocks"],
        "original_attributes": graph["attributes"],
        "control_flow_normalization": {
            "schema": "t2m.control-flow-normalization.v1",
            "entry": "kernel",
            "source_sha256": record(source)["sha256"],
            "grants_semantics": False,
            "artifacts": {name: record(raw) for name, raw in data.items()},
        },
    }
    return source, data, provenance


def test_complete_cfg_sources_are_copied_without_producer_or_evaluator_imports():
    source, data, provenance = fixture()
    assert checked_control_flow_sources(data.__getitem__, provenance, source) == data
    assert checked_control_flow_sources(data.__getitem__, {}, source) == {}


@pytest.mark.parametrize(
    "name",
    [
        "cfg-input.mlir",
        "cfg-lifted.mlir",
        "cfg-output.ttir",
        "cfg-source-graph.json",
        "cfg-stage.json",
        "cfg-stderr.log",
    ],
)
def test_changed_cfg_compiler_bytes_refuse(name):
    source, data, provenance = fixture()
    data[name] += b"changed"
    with pytest.raises(ValueError, match="compiler bytes"):
        checked_control_flow_sources(data.__getitem__, provenance, source)


@pytest.mark.parametrize("change", ["flags", "original", "graph", "authority", "optimizer"])
def test_rehashed_cfg_records_cannot_override_source_stage_links(change):
    source, data, provenance = fixture()
    report = provenance["control_flow_normalization"]
    stage = json.loads(data["cfg-stage.json"])
    if change == "flags":
        stage["flags"].remove("--lift-cf-to-scf")
    elif change == "optimizer":
        stage["optimizer"]["bytes"] = 0
    elif change == "graph":
        graph = json.loads(data["cfg-source-graph.json"])
        graph["operations"][0]["successors"] = []
        data["cfg-source-graph.json"] = json.dumps(graph).encode()
    elif change == "authority":
        report["grants_semantics"] = True
    elif change == "original":
        source += b"changed"
    data["cfg-stage.json"] = json.dumps(stage).encode()
    report["artifacts"] = {name: record(raw) for name, raw in data.items()}
    with pytest.raises(ValueError):
        checked_control_flow_sources(data.__getitem__, provenance, source)
