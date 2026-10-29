"""Index every original captured framework call without inventing target semantics.

This reuses the capture's original-graph accounting and producer lineage. The
index selects exact source bytes and value frontiers; it does not turn a node
into a standalone executable, prove a fusion, or equate another kernel's
quantizer with the captured recipe.
"""

from __future__ import annotations

import hashlib
import stat
from collections import defaultdict
from pathlib import Path

from merlin.capture.contraction_formats import ContractionFormatLimits, _json, _original
from merlin.common.jsonio import canonical_sha256
from merlin.common.pinned_files import PinnedFile, verify_files


def _references(value, path=""):
    if isinstance(value, dict):
        if "node_id" in value or "value_id" in value:
            yield path, value
        else:
            for key, member in value.items():
                yield from _references(member, f"{path}/{key}")
    elif isinstance(value, list):
        for index, member in enumerate(value):
            yield from _references(member, f"{path}/{index}")


def _load(path, max_bytes):
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode) or path.stat().st_size > max_bytes:
        raise ValueError("capture region input must be a bounded regular file")
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raise ValueError("capture region input exceeds its byte bound")
    pin = PinnedFile(path, hashlib.sha256(raw).hexdigest())
    return raw, pin


def _record(pin):
    return {"path": str(pin.path), "sha256": pin.sha256}


def capture_graph_regions(capture: Path, *, max_bytes: int = 64 * 1024 * 1024) -> dict:
    """Return all original call sites, source frontiers and recorded lowering lineage.

    No framework import, model construction, weight conversion or execution is
    performed. A selected downstream extractor must still derive standalone SSA
    and the exact numerical region. All original calls remain in the index,
    including noncontractions and calls without any lowering correspondence.
    """
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("capture region byte bound must be a positive integer")
    capture = Path(capture).resolve()
    limits = ContractionFormatLimits(file_bytes=max_bytes, metadata_items=max_bytes)
    files = {}
    documents = {}
    for name in (
        "frontend-trace.json",
        "meta.json",
        "model.mlir",
        "weights.safetensors",
        "weights.safetensors.manifest.json",
    ):
        raw, pin = _load(capture / name, max_bytes)
        files[name] = pin
        if name.endswith(".json"):
            documents[name] = _json(raw, limits)
    trace, meta = documents["frontend-trace.json"], documents["meta.json"]
    if trace.get("schema") != "m2m.frontend_trace.v1" or trace.get("status") != "complete":
        raise ValueError("capture region index requires a complete frontend trace")
    if meta.get("frontend_trace", {}).get("sha256") != files["frontend-trace.json"].sha256:
        raise ValueError("capture metadata does not pin the selected frontend trace")
    if trace.get("mlir", {}).get("sha256") != files["model.mlir"].sha256:
        raise ValueError("frontend trace does not pin the selected source module")
    graphs = trace.get("graphs", {})
    original = graphs.get("original", {})
    # The existing complete-original validator owns graph digest, node/value
    # identity, topological argument binding and complete call accounting.
    _original(original, limits)
    nodes = {node["id"]: node for node in original["nodes"]}
    values = {value["id"]: (node, value) for node in nodes.values() for value in node["results"]}
    callers = {node["id"] for node in nodes.values() if node["op"] in ("call_function", "call_method", "call_module")}
    lineage = defaultdict(lambda: defaultdict(list))
    other_nodes = {}
    for stage, graph in graphs.items():
        if stage == "original":
            continue
        if graph.get("schema") != "m2m.frontend_graph.v1" or graph.get("status") != "complete":
            raise ValueError("frontend lineage requires a complete captured graph")
        if graph.get("sha256") != canonical_sha256({k: v for k, v in graph.items() if k != "sha256"}):
            raise ValueError("frontend lineage graph digest differs from its selected bytes")
        if not isinstance(graph.get("nodes"), list) or len(graph["nodes"]) > ContractionFormatLimits().graph_nodes:
            raise ValueError("frontend lineage graph exceeds its node bound")
        members = {node["id"]: node for node in graph["nodes"]}
        if len(members) != len(graph["nodes"]):
            raise ValueError("frontend lineage graph repeats node identities")
        other_nodes[stage] = members
        for node in members.values():
            for origin in node.get("origin_node_ids", ()):
                if origin in nodes:
                    lineage[origin][stage].append(node["id"])
    mlir_lineage = defaultdict(list)
    operations = trace["mlir"].get("operations", [])
    for operation in operations:
        for origin in operation.get("origin_node_ids", ()):
            if origin in nodes:
                mlir_lineage[origin].append(operation["ordinal"])
    consumers = defaultdict(list)
    for node in nodes.values():
        for path, link in _references({"args": node.get("args", []), "kwargs": node.get("kwargs", {})}):
            consumers[link["value_id"]].append({"node_id": node["id"], "argument_path": path})
    manifest = documents["weights.safetensors.manifest.json"]
    buffers = {
        row["weight"]: {"entry_argument_index": int(index), **row}
        for index, row in manifest.items()
        if isinstance(row, dict) and "weight" in row
    }
    regions = []
    for node in nodes.values():
        if node["id"] not in callers:
            continue
        inputs = {}
        for path, link in _references({"args": node.get("args", []), "kwargs": node.get("kwargs", {})}):
            producer, value = values[link["value_id"]]
            row = inputs.setdefault(
                value["id"],
                {
                    "producer_node_id": producer["id"],
                    "value": value,
                    "input_kind": producer.get("input_kind"),
                    "input_target": producer.get("input_target"),
                    "argument_paths": [],
                },
            )
            row["argument_paths"].append(path)
        constant_bindings = {}
        for stage, ids in lineage[node["id"]].items():
            members = other_nodes[stage]
            selected = set(ids)
            for identity in ids:
                selected.update(link.get("node_id") for _, link in _references(members[identity]))
            for identity in sorted(selected, key=str):
                member = members.get(identity, {})
                name = member.get("input_target")
                if name in buffers:
                    constant_bindings[name] = {"stage": stage, "node_id": identity, "manifest": buffers[name]}
        regions.append(
            {
                "region_id": node["id"],
                "ordinal": node["ordinal"],
                "operation": node,
                "inputs": list(inputs.values()),
                "outputs": node["results"],
                "output_consumers": {v["id"]: consumers[v["id"]] for v in node["results"]},
                "dependency_region_ids": list(
                    dict.fromkeys(
                        row["producer_node_id"] for row in inputs.values() if row["producer_node_id"] in callers
                    )
                ),
                "lineage": {"graphs": dict(lineage[node["id"]]), "mlir_operation_ordinals": mlir_lineage[node["id"]]},
                "buffer_bindings": list(constant_bindings.values()),
                "source_binding": {
                    "module": _record(files["model.mlir"]),
                    "standalone": False,
                    "original_node_ids": [node["id"]],
                },
                "source_region": None,
                "reference_ids": [],
                "gaps": [
                    {
                        "kind": "source_region_extraction_gap",
                        "reason": "Standalone SSA and numerical frontier require an executed source extractor.",
                    },
                    {
                        "kind": "reference_alignment_gap",
                        "reason": "No expert instruction region correspondence or quantizer equality is established.",
                    },
                ],
            }
        )
    verify_files(tuple(files.values()))
    return {
        "schema": "merlin.capture_graph_regions.v1",
        "source_module": _record(files["model.mlir"]),
        "frontend_trace": _record(files["frontend-trace.json"]),
        "source_pins": {k: _record(v) for k, v in files.items()},
        "graph": {k: original[k] for k in ("graph_id", "sha256", "stage", "call_count", "by_target") if k in original},
        "lineage_graphs": {
            stage: {k: graph[k] for k in ("graph_id", "sha256", "stage", "call_count") if k in graph}
            for stage, graph in graphs.items()
            if stage != "original"
        },
        "recipe": meta.get("recipe"),
        "recipe_sha256": meta.get("recipe_sha256"),
        "quantization_stats": meta.get("quantization_stats"),
        "regions": regions,
        "not_established": [
            "expert_region_equivalence",
            "fusion_equivalence",
            "standalone_source_execution",
            "numerical_equivalence",
            "target_execution",
            "performance",
        ],
    }
