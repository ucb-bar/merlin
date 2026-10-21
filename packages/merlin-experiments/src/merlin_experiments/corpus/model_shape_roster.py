"""Value-free graph shapes and connected-region proposals for a new source epoch.

This projection admits no executable capsule and exposes no captured arrays,
weights, numerical outputs or original filesystem paths. Exact graph identities
remain attached so separately admitted cuts can be joined to their source.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "merlin.corpus.model_shape_roster.v1"
_NODE_KEYS = ("id", "op", "target", "ordinal", "origin_node_ids")
_VALUE_KEYS = ("id", "kind", "dtype", "shape", "stride", "layout", "storage_dtype", "compute_dtype")
_EDGE_KEYS = ("producer_node_id", "producer_value_id", "consumer_node_id", "argument_path", "dtype", "shape", "value_kind")


def project(trace: dict, *, model: str, program: str, source_sha256: str, max_region_nodes: int = 8) -> dict:
    """Project explicit metadata fields; tensor payload and arbitrary args never cross."""
    if type(max_region_nodes) is not int or not 2 <= max_region_nodes <= 32:
        raise ValueError("connected-region budget must be between two and 32 nodes")
    if not isinstance(trace.get("graphs"), dict) or not trace["graphs"]:
        raise ValueError("selected capture has no frontend graph")
    graphs = []
    for stage, graph in sorted(trace["graphs"].items()):
        nodes = []
        for node in graph["nodes"]:
            if node.get("op") != "call_function":
                continue
            row = {key: node[key] for key in _NODE_KEYS if key in node}
            row["results"] = [{key: value[key] for key in _VALUE_KEYS if key in value}
                              for value in node.get("results", [])]
            nodes.append(row)
        ids = {node["id"] for node in nodes}
        if len(ids) != len(nodes):
            raise ValueError("graph node identity is not unique")
        edges = [{key: edge[key] for key in _EDGE_KEYS if key in edge}
                 for edge in graph.get("edges", []) if edge.get("consumer_node_id") in ids]
        by_id = {node["id"]: node for node in nodes}
        successors = {name: [] for name in ids}
        incoming, outgoing = {}, {}
        for edge in edges:
            producer, consumer = edge["producer_node_id"], edge["consumer_node_id"]
            incoming.setdefault(consumer, []).append(edge)
            outgoing.setdefault(producer, []).append(edge)
            if producer in successors and consumer not in successors[producer]:
                successors[producer].append(consumer)
        regions = []
        seen = set()
        for node in nodes:
            # Every connected neighborhood is a proposal; no hand-selected model
            # op determines compiler placement, legal fusion or numerical policy.
            selected, pending = [], [node["id"]]
            while pending and len(selected) < max_region_nodes:
                current = pending.pop(0)
                if current in selected:
                    continue
                selected.append(current)
                pending.extend(successors[current])
            key = tuple(sorted(selected))
            if len(selected) < 2 or key in seen:
                continue
            seen.add(key)
            region_ids = set(selected)
            boundary_inputs = [edge for name in selected for edge in incoming.get(name, [])
                               if edge["producer_node_id"] not in region_ids]
            boundary_outputs = [edge for name in selected for edge in outgoing.get(name, [])
                                if edge["consumer_node_id"] not in region_ids]
            regions.append({"id": hashlib.sha256(json.dumps([source_sha256, stage, key]).encode()).hexdigest(),
                            "nodes": selected, "operations": [by_id[name]["target"] for name in selected],
                            "boundary_inputs": boundary_inputs, "boundary_outputs": boundary_outputs,
                            "execution_status": "not_admitted", "numerical_contract": "requires_cut_admission"})
            if len(regions) >= 64:
                break  # Complete node/edge roster remains available for other cuts.
        graphs.append({"stage": stage, "graph_sha256": graph.get("sha256"), "nodes": nodes,
                       "edges": edges, "connected_region_proposals": regions})
    return {"model": model, "program": program, "source_sha256": source_sha256, "graphs": graphs,
            "role": "public_development_source_shapes", "historical_holdout_claim": False,
            "execution_status": "not_executed", "full_model_engine": "spike", "full_model_stage": "final"}


def compact_projection(record: dict) -> dict:
    """Keep every distinct operation/input/output shape and its source occurrences."""
    graphs = []
    for graph in record["graphs"]:
        incoming = {}
        for edge in graph["edges"]:
            incoming.setdefault(edge["consumer_node_id"], []).append(
                {key: edge[key] for key in ("argument_path", "dtype", "shape", "value_kind") if key in edge})
        groups = {}
        for node in graph["nodes"]:
            signature = {"operation": node["target"], "inputs": incoming.get(node["id"], []),
                         "results": [{key: value for key, value in result.items() if key != "id"}
                                     for result in node["results"]]}
            key = json.dumps(signature, sort_keys=True)
            group = groups.setdefault(key, {**signature, "source_node_ids": []})
            group["source_node_ids"].append(node["id"])
        graphs.append({"stage": graph["stage"], "graph_sha256": graph["graph_sha256"],
                       "node_count": len(graph["nodes"]), "shape_signatures": list(groups.values()),
                       "connected_region_proposals": graph["connected_region_proposals"]})
    return {**record, "graphs": graphs, "representation": "complete_shape_signatures_and_source_occurrences"}


def prepare(selections: list[dict], destination: Path, *, compact: bool = False) -> dict:
    """Read explicit pinned traces only; the caller owns the source-role transition."""
    if destination.exists():
        raise ValueError("shape roster needs a fresh destination")
    destination.mkdir(parents=True)
    records, sources = [], []
    for selection in selections:
        path = Path(selection["path"])
        if path != path.resolve() or not path.is_file():
            raise ValueError("shape source must be an ordinary canonical file")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != selection["sha256"]:
            raise ValueError("shape source bytes changed")
        record = project(json.loads(raw), model=selection["model"], program=selection["program"], source_sha256=digest)
        graph_counts = [{"stage": row["stage"], "nodes": len(row["nodes"]),
                         "connected_regions": len(row["connected_region_proposals"])} for row in record["graphs"]]
        if compact:
            record = compact_projection(record)
        name = f"{len(records):03d}.json"
        (destination / name).write_text(json.dumps(record, indent=2) + "\n")
        records.append({"model": selection["model"], "program": selection["program"], "member": name,
                        "sha256": hashlib.sha256((destination / name).read_bytes()).hexdigest(),
                        "source_sha256": digest,
                        "graphs": graph_counts})
        sources.append(selection)
    summary = {"schema": SCHEMA, "role": "new_epoch_public_development", "programs": records,
               "scope": "Source shapes and proposed connected cuts; no executable or numerical admission"}
    (destination / "roster.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {"public": summary, "private_source_pins": sources}
