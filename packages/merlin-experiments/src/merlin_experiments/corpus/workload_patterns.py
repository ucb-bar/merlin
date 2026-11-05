"""Source graph pattern analysis for development capsule planning.

Consumes the value-free public graph projection. Patterns are proposals, not
legal fusion, placement or executable-capsule admission. No simulator runs here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from math import prod


def _bytes(edge):
    widths = {"int8": 1, "uint8": 1, "bool": 1, "int16": 2, "float16": 2,
              "bfloat16": 2, "int32": 4, "float32": 4, "int64": 8, "float64": 8}
    shape, width = edge.get("shape"), widths.get(edge.get("dtype"))
    if width is None or not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
        return None
    return prod(shape) * width


def _footprint(edges):
    # Distinct logical values, not distinct backing stores. Aliasing and views
    # must be established by the original IR before interpreting this as traffic.
    values = {edge["producer_value_id"]: edge for edge in edges}
    sizes = [_bytes(edge) for edge in values.values()]
    return {"logical_values": len(values), "known_logical_bytes": sum(x for x in sizes if x is not None),
            "unknown_size_values": sum(x is None for x in sizes), "physical_transfer_bytes": None}


def analyze(record: dict, *, max_patterns: int = 64) -> dict:
    """Count repeated connected pairs/triples across every node of each stage."""
    if type(max_patterns) is not int or not 1 <= max_patterns <= 256:
        raise ValueError("pattern report bound must be between one and 256")
    stages = []
    for graph in record["graphs"]:
        nodes = {node["id"]: node for node in graph["nodes"]}
        incoming, outgoing, successors = defaultdict(list), defaultdict(list), defaultdict(set)
        for edge in graph["edges"]:
            producer, consumer = edge["producer_node_id"], edge["consumer_node_id"]
            incoming[consumer].append(edge)
            outgoing[producer].append(edge)
            if producer in nodes and consumer in nodes:
                successors[producer].add(consumer)
        groups = {}
        for first in nodes:
            for second in sorted(successors[first]):
                paths = [(first, second)]
                paths.extend((first, second, third) for third in sorted(successors[second])
                             if third not in {first, second})
                for path in paths:
                    operations = tuple(nodes[name]["target"] for name in path)
                    item = groups.setdefault(operations, {"operations": list(operations),
                                                        "occurrences": 0, "examples": []})
                    item["occurrences"] += 1
                    if len(item["examples"]) >= 2:
                        continue
                    selected = set(path)
                    inputs = [edge for name in path for edge in incoming[name]
                              if edge["producer_node_id"] not in selected]
                    outputs = [edge for name in path for edge in outgoing[name]
                               if edge["consumer_node_id"] not in selected]
                    internal = [edge for name in path for edge in outgoing[name]
                                if edge["consumer_node_id"] in selected]
                    item["examples"].append({
                        "id": hashlib.sha256(json.dumps([record["source_sha256"], graph["stage"], path]).encode()).hexdigest(),
                        "nodes": list(path), "boundary_inputs": inputs, "boundary_outputs": outputs,
                        "input_footprint": _footprint(inputs), "output_footprint": _footprint(outputs),
                        "internal_footprint": _footprint(internal),
                        "execution_status": "not_admitted", "fusion_status": "requires_semantic_and_resource_checks",
                    })
        ordered = sorted(groups.values(), key=lambda item: (-item["occurrences"], item["operations"]))
        contractions = {"aten._int_mm.default", "aten.mm.default", "aten.matmul.default",
                        "aten.bmm.default", "aten.addmm.default", "aten.conv2d.default", "aten.convolution.default"}
        seams = Counter()
        for edge in graph["edges"]:
            producer, consumer = nodes.get(edge["producer_node_id"]), nodes.get(edge["consumer_node_id"])
            if producer and consumer and ((producer["target"] in contractions) != (consumer["target"] in contractions)):
                seams[(producer["target"], consumer["target"])] += 1
        stages.append({
            "stage": graph["stage"], "graph_sha256": graph.get("graph_sha256"),
            "nodes_analyzed": len(nodes), "connected_pattern_families": len(groups),
            "operation_counts": dict(Counter(node["target"] for node in nodes.values())),
            "repeated_patterns": [item for item in ordered if item["occurrences"] > 1][:max_patterns],
            "contraction_boundary_patterns": [{"producer": a, "consumer": b, "occurrences": count}
                                               for (a, b), count in seams.most_common(max_patterns)],
        })
    return {"schema": "merlin.corpus.workload_patterns.v1", "model": record["model"],
            "program": record["program"], "source_sha256": record["source_sha256"], "graphs": stages,
            "scope": "source-pattern and logical-boundary analysis; no placement, physical traffic or fusion claim",
            "host_owner": "Merlin shared host codegen and runtime",
            "accelerator_owner": "OOT target mapping, legality, instruction lowering and target scheduling",
            "required_cut_checks": ["original source region and complete live-outs", "aliasing, mutation and carried state",
                                    "layout and quantization semantics", "shared-pass legality and actual transformed IR",
                                    "buffer lifetime, capacity and resource overlap", "all-output numerical equivalence"],
            "stage_counts_are_alternative_representations": True}
