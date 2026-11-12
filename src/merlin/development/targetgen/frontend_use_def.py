"""Replay original frontend SSA uses before deriving shape-free interactions.

The existing operation-roster reader remains compatible with historical traces.
This stricter reader requires the complete serialized argument and edge roster.
It establishes source relations, not operator effects, aliasing or target support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from merlin.development.targetgen.frontend_trace import original_operation_semantics

_CALLS = {"call_function", "call_method", "call_module"}
_NODES = {*_CALLS, "placeholder", "get_attr", "output"}


def _references(value, path):
    if isinstance(value, dict):
        if set(value) & {"node_id", "value_id"}:
            if set(value) != {"node_id", "value_id"} or any(
                not isinstance(item, str) or not item for item in value.values()
            ):
                raise ValueError("use-def arguments require complete concrete value references")
            yield path, value["node_id"], value["value_id"]
        else:
            for key, child in value.items():
                if not isinstance(key, str) or not key or "/" in key:
                    raise ValueError("use-def argument mappings require unambiguous path keys")
                yield from _references(child, path + "/" + key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _references(child, path + "/" + str(index))
    elif value is not None and type(value) not in {bool, int, float, str}:
        raise ValueError("use-def argument serialization is unsupported")


@dataclass(frozen=True)
class OriginalUseDef:
    graph_sha256: str
    operations: tuple[str, ...]
    interaction_classes: tuple[str, ...]
    witnesses_json: str

    def public_semantics(self):
        """Exclude node names, original shapes, topology size and frequencies."""
        return {
            "graph_sha256": self.graph_sha256,
            "operation_semantics": list(self.operations),
            "interaction_classes": list(self.interaction_classes),
        }

    def witnesses(self):
        """Private replay information is never a generated topology template."""
        return json.loads(self.witnesses_json)


def original_use_def_semantics(trace: dict) -> OriginalUseDef:
    """Reconstruct every argument use and require exact bidirectional edges.

    Every result belongs to one producer. All producer references precede their
    consumer, including output publication; there is one final output node.
    Unknown serialization or missing ownership refuses rather than guessing.
    """
    graph_sha256, operations = original_operation_semantics(trace)
    graph = trace["graphs"]["original"]
    nodes, edges = graph["nodes"], graph.get("edges")
    if not isinstance(edges, list):
        raise ValueError("use-def requires an explicit complete edge roster")
    indexed, values, positions = {}, {}, {}
    for index, node in enumerate(nodes):
        if node.get("op") not in _NODES or type(node.get("ordinal")) is not int or node["ordinal"] != index:
            raise ValueError("use-def requires supported nodes in exact source order")
        if not isinstance(node.get("args"), list) or not isinstance(node.get("kwargs"), dict):
            raise ValueError("use-def requires complete original args and kwargs")
        results = node.get("results")
        if not isinstance(results, list):
            raise ValueError("use-def requires an explicit producer result roster")
        indexed[node["id"]], positions[node["id"]] = node, index
        for result in results:
            if not isinstance(result, dict) or not isinstance(result.get("id"), str) or not result["id"]:
                raise ValueError("use-def producer result has no concrete identity")
            if result["id"] in values:
                raise ValueError("use-def producer result identity is duplicated")
            values[result["id"]] = node["id"], result
    outputs = [node["id"] for node in nodes if node["op"] == "output"]
    if outputs != [nodes[-1]["id"]]:
        raise ValueError("use-def requires one final output publication node")
    reconstructed, uses = {}, {}
    for node in nodes:
        for root in ("args", "kwargs"):
            for path, producer, value in _references(node[root], root):
                if value not in values or values[value][0] != producer:
                    raise ValueError("use-def argument value disagrees with its exact producer owner")
                if positions[producer] >= positions[node["id"]]:
                    raise ValueError("use-def argument producer does not precede its consumer")
                key = node["id"], path
                if key in reconstructed:
                    raise ValueError("use-def argument path is duplicated")
                reconstructed[key] = producer, value
                uses.setdefault(value, []).append((node["id"], path))
    recorded = {}
    for edge in edges:
        path = edge.get("argument_path")
        if not isinstance(path, str) or not path:
            raise ValueError("use-def edge has no exact argument path")
        key = edge["consumer_node_id"], path
        if key in recorded:
            raise ValueError("use-def edge argument path is duplicated")
        producer, value = edge["producer_node_id"], edge.get("producer_value_id")
        if value not in values or values[value][0] != producer:
            raise ValueError("use-def edge value disagrees with its exact producer owner")
        result = values[value][1]
        if result.get("kind") != edge.get("value_kind"):
            raise ValueError("use-def edge value kind disagrees with its producer result")
        recorded[key] = producer, value
    if recorded != reconstructed:
        raise ValueError("use-def edge roster differs from reconstructed original args and kwargs")
    witnesses = []
    for value, consumers in uses.items():
        producer, result = values[value]
        if result.get("kind") != "tensor":
            continue
        calls = sorted({consumer for consumer, _ in consumers if indexed[consumer]["op"] in _CALLS})
        published = sorted({consumer for consumer, _ in consumers if indexed[consumer]["op"] == "output"})
        kinds = []
        if len(calls) > 1:
            if indexed[producer]["op"] in {"placeholder", "get_attr"}:
                kinds.append("shared_input_multiple_consumers")
            elif indexed[producer]["op"] in _CALLS:
                kinds.append("shared_producer_multiple_consumers")
        if calls and published and indexed[producer]["op"] in _CALLS:
            kinds.append("publication_and_further_use")
        for kind in kinds:
            witnesses.append(
                {"kind": kind, "value": value, "producer": producer, "consumers": calls, "publications": published}
            )
    return OriginalUseDef(
        graph_sha256,
        operations,
        tuple(sorted({row["kind"] for row in witnesses})),
        json.dumps(witnesses, sort_keys=True, separators=(",", ":")),
    )
