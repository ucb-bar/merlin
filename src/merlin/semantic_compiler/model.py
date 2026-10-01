"""Immutable source semantics and the engine-neutral kernel boundary.

Only JSON scalar attributes are admitted. In particular, a contract cannot
smuggle executable Python into an instruction rule or numerical policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

JsonScalar = str | int | float | bool | None


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class TensorType:
    shape: tuple[int, ...]
    dtype: str
    numerical_policy: str

    def __post_init__(self) -> None:
        if not self.shape or any(not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in self.shape):
            raise ValueError("tensor shape must have positive static dimensions")
        if any(not isinstance(value, str) or not value for value in (self.dtype, self.numerical_policy)):
            raise ValueError("dtype and numerical policy must be explicit")

    def record(self) -> dict[str, Any]:
        return {"shape": list(self.shape), "dtype": self.dtype, "numerical_policy": self.numerical_policy}

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> TensorType:
        if set(row) != {"shape", "dtype", "numerical_policy"}:
            raise ValueError("tensor type has missing or unknown fields")
        if not isinstance(row["shape"], list):
            raise ValueError("tensor shape must be a list")
        return cls(tuple(row["shape"]), row["dtype"], row["numerical_policy"])


@dataclass(frozen=True)
class IndexMap:
    """Restricted integer-affine map from logical loop indices to one tensor."""

    loop_rank: int
    coefficients: tuple[tuple[int, ...], ...]
    offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.loop_rank) is not int or self.loop_rank <= 0:
            raise ValueError("index map needs a positive logical loop rank")
        if not self.coefficients or len(self.coefficients) != len(self.offsets):
            raise ValueError("index map needs one offset per result dimension")
        if any(len(row) != self.loop_rank or any(type(value) is not int for value in row) for row in self.coefficients):
            raise ValueError("index map coefficients have invalid rank or type")
        if any(type(value) is not int for value in self.offsets):
            raise ValueError("index map offsets must be integers")

    def apply(self, indices: tuple[int, ...]) -> tuple[int, ...]:
        if len(indices) != self.loop_rank or any(type(value) is not int for value in indices):
            raise ValueError("logical index has wrong rank or type")
        return tuple(sum(coefficient * index for coefficient, index in zip(row, indices)) + offset
                     for row, offset in zip(self.coefficients, self.offsets))

    def record(self) -> dict[str, Any]:
        return {
            "loop_rank": self.loop_rank,
            "coefficients": [list(row) for row in self.coefficients],
            "offsets": list(self.offsets),
        }

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> IndexMap:
        if set(row) != {"loop_rank", "coefficients", "offsets"}:
            raise ValueError("index map has missing or unknown fields")
        return cls(row["loop_rank"], tuple(tuple(result) for result in row["coefficients"]), tuple(row["offsets"]))


@dataclass(frozen=True)
class SemanticNode:
    id: str
    op: str
    inputs: tuple[str, ...]
    type: TensorType
    attrs: tuple[tuple[str, JsonScalar], ...] = ()
    effect: str = "pure"
    index_maps: tuple[IndexMap, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value for value in (self.id, self.op)):
            raise ValueError("node id and operation must be nonempty")
        if any(not isinstance(child, str) or not child for child in self.inputs):
            raise ValueError("node operands must be nonempty identifiers")
        if len({key for key, _ in self.attrs}) != len(self.attrs):
            raise ValueError("duplicate semantic attribute")
        if any(not isinstance(key, str) or not key for key, _ in self.attrs):
            raise ValueError("semantic attribute names must be nonempty strings")
        for _, value in self.attrs:
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError("semantic attributes must be JSON scalars")
            _json_bytes(value)
        if self.effect not in {"pure", "input", "constant", "state"}:
            raise ValueError("unknown effect class")
        if self.op == "input" and (self.inputs or self.effect != "input"):
            raise ValueError("input nodes must have input effect and no operands")
        if self.op == "constant" and (self.inputs or self.effect != "constant"):
            raise ValueError("constant nodes must have constant effect and no operands")
        if self.effect != "pure" and self.op not in {"input", "constant"}:
            raise ValueError("stateful computation requires an explicit state/token interface")
        if self.index_maps and len(self.index_maps) != len(self.inputs) + 1:
            raise ValueError("index maps need one entry per operand and one result")
        if self.index_maps and len({index_map.loop_rank for index_map in self.index_maps}) != 1:
            raise ValueError("all index maps must share one logical loop domain")

    def semantic_key(self) -> str:
        """Exclude source id and provenance, which do not change pure semantics."""
        return sha256(_json_bytes((
            self.op, self.type.record(), sorted(self.attrs), [index_map.record() for index_map in self.index_maps],
        ))).hexdigest()

    def record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "op": self.op,
            "inputs": list(self.inputs),
            "type": self.type.record(),
            "attrs": dict(self.attrs),
            "effect": self.effect,
            "index_maps": [index_map.record() for index_map in self.index_maps],
        }

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> SemanticNode:
        if set(row) != {"id", "op", "inputs", "type", "attrs", "effect", "index_maps"}:
            raise ValueError("semantic node has missing or unknown fields")
        if not isinstance(row["inputs"], list) or not isinstance(row["attrs"], dict):
            raise ValueError("semantic operands and attributes have invalid structure")
        return cls(
            row["id"],
            row["op"],
            tuple(row["inputs"]),
            TensorType.from_record(row["type"]),
            tuple(sorted(row["attrs"].items())),
            row["effect"],
            tuple(IndexMap.from_record(index_map) for index_map in row["index_maps"]),
        )


@dataclass(frozen=True)
class KernelRequest:
    nodes: tuple[SemanticNode, ...]
    outputs: tuple[str, ...]
    output_storages: tuple[str, ...]
    input_storages: tuple[tuple[str, str], ...]
    target_identity: str
    lowering_policy: str = "strict-native"
    source_identity: str = ""
    _by_id: dict[str, SemanticNode] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        seen: dict[str, SemanticNode] = {}
        for node in self.nodes:
            if node.id in seen:
                raise ValueError(f"duplicate node id: {node.id}")
            if any(child not in seen for child in node.inputs):
                raise ValueError(f"node {node.id} is not in topological order")
            if node.index_maps:
                tensor_ranks = [len(seen[child].type.shape) for child in node.inputs]
                tensor_ranks.append(len(node.type.shape))
                if any(len(index_map.coefficients) != rank for index_map, rank in zip(node.index_maps, tensor_ranks)):
                    raise ValueError(f"node {node.id} index-map result rank differs from tensor rank")
            seen[node.id] = node
        if not self.outputs or len(self.outputs) != len(self.output_storages):
            raise ValueError("all ordered outputs need a required storage")
        if any(not isinstance(value, str) or not value for value in (*self.outputs, *self.output_storages)):
            raise ValueError("outputs and storages must be nonempty identifiers")
        if any(output not in seen for output in self.outputs):
            raise ValueError("unknown output")
        boundary = dict(self.input_storages)
        if len(boundary) != len(self.input_storages):
            raise ValueError("duplicate input boundary")
        if any(
            not isinstance(key, str) or not isinstance(value, str) or not key or not value
            for key, value in self.input_storages
        ):
            raise ValueError("input boundary names and storages must be nonempty strings")
        if {n.id for n in self.nodes if n.effect == "input"} != set(boundary):
            raise ValueError("each input needs exactly one declared boundary representation")
        if not isinstance(self.target_identity, str) or not self.target_identity or self.lowering_policy not in {
            "strict-native", "hybrid", "diagnostic",
        }:
            raise ValueError("target identity and valid lowering policy are required")
        if not isinstance(self.source_identity, str):
            raise ValueError("source identity must be a string")
        object.__setattr__(self, "_by_id", seen)

    def node(self, node_id: str) -> SemanticNode:
        return self._by_id[node_id]

    def record(self) -> dict[str, Any]:
        return {
            "schema": "merlin.semantic_kernel.v1",
            "nodes": [node.record() for node in self.nodes],
            "outputs": list(self.outputs),
            "output_storages": list(self.output_storages),
            "input_storages": dict(self.input_storages),
            "target_identity": self.target_identity,
            "lowering_policy": self.lowering_policy,
            "source_identity": self.source_identity,
        }

    def digest(self) -> str:
        return sha256(_json_bytes(self.record())).hexdigest()

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> KernelRequest:
        expected = {
            "schema", "nodes", "outputs", "output_storages", "input_storages",
            "target_identity", "lowering_policy", "source_identity",
        }
        if set(row) != expected or row["schema"] != "merlin.semantic_kernel.v1":
            raise ValueError("unexpected semantic kernel schema or fields")
        if not isinstance(row["nodes"], list) or not isinstance(row["input_storages"], dict):
            raise ValueError("semantic kernel graph or boundary has invalid structure")
        return cls(
            nodes=tuple(SemanticNode.from_record(node) for node in row["nodes"]),
            outputs=tuple(row["outputs"]),
            output_storages=tuple(row["output_storages"]),
            input_storages=tuple(sorted(row["input_storages"].items())),
            target_identity=row["target_identity"],
            lowering_policy=row["lowering_policy"],
            source_identity=row["source_identity"],
        )
