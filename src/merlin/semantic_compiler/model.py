"""Immutable source semantics and the engine-neutral kernel boundary.

Only JSON scalar attributes are admitted. In particular, a contract cannot
smuggle executable Python into an instruction rule or numerical policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from math import prod
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
        if not isinstance(self.shape, tuple) or any(
            not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in self.shape
        ):
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
        # A rank-zero tensor has an empty result tuple, e.g. a vector sum's
        # accumulator map. The surrounding loop domain still has positive rank.
        if len(self.coefficients) != len(self.offsets):
            raise ValueError("index map needs one offset per result dimension")
        if any(len(row) != self.loop_rank or any(type(value) is not int for value in row) for row in self.coefficients):
            raise ValueError("index map coefficients have invalid rank or type")
        if any(type(value) is not int for value in self.offsets):
            raise ValueError("index map offsets must be integers")

    def apply(self, indices: tuple[int, ...]) -> tuple[int, ...]:
        if len(indices) != self.loop_rank or any(type(value) is not int for value in indices):
            raise ValueError("logical index has wrong rank or type")
        return tuple(
            sum(coefficient * index for coefficient, index in zip(row, indices)) + offset
            for row, offset in zip(self.coefficients, self.offsets)
        )

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


def pointwise_identity_maps(rank: int, inputs: int) -> tuple[IndexMap, ...]:
    """Canonical logical maps for an elementwise tensor operation."""
    if type(rank) is not int or rank <= 0 or type(inputs) is not int or inputs <= 0:
        raise ValueError("pointwise maps need positive rank and input count")
    identity = IndexMap(
        rank,
        tuple(tuple(int(axis == result) for axis in range(rank)) for result in range(rank)),
        (0,) * rank,
    )
    return (identity,) * (inputs + 1)


@dataclass(frozen=True)
class SemanticNode:
    """Typed source node.

    Input nodes default to retained storage. ``input_retention=reusable`` is an
    explicit boundary promise that their storage may be overwritten after the
    last qualified read, including a delayed execution-time read.
    """

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
        input_retention = dict(self.attrs).get("input_retention", "preserve")
        if self.op == "input" and input_retention not in {"preserve", "reusable"}:
            raise ValueError("input retention must be preserve or reusable")
        if self.op != "input" and "input_retention" in dict(self.attrs):
            raise ValueError("input retention applies only to input nodes")
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
        return sha256(
            _json_bytes(
                (
                    self.op,
                    self.type.record(),
                    sorted(self.attrs),
                    [index_map.record() for index_map in self.index_maps],
                )
            )
        ).hexdigest()

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
class ConstantBinding:
    """Declared exact tensor bytes; target packing remains a separate obligation."""

    node_id: str
    encoding: str
    data_hex: str

    def __post_init__(self) -> None:
        if not self.node_id or self.encoding not in {
            "i1-raw",
            "i8",
            "i32-le",
            "i64-le",
            "fp8_e4m3-raw",
            "bf16-le-raw",
            "f32-le-raw",
            "f64-le-raw",
        }:
            raise ValueError("constant needs a named, admitted byte encoding")
        if len(self.data_hex) % 2 or any(char not in "0123456789abcdef" for char in self.data_hex):
            raise ValueError("constant bytes must be canonical lowercase hex")

    def validate_type(self, type_: TensorType) -> None:
        expected = {
            "i1-raw": ("i1", 1),
            "i8": ("i8", 1),
            "i32-le": ("i32", 4),
            "i64-le": ("i64", 8),
            "fp8_e4m3-raw": ("fp8_e4m3", 1),
            "bf16-le-raw": ("bf16", 2),
            "f32-le-raw": ("f32", 4),
            "f64-le-raw": ("f64", 8),
        }[self.encoding]
        if type_.dtype != expected[0] or len(self.data_hex) // 2 != prod(type_.shape) * expected[1]:
            raise ValueError("constant bytes differ from declared tensor type")
        if self.encoding == "i64-le" and type_.numerical_policy != "signed-i64":
            raise ValueError("signed i64 constant needs its exact numerical policy")
        if self.encoding == "i1-raw" and (
            type_.numerical_policy != "boolean" or any(value not in (0, 1) for value in bytes.fromhex(self.data_hex))
        ):
            raise ValueError("Boolean constant needs canonical zero/one bytes and its Boolean policy")

    def record(self) -> dict[str, str]:
        return {"node_id": self.node_id, "encoding": self.encoding, "data_hex": self.data_hex}

    @classmethod
    def from_record(cls, row: dict[str, str]) -> ConstantBinding:
        if set(row) != {"node_id", "encoding", "data_hex"}:
            raise ValueError("constant binding has missing or unknown fields")
        return cls(**row)


@dataclass(frozen=True)
class KernelRequest:
    nodes: tuple[SemanticNode, ...]
    outputs: tuple[str, ...]
    output_storages: tuple[str, ...]
    input_storages: tuple[tuple[str, str], ...]  # Ordered boundary ports, not a set.
    target_identity: str
    lowering_policy: str = "strict-native"
    source_identity: str = ""
    constants: tuple[ConstantBinding, ...] = ()
    # Extents are measured in the named storage bank's address unit, not tensor
    # elements. Legacy v2 requests use one unit per boundary value.
    physical_extents: tuple[tuple[str, int], ...] = ()
    # A qualified no-copy view needs the element order at every boundary.
    # Empty fields retain older, layout-unknown requests, which cannot supply
    # a source for a layout-dependent view.
    boundary_layouts: tuple[tuple[str, str], ...] = ()
    output_layouts: tuple[str, ...] = ()
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
        if (
            not isinstance(self.target_identity, str)
            or not self.target_identity
            or self.lowering_policy
            not in {
                "strict-native",
                "hybrid",
                "diagnostic",
            }
        ):
            raise ValueError("target identity and valid lowering policy are required")
        if not isinstance(self.source_identity, str):
            raise ValueError("source identity must be a string")
        bindings = {binding.node_id: binding for binding in self.constants}
        if len(bindings) != len(self.constants) or set(bindings) != {
            node.id for node in self.nodes if node.effect == "constant"
        }:
            raise ValueError("each semantic constant needs exactly one declared byte binding")
        for node_id, binding in bindings.items():
            binding.validate_type(seen[node_id].type)
        extents = dict(self.physical_extents)
        if self.physical_extents and (
            len(extents) != len(self.physical_extents)
            or set(extents) != set(boundary) | set(bindings)
            or any(type(extent) is not int or extent <= 0 for extent in extents.values())
        ):
            raise ValueError("physical extents must cover every input and constant in storage units")
        layouts = dict(self.boundary_layouts)
        if self.boundary_layouts and (
            len(layouts) != len(self.boundary_layouts)
            or set(layouts) != set(boundary) | set(bindings)
            or any(not isinstance(layout, str) or not layout for layout in layouts.values())
        ):
            raise ValueError("physical layouts must cover every input and constant")
        if self.output_layouts and (
            len(self.output_layouts) != len(self.outputs)
            or any(not isinstance(layout, str) or not layout for layout in self.output_layouts)
        ):
            raise ValueError("physical output layouts must cover every ordered output")
        if bool(self.boundary_layouts) != bool(self.output_layouts):
            raise ValueError("physical boundary and output layouts must be declared together")
        if self.boundary_layouts and not self.physical_extents:
            raise ValueError("physical layouts need explicit boundary extents")
        object.__setattr__(self, "physical_extents", tuple(sorted(extents.items())))
        object.__setattr__(self, "boundary_layouts", tuple(sorted(layouts.items())))
        object.__setattr__(self, "_by_id", seen)

    def node(self, node_id: str) -> SemanticNode:
        return self._by_id[node_id]

    def physical_extent(self, node_id: str) -> int:
        if node_id not in dict(self.input_storages) and node_id not in {binding.node_id for binding in self.constants}:
            raise ValueError("physical extent applies only to boundary values")
        return dict(self.physical_extents).get(node_id, 1)

    def boundary_layout(self, node_id: str) -> str:
        return dict(self.boundary_layouts).get(node_id, "")

    def record(self) -> dict[str, Any]:
        # Older records only contain a storage map, whose canonical key order is
        # lexical. Preserve an explicit calling convention when it differs.
        ordered_inputs = self.input_storages != tuple(sorted(self.input_storages))
        record = {
            "schema": (
                "merlin.semantic_kernel.v5"
                if ordered_inputs
                else "merlin.semantic_kernel.v4"
                if self.boundary_layouts
                else "merlin.semantic_kernel.v3"
                if self.physical_extents
                else "merlin.semantic_kernel.v2"
            ),
            "nodes": [node.record() for node in self.nodes],
            "outputs": list(self.outputs),
            "output_storages": list(self.output_storages),
            "input_storages": dict(self.input_storages),
            "target_identity": self.target_identity,
            "lowering_policy": self.lowering_policy,
            "source_identity": self.source_identity,
            "constants": [binding.record() for binding in sorted(self.constants, key=lambda item: item.node_id)],
        }
        if self.physical_extents:
            record["physical_extents"] = dict(sorted(self.physical_extents))
        if self.boundary_layouts:
            record["boundary_layouts"] = dict(self.boundary_layouts)
            record["output_layouts"] = list(self.output_layouts)
        if ordered_inputs:
            record.update(
                input_order=[key for key, _ in self.input_storages],
                physical_extents=dict(self.physical_extents),
                boundary_layouts=dict(self.boundary_layouts),
                output_layouts=list(self.output_layouts),
            )
        return record

    def digest(self) -> str:
        return sha256(_json_bytes(self.record())).hexdigest()

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> KernelRequest:
        expected = {
            "schema",
            "nodes",
            "outputs",
            "output_storages",
            "input_storages",
            "target_identity",
            "lowering_policy",
            "source_identity",
            "constants",
        }
        schema = row.get("schema")
        if schema == "merlin.semantic_kernel.v5":
            expected.update({"input_order", "boundary_layouts", "output_layouts", "physical_extents"})
        elif schema == "merlin.semantic_kernel.v4":
            expected.update({"boundary_layouts", "output_layouts", "physical_extents"})
        elif schema == "merlin.semantic_kernel.v3":
            expected.add("physical_extents")
        elif schema != "merlin.semantic_kernel.v2":
            raise ValueError("unexpected semantic kernel schema or fields")
        if set(row) != expected:
            raise ValueError("unexpected semantic kernel schema or fields")
        if not isinstance(row["nodes"], list) or not isinstance(row["input_storages"], dict):
            raise ValueError("semantic kernel graph or boundary has invalid structure")
        has_extents = schema in {"merlin.semantic_kernel.v3", "merlin.semantic_kernel.v4", "merlin.semantic_kernel.v5"}
        has_layouts = schema in {"merlin.semantic_kernel.v4", "merlin.semantic_kernel.v5"}
        if has_extents and (
            not isinstance(row["physical_extents"], dict)
            or (schema != "merlin.semantic_kernel.v5" and not row["physical_extents"])
        ):
            raise ValueError("semantic kernel physical extents have invalid structure")
        if has_layouts and (
            not isinstance(row["boundary_layouts"], dict) or not isinstance(row["output_layouts"], list)
        ):
            raise ValueError("semantic kernel physical layouts have invalid structure")
        input_order = row.get("input_order", sorted(row["input_storages"]))
        if (
            not isinstance(input_order, list)
            or any(not isinstance(key, str) for key in input_order)
            or len(input_order) != len(row["input_storages"])
            or set(input_order) != set(row["input_storages"])
        ):
            raise ValueError("semantic kernel input order must cover each boundary exactly once")
        return cls(
            nodes=tuple(SemanticNode.from_record(node) for node in row["nodes"]),
            outputs=tuple(row["outputs"]),
            output_storages=tuple(row["output_storages"]),
            input_storages=tuple((key, row["input_storages"][key]) for key in input_order),
            target_identity=row["target_identity"],
            lowering_policy=row["lowering_policy"],
            source_identity=row["source_identity"],
            constants=tuple(ConstantBinding.from_record(item) for item in row["constants"]),
            physical_extents=tuple(sorted(row["physical_extents"].items())) if has_extents else (),
            boundary_layouts=tuple(sorted(row["boundary_layouts"].items())) if has_layouts else (),
            output_layouts=tuple(row["output_layouts"]) if has_layouts else (),
        )
