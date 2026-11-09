"""Declared physical view contracts admitted by generated selection rules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import prod
from typing import Any

from .instruction import AddressConstraint
from .model import IndexMap, KernelRequest, SemanticNode, TensorType


@dataclass(frozen=True)
class ViewDescriptor:
    """Target assertion for a contiguous reshape or outermost-axis slice view.

    Exact types and the extent in the named bank's unit are intentional: a
    shape-changing tensor operation is not a free view merely because its
    logical elements have the same count. The target must qualify this layout
    for both shapes before publishing the descriptor. A slice also declares
    its source extent and offset in that same physical unit.
    """

    name: str
    storage: str
    source_type: TensorType
    result_type: TensorType
    extent: int
    physical_layout: str = "contiguous_element_order"
    kind: str = "reshape"
    source_extent: int | None = None
    physical_offset: int = 0

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.storage
            or type(self.extent) is not int
            or self.extent <= 0
            or type(self.physical_offset) is not int
        ):
            raise ValueError("view needs a name, storage and positive physical extent")
        if self.physical_layout != "contiguous_element_order":
            raise ValueError("view needs a declared contiguous element order")
        if self.kind == "reshape":
            if (
                self.source_extent is not None
                or self.physical_offset != 0
                or self.source_type.dtype != self.result_type.dtype
                or self.source_type.numerical_policy != self.result_type.numerical_policy
                or prod(self.source_type.shape) != prod(self.result_type.shape)
            ):
                raise ValueError("reshape view must preserve element bits and count")
        elif self.kind == "slice":
            if (
                self.source_type.dtype != self.result_type.dtype
                or self.source_type.numerical_policy != self.result_type.numerical_policy
            ):
                raise ValueError("slice view must preserve element bits and numerical policy")
            if (
                len(self.source_type.shape) != len(self.result_type.shape)
                or self.source_type.shape[1:] != self.result_type.shape[1:]
                or type(self.source_extent) is not int
                or self.source_extent <= 0
                or type(self.physical_offset) is not int
                or self.physical_offset < 0
                or self.source_extent % prod(self.source_type.shape)
            ):
                raise ValueError("slice view needs outermost-axis contiguous physical geometry")
            units_per_element = self.source_extent // prod(self.source_type.shape)
            units_per_outer_slice = units_per_element * prod(self.source_type.shape[1:])
            if (
                self.extent != prod(self.result_type.shape) * units_per_element
                or self.physical_offset % units_per_outer_slice
                or self.physical_offset + self.extent > self.source_extent
            ):
                raise ValueError("slice view offset or extent differs from contiguous elements")
        else:
            raise ValueError("unknown view kind")

    def accepts(self, node: SemanticNode, source: SemanticNode | tuple[SemanticNode, ...]) -> bool:
        if isinstance(source, tuple):
            if len(source) != 1:
                return False
            source = source[0]
        common = (
            node.effect == "pure"
            and not node.attrs
            and node.inputs == (source.id,)
            and source.type == self.source_type
            and node.type == self.result_type
        )
        if not common:
            return False
        if self.kind == "reshape":
            return node.op == "reshape" and not node.index_maps
        rank = len(self.source_type.shape)
        units_per_element = self.source_extent // prod(self.source_type.shape)
        start = self.physical_offset // (units_per_element * prod(self.source_type.shape[1:]))
        identity = tuple(tuple(int(axis == dim) for axis in range(rank)) for dim in range(rank))
        return node.op == "gather" and node.index_maps == (
            IndexMap(rank, identity, (start, *(0 for _ in range(rank - 1)))),
            IndexMap(rank, identity, (0,) * rank),
        )

    def record(self) -> dict[str, Any]:
        row = {
            "name": self.name,
            "storage": self.storage,
            "source_type": self.source_type.record(),
            "result_type": self.result_type.record(),
            "extent": self.extent,
            "physical_layout": self.physical_layout,
        }
        if self.kind == "slice":
            row.update(
                {"kind": self.kind, "source_extent": self.source_extent, "physical_offset": self.physical_offset}
            )
        return row

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> ViewDescriptor | ConcatViewDescriptor:
        if row.get("kind") == "concat":
            return ConcatViewDescriptor.from_record(row)
        base = {"name", "storage", "source_type", "result_type", "extent", "physical_layout"}
        kind = row.get("kind", "reshape")
        expected = base if kind == "reshape" else base | {"kind", "source_extent", "physical_offset"}
        if set(row) != expected:
            raise ValueError("view has missing or unknown fields")
        return cls(
            row["name"],
            row["storage"],
            TensorType.from_record(row["source_type"]),
            TensorType.from_record(row["result_type"]),
            row["extent"],
            row["physical_layout"],
            kind,
            row.get("source_extent"),
            row.get("physical_offset", 0),
        )


@dataclass(frozen=True)
class ConcatViewDescriptor:
    """Outermost-axis concatenation of contiguous row-major storage spans.

    Every source and the result use the same bank and physical unit width per
    logical element. The allocator keeps this view until all adjacency
    obligations are checked; it never initializes or copies a child implicitly.
    """

    name: str
    storage: str
    source_types: tuple[TensorType, ...]
    result_type: TensorType
    source_extents: tuple[int, ...]
    extent: int
    physical_layout: str = "contiguous_element_order"

    def __post_init__(self) -> None:
        if not self.name or not self.storage or self.physical_layout != "contiguous_element_order":
            raise ValueError("concat view needs named contiguous storage")
        if (
            len(self.source_types) < 2
            or len(self.source_types) != len(self.source_extents)
            or not self.result_type.shape
            or any(len(source.shape) != len(self.result_type.shape) for source in self.source_types)
            or any(type(size) is not int or size <= 0 for size in self.source_extents)
            or type(self.extent) is not int
            or self.extent <= 0
        ):
            raise ValueError("concat view needs equal-rank positive source extents")
        if (
            any(
                (source.dtype, source.numerical_policy) != (self.result_type.dtype, self.result_type.numerical_policy)
                for source in self.source_types
            )
            or any(source.shape[1:] != self.result_type.shape[1:] for source in self.source_types)
            or sum(source.shape[0] for source in self.source_types) != self.result_type.shape[0]
        ):
            raise ValueError("concat view changes element order, type, or numerical policy")
        widths = []
        for source, size in zip(self.source_types, self.source_extents, strict=True):
            if size % prod(source.shape):
                raise ValueError("concat view has a nonintegral physical element width")
            widths.append(size // prod(source.shape))
        if len(set(widths)) != 1 or sum(self.source_extents) != self.extent:
            raise ValueError("concat view extents disagree with contiguous source elements")

    def accepts(self, node: SemanticNode, sources: tuple[SemanticNode, ...]) -> bool:
        return (
            node.effect == "pure"
            and node.op == "concat"
            and node.attrs == (("axis", 0),)
            and not node.index_maps
            and node.inputs == tuple(source.id for source in sources)
            and tuple(source.type for source in sources) == self.source_types
            and node.type == self.result_type
        )

    def record(self) -> dict[str, Any]:
        return {
            "kind": "concat",
            "name": self.name,
            "storage": self.storage,
            "source_types": [source.record() for source in self.source_types],
            "result_type": self.result_type.record(),
            "source_extents": list(self.source_extents),
            "extent": self.extent,
            "physical_layout": self.physical_layout,
        }

    @classmethod
    def from_record(cls, row: dict[str, Any]) -> ConcatViewDescriptor:
        if (
            set(row)
            != {
                "kind",
                "name",
                "storage",
                "source_types",
                "result_type",
                "source_extents",
                "extent",
                "physical_layout",
            }
            or row["kind"] != "concat"
        ):
            raise ValueError("concat view has missing or unknown fields")
        if not isinstance(row["source_types"], list) or not isinstance(row["source_extents"], list):
            raise ValueError("concat view sources must be lists")
        return cls(
            row["name"],
            row["storage"],
            tuple(TensorType.from_record(source) for source in row["source_types"]),
            TensorType.from_record(row["result_type"]),
            tuple(row["source_extents"]),
            row["extent"],
            row["physical_layout"],
        )
