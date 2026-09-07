"""Checked logical-axis grouping into padded, strided physical tensor storage.

This is an explicit caller/compiler ABI contract, not an inference from equal
element counts or a target-specific layout name. It proves an injective bounded
address map and supplies a bit-preserving reference pack/unpack operation. It
does NOT prove that an arbitrary emitted consumer uses that address map, that
padding is numerically neutral, or that a caller actually packed its arguments.
Those obligations need artifact-bound evidence at the consuming boundary.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from math import prod

from .structural_transitions import StaticStridedLayout, _element_bytes


@dataclass(frozen=True)
class GroupedAxesStorage:
    logical_shape: tuple[int, ...]
    dtype: str
    axis_groups: tuple[tuple[int, ...], ...]
    physical_shape: tuple[int, ...]
    strides_elements: tuple[int, ...]
    storage_elements: int
    offset_elements: int = 0

    def validate(self) -> None:
        if (not isinstance(self.logical_shape, tuple)
                or not isinstance(self.physical_shape, tuple)
                or not isinstance(self.strides_elements, tuple)
                or not isinstance(self.axis_groups, tuple)
                or any(not isinstance(group, tuple) for group in self.axis_groups)
                or any(type(dim) is not int or dim <= 0
                       for dim in (*self.logical_shape, *self.physical_shape))
                or not isinstance(self.dtype, str)
                or type(self.storage_elements) is not int or self.storage_elements <= 0):
            raise ValueError("storage encoding requires immutable static positive dimensions")
        if len(self.axis_groups) != len(self.physical_shape):
            raise ValueError("each physical axis requires one logical axis group")
        axes = tuple(axis for group in self.axis_groups for axis in group)
        if (any(type(axis) is not int for axis in axes)
                or sorted(axes) != list(range(len(self.logical_shape)))):
            raise ValueError("axis groups must partition every logical axis exactly once")
        expected = tuple(prod(self.logical_shape[axis] for axis in group)
                         for group in self.axis_groups)
        if self.physical_shape != expected:
            raise ValueError("physical shape differs from grouped logical extents")
        width = _element_bytes(self.dtype)
        StaticStridedLayout(self.physical_shape, self.strides_elements,
                            self.storage_elements * width,
                            self.offset_elements).validate(self.dtype)
        # Prove the derived logical address map too. Grouping is mixed-radix,
        # with the last axis in each group varying fastest; it is not a sort.
        StaticStridedLayout(self.logical_shape, self._logical_strides(),
                            self.storage_elements * width,
                            self.offset_elements).validate(self.dtype)

    def _logical_strides(self) -> tuple[int, ...]:
        result = [0] * len(self.logical_shape)
        for group, stride in zip(self.axis_groups, self.strides_elements, strict=True):
            for axis in reversed(group):
                result[axis] = stride
                stride *= self.logical_shape[axis]
        return tuple(result)

    @property
    def logical_strides_elements(self) -> tuple[int, ...]:
        self.validate()
        return self._logical_strides()

    def element_offset(self, indices: Sequence[int]) -> int:
        self.validate()
        if (len(indices) != len(self.logical_shape)
                or any(type(index) is not int or not 0 <= index < dim
                       for index, dim in zip(indices, self.logical_shape, strict=True))):
            raise ValueError("logical index is outside the storage encoding domain")
        return self.offset_elements + sum(index * stride for index, stride
                                         in zip(indices, self._logical_strides(), strict=True))

    def to_dict(self) -> dict:
        self.validate()
        return {"schema": "grouped_axes_storage_v1", "logical_shape": list(self.logical_shape),
                "dtype": self.dtype, "axis_groups": [list(group) for group in self.axis_groups],
                "physical_shape": list(self.physical_shape),
                "strides_elements": list(self.strides_elements),
                "storage_elements": self.storage_elements, "offset_elements": self.offset_elements}

    @classmethod
    def from_dict(cls, record: Mapping) -> GroupedAxesStorage:
        fields = {"schema", "logical_shape", "dtype", "axis_groups", "physical_shape",
                  "strides_elements", "storage_elements", "offset_elements"}
        if (not isinstance(record, Mapping) or set(record) != fields
                or record.get("schema") != "grouped_axes_storage_v1"):
            raise ValueError("unsupported or incomplete storage encoding contract")
        if (any(not isinstance(record[name], list) for name in
                ("logical_shape", "physical_shape", "strides_elements", "axis_groups"))
                or any(not isinstance(group, list) for group in record["axis_groups"])):
            raise ValueError("storage encoding axes and extents must be JSON arrays")
        result = cls(tuple(record["logical_shape"]), record["dtype"],
                     tuple(tuple(group) for group in record["axis_groups"]),
                     tuple(record["physical_shape"]), tuple(record["strides_elements"]),
                     record["storage_elements"], record["offset_elements"])
        result.validate()
        return result

    def _byte_contract(self, max_storage_bytes: int) -> tuple[int, int, int]:
        self.validate()
        width = _element_bytes(self.dtype)
        size = self.storage_elements * width
        if type(max_storage_bytes) is not int or max_storage_bytes <= 0 or size > max_storage_bytes:
            raise ValueError("storage encoding exceeds the explicit reference-copy byte budget")
        return width, size, prod(self.logical_shape) * width

    def pack_bytes(self, logical: bytes, *, max_storage_bytes: int) -> bytes:
        """Reference copy only, with explicit allocation bound; padding is zero bits.

        No arithmetic, casting, quantization or inference execution occurs here.
        Zero-filled padding is not permission for a consumer to reduce over it.
        """
        width, size, logical_size = self._byte_contract(max_storage_bytes)
        if not isinstance(logical, bytes) or len(logical) != logical_size:
            raise ValueError("logical byte payload disagrees with the exact source type")
        result = bytearray(size)
        strides = self._logical_strides()
        for linear, indices in enumerate(product(*(range(dim) for dim in self.logical_shape))):
            destination = (self.offset_elements + sum(i * s for i, s in zip(indices, strides))) * width
            result[destination:destination + width] = logical[linear * width:(linear + 1) * width]
        return bytes(result)

    def unpack_bytes(self, physical: bytes, *, max_storage_bytes: int) -> bytes:
        """Read logical values only; allocated padding is never interpreted."""
        width, size, logical_size = self._byte_contract(max_storage_bytes)
        if not isinstance(physical, bytes) or len(physical) != size:
            raise ValueError("physical byte payload disagrees with allocated storage")
        result = bytearray(logical_size)
        strides = self._logical_strides()
        for linear, indices in enumerate(product(*(range(dim) for dim in self.logical_shape))):
            source = (self.offset_elements + sum(i * s for i, s in zip(indices, strides))) * width
            result[linear * width:(linear + 1) * width] = physical[source:source + width]
        return bytes(result)
