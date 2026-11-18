"""Static strided address maps: scalar widths and a conservative injectivity proof.

Kept free of compiler-lowering imports so data contracts (interfaces, storage
encodings) can check a layout without depending on dispatch or transition
planning. It proves only bounded, injective addressing of declared extents; it
says nothing about emission, padding semantics or timing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from math import prod

# ---- strided_layout ----------------------------------------------------------------------------


def _element_bytes(dtype: str) -> int:
    fixed = {"f16": 2, "bf16": 2, "f32": 4, "f64": 8}
    if dtype in fixed:
        return fixed[dtype]
    if dtype.startswith("i") and dtype[1:].isdigit():
        bits = int(dtype[1:])
        if bits > 0 and bits % 8 == 0:
            return bits // 8
    raise ValueError(f"byte-addressable scalar representation unknown for {dtype!r}")


def check_static_strided(shape, strides_elements, storage_bytes, offset_elements, dtype: str) -> None:
    """Refuse a layout that is not positive, static, injective and inside its allocation."""
    if (
        len(shape) != len(strides_elements)
        or any(type(dim) is not int or dim <= 0 for dim in shape)
        or any(type(stride) is not int or stride <= 0 for stride in strides_elements)
        or type(offset_elements) is not int
        or offset_elements < 0
        or type(storage_bytes) is not int
        or storage_bytes <= 0
    ):
        raise ValueError("layout requires positive static extents/strides and a bounded allocation")
    # A conservative injectivity proof, including axis permutations and row
    # padding. Unknown overlapping/interleaved affine layouts fail closed.
    span = 1
    for stride, dim in sorted(zip(strides_elements, shape)):
        if dim > 1 and stride < span:
            raise ValueError("layout address mapping is not proved injective")
        span += (dim - 1) * stride
    end = offset_elements + 1 + sum((dim - 1) * stride for dim, stride in zip(shape, strides_elements))
    if end * _element_bytes(dtype) > storage_bytes:
        raise ValueError("layout touches storage outside its declared allocation")


# ---- storage_encoding --------------------------------------------------------------------------
# Checked logical-axis grouping into padded, strided physical tensor storage.
#
# This is an explicit caller/compiler ABI contract, not an inference from equal
# element counts or a target-specific layout name. It proves an injective bounded
# address map and supplies a bit-preserving reference pack/unpack operation. It
# does NOT prove that an arbitrary emitted consumer uses that address map, that
# padding is numerically neutral, or that a caller actually packed its arguments.
# Those obligations need artifact-bound evidence at the consuming boundary.


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
        if (
            not isinstance(self.logical_shape, tuple)
            or not isinstance(self.physical_shape, tuple)
            or not isinstance(self.strides_elements, tuple)
            or not isinstance(self.axis_groups, tuple)
            or any(not isinstance(group, tuple) for group in self.axis_groups)
            or any(type(dim) is not int or dim <= 0 for dim in (*self.logical_shape, *self.physical_shape))
            or not isinstance(self.dtype, str)
            or type(self.storage_elements) is not int
            or self.storage_elements <= 0
        ):
            raise ValueError("storage encoding requires immutable static positive dimensions")
        if len(self.axis_groups) != len(self.physical_shape):
            raise ValueError("each physical axis requires one logical axis group")
        axes = tuple(axis for group in self.axis_groups for axis in group)
        if any(type(axis) is not int for axis in axes) or sorted(axes) != list(range(len(self.logical_shape))):
            raise ValueError("axis groups must partition every logical axis exactly once")
        expected = tuple(prod(self.logical_shape[axis] for axis in group) for group in self.axis_groups)
        if self.physical_shape != expected:
            raise ValueError("physical shape differs from grouped logical extents")
        width = _element_bytes(self.dtype)
        check_static_strided(
            self.physical_shape, self.strides_elements, self.storage_elements * width, self.offset_elements, self.dtype
        )
        # Prove the derived logical address map too. Grouping is mixed-radix,
        # with the last axis in each group varying fastest; it is not a sort.
        check_static_strided(
            self.logical_shape, self._logical_strides(), self.storage_elements * width, self.offset_elements, self.dtype
        )

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
        if len(indices) != len(self.logical_shape) or any(
            type(index) is not int or not 0 <= index < dim
            for index, dim in zip(indices, self.logical_shape, strict=True)
        ):
            raise ValueError("logical index is outside the storage encoding domain")
        return self.offset_elements + sum(
            index * stride for index, stride in zip(indices, self._logical_strides(), strict=True)
        )

    def to_dict(self) -> dict:
        self.validate()
        return {
            "schema": "grouped_axes_storage_v1",
            "logical_shape": list(self.logical_shape),
            "dtype": self.dtype,
            "axis_groups": [list(group) for group in self.axis_groups],
            "physical_shape": list(self.physical_shape),
            "strides_elements": list(self.strides_elements),
            "storage_elements": self.storage_elements,
            "offset_elements": self.offset_elements,
        }

    @classmethod
    def from_dict(cls, record: Mapping) -> GroupedAxesStorage:
        if isinstance(record, Mapping) and record.get("schema") == "factored_tensor_storage_v1":
            return FactoredTensorStorage.from_dict(record)
        fields = {
            "schema",
            "logical_shape",
            "dtype",
            "axis_groups",
            "physical_shape",
            "strides_elements",
            "storage_elements",
            "offset_elements",
        }
        if (
            not isinstance(record, Mapping)
            or set(record) != fields
            or record.get("schema") != "grouped_axes_storage_v1"
        ):
            raise ValueError("unsupported or incomplete storage encoding contract")
        if any(
            not isinstance(record[name], list)
            for name in ("logical_shape", "physical_shape", "strides_elements", "axis_groups")
        ) or any(not isinstance(group, list) for group in record["axis_groups"]):
            raise ValueError("storage encoding axes and extents must be JSON arrays")
        result = cls(
            tuple(record["logical_shape"]),
            record["dtype"],
            tuple(tuple(group) for group in record["axis_groups"]),
            tuple(record["physical_shape"]),
            tuple(record["strides_elements"]),
            record["storage_elements"],
            record["offset_elements"],
        )
        result.validate()
        return result

    def linear_index_terms(self):
        self.validate()
        return tuple(
            (prod(self.logical_shape[axis + 1 :]), extent, stride)
            for axis, (extent, stride) in enumerate(zip(self.logical_shape, self.logical_strides_elements, strict=True))
        )

    @property
    def reorders_logical_elements(self):
        active = [
            (axis, stride)
            for axis, (extent, stride) in enumerate(zip(self.logical_shape, self.logical_strides_elements, strict=True))
            if extent > 1
        ]
        return [axis for axis, _ in active] != [axis for axis, _ in sorted(active, key=lambda pair: -pair[1])]

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
            result[destination : destination + width] = logical[linear * width : (linear + 1) * width]
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
            result[linear * width : (linear + 1) * width] = physical[source : source + width]
        return bytes(result)


# ---- factored_storage --------------------------------------------------------------------------
# Dense factored-axis storage permutations, independent of a target geometry.
#
# Dependency-free so interface and storage-encoding contracts can check it
# without importing scheduling passes; ``merlin.analysis.programs.storage_encoding``
# re-exports it under its scheduling API name.
#
# This is a byte-layout declaration and bijection checker. Source ownership,
# immutable preparation, actual emitted addresses and target execution remain
# independent obligations. No padding, arithmetic or quantization is performed.

_WIDTHS = {
    "i8": 1,
    "ui8": 1,
    "i16": 2,
    "ui16": 2,
    "i32": 4,
    "ui32": 4,
    "i64": 8,
    "ui64": 8,
    "f16": 2,
    "bf16": 2,
    "f32": 4,
    "f64": 8,
}


@dataclass(frozen=True)
class FactoredTensorStorage:
    logical_shape: tuple[int, ...]
    dtype: str
    axis_factors: tuple[tuple[int, ...], ...]
    physical_axis_order: tuple[int, ...]

    def validate(self):
        if (
            type(self.logical_shape) is not tuple
            or not self.logical_shape
            or any(type(v) is not int or v <= 0 for v in self.logical_shape)
            or type(self.axis_factors) is not tuple
            or len(self.axis_factors) != len(self.logical_shape)
            or any(
                type(g) is not tuple or not g or any(type(v) is not int or v <= 0 for v in g) for g in self.axis_factors
            )
            or tuple(map(prod, self.axis_factors)) != self.logical_shape
            or type(self.physical_axis_order) is not tuple
            or any(type(v) is not int for v in self.physical_axis_order)
            or sorted(self.physical_axis_order) != list(range(sum(map(len, self.axis_factors))))
            or type(self.dtype) is not str
            or self.dtype not in _WIDTHS
            or prod(self.logical_shape) * _WIDTHS[self.dtype] >= 1 << 63
        ):
            raise ValueError("factored storage needs exact positive axis factors and a complete dense permutation")

    @property
    def factored_shape(self):
        return tuple(v for group in self.axis_factors for v in group)

    @property
    def physical_shape(self):
        self.validate()
        return tuple(self.factored_shape[i] for i in self.physical_axis_order)

    @property
    def storage_elements(self):
        self.validate()
        return prod(self.logical_shape)

    @property
    def offset_elements(self):
        return 0

    def linear_index_terms(self):
        """(linear divisor, factored extent, physical stride) for exact indexing."""
        self.validate()
        shape = self.factored_shape
        strides = [0] * len(shape)
        for at, axis in enumerate(self.physical_axis_order):
            strides[axis] = prod(shape[i] for i in self.physical_axis_order[at + 1 :])
        return tuple((prod(shape[i + 1 :]), extent, strides[i]) for i, extent in enumerate(shape))

    @property
    def logical_strides_elements(self):
        """Retain affine-only callers only when factoring collapses exactly."""
        terms = self.linear_index_terms()
        result, cursor = [], 0
        for factors in self.axis_factors:
            selected = terms[cursor : cursor + len(factors)]
            stride = selected[-1][2]
            if any(
                extent > 1 and physical != stride * prod(factors[i + 1 :])
                for i, (_, extent, physical) in enumerate(selected)
            ):
                raise ValueError("factored logical storage is nonaffine; use the exact linear index terms")
            result.append(stride)
            cursor += len(factors)
        return tuple(result)

    @property
    def reorders_logical_elements(self):
        return any(extent > 1 and divisor != stride for divisor, extent, stride in self.linear_index_terms())

    def element_offset(self, indices):
        self.validate()
        if len(indices) != len(self.logical_shape) or any(
            type(i) is not int or not 0 <= i < extent for i, extent in zip(indices, self.logical_shape, strict=True)
        ):
            raise ValueError("logical index is outside the exact factored storage domain")
        linear = sum(i * prod(self.logical_shape[a + 1 :]) for a, i in enumerate(indices))
        return sum((linear // divisor % extent) * stride for divisor, extent, stride in self.linear_index_terms())

    def emit_linear_offset(self, builder, linear):
        """Emit the same checked pure integer map; supplies no pointer authority."""
        result = builder.value(0)
        for divisor, extent, stride in self.linear_index_terms():
            term = builder.op("urem", builder.op("udiv", linear, divisor), extent)
            result = builder.op("add", result, builder.op("mul", term, stride))
        return result

    def to_dict(self):
        self.validate()
        return dict(
            schema="factored_tensor_storage_v1",
            logical_shape=list(self.logical_shape),
            dtype=self.dtype,
            axis_factors=[list(g) for g in self.axis_factors],
            physical_axis_order=list(self.physical_axis_order),
        )

    @classmethod
    def from_dict(cls, record):
        if (
            not isinstance(record, dict)
            or set(record) != {"schema", "logical_shape", "dtype", "axis_factors", "physical_axis_order"}
            or record["schema"] != "factored_tensor_storage_v1"
            or any(not isinstance(record[k], list) for k in ("logical_shape", "axis_factors", "physical_axis_order"))
            or any(not isinstance(g, list) for g in record["axis_factors"])
        ):
            raise ValueError("unsupported or incomplete factored tensor storage declaration")
        result = cls(
            tuple(record["logical_shape"]),
            record["dtype"],
            tuple(tuple(g) for g in record["axis_factors"]),
            tuple(record["physical_axis_order"]),
        )
        result.validate()
        return result

    def _copy(self, payload, *, max_storage_bytes, inverse):
        self.validate()
        width = _WIDTHS[self.dtype]
        size = self.storage_elements * width
        if (
            type(max_storage_bytes) is not int
            or max_storage_bytes <= 0
            or size > max_storage_bytes
            or type(payload) is not bytes
            or len(payload) != size
        ):
            raise ValueError("factored tensor byte copy exceeds its exact source extent or explicit budget")
        terms = self.linear_index_terms()
        # Keep the largest suffix contiguous in both layouts. This includes
        # arbitrary dense permutations and unit factors; no target tile size
        # or element value is involved in the copy granularity.
        run = 1
        for divisor, extent, stride in reversed(terms):
            if extent == 1:
                continue
            if divisor != run or stride != run:
                break
            run *= extent
        if run == self.storage_elements:
            return payload
        result = bytearray(size)
        terms = tuple(term for term in terms if term[1] > 1)
        for linear in range(0, self.storage_elements, run):
            physical = sum((linear // d % n) * s for d, n, s in terms)
            source, destination = (physical, linear) if inverse else (linear, physical)
            result[destination * width : (destination + run) * width] = payload[source * width : (source + run) * width]
        return bytes(result)

    def pack_bytes(self, logical, *, max_storage_bytes):
        return self._copy(logical, max_storage_bytes=max_storage_bytes, inverse=False)

    def unpack_bytes(self, physical, *, max_storage_bytes):
        return self._copy(physical, max_storage_bytes=max_storage_bytes, inverse=True)


def column_panel_storage(rows, columns, panel_columns, dtype):
    """Source [K,N] -> dense [N/P,K,P]; an indivisible N keeps its old route."""
    if any(type(v) is not int or v <= 0 for v in (rows, columns, panel_columns)):
        raise ValueError("column panels need explicit positive source extents and panel width")
    if columns % panel_columns:
        return None
    result = FactoredTensorStorage(
        (rows, columns), dtype, ((rows,), (columns // panel_columns, panel_columns)), (1, 0, 2)
    )
    result.validate()
    return result
