"""Dense factored-axis storage permutations, independent of a target geometry.

This is a byte-layout declaration and bijection checker. Source ownership,
immutable preparation, actual emitted addresses and target execution remain
independent obligations. No padding, arithmetic or quantization is performed.
"""

from dataclasses import dataclass
from math import prod


_WIDTHS = {"i8": 1, "ui8": 1, "i16": 2, "ui16": 2, "i32": 4, "ui32": 4,
           "i64": 8, "ui64": 8, "f16": 2, "bf16": 2, "f32": 4, "f64": 8}


@dataclass(frozen=True)
class FactoredTensorStorage:
    logical_shape: tuple[int, ...]
    dtype: str
    axis_factors: tuple[tuple[int, ...], ...]
    physical_axis_order: tuple[int, ...]

    def validate(self):
        if (type(self.logical_shape) is not tuple or not self.logical_shape
                or any(type(v) is not int or v <= 0 for v in self.logical_shape)
                or type(self.axis_factors) is not tuple or len(self.axis_factors) != len(self.logical_shape)
                or any(type(g) is not tuple or not g or any(type(v) is not int or v <= 0 for v in g)
                       for g in self.axis_factors)
                or tuple(map(prod, self.axis_factors)) != self.logical_shape
                or type(self.physical_axis_order) is not tuple
                or any(type(v) is not int for v in self.physical_axis_order)
                or sorted(self.physical_axis_order) != list(range(sum(map(len, self.axis_factors))))
                or type(self.dtype) is not str or self.dtype not in _WIDTHS
                or prod(self.logical_shape) * _WIDTHS[self.dtype] >= 1 << 63):
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
            strides[axis] = prod(shape[i] for i in self.physical_axis_order[at + 1:])
        return tuple((prod(shape[i + 1:]), extent, strides[i]) for i, extent in enumerate(shape))

    @property
    def logical_strides_elements(self):
        """Retain affine-only callers only when factoring collapses exactly."""
        terms = self.linear_index_terms()
        result, cursor = [], 0
        for factors in self.axis_factors:
            selected = terms[cursor:cursor + len(factors)]
            stride = selected[-1][2]
            if any(extent > 1 and physical != stride * prod(factors[i + 1:])
                   for i, (_, extent, physical) in enumerate(selected)):
                raise ValueError("factored logical storage is nonaffine; use the exact linear index terms")
            result.append(stride)
            cursor += len(factors)
        return tuple(result)

    @property
    def reorders_logical_elements(self):
        return any(extent > 1 and divisor != stride for divisor, extent, stride in self.linear_index_terms())

    def element_offset(self, indices):
        self.validate()
        if (len(indices) != len(self.logical_shape) or any(type(i) is not int or not 0 <= i < extent
                for i, extent in zip(indices, self.logical_shape, strict=True))):
            raise ValueError("logical index is outside the exact factored storage domain")
        linear = sum(i * prod(self.logical_shape[a + 1:]) for a, i in enumerate(indices))
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
        return dict(schema="factored_tensor_storage_v1", logical_shape=list(self.logical_shape), dtype=self.dtype,
                    axis_factors=[list(g) for g in self.axis_factors],
                    physical_axis_order=list(self.physical_axis_order))

    @classmethod
    def from_dict(cls, record):
        if (not isinstance(record, dict) or set(record) != {"schema", "logical_shape", "dtype", "axis_factors",
                "physical_axis_order"} or record["schema"] != "factored_tensor_storage_v1"
                or any(not isinstance(record[k], list) for k in ("logical_shape", "axis_factors", "physical_axis_order"))
                or any(not isinstance(g, list) for g in record["axis_factors"])):
            raise ValueError("unsupported or incomplete factored tensor storage declaration")
        result = cls(tuple(record["logical_shape"]), record["dtype"],
                     tuple(tuple(g) for g in record["axis_factors"]), tuple(record["physical_axis_order"]))
        result.validate()
        return result

    def _copy(self, payload, *, max_storage_bytes, inverse):
        self.validate()
        width = _WIDTHS[self.dtype]
        size = self.storage_elements * width
        if (type(max_storage_bytes) is not int or max_storage_bytes <= 0 or size > max_storage_bytes
                or type(payload) is not bytes or len(payload) != size):
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
            result[destination * width:(destination + run) * width] = payload[source * width:(source + run) * width]
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
    result = FactoredTensorStorage((rows, columns), dtype, ((rows,), (columns // panel_columns, panel_columns)),
                                   (1, 0, 2))
    result.validate()
    return result
