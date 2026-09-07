"""Execute a host-built short artifact ONLY inside the existing compiler sandbox.

This file contains no golden outputs. It is copied into dedicated sandbox scratch by
the host qualifier; the parent compares returned data independently outside the sandbox.
"""
from __future__ import annotations

import ctypes
from itertools import product
import json
import math
import os
from pathlib import Path
import sys


_KINDS = {"i8": ctypes.c_int8, "i16": ctypes.c_int16, "i32": ctypes.c_int32,
          "i64": ctypes.c_int64, "f32": ctypes.c_float}
# Host process safety limits, not target geometry or performance assumptions.
_MAX_STORAGE_ELEMENTS = 1_000_000
_MAX_ARGUMENT_ELEMENTS = 4096
_MAX_TOTAL_STORAGE_BYTES = 64 * 1024 * 1024
_MAX_TOTAL_LOGICAL_ELEMENTS = 65536


def _positive_int(value, label, maximum):
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError(f"{label} exceeds the short host witness budget")
    return value


def _logical_offsets(spec):
    mapping = spec.get("logical_address_map")
    if "logical_address_map" not in spec:
        rows = _positive_int(spec.get("rows"), "rows", _MAX_ARGUMENT_ELEMENTS)
        cols = _positive_int(spec.get("cols"), "cols", _MAX_ARGUMENT_ELEMENTS)
        if rows * cols > _MAX_ARGUMENT_ELEMENTS:
            raise ValueError("logical element count exceeds short witness budget")
        stride = _positive_int(spec.get("row_stride"), "row stride", _MAX_STORAGE_ELEMENTS)
        if stride < cols:
            raise ValueError("legacy row stride aliases logical elements")
        return tuple(row * stride + col for row in range(rows) for col in range(cols))
    if not isinstance(mapping, dict) or mapping.get("schema") != "logical_address_map_v1":
        raise ValueError("unsupported logical address map")
    if "offsets_elements" in mapping:
        if set(mapping) != {"schema", "offsets_elements"}:
            raise ValueError("explicit offset map has ambiguous extra fields")
        offsets = mapping["offsets_elements"]
        if not isinstance(offsets, list) or not 0 < len(offsets) <= _MAX_ARGUMENT_ELEMENTS:
            raise ValueError("explicit offset count exceeds short witness budget")
        if any(type(offset) is not int for offset in offsets):
            raise ValueError("logical offsets must be integers")
        return tuple(offsets)
    if set(mapping) != {"schema", "logical_shape", "strides_elements", "offset_elements"}:
        raise ValueError("strided logical address map is incomplete or ambiguous")
    shape, strides, base = (mapping[key] for key in
                            ("logical_shape", "strides_elements", "offset_elements"))
    if (not isinstance(shape, list) or not isinstance(strides, list)
            or len(shape) != len(strides) or len(shape) > _MAX_ARGUMENT_ELEMENTS
            or any(type(dim) is not int or dim <= 0 for dim in shape)
            or any(type(stride) is not int for stride in strides) or type(base) is not int):
        raise ValueError("invalid strided logical address map")
    if math.prod(shape) > _MAX_ARGUMENT_ELEMENTS:
        raise ValueError("strided logical element count exceeds short witness budget")
    return tuple(base + sum(index * stride for index, stride in zip(indices, strides))
                 for indices in product(*(range(dim) for dim in shape)))


def _validate_arguments(request, alignment):
    """Validate the complete host request before loading code or allocating buffers.

    An explicit map is host-supplied witness setup, not evidence that a candidate
    implements that map or that runtime activation conversion may be excluded
    from a performance measurement. This runner never reports performance.
    """
    if not isinstance(request, dict):
        raise ValueError("host witness request must be an object")
    specs = request.get("arguments")
    if not isinstance(specs, list) or not specs:
        raise ValueError("host witness requires explicit argument records")
    explicit = any(isinstance(spec, dict) and "logical_address_map" in spec for spec in specs)
    limits = request.get("limits")
    if limits is None and explicit:
        raise ValueError("logical address maps require explicit host allocation/element limits")
    if limits is None:
        limits = {"max_total_storage_bytes": _MAX_TOTAL_STORAGE_BYTES,
                  "max_total_logical_elements": _MAX_TOTAL_LOGICAL_ELEMENTS}
    if not isinstance(limits, dict) or set(limits) != {
            "max_total_storage_bytes", "max_total_logical_elements"}:
        raise ValueError("unsupported host witness limits")
    byte_limit = _positive_int(limits["max_total_storage_bytes"], "total storage bytes",
                               _MAX_TOTAL_STORAGE_BYTES)
    element_limit = _positive_int(limits["max_total_logical_elements"], "total logical elements",
                                  _MAX_TOTAL_LOGICAL_ELEMENTS)
    total_bytes = total_elements = 0
    layouts = []
    for spec in specs:
        if (not isinstance(spec, dict) or not isinstance(spec.get("dtype"), str)
                or spec["dtype"] not in _KINDS or spec.get("access") not in ("read", "write")):
            raise ValueError("unsupported host witness dtype/access")
        ctype = _KINDS[spec["dtype"]]
        count = _positive_int(spec.get("storage_elements"), "storage elements", _MAX_STORAGE_ELEMENTS)
        offsets = _logical_offsets(spec)
        if len(set(offsets)) != len(offsets) or any(not 0 <= offset < count for offset in offsets):
            raise ValueError("logical address map is not injective and bounded")
        values = spec.get("values")
        if "logical_address_map" in spec and spec["access"] == "read" and values is None:
            raise ValueError("mapped read input requires explicit source logical values")
        if values is not None:
            if not isinstance(values, list) or len(values) != len(offsets):
                raise ValueError("source logical value count differs from address map")
            for value in values:
                if spec["dtype"] == "f32":
                    if (type(value) not in (int, float) or not math.isfinite(value)
                            or abs(value) > 3.4028234663852886e38):
                        raise ValueError("host witness f32 input is not finite/representable")
                else:
                    bits = ctypes.sizeof(ctype) * 8
                    if type(value) is not int or not -(1 << (bits-1)) <= value < (1 << (bits-1)):
                        raise ValueError("host witness integer input differs from declared dtype")
        total_bytes += ctypes.sizeof(ctype) * count + alignment
        total_elements += len(offsets)
        if total_bytes > byte_limit or total_elements > element_limit:
            raise ValueError("aggregate host allocation/element limit exceeded")
        layouts.append((ctype, count, offsets, values))
    return layouts


def main(request_path: Path) -> None:
    request = json.loads(request_path.read_text())
    alignment = int(os.sysconf("SC_PAGE_SIZE"))
    layouts = _validate_arguments(request, alignment)
    library = ctypes.CDLL(request["library"])
    function = getattr(library, request["symbol"])
    function.argtypes = [ctypes.c_void_p] * len(request["arguments"])
    function.restype = None
    storage, pointers, arrays, initial = [], [], [], []
    for ctype, count, offsets, values in layouts:
        raw = ctypes.create_string_buffer(ctypes.sizeof(ctype)*count + alignment)
        address = (ctypes.addressof(raw) + alignment-1) // alignment * alignment
        array = (ctype*count).from_address(address)
        if values is not None:
            for offset, value in zip(offsets, values):
                array[offset] = value
        storage.append(raw)
        pointers.append(address)
        arrays.append(array)
        initial.append(bytes(array))
    # Same warm-before-observed call discipline, but this is correctness only: no cycles.
    function(*pointers)
    function(*pointers)
    outputs = []
    for spec, array, original, layout in zip(request["arguments"], arrays, initial, layouts):
        if spec["access"] == "read":
            if bytes(array) != original:
                raise ValueError("submitted kernel modified a read-only input")
        elif spec["access"] == "write":
            outputs.append([array[offset] for offset in layout[2]])
    print(json.dumps({"outputs": outputs, "warmup_calls": 1, "observed_calls": 1,
                      "timing_measured": False}, allow_nan=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
