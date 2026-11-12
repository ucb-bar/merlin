"""Static strided address maps: scalar widths and a conservative injectivity proof.

Kept free of compiler-lowering imports so data contracts (interfaces, storage
encodings) can check a layout without depending on dispatch or transition
planning. It proves only bounded, injective addressing of declared extents; it
says nothing about emission, padding semantics or timing.
"""

from __future__ import annotations


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
