"""Explicit grouped bit storage for low-precision float codes and block scales.

This is packing, not quantization: callers supply already selected codes and
E8M0 scales. Targets choose lane order, group width and alignment explicitly.
No scale geometry, padding, lookup table or physical instruction is inferred.
"""

from dataclasses import dataclass
from numbers import Integral

from .fp8_formats import canonical_float, storage_bits


@dataclass(frozen=True)
class PackedFloatLayout:
    format: str
    lanes_per_group: int
    group_bytes: int
    lane_order: str
    byte_order: str
    scale_group_elements: int | None = None

    def __post_init__(self):
        canonical_float(self.format)
        if (type(self.lanes_per_group) is not int or self.lanes_per_group <= 0 or
                type(self.group_bytes) is not int or self.group_bytes <= 0 or
                self.lanes_per_group * storage_bits(self.format) > self.group_bytes * 8 or
                self.lane_order not in ("lsb-first", "msb-first") or self.byte_order not in ("little", "big") or
                self.scale_group_elements is not None and (
                    type(self.scale_group_elements) is not int or self.scale_group_elements <= 0)):
            raise ValueError("invalid explicit packed float layout")

    def storage_bytes(self, elements):
        if type(elements) is not int or elements < 0:
            raise ValueError("element count must be nonnegative")
        return ((elements + self.lanes_per_group - 1) // self.lanes_per_group) * self.group_bytes

    def scale_bytes(self, elements):
        self.storage_bytes(elements)
        return 0 if self.scale_group_elements is None else (
            elements + self.scale_group_elements - 1) // self.scale_group_elements

    def _shift(self, lane):
        bits = storage_bits(self.format)
        return lane * bits if self.lane_order == "lsb-first" else self.group_bytes * 8 - (lane + 1) * bits


def pack_codes(codes, layout: PackedFloatLayout, *, scales=b""):
    """Return separately bound payload/scale bytes; unused bits are positive zero."""
    if type(layout) is not PackedFloatLayout:
        raise ValueError("explicit packed float layout required")
    codes = tuple(codes)
    limit = 1 << storage_bits(layout.format)
    if any(not isinstance(code, Integral) or isinstance(code, bool) or not 0 <= code < limit for code in codes):
        raise ValueError("code outside selected float format")
    codes = tuple(int(code) for code in codes)
    if type(scales) is not bytes or len(scales) != layout.scale_bytes(len(codes)) or 255 in scales:
        raise ValueError("one finite E8M0 code per declared scale group required")
    payload = bytearray()
    for start in range(0, len(codes), layout.lanes_per_group):
        word = sum(code << layout._shift(lane) for lane, code in enumerate(codes[start:start+layout.lanes_per_group]))
        payload.extend(word.to_bytes(layout.group_bytes, layout.byte_order))
    return bytes(payload), scales


def unpack_codes(payload, elements, layout: PackedFloatLayout, *, scales=b""):
    """Decode codes with exact extent and canonical padding checks, without scaling."""
    if type(layout) is not PackedFloatLayout or type(payload) is not bytes or len(payload) != layout.storage_bytes(elements):
        raise ValueError("packed payload extent differs")
    codes = []
    mask = (1 << storage_bits(layout.format)) - 1
    for start in range(0, len(payload), layout.group_bytes):
        word = int.from_bytes(payload[start:start+layout.group_bytes], layout.byte_order)
        codes.extend((word >> layout._shift(lane)) & mask for lane in range(layout.lanes_per_group))
    result = tuple(codes[:elements])
    if pack_codes(result, layout, scales=scales)[0] != payload:
        raise ValueError("nonzero unused lane or padding bits")
    return result
