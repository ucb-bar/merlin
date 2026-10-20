"""Exact lazy host input bytes for the existing logical pointer ABI.

This transports frozen scalar storage. It supplies no packing, conversion,
source-equivalence or device capability authority.
"""

from dataclasses import dataclass
from math import prod

from merlin.common.quant_formats import get
from merlin.llvmlower.weight_payload import WeightPayload


@dataclass(frozen=True)
class TensorPayload:
    shape: tuple[int, ...]
    dtype: str
    payload: WeightPayload
    byte_order: str = "little"

    def bind(self, *, shape, dtype, byte_order):
        try:
            selected, expected = get(self.dtype), get(dtype)
        except (KeyError, TypeError) as error:
            raise ValueError("streamed input has an unsupported scalar dtype") from error
        if (
            type(self) is not TensorPayload
            or type(self.shape) is not tuple
            or any(type(d) is not int or d <= 0 for d in self.shape)
            or self.shape != tuple(shape)
            or type(self.dtype) is not str
            or selected.name != expected.name
            or self.byte_order != byte_order
            or byte_order != "little"
            or type(self.payload) is not WeightPayload
        ):
            raise ValueError("streamed input differs from its exact logical tensor storage")
        scalar = selected
        if (
            scalar.kind not in ("int_affine", "float_ieee")
            or scalar.element_bits not in (8, 16, 32, 64)
            or scalar.pack_bits
            or len(self.payload) != prod(self.shape) * (scalar.element_bits // 8)
        ):
            raise ValueError("streamed input omits or adds scalar storage bytes")
        # Freeze the span roster for this emission. Extending the caller's
        # mutable builder later cannot change the declared pointer extent.
        return self.payload[:len(self.payload)]
