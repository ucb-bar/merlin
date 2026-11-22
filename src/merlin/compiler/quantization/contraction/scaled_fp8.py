"""Target-neutral logical FP8 codes and binary scales, before physical packing.

This admission boundary describes values and compares declared numerical routes.
It does not implement or qualify a device accumulator or authorize approximation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ScaledFP8Spec:
    """E4M3FN (bias 7, cap448) i8 bits paired with signed i32 exponents.

    Exponent dimensions map, in order, to explicit logical code axes. All other
    axes broadcast. A mapped dimension can be singleton; no implicit trailing
    axis inference, block size, or transpose is permitted.
    """

    exponent_axes: tuple[int, ...]

    def __post_init__(self):
        if (type(self.exponent_axes) is not tuple
                or any(type(axis) is not int or axis < 0 for axis in self.exponent_axes)
                or tuple(sorted(set(self.exponent_axes))) != self.exponent_axes):
            raise ValueError("exponent axes must be unique, increasing nonnegative integers")

    def decode(self, codes: np.ndarray, exponents: np.ndarray) -> np.ndarray:
        """Decode exact logical values in binary64; refuse any lossy conversion.

        The representation admits signed zero and subnormals. A particular
        implementation can impose a narrower domain before accepting execution.
        Reserved NaN encodings are refused rather than assigned payload meaning.
        """
        if (not isinstance(codes, np.ndarray) or codes.dtype != np.int8
                or not codes.size or not isinstance(exponents, np.ndarray)
                or exponents.dtype != np.int32 or not exponents.size):
            raise ValueError("scaled FP8 requires nonempty i8 codes and i32 exponents")
        if len(self.exponent_axes) != exponents.ndim or any(axis >= codes.ndim for axis in self.exponent_axes):
            raise ValueError("exponent rank differs from explicit logical axes")
        shape = [1] * codes.ndim
        for axis, extent in zip(self.exponent_axes, exponents.shape, strict=True):
            if extent not in (1, codes.shape[axis]):
                raise ValueError("exponent extent does not broadcast on its declared axis")
            shape[axis] = extent
        raw = codes.view(np.uint8)
        magnitude = raw & 0x7f
        if np.any(magnitude == 0x7f):
            raise ValueError("reserved E4M3FN NaN code")
        exponent = (magnitude >> 3).astype(np.int32)
        fraction = (magnitude & 7).astype(np.float64)
        decoded = np.where(exponent == 0, fraction * 2.0**-9,
                           np.ldexp(1 + fraction / 8, exponent - 7))
        decoded = np.copysign(decoded, np.where(raw & 0x80, -1., 1.))
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            scales = exponents.astype(np.int64).reshape(shape)
            result = np.ldexp(decoded, scales)
            recovered = np.ldexp(result, -scales)
        if not np.isfinite(result).all() or np.any(recovered != decoded):
            raise ValueError("logical value is outside exact finite binary64 decoding range")
        return result

    def record(self):
        return {"schema": "merlin.scaled_fp8.v1", "encoding": "e4m3fn",
                "code_storage": "i8_bits", "exponent_storage": "i32",
                "value": "decode(code)*2**exponent", "exponent_axes": list(self.exponent_axes)}


@dataclass(frozen=True)
class ContractionNumerics:
    """An exact declared arithmetic sequence, never an equality proof by itself.

    Accumulation/readout identities must resolve to the selected implementation's
    full contract. Stage order and separate bias/activation semantics are part of
    identity, so a provider cannot silently fuse, reorder, or change rounding.
    """

    accumulation: str
    readout: str
    stages: tuple[str, ...]
    bias_rounding: str = "none"
    activation: str = "none"

    def __post_init__(self):
        if (any(type(value) is not str or not value.strip() for value in
                (self.accumulation, self.readout, self.bias_rounding, self.activation))
                or type(self.stages) is not tuple or not self.stages
                or any(type(stage) is not str or not stage.strip() for stage in self.stages)):
            raise ValueError("complete immutable numerical stage declarations are required")

    def record(self):
        return {"schema": "merlin.contraction_numerics.v1", **asdict(self)}

    @classmethod
    def from_record(cls, row):
        """Read only the complete versioned numerical contract."""
        if (not isinstance(row, dict)
                or set(row) != {"schema", "accumulation", "readout", "stages", "bias_rounding", "activation"}
                or row["schema"] != "merlin.contraction_numerics.v1"
                or type(row["stages"]) not in (list, tuple)):
            raise ValueError("invalid contraction numerical declaration")
        return cls(row["accumulation"], row["readout"], tuple(row["stages"]),
                   row["bias_rounding"], row["activation"])


def require_same_numerics(source: ContractionNumerics, candidate: ContractionNumerics) -> None:
    """Admit only identical declared semantics; differing routes need other proof.

    No tolerance, diagnostic family availability, or operation name grants a
    numerical change. Source custody and implementation qualification are separate.
    """
    if type(source) is not ContractionNumerics or type(candidate) is not ContractionNumerics:
        raise ValueError("typed source and candidate numerical contracts required")
    source.__post_init__()
    candidate.__post_init__()
    if source != candidate:
        raise ValueError("candidate changes declared contraction numerical semantics")
