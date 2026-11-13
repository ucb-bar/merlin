"""Independent NumPy execution of the pinned Atlas integer datapaths.

FP8 storage bytes are never rewritten by decoding. MXU operands use the custom
E4M3 multiplier, whose exponent-zero inputs flush and whose 0x7f/0xff bytes
participate as numeric significands. This differs from E4M3FN storage decoding.
All public BF16 arithmetic inputs/outputs are raw uint16 encodings.
"""
from __future__ import annotations

import numpy as np


def _raw(value, width):
    array = np.asarray(value)
    if array.dtype.kind not in "ui" or np.any(array < 0) or np.any(array > (1 << width) - 1):
        raise ValueError(f"expected unsigned {width}-bit integer encodings")
    return array.astype(np.uint8 if width == 8 else np.uint16, copy=False)


def bf16_values(bits):
    """Decode raw BF16 bits to float32, preserving signed zero and payloads."""
    return (_raw(bits, 16).astype(np.uint32) << 16).view(np.float32)


def bf16_bits(values):
    """Float32 to BF16, round nearest ties even, canonical positive quiet NaN."""
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(values, dtype=np.float32)
    raw = values.view(np.uint32)
    rounded = (raw + np.uint32(0x7fff) + ((raw >> 16) & 1)) >> 16
    return np.where(np.isnan(values), 0x7fc0, rounded).astype(np.uint16)


def decode_e4m3(codes):
    """Decode E4M3FN storage, including subnormals, signed zero, and NaN."""
    raw = _raw(codes, 8)
    exponent = ((raw >> 3) & 15).astype(np.int32)
    mantissa = (raw & 7).astype(np.float32)
    value = np.ldexp(np.where(exponent == 0, mantissa, 8 + mantissa),
                     np.where(exponent == 0, -9, exponent - 10))
    value = np.where(raw & 128, -value, value).astype(np.float32)
    return np.where((raw & 127) == 127, np.float32(np.nan), value).astype(np.float32)


def _scale_shift(scale_code):
    code = np.asarray(scale_code)
    if code.dtype.kind not in "iu" or np.any(code < 0) or np.any(code > 255):
        raise ValueError("scale_code must contain raw ERF bytes in [0, 255]")
    return np.clip(code.astype(np.int32) - 127, -128, 127)


def pack_bf16(bits, scale_code, *, mxu=False):
    """FP8Pack (divide) or BF16ScaleToE4M3 accumulator pop (multiply).

    Both flush BF16 denormals and NaN to positive zero and saturate infinity.
    MXU pop alone preserves encoding 0x7f/0xff when finite input rounds to 480.
    """
    raw, shift = np.broadcast_arrays(_raw(bits, 16).astype(np.int32), _scale_shift(scale_code))
    sign, exp, mant = raw >> 15, (raw >> 7) & 255, raw & 127
    adjusted = exp - 127 + (shift if mxu else -shift)
    sig = 128 | mant
    rounded = (sig >> 4) + (((sig & 8) != 0) & (((sig & 7) != 0) | (((sig >> 4) & 1) != 0)))
    adjusted = adjusted + (rounded == 16)
    frac = (np.where(rounded == 16, 8, rounded) - 8) & 7
    saturated = (adjusted > 8) | ((not mxu) & (adjusted == 8) & (frac == 7))
    output = (sign << 7) | ((adjusted + 7) << 3) | frac
    output = np.where(saturated, (sign << 7) | 126, output)
    output = np.where(adjusted < -6, 0, output)
    output = np.where(exp == 255, np.where(mant != 0, 0, (sign << 7) | 126), output)
    return np.where(exp == 0, 0, output).astype(np.uint8)


def unpack_fp8(codes, scale_code=127):
    """FP8Unpack: flush exponent-zero and NaN inputs to signed BF16 zero."""
    raw, shift = np.broadcast_arrays(_raw(codes, 8).astype(np.int32), _scale_shift(scale_code))
    sign, exp, mant = raw >> 7, (raw >> 3) & 15, raw & 7
    adjusted = exp + shift + 120
    output = (sign << 15) | (adjusted << 7) | (mant << 4)
    output = np.where(adjusted >= 255, (sign << 15) | 0x7f7f, output)
    flush = (exp == 0) | ((exp == 15) & (mant == 7)) | (adjusted <= 0)
    return np.where(flush, sign << 15, output).astype(np.uint16)


def multiply_bf16(a_bits, b_bits):
    """VMUL_BF16: BF16 RNE multiplication with HardFloat's default NaN."""
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return bf16_bits(bf16_values(a_bits) * bf16_values(b_bits))


def readout_bf16(acc_bits, exponent):
    """BF16 accumulator pop followed by VMUL with BF16(2**exponent).

    Exponent is the signed dequantization exponent, usually sx + sw, rather
    than the biased ERF encoding accepted by pack_bf16 and unpack_fp8.
    """
    exponent = np.asarray(exponent)
    if exponent.dtype.kind not in "iu":
        raise ValueError("readout exponent must be integral")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scale = bf16_bits(np.ldexp(np.ones(exponent.shape, dtype=np.float32),
                                  np.clip(exponent, -1000, 1000).astype(np.int32)))
    return multiply_bf16(acc_bits, scale)


def _product(a, b):
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    ae, be = (a >> 3) & 15, (b >> 3) & 15
    sig = (8 | (a & 7)) * (8 | (b & 7))
    carry = sig >> 7
    valid = (ae != 0) & (be != 0)
    return (a ^ b) >> 7, np.where(valid, ae + be - 1 + carry, 0), np.where(valid, np.where(carry != 0, sig, sig << 1) & 127, 0)


# Small exact lookup avoids floating logarithms in cancellation normalization.
_BIT_LENGTH = np.array([x.bit_length() for x in range(4096)], dtype=np.int64)


def _systolic_step(a, b, accumulator):
    sign, exponent, fraction = _product(a, b)
    exponent += 114
    acc = accumulator.astype(np.int64)
    ae, am, az = (acc >> 7) & 255, acc & 127, acc >> 15
    p, q = (128 | fraction) << 3, (128 | am) << 3
    product_larger = (exponent > ae) | ((exponent == ae) & (p >= q))
    e = np.where(product_larger, exponent, ae)
    diff = np.abs(exponent - ae)
    result_sign = np.where(product_larger, sign, az)
    big, small = np.where(product_larger, p, q), np.where(product_larger, q, p)
    shift = np.minimum(diff, 11)
    sticky = (small & ((1 << shift) - 1)) != 0
    small = (small >> shift) | sticky
    sig = np.where(sign == az, big + small, big - small)
    zero = sig == 0
    carry = (sig & 2048) != 0
    sticky = sticky | (carry & ((sig & 1) != 0))
    sig = np.where(carry, sig >> 1, sig)
    e = np.where(carry, (e + 1) & 255, e)
    normalize = (~carry) & ((sig & 1024) == 0) & (~zero)
    shift = np.where(normalize, np.minimum(11 - _BIT_LENGTH[sig], e), 0)
    sig, e = (sig << shift) & 2047, e - shift
    frac = ((sig >> 3) & 127) + (((sig & 4) != 0) & (((sig & 3) != 0) | sticky | ((sig & 8) != 0)))
    rounded_carry = frac == 128
    e = np.where(rounded_carry, (e + 1) & 255, e)
    frac = np.where(rounded_carry, 0, frac)
    result = (result_sign << 15) | (np.minimum(e, 254) << 7) | np.where(e > 254, 127, frac)
    result = np.where(zero, 0, result)
    result = np.where(ae == 0, (sign << 15) | (exponent << 7) | fraction, result)
    return np.where(exponent == 114, acc, result).astype(np.uint16)


def _inner_chunk(a, b, accumulator):
    """One 32-lane AnchorAccumulationTree, including incoming BF16 partial."""
    acc = accumulator.astype(np.int64)
    pe, pm = (acc >> 7) & 255, acc & 127
    anchor = np.where((acc & 32767) != 0, pe - 127, -1024)
    products = []
    for k in range(a.shape[-1]):
        product = _product(a[..., :, k, None], b[..., None, k, :])
        products.append(product)
        anchor = np.maximum(anchor, np.where(product[1] != 0, product[1] - 13, -1024))
    anchor += 7
    total = np.zeros(acc.shape, dtype=np.int64)
    for sign, exponent, fraction in products:
        # The 7-bit headroom bounds every valid shift below 64.
        shift = np.clip(anchor - (exponent - 13), 0, 63)
        magnitude = np.where(exponent != 0, ((128 | fraction) << 24) >> shift, 0)
        total += np.where(sign != 0, -magnitude, magnitude)
    shift = anchor - (pe - 127) - 24
    sig = np.where(pe != 0, 128, 0) | pm
    magnitude = np.where(shift >= 0, sig >> np.clip(shift, 0, 63),
                         (sig << np.clip(-shift, 0, 31)) & 0xffffffff)
    magnitude = np.where((shift >= 32) | (shift < -31), 0, magnitude)
    total += np.where((acc & 32768) != 0, -magnitude, magnitude)
    total = ((total + (1 << 31)) & 0xffffffff) - (1 << 31)
    mag = np.abs(total)
    # Every positive 32-bit magnitude has an exact double representation.
    lead = np.frexp(mag.astype(np.float64))[1].astype(np.int64) - 1
    shift = np.maximum(0, lead - 7)
    sig = (mag >> shift) << np.maximum(0, 7 - lead)  # the RTL normalizes sums shorter than eight bits
    remainder = mag & ((1 << shift) - 1)
    half = 1 << np.maximum(0, shift - 1)
    sig += (shift != 0) & ((remainder > half) | ((remainder == half) & ((sig & 1) != 0)))
    carry = sig >= 256
    sig, lead = np.where(carry, sig >> 1, sig), lead + carry
    exponent = lead + anchor - 31 + 127
    result = ((total < 0).astype(np.int64) << 15) | (exponent << 7) | (sig & 127)
    result = np.where(exponent >= 255, ((total < 0).astype(np.int64) << 15) | 0x7f7f, result)
    return np.where((total == 0) | (exponent <= 0), 0, result).astype(np.uint16)


def matmul_codes(a, b, unit="mxu0", *, initial=None, chunk_size=32):
    """A[..., M, K] @ B[..., K, N], yielding raw BF16 accumulators.

    K advances in ascending order. Each 32-lane instruction continues from
    the preceding BF16 partial. MXU0 rounds each custom FMA; MXU1 rounds each
    32-lane anchor tree. A short final instruction is implicitly zero padded.
    initial, if supplied, is an incoming raw BF16 accumulator tile.
    """
    if unit not in ("mxu0", "mxu1"):
        raise ValueError("unit must be mxu0 or mxu1")
    if chunk_size != 32:
        raise ValueError("only the pinned 32-lane instruction schedule is supported")
    a, b = _raw(a, 8), _raw(b, 8)
    if a.ndim < 2 or b.ndim < 2 or a.shape[-1] != b.shape[-2]:
        raise ValueError("matmul_codes requires [..., M, K] and [..., K, N]")
    batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a = np.broadcast_to(a, batch + a.shape[-2:])
    b = np.broadcast_to(b, batch + b.shape[-2:])
    shape = batch + (a.shape[-2], b.shape[-1])
    acc = np.zeros(shape, dtype=np.uint16) if initial is None else np.broadcast_to(_raw(initial, 16), shape).copy()
    for start in range(0, a.shape[-1], 32):
        stop = min(start + 32, a.shape[-1])
        if unit == "mxu1":
            acc = _inner_chunk(a[..., start:stop], b[..., start:stop, :], acc)
        else:
            for k in range(start, stop):
                acc = _systolic_step(a[..., :, k, None], b[..., None, k, :], acc)
    return acc
