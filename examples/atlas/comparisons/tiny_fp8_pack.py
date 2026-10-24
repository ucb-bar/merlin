"""Exact integer VPU FP8Pack for finite BF16 host tensors; no float approximation."""
import numpy as np


def admit_words(value):
    if (not isinstance(value, np.ndarray) or value.dtype != np.uint16 or value.ndim != 2 or not value.size
            or np.any((value & np.uint16(0x7f80)) == np.uint16(0x7f80))):
        raise ValueError('require finite nonempty rank 2 raw BF16 words')
    return value


def tensor_scale_bits(value):
    value = admit_words(value)
    maximum = int((value & np.uint16(32767)).max())
    if not maximum:
        return 0
    exponent = maximum >> 7 & 255
    if exponent:
        significand = 128 + (maximum & 127)
        exponent -= 127
    else:
        fraction = maximum & 127
        top = fraction.bit_length() - 1
        significand = fraction << 7 - top
        exponent = top - 133
    shift = exponent - 8 + int(significand > 224)
    if not -126 <= shift <= 127:
        raise ValueError('tensor scale outside finite admitted E8M0')
    return shift


def pack_words(value, scale):
    value = admit_words(value)
    if type(scale) is not int or not -126 <= scale <= 127:
        raise ValueError('scale outside normal finite BF16 rescale domain')
    field = (value >> 7 & 255).astype(np.int32)
    significand = 128 + (value & 127).astype(np.int32)
    rounded = significand + 7 + (significand >> 4 & 1) >> 4
    exponent = field - 127 - scale + (rounded == 16)
    rounded = np.where(rounded == 16, 8, rounded)
    sign = (value >> 8 & 128).astype(np.uint8)
    encoded = sign | (np.clip(exponent, -6, 8) + 7).astype(np.uint8) << 3 | (rounded - 8).astype(np.uint8)
    encoded = np.where((exponent > 8) | (exponent == 8) & (rounded >= 15), sign | np.uint8(126), encoded)
    return np.where((field == 0) | (exponent < -6), np.uint8(0), encoded).astype(np.uint8)


def encode(value):
    scale = tensor_scale_bits(value)
    return (pack_words(value, scale), scale)
