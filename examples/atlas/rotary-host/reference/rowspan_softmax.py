"""Independent unchanged-pad softmax under an explicit row-relative domain."""
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import attention_reference as attention

POLICY = "bf16-minfinite-zero-rowspan80-nonempty-v1"
DOMAIN = "bf16-row-relative-span80-pad32-v1"


def check_domain(scores, keep):
    scores = attention._raw_bf16(scores)
    keep = np.asarray(keep)
    if keep.dtype != np.bool_ or keep.shape != scores.shape or scores.ndim < 2 or not 0 < scores.shape[-1] <= 1024:
        raise ValueError("rowspan80 softmax requires a same-shape Boolean keep mask and 1..1024 keys")
    if not np.all(np.any(keep, axis=-1)):
        raise ValueError("rowspan80 softmax requires every row to retain at least one key")
    exponent = (scores >> 7) & 255
    magnitude = scores & 0x7fff
    values = (scores.astype(np.uint32) << 16).view(np.float32)
    if np.any(keep & ((magnitude != 0) & (exponent < 8))) or np.any(keep & (~np.isfinite(values))):
        raise ValueError("rowspan80 kept scores must be normal with exponent>=8 or signed zero")
    maximum = np.max(np.where(keep, values, -np.inf), axis=-1)
    minimum = np.min(np.where(keep, values, np.inf), axis=-1)
    span = maximum.astype(np.float64) - minimum.astype(np.float64)
    maximum_codes = np.take_along_axis(scores, np.argmax(np.where(keep, values, -np.inf), axis=-1)[..., None], axis=-1)[..., 0]
    minimum_codes = np.take_along_axis(scores, np.argmin(np.where(keep, values, np.inf), axis=-1)[..., None], axis=-1)[..., 0]
    def units(code):
        code = int(code)
        exponent, fraction = (code >> 7) & 255, code & 127
        magnitude = fraction if not exponent else (128 + fraction) << (exponent - 1)
        return -magnitude if code & 0x8000 else magnitude
    span_exceeds = any(units(high) - units(low) > (80 << 133)
                       for high, low in zip(maximum_codes.flat, minimum_codes.flat, strict=True))
    if np.any((maximum < -32) | (maximum > 48)) or span_exceeds:
        raise ValueError("rowspan80 kept row maximum or span leaves its explicit domain: "
                         + str({"rowmax_min": float(maximum.min()), "rowmax_max": float(maximum.max()),
                                "span_max_approximate": float(span.max()), "exact_span_exceeds80": span_exceeds,
                                "score_min": float(minimum.min())}))
    # Exact instruction subtraction rounds to F32, then chops to BF16.
    shifted = (values - maximum[..., None]).view(np.uint32) >> 16
    shifted_values = (shifted.astype(np.uint32) << 16).view(np.float32)
    if np.any(keep & ((shifted_values < -80) | (shifted_values > 0))):
        raise ValueError("rowspan80 actual chopped subtraction leaves [-80,0]")
    return scores, keep


def normalized_softmax(scores, keep):
    """Compose pinned model primitives; keep the inherited -32 masked pad."""
    from npu_model.configs.numerics import RtlNumerics as rtl

    scores, keep = check_domain(scores, keep)
    tiles, masks, maxima = [], [], []
    for start in range(0, scores.shape[-1], 32):
        count = min(32, scores.shape[-1] - start)
        tile = np.full((*scores.shape[:-1], 32), 0xc200, np.uint16)
        valid = np.zeros(tile.shape, np.bool_)
        tile[..., :count] = np.where(keep[..., start:start + count], scores[..., start:start + count], 0xc200)
        valid[..., :count] = keep[..., start:start + count]
        tensor = attention._torch_bf16(tile)
        tiles.append(tensor)
        masks.append(attention._torch_bf16(np.where(valid, 0x3f80, 0).astype(np.uint16)))
        maxima.append(rtl.amax(tensor, -1))
    maximum = maxima[0]
    for partial in maxima[1:]:
        maximum = rtl.maximum(maximum, partial)
    shifted, exponentials, sums = [], [], []
    for tile, mask in zip(tiles, masks, strict=True):
        offset = rtl.sub(tile, maximum)
        raw_offset = attention._bits(offset)
        offset_values = (raw_offset.astype(np.uint32) << 16).view(np.float32)
        if np.any(~np.isfinite(offset_values)) or np.any(offset_values < -80) or np.any(offset_values > 0):
            raise ValueError("rowspan80 instruction subtraction leaves [-80,0]")
        exponential = rtl.mul(rtl.unary("exp", offset), mask)
        shifted.append(offset)
        exponentials.append(exponential)
        sums.append(rtl.row_sum(exponential))
    denominator = sums[0]
    for partial in sums[1:]:
        denominator = rtl.add(denominator, partial)
    reciprocal = rtl.unary("rcp", denominator)
    probabilities = [rtl.mul(exponential, reciprocal) for exponential in exponentials]
    trim = lambda values: np.concatenate([attention._bits(value) for value in values], axis=-1)[..., :scores.shape[-1]]
    return {"scores_bf16": trim(tiles), "keep": keep.copy(), "row_max_bf16": attention._bits(maximum),
            "shifted_bf16": trim(shifted), "exp_bf16": trim(exponentials),
            "tile_sums_bf16": np.concatenate([attention._bits(value) for value in sums], axis=-1),
            "denominator_bf16": attention._bits(denominator), "reciprocal_bf16": attention._bits(reciprocal),
            "probabilities_bf16": trim(probabilities)}


def source_hash():
    return sha256(Path(__file__).read_bytes()).hexdigest()
