"""Activation scale policy from the quantized graph, with independent Atlas packing.

The policy matches the graph's BF16 maxima and scale grouping. The byte encoder
is the independent arithmetic implementation, rather than the framework packer.
"""

from __future__ import annotations

import operator

import numpy as np

from .arithmetic import bf16_bits, bf16_values, pack_bf16


def _scale_exp(values: np.ndarray, axes, rule: str) -> np.ndarray:
    # Nonfinite values do not contribute to the graph's scale selection.
    magnitude = np.abs(values)
    finite = np.where(np.isfinite(magnitude), magnitude, np.float32(0))
    maximum = np.max(finite, axis=axes, keepdims=axes is not None)
    _, exponent = np.frexp(maximum)
    scale = np.where(maximum == 0, 0, exponent - 9).astype(np.int32)
    if rule == "nc":
        # Float64 represents every BF16 maximum and this power-of-two product
        # exactly, including tiny inputs whose requested scales we later refuse.
        limit = np.ldexp(np.float64(448), scale)
        scale = scale + (maximum > limit).astype(np.int32)
    return np.asarray(scale, dtype=np.int32)


def _row_group_exp(values: np.ndarray, rows: int, rule: str) -> np.ndarray:
    lead, m, k = values.shape[:-2], values.shape[-2], values.shape[-1]
    result = np.empty((*lead, m, 1), dtype=np.int32)
    for start in range(0, m, rows):
        stop = min(start + rows, m)
        result[..., start:stop, :] = _scale_exp(
            values[..., start:stop, :], (-2, -1), rule
        )
    return result


def quantize_act(
    x,
    mode: str,
    reduce: str,
    rule: str,
    static_exp: int | None,
    rows: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Atlas uint8 codes and the graph's int32 power-of-two exponent.

    Floating inputs are rounded to BF16 before selecting scales. A uint16 input
    explicitly supplies BF16 bit patterns. ``block`` groups flattened rows;
    ``head_block`` groups rows independently within each leading matrix.
    Scales outside -127..127 are refused, never clamped. Code 255 would
    encode exponent 128 but the pinned Atlas model clamps its shift to 127.
    """
    rows = operator.index(rows)
    if rows < 1:
        raise ValueError("rows must be >= 1")
    if mode not in ("dynamic", "static"):
        raise ValueError(f"unsupported activation mode: {mode!r}")
    if reduce not in ("tensor", "token", "head", "block", "head_block"):
        raise ValueError(f"unsupported activation reduction: {reduce!r}")
    if rule not in ("ref", "nc"):
        raise ValueError(f"unsupported activation rule: {rule!r}")
    if (mode == "static") != (static_exp is not None):
        raise ValueError("static_exp is required exactly when mode='static'")

    raw = np.asarray(x)
    bits = raw if raw.dtype == np.uint16 else bf16_bits(raw)
    values = bf16_values(bits)
    if mode == "static":
        requested = operator.index(static_exp)
        if not -127 <= requested <= 127:
            raise ValueError("activation exponent has no exact supported E8M0 scale")
        scale = np.asarray(requested, dtype=np.int32)
    else:
        minimum_rank = {"tensor": 0, "token": 1, "head": 2, "block": 1, "head_block": 2}[reduce]
        if values.ndim < minimum_rank:
            raise ValueError(f"{reduce!r} reduction requires rank >= {minimum_rank}")
        if values.size == 0:
            raise ValueError("dynamic activation quantization requires a nonempty tensor")
        if reduce == "block":
            scale = _row_group_exp(values.reshape(-1, values.shape[-1]), rows, rule)
            scale = scale.reshape(*values.shape[:-1], 1)
        elif reduce == "head_block":
            scale = _row_group_exp(values, rows, rule)
        else:
            axes = {"tensor": None, "token": (-1,), "head": (-2, -1)}[reduce]
            scale = _scale_exp(values, axes, rule)
        if np.any((scale < -127) | (scale > 127)):
            raise ValueError("activation exponent has no exact supported E8M0 scale")
    codes = pack_bf16(bits, scale_code=scale + np.int32(127))
    return np.asarray(codes, dtype=np.uint8), np.asarray(scale, dtype=np.int32)
