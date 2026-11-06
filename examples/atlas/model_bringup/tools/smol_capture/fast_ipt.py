"""Optional development-only NumPy implementation of the literal IPT32 datapath.

This module never selects or patches a model backend. Explicitly wrapping an
existing RTL math object changes execution provenance and is not a gate run.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch


BASELINE_NPU_MODEL_COMMIT = "47b8599d8ffbe5ddd96059ffe2b5be2427a410f5"
BACKEND_NAME = "numpy-ipt32-development-only"


def _leading_bit(values):
    """Integer floor(log2), with zero mapped to zero for the masked output lane."""
    lead = np.zeros_like(values)
    rest = values.copy()
    for shift in (16, 8, 4, 2, 1):
        upper = rest >> shift
        take = upper != 0
        lead += take * shift
        rest = np.where(take, upper, rest)
    return lead


def fast_inner_product_matmul(a_fp8, b_fp8, c_bf16):
    """Return literal MXU1 c + a@b, reducing exactly 32 FP8 terms per output.

    The public inputs and output are CPU Torch tensors, matching RtlNumerics.
    FP8 bytes, including subnormal and NaN encodings, and all BF16 partial bits
    follow rtl_math.ipt_row's integer rules rather than NumPy floating math.
    """
    if any(not isinstance(x, torch.Tensor) or x.device.type != "cpu" for x in (a_fp8, b_fp8, c_bf16)):
        raise ValueError("development IPT32 requires CPU Torch tensors")
    if a_fp8.dtype != torch.float8_e4m3fn or b_fp8.dtype != torch.float8_e4m3fn or c_bf16.dtype != torch.bfloat16:
        raise ValueError("require FP8 E4M3 A/B and BF16 C")
    if any(x.ndim != 2 for x in (a_fp8, b_fp8, c_bf16)):
        raise ValueError("require matrix operands")
    if a_fp8.shape[1] != 32 or b_fp8.shape[0] != 32 or c_bf16.shape != (a_fp8.shape[0], b_fp8.shape[1]):
        raise ValueError("require A[R,32], B[32,C], and partial[R,C]")
    a = a_fp8.detach().contiguous().view(torch.uint8).numpy().astype(np.int64)[:, :, None]
    b = b_fp8.detach().contiguous().view(torch.uint8).numpy().astype(np.int64)[None, :, :]
    partial = c_bf16.detach().contiguous().view(torch.uint16).numpy().astype(np.int64)

    ae, be = (a >> 3) & 15, (b >> 3) & 15
    product_nonzero = (ae != 0) & (be != 0)
    sig = (8 | (a & 7)) * (8 | (b & 7))
    carry = sig >> 7
    product_exp = np.where(product_nonzero, ae + be - 1 + carry, 0)
    product_mant = np.where(carry != 0, sig, sig << 1) & 127
    product_sign = (a ^ b) >> 7
    pe, pm = (partial >> 7) & 255, partial & 127
    anchor = np.maximum(np.where(product_exp != 0, product_exp - 13, -1024).max(axis=1),
                        np.where((partial & 32767) != 0, pe - 127, -1024)) + 7

    product_shift = anchor[:, None, :] - (product_exp - 13)
    # Python's unbounded right shift returns zero for counts >= 64 here.
    # Clamp shift operands before NumPy evaluates either side of where.
    magnitude = ((128 | product_mant) << 24) >> np.clip(product_shift, 0, 63)
    magnitude = np.where(product_nonzero & (product_shift < 64), magnitude, 0)
    total = np.where(product_sign != 0, -magnitude, magnitude).sum(axis=1, dtype=np.int64)

    shift = anchor - (pe - 127) - 24
    sig = np.where(pe != 0, 128, 0) | pm
    right = sig >> np.clip(shift, 0, 63)
    left = (sig << np.clip(-shift, 0, 31)) & 0xffffffff
    magnitude = np.where((shift >= -31) & (shift < 32), np.where(shift >= 0, right, left), 0)
    total += np.where((partial & 32768) != 0, -magnitude, magnitude)
    total = ((total + (1 << 31)) & 0xffffffff) - (1 << 31)

    sign, magnitude = (total < 0).astype(np.int64), np.abs(total)
    lead = _leading_bit(magnitude)
    shift = np.maximum(0, lead - 7)
    sig = magnitude >> shift
    remainder = magnitude & ((1 << shift) - 1)
    half = 1 << np.maximum(shift - 1, 0)
    round_up = (shift != 0) & ((remainder > half) | ((remainder == half) & ((sig & 1) != 0)))
    sig += round_up
    carry = sig >= 256
    sig = np.where(carry, sig >> 1, sig)
    lead += carry
    exponent = lead + anchor - 31 + 127
    bits = np.where(exponent >= 255, (sign << 15) | 0x7f7f,
                    (sign << 15) | (exponent << 7) | (sig & 127))
    bits = np.where((total == 0) | (exponent <= 0), 0, bits).astype(np.uint16)
    return torch.from_numpy(bits).view(torch.bfloat16)


class DevelopmentOnlyMathOverride:
    """Explicit external wrapper; delegates every operation except MXU1 IPT32."""

    def __init__(self, original_rtl_math):
        if type(original_rtl_math).__name__ != "RtlNumerics":
            raise ValueError("wrap an explicitly selected RtlNumerics instance")
        self.original_rtl_math = original_rtl_math
        self.provenance = dict(backend=BACKEND_NAME, baseline_npu_model_commit=BASELINE_NPU_MODEL_COMMIT,
                               helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               qualification="development primitives only; not original execution or a gate backend")

    def __getattr__(self, name):
        return getattr(self.original_rtl_math, name)

    inner_product_matmul = staticmethod(fast_inner_product_matmul)


def development_only_math_override(original_rtl_math):
    """Opt in explicitly and retain the returned object's provenance in receipts."""
    return DevelopmentOnlyMathOverride(original_rtl_math)
