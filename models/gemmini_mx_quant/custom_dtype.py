"""Stage 6.B — bit-exact mxGemmini custom dtype subclasses.

mxGemmini's per-element formats are **unsigned** — the sign of the
original value is carried by the per-block shared scale (the scale
itself is signed). This is *not* what stock torchao provides:

    +---------+--------+----------+-------------+----------+--------------+
    | format  | bits   | exp/mant | signed?     |  pmax    | merlin alias |
    +=========+========+==========+=============+==========+==============+
    | E4M4    | 8      | 4 / 4    | unsigned    |   ±448   |  fp8_0       |
    | E2M2    | 4      | 2 / 2    | unsigned    |    ±6    |  fp4         |
    +---------+--------+----------+-------------+----------+--------------+

Bit layout — E4M4 (one byte, ``[7:0]``)::

    bit 7  6  5  4  3  2  1  0
    ┌──┬──┬──┬──┬──┬──┬──┬──┐
    │ exp[3:0]  │ mant[3:0]  │
    └───────────┴────────────┘

Bit layout — E2M2 (one nibble, ``[3:0]``)::

    bit 3  2  1  0
    ┌──┬──┬──┬──┐
    │exp│ mant │
    └───┴──────┘

Both formats share the IEEE-style "biased exponent" convention with
``bias = 2**(E-1) - 1``. Encoding of value ``v ≥ 0`` is::

    if v == 0:               raw = 0
    else:
      let m, e = decompose v = 2**e * (1 + frac)   # frac in [0,1)
      mantissa = round(frac * 2**M)
      exp_field = e + bias
      raw = (exp_field << M) | mantissa

Subnormals: ``exp_field == 0`` => value = ``2**(1-bias) * frac`` where
``frac = mantissa / 2**M`` (no implicit leading 1).

Saturation: clamped at ``pmax = (2 - 2**-M) * 2**(2**E - 1 - bias)``
(equals ±448 for E4M4, ±6 for E2M2 — matches MxRequantizer.scala:7-44).

Per-block (16-elem) shared scale uses E8M0 power-of-two encoding
(matching torchao's MXTensor.scale convention) plus a sign bit kept in
the high bit of the first scale byte. This way the dialect's i8
buffer-level interface is preserved.

The Tensor subclass exposes ``.qdata`` (unsigned uint8 quantized
elements; nibbles for E2M2) and ``.scale`` (signed bf16 scale per
16-elem block) so downstream export consumers can pack however they
need.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# Saturation constants from MxRequantizer.scala:7-44
# --------------------------------------------------------------------------

E4M4_E_BITS = 4
E4M4_M_BITS = 4
E4M4_BIAS = (1 << (E4M4_E_BITS - 1)) - 1  # = 7
# pmax = (2 - 2^-M) * 2^(2^E - 1 - bias) = (2 - 1/16) * 2^(15-7) = 1.9375*256 = 496
# But MxRequantizer.scala saturates to ±448 to match nVidia's E4M3 finite
# range convention (pmax = 1.75 * 2^8 = 448). We follow the hardware.
E4M4_PMAX = 448.0

E2M2_E_BITS = 2
E2M2_M_BITS = 2
E2M2_BIAS = (1 << (E2M2_E_BITS - 1)) - 1  # = 1
# pmax = (2 - 2^-M) * 2^(2^E - 1 - bias) = (2 - 1/4) * 2^(3-1) = 1.75*4 = 7
# Hardware clamps to ±6 (matches MxRequantizer.scala line 7).
E2M2_PMAX = 6.0


# --------------------------------------------------------------------------
# Quantization helpers — pure tensor ops (work on any device)
# --------------------------------------------------------------------------


def _quantize_unsigned(
    x_abs: torch.Tensor,
    e_bits: int,
    m_bits: int,
    pmax: float,
) -> torch.Tensor:
    """Quantize a non-negative float tensor to the smallest "unsigned EaMb"
    representation, returning the *integer* code as ``int32``.

    Returned codes fit in ``e_bits + m_bits`` bits and represent::

        value = 2 ** (e - bias) * (1 + frac)  # normal
        value = 2 ** (1 - bias) * frac  # subnormal (e == 0)
    """
    bias = (1 << (e_bits - 1)) - 1
    max_e = (1 << e_bits) - 1  # all-ones exponent

    x = x_abs.clamp(min=0.0, max=pmax).to(torch.float32)
    nz = x > 0

    # log2 with safe handling of zero
    safe = torch.where(nz, x, torch.ones_like(x))
    log2 = torch.log2(safe)
    e = torch.floor(log2).to(torch.int32)

    # frac = x / 2^e - 1
    pow_e = torch.pow(torch.full_like(x, 2.0), e.to(torch.float32))
    frac = x / pow_e - 1.0

    # round mantissa to m_bits, with carry handling
    mant = torch.round(frac * (1 << m_bits)).to(torch.int32)
    # If mant overflows past the implicit 1 (== 1<<m_bits), bump exponent.
    overflow = mant >= (1 << m_bits)
    e = torch.where(overflow, e + 1, e)
    mant = torch.where(overflow, torch.zeros_like(mant), mant)

    # exp_field = e + bias, clamped to [0, max_e].
    exp_field = e + bias
    # Saturation: anything that exceeds max_e becomes the all-ones exp +
    # all-ones mantissa (= pmax representation).
    sat = exp_field >= max_e
    exp_field = torch.where(sat, torch.full_like(exp_field, max_e - 1), exp_field)
    mant = torch.where(sat, torch.full_like(mant, (1 << m_bits) - 1), mant)

    # Subnormals: exp_field <= 0  =>  shift mantissa appropriately.
    sub = exp_field <= 0
    if sub.any():
        # value = 2^(1-bias) * frac_sub  =>  raw mantissa = round(x / 2^(1-bias) * 2^M)
        scale_sub = 2.0 ** (1 - bias)
        mant_sub = torch.round(x_abs.to(torch.float32) / scale_sub * (1 << m_bits)).to(torch.int32)
        mant_sub = mant_sub.clamp(min=0, max=(1 << m_bits) - 1)
        exp_field = torch.where(sub, torch.zeros_like(exp_field), exp_field)
        mant = torch.where(sub, mant_sub, mant)

    # Zero stays zero
    exp_field = torch.where(nz, exp_field, torch.zeros_like(exp_field))
    mant = torch.where(nz, mant, torch.zeros_like(mant))

    raw = (exp_field << m_bits) | mant
    return raw


def _dequantize_unsigned(
    raw: torch.Tensor,
    e_bits: int,
    m_bits: int,
) -> torch.Tensor:
    """Inverse of :func:`_quantize_unsigned`. Returns float32, non-negative."""
    bias = (1 << (e_bits - 1)) - 1
    raw_i = raw.to(torch.int32)
    exp_field = (raw_i >> m_bits) & ((1 << e_bits) - 1)
    mant = raw_i & ((1 << m_bits) - 1)

    # Normal: 2^(exp_field - bias) * (1 + mant/2^M)
    normal = torch.pow(
        torch.full_like(exp_field, 2.0, dtype=torch.float32),
        (exp_field - bias).to(torch.float32),
    ) * (1.0 + mant.to(torch.float32) / (1 << m_bits))
    # Subnormal: 2^(1 - bias) * (mant/2^M)
    sub = (2.0 ** (1 - bias)) * (mant.to(torch.float32) / (1 << m_bits))
    val = torch.where(exp_field == 0, sub, normal)
    val = torch.where((exp_field == 0) & (mant == 0), torch.zeros_like(val), val)
    return val


def quantize_to_e4m4(x: torch.Tensor, block_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to mxGemmini E4M4 with per-``block_size`` shared scales.

    Returns
    -------
    qdata: ``uint8`` tensor of unsigned E4M4 codes, same shape as ``x``.
    scale: ``float32`` tensor of signed per-block scales, with last dim
        replaced by ``last_dim // block_size``.
    """
    return _quantize_blocked(x, block_size, E4M4_E_BITS, E4M4_M_BITS, E4M4_PMAX)


def quantize_to_e2m2(x: torch.Tensor, block_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to mxGemmini E2M2 with per-``block_size`` shared scales.

    Returns
    -------
    qdata: ``uint8`` tensor of unsigned E2M2 codes (in ``[0, 15]``),
        same shape as ``x``. Caller is responsible for nibble-packing
        if required.
    scale: ``float32`` tensor of signed per-block scales.
    """
    return _quantize_blocked(x, block_size, E2M2_E_BITS, E2M2_M_BITS, E2M2_PMAX)


def _quantize_blocked(
    x: torch.Tensor,
    block_size: int,
    e_bits: int,
    m_bits: int,
    pmax: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block quantize: split last dim into blocks of ``block_size``,
    compute a signed per-block power-of-two scale (signed E8M0), and
    encode each element's magnitude in unsigned EaMb.

    The block scale ``s`` is chosen so that ``max(|x_block|) / s ≈
    pmax``; the sign of the largest-magnitude element of the block
    determines the sign of ``s``.
    """
    orig_shape = x.shape
    last = orig_shape[-1]
    if last % block_size != 0:
        raise ValueError(f"last dim {last} must be divisible by block_size {block_size}")
    n_blocks = last // block_size

    x32 = x.to(torch.float32).reshape(*orig_shape[:-1], n_blocks, block_size)
    abs_blk = x32.abs()
    max_abs = abs_blk.amax(dim=-1, keepdim=True).clamp(min=1e-30)

    # Power-of-two scale chosen so max_abs/scale_mag <= pmax.
    # log2(scale_mag) = ceil(log2(max_abs / pmax))
    log2_scale = torch.ceil(torch.log2(max_abs / pmax))
    scale_mag = torch.pow(torch.full_like(log2_scale, 2.0), log2_scale)

    # Sign: take sign of the element with largest |x| in the block.
    argmax = abs_blk.argmax(dim=-1, keepdim=True)
    block_signs = torch.gather(torch.sign(x32), -1, argmax)
    # Treat exact-zero blocks as positive scale.
    block_signs = torch.where(block_signs == 0, torch.ones_like(block_signs), block_signs)

    scale = (block_signs * scale_mag).squeeze(-1)  # last dim n_blocks
    # Quantize each element's magnitude with the *block's* sign convention.
    # value = scale * dequant(qdata)  =>  dequant(qdata) = x / scale
    # Negative-scale blocks flip the sign of the elements before
    # encoding so qdata stays non-negative.
    norm = x32 / (scale_mag * block_signs)  # >= 0 ideally
    norm = norm.clamp(min=0.0)  # numerical safety; small negatives -> 0
    qdata = _quantize_unsigned(norm, e_bits, m_bits, pmax)
    qdata = qdata.to(torch.uint8).reshape(orig_shape)
    return qdata, scale


def _dequantize_blocked(
    qdata: torch.Tensor,
    scale: torch.Tensor,
    block_size: int,
    e_bits: int,
    m_bits: int,
) -> torch.Tensor:
    orig_shape = qdata.shape
    n_blocks = orig_shape[-1] // block_size
    q = qdata.to(torch.int32).reshape(*orig_shape[:-1], n_blocks, block_size)
    mag = _dequantize_unsigned(q, e_bits, m_bits)
    out = mag * scale.unsqueeze(-1)
    return out.reshape(orig_shape).to(torch.float32)


# --------------------------------------------------------------------------
# Tensor subclasses — torchao-style ``.qdata`` / ``.scale`` / ``.dequantize()``
# --------------------------------------------------------------------------


@dataclass
class _MxGemminiTensorBase:
    """Common state for :class:`MxGemminiE4M4Tensor` /
    :class:`MxGemminiE2M2Tensor`. Plain dataclass on purpose — these
    are not full ``torch.Tensor`` subclasses (no ``__torch_dispatch__``
    plumbing) because we only need them to *export* and to *reference-
    dequantize* on the CPU side. The runtime path is the dialect-level
    matmul on i8 buffers, not torch ops.
    """

    qdata: torch.Tensor
    scale: torch.Tensor
    block_size: int

    @property
    def shape(self) -> torch.Size:
        return self.qdata.shape

    @property
    def dtype(self):  # match torch attribute API
        return self.qdata.dtype

    def numel(self) -> int:
        return self.qdata.numel()


class MxGemminiE4M4Tensor(_MxGemminiTensorBase):
    """mxGemmini FP8_0 — unsigned E4M4, ``±448`` saturation, block 16."""

    elem_dtype = "e4m4_unsigned"
    e_bits = E4M4_E_BITS
    m_bits = E4M4_M_BITS
    pmax = E4M4_PMAX

    @classmethod
    def from_float(cls, x: torch.Tensor, block_size: int = 16) -> MxGemminiE4M4Tensor:
        q, s = quantize_to_e4m4(x, block_size=block_size)
        return cls(qdata=q, scale=s, block_size=block_size)

    def dequantize(self, target_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return _dequantize_blocked(self.qdata, self.scale, self.block_size, self.e_bits, self.m_bits).to(target_dtype)


class MxGemminiE2M2Tensor(_MxGemminiTensorBase):
    """mxGemmini FP4 — unsigned E2M2, ``±6`` saturation, block 16."""

    elem_dtype = "e2m2_unsigned"
    e_bits = E2M2_E_BITS
    m_bits = E2M2_M_BITS
    pmax = E2M2_PMAX

    @classmethod
    def from_float(cls, x: torch.Tensor, block_size: int = 16) -> MxGemminiE2M2Tensor:
        q, s = quantize_to_e2m2(x, block_size=block_size)
        return cls(qdata=q, scale=s, block_size=block_size)

    def dequantize(self, target_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return _dequantize_blocked(self.qdata, self.scale, self.block_size, self.e_bits, self.m_bits).to(target_dtype)


def quantize_linear_to_mxgemmini(linear: nn.Linear, fmt: str, block_size: int = 16):
    """Replace ``linear.weight`` with a Stage-6.B custom-dtype subclass
    according to ``fmt`` ∈ ``{"fp8", "fp4"}``. The bias is untouched.

    Returns the subclass instance for inspection.
    """
    fmt = fmt.lower()
    w = linear.weight.detach().to(torch.float32)
    if fmt in {"fp8", "fp8_0", "e4m4"}:
        t = MxGemminiE4M4Tensor.from_float(w, block_size=block_size)
    elif fmt in {"fp4", "e2m2"}:
        t = MxGemminiE2M2Tensor.from_float(w, block_size=block_size)
    else:
        raise ValueError(f"Unsupported mxGemmini format: {fmt!r}")

    # Stash the subclass instance on the module so the export path can
    # see it. We deliberately don't replace ``Parameter`` with a
    # subclass tensor (torch.compile / fx-trace don't yet handle that
    # uniformly across torchao versions).
    linear.mxgemmini_weight = t  # type: ignore[attr-defined]
    return t


__all__ = [
    "E2M2_PMAX",
    "E4M4_PMAX",
    "MxGemminiE2M2Tensor",
    "MxGemminiE4M4Tensor",
    "quantize_linear_to_mxgemmini",
    "quantize_to_e2m2",
    "quantize_to_e4m4",
]
