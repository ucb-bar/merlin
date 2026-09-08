"""Pinned Gemmini ``LOOP_CONV_WS`` descriptor encoding.

The five static records below are a literal transcription of the public
``gemmini_loop_conv_ws`` macro.  Address-bearing CONFIG_5/CONFIG_6 records are emitted by the
LLVM layer because their values are runtime pointers.  Keeping all other bit packing here gives
the verifier and code generator one auditable source of truth.
"""
from __future__ import annotations

from . import isa
from . import rtl_facts as F


K_LOOP_CONV_WS = 15
K_CONFIG_1 = 16
K_CONFIG_2 = 17
K_CONFIG_3 = 18
K_CONFIG_4 = 19
K_CONFIG_5 = 20
K_CONFIG_6 = 21


def _u64(value: int) -> int:
    return int(value) & isa.MASK64


def static_descriptor(*, batch_size: int, in_row_dim: int, in_col_dim: int,
                      in_channels: int, out_channels: int, out_row_dim: int,
                      out_col_dim: int, pool_out_row_dim: int, pool_out_col_dim: int,
                      stride: int, padding: int, kernel_dim: int, kernel_dilation: int,
                      pool_size: int, pool_stride: int, pool_padding: int,
                      batches: int, porows: int, pocols: int, pochs: int,
                      krows: int, kcols: int, kchs: int,
                      lpad: int, rpad: int, upad: int, dpad: int,
                      plpad: int, prpad: int, pupad: int, pdpad: int,
                      orows: int, ocols: int, in_stride: int, weight_stride: int,
                      out_stride: int, no_bias: bool, no_pool: bool,
                      downsample: bool, wrot180: bool, input_dilated: bool,
                      activation: int, trans_output_1203: bool,
                      trans_weight_1203: bool, trans_weight_0132: bool,
                      trans_input_3120: bool, max_pixels_per_row: int,
                      dw: bool, a_spad_id: int, b_spad_id: int
                      ) -> list[tuple[int, int, int]]:
    """Return CONFIG_1..4 plus launch, exactly as the public macro packs them."""
    config1 = (
        K_CONFIG_1,
        _u64((out_channels << 48) | (in_channels << 32) | (in_row_dim << 16)
             | batch_size),
        _u64((padding << 56) | (stride << 48) | (out_col_dim << 32)
             | (pool_out_row_dim << 16) | out_row_dim),
    )
    config2 = (
        K_CONFIG_2,
        _u64((kernel_dim << 48) | (pool_out_col_dim << 32) | (pool_size << 16)
             | (pool_stride << 8) | pool_padding),
        _u64((batches << 48) | (porows << 32) | (pocols << 16) | pochs),
    )
    config3 = (
        K_CONFIG_3,
        _u64((krows << 48) | (kcols << 32) | (kchs << 16) | lpad),
        _u64((rpad << 48) | (upad << 32) | (dpad << 24) | (plpad << 16)
             | in_col_dim),
    )
    config4 = (
        K_CONFIG_4,
        _u64((orows << 48) | (prpad << 32) | (pupad << 21) | (pdpad << 10)
             | kernel_dilation),
        _u64((in_stride << 48) | (weight_stride << 32) | (out_stride << 16)
             | ocols),
    )
    launch = (
        K_LOOP_CONV_WS,
        _u64((a_spad_id << 18) | (b_spad_id << 16) | (max_pixels_per_row << 8)
             | (int(dw) << 6) | (int(trans_input_3120) << 5)
             | (int(trans_weight_0132) << 4) | (int(trans_weight_1203) << 3)
             | (int(trans_output_1203) << 2) | (int(wrot180) << 1)
             | int(no_bias)),
        _u64((int(activation) << 3) | (int(input_dilated) << 2)
             | (int(downsample) << 1) | int(no_pool)),
    )
    for funct, _rs1, _rs2 in (config1, config2, config3, config4, launch):
        isa.assert_legal(funct)
    return [config1, config2, config3, config4, launch]


def total_rows(*, accumulator: bool, stride: int, kernel_dilation: int,
               batches: int, porows: int, pocols: int, pochs: int,
               krows: int, kcols: int, kchs: int) -> int:
    """Rows used by one no-pool NHWC/HWIO tile, matching tiled_conv_total_spad_rows."""
    dilated_krows = krows + (kernel_dilation - 1) * (krows - 1)
    dilated_kcols = kcols + (kernel_dilation - 1) * (kcols - 1)
    irows = porows * stride + dilated_krows - 1
    icols = pocols * stride + dilated_kcols - 1
    in_channel_tiles = (kchs + F.DIM - 1) // F.DIM
    out_channel_tiles = (pochs + F.DIM - 1) // F.DIM
    a_rows = in_channel_tiles * batches * irows * icols
    b_rows = out_channel_tiles * kcols * krows * kchs
    c_rows = out_channel_tiles * batches * porows * pocols
    return c_rows if accumulator else a_rows + b_rows


def auto_tile(*, batch: int, ho: int, wo: int, co: int, kh: int, kw: int,
              ci: int, stride: int, kernel_dilation: int) -> tuple[int, ...]:
    """Capacity-select a native tile using Gemmini's public ``tiled_conv_stride_auto`` rule."""
    args = [batch, ho, wo, co, kh, kw, ci]
    maxima = list(args)
    max_spad_rows, max_acc_rows = F.SPAD_ROWS // 2, F.ACC_ROWS // 2

    def fits(candidate: list[int]) -> bool:
        common = dict(stride=stride, kernel_dilation=kernel_dilation,
                      batches=candidate[0], porows=candidate[1], pocols=candidate[2],
                      pochs=candidate[3], krows=candidate[4], kcols=candidate[5],
                      kchs=candidate[6])
        return (total_rows(accumulator=False, **common) <= max_spad_rows
                and total_rows(accumulator=True, **common) <= max_acc_rows)

    while not fits(args):
        candidates = [i for i, value in enumerate(args)
                      if not (i == 2 and value <= F.DIM and args[1] > 1)]
        index = max(candidates, key=lambda i: args[i])
        if index in (3, 6):
            args[index] = ((args[index] // F.DIM) * F.DIM
                           if args[index] % F.DIM else args[index] - F.DIM)
        else:
            args[index] -= 1
        args[index] = max(1, args[index])

    # Preserve width utilization first, then greedily recover every legal extent.
    while args[2] < maxima[2]:
        candidate = list(args)
        candidate[2] += 1
        if not fits(candidate):
            break
        args = candidate
    changed = True
    while changed:
        changed = False
        for i in range(len(args)):
            if args[i] >= maxima[i]:
                continue
            candidate = list(args)
            candidate[i] += 1
            if fits(candidate):
                args, changed = candidate, True
    return tuple(args)
