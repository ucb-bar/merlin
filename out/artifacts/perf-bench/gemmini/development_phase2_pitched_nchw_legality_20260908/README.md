# Pitched batch-one NCHW legality for Gemmini LOOP_CONV

This development checkpoint separates **semantic representability** from **measured
profitability**.  It does not modify the sealed 92/96 compiler, private model payloads, or any
FireSim job.

## Result

All **53/53** captured ResNet-50 convolutions are addressable by the existing LOOP_CONV descriptor
after a compiler-only pitched-input correction.  The current 1/53 count is not an ISA limit: the
stem has `W=224`, already a multiple of DIM, while every later activation has an ordinary batch-one
NCHW row padded to DIM:

| logical W | physical row pitch | convolution count |
|---:|---:|---:|
| 224 | 224 | 1 |
| 56 | 64 | 13 |
| 28 | 32 | 13 |
| 14 | 16 | 19 |
| 7 | 16 | 7 |

The pinned RTL's `LoopConvLdInput` computes a transposed-input DRAM address as
`(ich * in_col_dim * in_row_dim + irow * in_col_dim + icol) * batches + b`.  In this mode it ignores
`in_stride`.  Therefore, for batch one, `in_col_dim` is exactly the physical NCHW row pitch.  The
tile-local `icols` remains separate and controls scratchpad/compute geometry, so using pitch 64 in
the address field does not turn a logical 56-wide convolution into a 64-wide convolution.

Neither convolution padding nor input dilation can encode the row pitch: both change source
coordinates and convolution semantics.  `trans_input_3120` is still required because ordinary
NCHW byte order equals CHWN only when `N=1`.

## Compiler correction

The isolated development compiler accepts a pitched input only when its complete encoding is the
canonical no-offset NCHW form:

`strides = [C * H * pitch, H * pitch, pitch, 1]`, with `pitch >= W`.

It then emits:

- descriptor `in_col_dim = pitch`;
- tile base `kch * H * pitch + row * pitch + col`;
- logical `out_col_dim = Wout` unchanged.

Malformed or noncanonical encodings fail closed.  The automatic whole-model selector intentionally
still requires `pitch == W`: queue job q545 showed that the batch-one transposed-input mechanism is
5.1106% slower than the q535 LOOP_WS baseline even for the dense stem.  This patch proves legality
and removes a future input copy, but it does **not** promote a hardware-rejected schedule.

The remaining performance route is an NHWC-propagated schedule with `trans_input_3120=false`, plus
native full-width/no-bias overwrite support or an equivalent fused epilogue.  Until that exists,
the 52 later convolutions correctly remain on accelerated streamed-row im2col + LOOP_WS.
