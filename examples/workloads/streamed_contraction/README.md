# Projection through a bounded device window

The source computes `relu(values @ weights)` with shapes `[257,32]` and
`[32,513]`. Both matrices are runtime inputs. The logical BF16 output is
263,682 bytes; tile padding increases it to 313,344 bytes. It cannot fit a
256-KiB staging window together with its operands.

A backend can select a reusable full-K body and stream independent M/N panels
through that window. The host packs each panel, launches the same body, waits
and scatters valid output cells into private staging. Only after every panel
succeeds does it publish the source tensor. K order and arithmetic stay within
one launch for each output cell. No reduction reassociation is needed.

The source example does not choose a target, staging capacity or precision.
Those are explicit compilation settings. A target precision conversion still
needs its own quality qualification. This example exercises temporary buffer
reuse; it does not establish persistent device weights or KV storage.

`static_weights.py` provides a second capture with `[1,128]` input and an
immutable `[128,2049]` parameter. Its encoded weight data exceeds 256 KiB before
padding. The application route embeds declared parameters in host code and
streams the required panels; it must not place the complete weight tensor at a
single panel address. Weights are compiler inputs, not evaluation samples.

`wide_reduction.py` instead uses `[3,8193]` input and an immutable `[8193,2]`
parameter. Even the smallest padded full-K panel exceeds 256 KiB. This requires
ordered K chunks with completed BF16 partial sums carried between launches.
Carry bytes must retain their precision; converting them to FP8, resetting the
accumulator, or applying ReLU before the last chunk changes the result. The
source remains target independent; buffer capacity and precision are compilation
settings, and reference inputs stay outside the compiler process.
