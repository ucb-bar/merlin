# Exact compute-only LOOP_CONV checkpoint

This isolated compiler accelerates the first convolution of the canonical q535 PT2E W8A8
ResNet-50 without changing its source arithmetic. It uses native `LOOP_CONV` for the convolution,
explicitly clears the accumulator through `LOAD3`, moves the full-width i32 result out, and leaves
the existing Merlin per-channel bias/requant/ReLU host epilogue intact.

The canonical graph passes Spike exactly: 1,000/1,000 logits, zero mismatches, finite outputs,
top-1 258, and checksum `c6e777c3fe0aae90`. The clean warm-progress run falls from q535's
506,265,226 cycles to 496,240,002 cycles (1.98% fewer, 1.0202x). These are Spike comparisons, not
FireSim/FPGA performance claims.

Coverage remains deliberately narrow: 1 of 53 convolutions is native and 52 fall back. Only the
dense-pitch stem is currently legal; later NCHW activations have padded row pitches that the
target's `trans_input_3120` path cannot express. The compiler fails closed on those layers.

## Public verification

Run from this directory with the repository environment:

```sh
PYTHONDONTWRITEBYTECODE=1 /scratch/agustin/projects/oscar-merlin/.venv/bin/python verify.py
```

The verifier checks the compiler tree, the payload-free two-descriptor exact Spike witness, the
1/53 census, q545's hardware rejection receipt and sanitized UART summary, and all 51 unit tests
(plus six MVOUT-width subtests). The full model input, trusted 1,000-logit vector, constant blob,
and diagnostic working tree are intentionally excluded from the public proof bundle. Their hashes
and aggregate results are recorded in `validation/compute_only_loop_conv_receipt.json`.

## Hardware state

FireSim queue job 545 completed successfully and preserved exact correctness, but measured
1,383,906,735 cycles versus q535's 1,316,619,699: 67,287,036 additional cycles, a 5.11059%
regression. The candidate is therefore rejected for performance, is not promoted, and the exact ELF
is blocked from resubmission. Although `instret` fell by 10,025,224, `main_ex` rose by 39,100,793
(1097.0%), accelerator-active cycles by 36,306,485 (116.2%), and reservation-station-active cycles
by 40,037,604 (74.23%). Those counters explain why the extra zero-fill/MVOUT command traffic erased
the host-side saving. q535 remains the hardware champion.
