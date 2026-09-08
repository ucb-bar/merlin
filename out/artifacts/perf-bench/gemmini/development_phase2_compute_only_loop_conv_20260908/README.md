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
1/53 census, the non-hardware qualification label, and all 51 unit tests (plus six MVOUT-width
subtests). The full model input, trusted 1,000-logit vector, constant blob, and diagnostic working
tree are intentionally excluded from the public proof bundle. Their hashes and aggregate results
are recorded in `validation/compute_only_loop_conv_receipt.json`.

## Hardware state

No FireSim/FPGA run has completed for this compiler. Independent audit passed, and the candidate is
ready for the root agent to submit using q535's immutable Jack Universal bitstream/runtime authority.
q535's successful hardware result remains baseline provenance, not qualification of this candidate.
