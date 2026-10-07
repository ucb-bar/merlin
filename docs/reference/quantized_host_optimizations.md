---
title: Quantized host optimizations
kind: reference
status: current
owner: core
last_verified: 2026-10-06
related: [architecture, lowering_pipeline]
code_refs: [src/merlin/llvmlower/requantization.py, src/merlin/llvmlower/integer_readout.py, src/merlin/llvmlower/guarded_quantized_mean.py]
---

# Quantized host optimizations

These explicit compiler utilities specialize numeric contracts from the current input,
independently of the accelerator or workload. They do not select behavior from model names,
captured operation IDs, golden outputs or calibration samples. Target providers retain
instruction selection, hardware legality, device schedules and ABI glue.

## Ordered requantization

`requantization.py` compares complete monotone signed-i8 output transitions over the stated
signed-i32 accumulator interval. Positive finite binary32 scales, separate ordered operations,
nearest-even rounding and signed saturation are required. Scale synthesis independently
rechecks all transitions; constant channel bias synthesis excludes accumulator overflow.
An infeasible scale returns conflicting constraints or a concrete witness. The error-bound
API reports the exact local output error and requires an explicit caller policy and full-model
accuracy gate before accepting approximation.

`integer_readout.py` derives every source threshold and emits CPU C code with binary search
or a proved fixed-point estimate and at most one neighboring-threshold correction. Emission
re-derives the proof, rejects mutated fields and traps inputs outside the proven interval.
The fixed-point variant requires arithmetic signed right shift and a checked signed64 product
bound. It does not assume an accelerator's scaling or rounding semantics.

## Guarded quantized mean and packed input

`guarded_quantized_mean.py` proves rational error bounds for a serial binary32 Q/DQ reduction
of signed-i8 values. A table covers every reachable integer sum; ambiguous sums replay the
original float operations in reduction-index order. Static reduction count1..128 and positive
finite scales/reciprocal are required; possible binary32 overflow is refused.

The contiguous emitter accepts a positive channel count. The packed NHWC emitter accepts
positive batches and channels divisible by8. It checks little-endian word layout, bounds
unsigned16 lane sums to prevent cross-lane carry, and uses a scalar path for unaligned input.
Both validate the certificate before emission. CPU readout and packing are shared compiler
infrastructure; provider-specific descriptors and device commands remain OOT.

## Verification

Core tests exhaust independent small accumulator domains, solve and refuse scale constraints,
check bias overflow/inexact bias, compare compiled readout with ordered C float arithmetic at
every transition and nearby value, and exhaust all65,536 two-element mean inputs. Packed and
contiguous tests include adversarial cancellation order, count128 lane bounds, multiple
batches, unaligned input, output guards and certificate tampering. These establish the stated
numeric contracts; model performance remains a separately measured result.
