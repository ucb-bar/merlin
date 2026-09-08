# Jack Universal Gemmini narrow-INT8 recovery v1

> **Erratum (2026-09-08):** the original report incorrectly treated
> `hardcode_d_to_garbage_address` as disabling convolution bias. It disables the
> ordinary execute-path D operand, but `LoopConvLdBias` independently loads i32
> per-channel bias into accumulator rows with `LOAD3_CMD`. The inventory remains
> 0/53 for the current compiler because every convolution needs per-channel scale
> partitioning (and some also cross residual/pool/global operations), not because
> bias is unavailable. The analyzer, generated inventory, and regression below
> have been corrected. A native LOOP_CONV recovery supersedes the conclusion in
> the original prose.

## Result

The exact prepared W8A8 ResNet-50 has **0/53 convolutions admitted** for a
semantics-preserving native narrow result on Jack's Universal Gemmini target.
This is a tested refusal result, not a FireSim candidate or performance claim.

The audit found 53/53 integer convolutions, 53/53 folded FP32 bias vectors with
every channel nonzero, and 53/53 with genuinely per-channel weight scales.
Twenty convolution paths participate in a residual add before quantization;
the stem requires max-pool; the final convolution reaches global reduction/FC.

The decisive remaining restriction is that each convolution uses genuinely
per-channel scaling, while one LOOP_CONV store descriptor supplies one scale.
The compiler must partition output channels by scale (or emit an equivalent
exact post-accumulation transformation). Twenty paths also cross residual adds,
the stem crosses max-pool, and the final path crosses global reduction/FC.

## Implemented safe vertical slice

`analyze_narrow_epilogues.py` is a fail-closed graph admission pass. It runs the
compiler's Q/DQ-to-integer preparation, recognizes the direct convolutions,
follows every result through requant/bias/activation/residual/pool/quantization,
resolves the actual frozen arrays through the argument manifest, and emits
per-convolution geometry and exact refusal reasons.

The complete 53-row result is
`resnet50_narrow_epilogue_inventory.json`. The copied compiler retains the
job-533 live-D refusal guard, preventing regression to invalid radix recovery.

## Reproduce

From the repository root:

```sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/out/artifacts/perf-bench/gemmini/universal_narrow_epilogue_recovery_v1_20260908/compiler"
.venv/bin/python -m pytest -q \
  out/artifacts/perf-bench/gemmini/universal_narrow_epilogue_recovery_v1_20260908/tests
```

Expected: `2 passed` (the target live-D guard and exact 53-convolution census).

## Hardware qualification

`not_run_no_candidate_admitted`. No FPGA/FireSim job was submitted. Job 533 is
the hardware evidence disproving the predecessor: 999/1000 logits mismatched
although Spike passed.

## Next legal boundary

Use the already implemented native LOOP_CONV mechanism, its dedicated LOAD3
bias path, and channel-partitioned store configuration to form an exact narrow
boundary. Then retain residual/global operations on Rocket until separately
proven fusible. Approximate fusion is deliberately not admitted under the exact
1000-logit gate.
