# Jack Universal Gemmini narrow-INT8 recovery v1

## Result

The exact prepared W8A8 ResNet-50 has **0/53 convolutions admitted** for a
semantics-preserving native narrow result on Jack's Universal Gemmini target.
This is a tested refusal result, not a FireSim candidate or performance claim.

The audit found 53/53 integer convolutions, 53/53 folded FP32 bias vectors with
every channel nonzero, and 53/53 with genuinely per-channel weight scales.
Twenty convolution paths participate in a residual add before quantization;
the stem requires max-pool; the final convolution reaches global reduction/FC.

The decisive target restriction is
`hardcode_d_to_garbage_address: true`. In weight-stationary execution, Jack's
RTL discards the operand carrying the accumulator preload. Therefore the bias
accepted by the LOOP_CONV programming interface cannot provide a legal bias
preload on this configuration. Merely changing the current output-row LOOP_WS
emitter to LOOP_CONV would still produce wrong results.

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

The same RTL needs a mechanism independent of D, or a target change exposing
live accumulator/bias preload. Channel-partitioned scale programming can handle
per-channel scales and LOOP_CONV can absorb the stem pool, but neither solves
nonzero bias on hardcoded-D hardware. Approximate fusion is deliberately not
admitted under the exact 1000-logit gate.
