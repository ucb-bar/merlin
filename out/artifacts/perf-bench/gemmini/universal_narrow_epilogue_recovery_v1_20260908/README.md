# Jack Universal Gemmini narrow-INT8 recovery v1

> **Erratum (2026-09-08):** the original report incorrectly treated
> `hardcode_d_to_garbage_address` as disabling convolution bias. It disables the
> ordinary execute-path D operand, but `LoopConvLdBias` independently loads i32
> per-channel bias into accumulator rows with `LOAD3_CMD`. The inventory remains
> 0/53 for the current capture because its FP32 bias is not an exact i32
> accumulator offset and every convolution also needs per-channel scale
> partitioning. The analyzer, generated inventory, and regressions below have
> been corrected. A native LOOP_CONV recovery supersedes the conclusion in the
> original prose, but it requires a separately declared native-aligned W8A8
> arithmetic contract rather than silently approximating the current PT2E graph.

## Result

The exact prepared W8A8 ResNet-50 has **0/53 convolutions admitted** for a
semantics-preserving native narrow result on Jack's Universal Gemmini target.
This is a tested refusal result, not a FireSim candidate or performance claim.

The audit found 53/53 integer convolutions, 53/53 folded FP32 bias vectors with
every channel nonzero, and 53/53 with genuinely per-channel weight scales.
Twenty convolution paths participate in a residual add before quantization;
the stem requires max-pool; the final convolution reaches global reduction/FC.

The dedicated LOAD3 mechanism solves bias *transport*, but not the captured
arithmetic. All 53 folded FP32 bias vectors have zero channels exactly
representable as integer accumulator offsets. Each convolution also uses
genuinely per-channel scaling, while one LOOP_CONV store descriptor supplies one
scale. Channel partitioning would require at least **26,560 LOOP_CONV launches**
(the sum of output channels), before spatial/K tiling, and would still not repair
the FP32 bias-rounding difference. Twenty paths also cross residual adds, the
stem crosses max-pool, and the final path crosses global reduction/FC.

## Deterministic first-layer proof

`prove_conv1_native_requant.py` executes conv1's real 7x7 i8 contraction on the
measured dog input, then compares:

1. the existing PT2E integer-reference order (`acc -> *sa -> *sw[c] -> +bias ->
   ReLU -> quantize`), and
2. native LOOP_CONV (`acc + round(bias/(sa*sw[c])) -> one CONFIG_ST scale ->
   RNE/saturate/ReLU`).

The standard native construction differs at **143/802,816** pre-pool values and
**33/200,704** post-pool values, across 41 and 17 channels respectively; every
difference is one quantization level. Searching bias ±3 and scale ±8 ULP makes
only 25/64 channels exact and still leaves 114 differences. Equal native i8
values require different correct PT2E i8 values in those same 41 channels (17
post-pool), so an elementwise host repair after narrow readout cannot recover the
lost information.

An explicit per-tensor-weight ablation also fails: after requantizing the
captured conv1 weight to one scalar scale, the two contracts differ at 364
pre-pool and 110 post-pool values, and 0/64 FP32 biases are exact accumulator
integers. Per-tensor weight scale removes the global CONFIG_ST channel conflict;
it does **not** by itself make the old PT2E arithmetic exact.

## Implemented safe vertical slice

`analyze_narrow_epilogues.py` is a fail-closed graph admission pass. It runs the
compiler's Q/DQ-to-integer preparation, recognizes the direct convolutions,
follows every result through requant/bias/activation/residual/pool/quantization,
resolves the actual frozen arrays through the argument manifest, and emits
per-convolution geometry and exact refusal reasons.

The complete 53-row result is
`resnet50_narrow_epilogue_inventory.json`; the first-layer numeric receipt is
`conv1_native_requant_proof.json`. The copied compiler retains the job-533
live-D refusal guard, preventing regression to invalid radix recovery.

## Reproduce

From the repository root:

```sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/out/artifacts/perf-bench/gemmini/universal_narrow_epilogue_recovery_v1_20260908/compiler"
.venv/bin/python -m pytest -q \
  out/artifacts/perf-bench/gemmini/universal_narrow_epilogue_recovery_v1_20260908/tests
```

Expected: `3 passed` (the target live-D guard, exact 53-convolution census, and
deterministic conv1 arithmetic proof).

## Hardware qualification

`not_run_no_candidate_admitted`. No FPGA/FireSim job was submitted. Job 533 is
the hardware evidence disproving the predecessor: 999/1000 logits mismatched
although Spike passed.

## Next legal boundary

Use the already implemented native LOOP_CONV mechanism with a distinct,
provenance-labelled native-aligned W8A8 contract: per-tensor symmetric i8
activation/weight, i32 bias quantized once in accumulator units, and one f32
CONFIG_ST multiply/RNE/saturation. Its independently generated golden and model
quality (1000 finite logits and expected top-1) must pass before any full compile
or Spike run. This is not interchangeable with the existing TorchAO PT2E golden.
Approximate fusion is deliberately not admitted under either exact gate.
