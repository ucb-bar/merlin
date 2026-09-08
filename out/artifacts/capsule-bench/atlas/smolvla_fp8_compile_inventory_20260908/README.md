# Atlas SmolVLA native-FP8 compile inventory

This is a bounded, compile-only end-to-end layer inventory for the captured SmolVLA model. It uses
Atlas' native `fp8_e4m3` operand and `bf16` accumulator path; it does not relabel INT8 data as FP8.

## Result

- The complete capture contains 8,570 routed operation demands: 470 mesh, 8,097 scalar/RVV, and 3
  fallback.
- Provenance-aware inspection finds 391 actual contraction kernels: 303 rank-2 matmuls and 88 batched
  matmuls. Their sources are 225 plain matmuls, 77 addmm contractions, 88 batch matmuls, and 1
  im2col convolution matmul.
- Those 391 layers reduce to 28 unique kernel shapes. The fixed 49/61 Atlas backend accepts and emits
  all 28 in 22.9 seconds wall time; summed backend subprocess time is 12.2 seconds. No shape timed out
  or was refused.
- Emission is not execution readiness. The generated RTL has a 32,768-word instruction memory. Only
  13/28 unique shapes fit it, accounting for 205/391 physical contraction layers. The other 15 unique
  shapes (186 layers) require compact runtime loops; the largest current program is 2,774,543 words.
- No simulator was launched. This artifact makes no halt, numeric-correctness, performance, or
  single-whole-image claim.

The difference between 470 mesh demands and 391 physical contraction kernels is real: model2MLIR
copies a source contraction's provenance tag onto bias and im2col support operations. `demands.jsonl`
preserves the router's raw view; `layers.jsonl` records only structurally identified contraction
kernels.

## Provenance

- Capture: `out/artifacts/recaptures/smolvla_fp32_consistent/model.mlir`, SHA-256 recorded in
  `summary.json`.
- Compiler package:
  `out/runs/atlas/capsule-bench/merlin_assisted/merlincirct_atlas_fresh_func_20260907/submission`,
  source-only tree SHA-256 recorded in `summary.json`.
- Capsule score: 49/61 from the run's hashed `plateau.json`.
- The compiled package is mechanically checked to contain all three `* 2` branch-displacement fixes
  and the compact-schedule threshold. The exact codegen file hash is in `summary.json`.
- Instruction-memory capacity comes from generated GSIM RTL
  `/scratch/agustin/tmp/gsim-atlas-core/AtlasCore.h:18413` (`uint32_t imem$mem[32768]`); its SHA-256 is
  recorded in `summary.json`.

## Reproduce and validate

From the repository root:

```bash
PYTHONPATH=merlin/python timeout --signal=TERM --kill-after=10s 600s \
  python out/artifacts/capsule-bench/atlas/smolvla_fp8_compile_inventory_20260908/run_inventory.py

python out/artifacts/capsule-bench/atlas/smolvla_fp8_compile_inventory_20260908/validate_inventory.py
```

`run_inventory.py` independently caps every backend invocation at 60 seconds and the entire run at
600 seconds. Emitted assembly is retained deterministically as `kernels/*.S.gz`; the original and
compressed hashes and byte counts are in `unique_shapes.json`.

## Files

- `summary.json`: headline counts, limits, hashes, backend score, control-flow-fix evidence, and IMEM fit.
- `demands.jsonl`: every one of the 8,570 routed captured operations, with unit/bucket/gap.
- `layers.jsonl`: every physical contraction layer mapped to a unique emitted kernel and IMEM verdict.
- `unique_shapes.json`: compile time, status, instruction words/bytes, hashes, and occurrence count.
- `interfaces/`: exact `merlin_iface` inputs to the Atlas compiler.
- `kernels/`: gzip-compressed emitted self-hosted Atlas assembly.
- `stderr/`: one backend stderr file per unique shape.
